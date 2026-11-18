"""Duplicate-aware train/validation split and isolated CPU model comparisons."""
import json
import logging
import multiprocessing as mp
import platform
import uuid
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from src.benchmarks import ResourceMonitor
from src.config import DEFAULTS, LABELS, MODEL_NAMES, TARGETS, directories
from src.database import connect, db_path, record_predictions
from src.evaluate import comparison_frame, evaluate_targets
from src.predict import load_bundle, probability_matrix
from src.utils import read_json, save_json, sha256_file, timestamp


def split_training(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stratify duplicate-text groups by any_toxic and never split a group."""
    groups = frame.groupby('text_group', sort=True).any_toxic.max()
    if len(groups) < 4:
        raise ValueError('At least four distinct nonempty training comments are required')
    counts = groups.value_counts()
    n_validation = int(np.ceil(len(groups) * config['validation_size']))
    n_train = len(groups) - n_validation
    stratify = groups if len(counts) == 2 and counts.min() >= 2 and min(n_train, n_validation) >= 2 else None
    note = ('Stratified by group-level any_toxic. Rare labels may remain uneven.' if stratify is not None
            else 'Unstratified fallback: too few any_toxic groups for valid stratification.')
    a, b = train_test_split(groups.index.to_numpy(), test_size=config['validation_size'],
                            random_state=config['seed'], stratify=stratify)
    train = frame.loc[frame.text_group.isin(a)].reset_index(drop=True)
    validation = frame.loc[frame.text_group.isin(b)].reset_index(drop=True)
    assert set(train.text_group).isdisjoint(validation.text_group)
    return train, validation, dict(note=note, train_rows=len(train), validation_rows=len(validation),
                                   train_groups=len(a), validation_groups=len(b))


def fit_bundle(name: str, train: pd.DataFrame, config: dict) -> dict:
    """Fit a shared sparse representation and seven independent classifiers."""
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown model {name}')
    options = dict(max_features=config['max_features'], ngram_range=tuple(config['ngram_range']),
                   min_df=config['min_df'], strip_accents=None, lowercase=True,
                   token_pattern=r'(?u)\b\w+\b', dtype=np.float32)
    if name == 'dummy':
        vectorizer = None
        matrix = sparse.csr_matrix((len(train), 1))
    elif name == 'bow_nb':
        vectorizer = CountVectorizer(**options)
        matrix = vectorizer.fit_transform(train.comment_text)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, **options)
        matrix = vectorizer.fit_transform(train.comment_text)
    models = []
    fallback_labels = []
    for label in TARGETS:
        y = train[label].to_numpy()
        if name == 'dummy' or len(np.unique(y)) < 2:
            estimator = DummyClassifier(strategy='prior')
            if name != 'dummy':
                fallback_labels.append(label)
        elif name == 'bow_nb':
            estimator = MultinomialNB(alpha=1.0)
        else:
            estimator = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000,
                                           class_weight='balanced' if name.endswith('balanced') else None,
                                           random_state=config['seed'])
        estimator.fit(matrix, y)
        models.append(estimator)
    return dict(model_name=name, vectorizer=vectorizer, estimators=models, targets=TARGETS,
                thresholds={label: 0.5 for label in TARGETS}, configuration=config,
                single_class_fallbacks=fallback_labels)


def _train_worker(root_string: str, name: str, experiment_id: str, config: dict) -> None:
    """Spawned process avoids cumulative memory from previously trained models."""
    root = Path(root_string)
    paths = directories(root)
    destination = paths['models'] / experiment_id
    destination.mkdir(exist_ok=True)
    frame = pd.read_parquet(paths['processed'] / 'train.parquet')
    assignments = pd.read_parquet(destination / 'split_assignments.parquet').set_index('id')['partition']
    train = frame.loc[frame.id.map(assignments).eq('train')].reset_index(drop=True)
    validation = frame.loc[frame.id.map(assignments).eq('validation')].reset_index(drop=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        with ResourceMonitor() as training:
            bundle = fit_bundle(name, train, config)
        with ResourceMonitor() as inference:
            parts = [probability_matrix(bundle, validation.comment_text.iloc[start:start + config['batch_size']].tolist())
                     for start in range(0, len(validation), config['batch_size'])]
            probabilities = np.vstack(parts)
    run_id = f'{experiment_id}_{name}'
    bundle.update(run_id=run_id, experiment_id=experiment_id)
    artifact = destination / f'{name}.joblib'
    joblib.dump(bundle, artifact, compress=3)
    run = dict(run_id=run_id, model_name=name, created_at=timestamp(), configuration=config,
               training_time=training.seconds, inference_time=inference.seconds,
               training_peak_rss_mb=training.peak_mb, inference_peak_rss_mb=inference.peak_mb,
               peak_memory_mb=max(training.peak_mb, inference.peak_mb),
               training_increment_rss_mb=training.increment_mb,
               artifact=str(artifact.relative_to(root).as_posix()), artifact_sha256=sha256_file(artifact),
               single_class_fallbacks=bundle['single_class_fallbacks'],
               warnings=sorted(set(str(w.message) for w in captured)),
               validation=evaluate_targets(validation[TARGETS], probabilities))
    save_json(destination / f'{name}.json', run)
    np.savez_compressed(destination / f'{name}_validation.npz', probabilities=probabilities)


def train_models(root: Path, config: dict | None = None) -> dict:
    paths = directories(root)
    settings = {**DEFAULTS, **(config or {})}
    if not 0 < settings['validation_size'] < 1 or settings['min_df'] < 1 or settings['max_features'] < 1:
        raise ValueError('Invalid validation fraction or vectorizer limits')
    manifest = read_json(paths['processed'] / 'manifest.json')
    if not manifest:
        raise FileNotFoundError('Run python -m scripts.prepare_data before training')
    frame = pd.read_parquet(paths['processed'] / 'train.parquet')
    train, validation, split_info = split_training(frame, settings)
    experiment_id = uuid.uuid4().hex[:12]
    destination = paths['models'] / experiment_id
    destination.mkdir()
    assignments = pd.concat([train[['id']].assign(partition='train'),
                             validation[['id']].assign(partition='validation')], ignore_index=True)
    assignments.to_parquet(destination / 'split_assignments.parquet', index=False)
    del train, frame
    runs = []
    for name in MODEL_NAMES:
        logging.info('Training %s in an isolated CPU process', name)
        process = mp.get_context('spawn').Process(target=_train_worker,
                      args=(str(root), name, experiment_id, settings))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f'{name} training failed. Inspect worker error; no winner has been published.')
        runs.append(read_json(destination / f'{name}.json'))
    # Stable declared order resolves an exact tie, after any_toxic F1.
    scored_runs = [run for run in runs if run['validation']['mean_roc_auc_six'] is not None]
    if not scored_runs:
        raise ValueError('No original-label validation ROC-AUC is defined. Supply more labeled data.')
    winner = max(scored_runs, key=lambda r: (r['validation']['mean_roc_auc_six'],
                 r['validation']['per_label']['any_toxic']['f1'], -MODEL_NAMES.index(r['model_name'])))
    experiment = dict(experiment_id=experiment_id, created_at=timestamp(), configuration=settings,
                      dataset_manifest=manifest, python=platform.python_version(), split=split_info,
                      split_assignments=str((destination / 'split_assignments.parquet').relative_to(root).as_posix()),
                      selection_criterion='Highest validation mean ROC-AUC over evaluable original six labels; '
                      'ties use any_toxic F1 at 0.5, then declared model order. Official test is untouched.',
                      memory_method='Each model runs in a fresh spawned process. Absolute process RSS sampled '
                      'every 10 ms during fit and inference, after loading data. Includes interpreter and data '
                      'already in memory. Short peaks may be missed. Timing excludes disk I/O and metrics.',
                      selected_model=winner['model_name'], runs=runs)
    # Verify the database came from the same processed data before attaching predictions.
    if db_path(root).exists():
        connection = connect(db_path(root), readonly=True)
        try:
            stored = json.loads(connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
        finally:
            connection.close()
        if stored['files'] != manifest['files']:
            raise ValueError('Database source mismatch. Rebuild the database before publishing this experiment.')
        for run in runs:
            with np.load(destination / f"{run['model_name']}_validation.npz") as saved:
                record_predictions(root, run, validation, saved['probabilities'])
    save_json(destination / 'experiment.json', experiment)
    save_json(paths['metrics'] / 'experiment.json', experiment)
    comparison_frame(experiment).to_csv(paths['metrics'] / 'comparison.csv', index=False)
    per_label = []
    for run in runs:
        for label, values in run['validation']['per_label'].items():
            per_label.append({'model': run['model_name'], 'label': label,
                              **{k: v for k, v in values.items() if k not in ['roc', 'pr']}})
    pd.DataFrame(per_label).to_csv(paths['metrics'] / 'per_label.csv', index=False)
    save_json(paths['models'] / 'selection.json', dict(experiment_id=experiment_id, selected_model=winner['model_name'],
              artifact=winner['artifact'], dataset_files=manifest['files']))
    return experiment


def evaluate_official(root: Path) -> dict:
    """Evaluate only the already selected model, once per experiment."""
    paths = directories(root)
    experiment = read_json(paths['metrics'] / 'experiment.json')
    if not experiment:
        raise FileNotFoundError('Train models first. Model selection uses validation only.')
    result_path = paths['metrics'] / 'official_test' / f"{experiment['experiment_id']}.json"
    if result_path.exists():
        return read_json(result_path)
    manifest = read_json(paths['processed'] / 'manifest.json')
    if manifest['files'] != experiment['dataset_manifest']['files']:
        raise ValueError('Prepared inputs changed after training. Train a new experiment first.')
    test_path = paths['processed'] / 'test.parquet'
    if not test_path.exists():
        return dict(status='unavailable', reason='Official test.csv is absent. No held-out metrics computed.')
    test = pd.read_parquet(test_path)
    test = test.loc[test[LABELS].isin([0, 1]).all(axis=1)].reset_index(drop=True)
    if test.empty:
        return dict(status='unavailable', reason='No fully scored test rows. Supply test_labels.csv; -1 is never negative.')
    run = next(r for r in experiment['runs'] if r['model_name'] == experiment['selected_model'])
    bundle = load_bundle(root / run['artifact'])
    with ResourceMonitor() as monitor:
        p = np.vstack([probability_matrix(bundle, test.comment_text.iloc[start:start + 2048].tolist())
                       for start in range(0, len(test), 2048)])
    result = dict(status='evaluated', experiment_id=experiment['experiment_id'], selected_model=run['model_name'],
                  evaluation=evaluate_targets(test[TARGETS], p), inference_time=monitor.seconds,
                  peak_memory_mb=monitor.peak_mb, created_at=timestamp(),
                  train_test_duplicate_text_rows=manifest['train_test_duplicate_text_rows'])
    save_json(result_path, result)
    record_predictions(root, run, test, p)
    return result

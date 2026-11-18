"""Prespecified training-source-only context audit and threshold experiment.

No official-test file or labels are read here. All five candidates are refitted
on the same new fit partition; historical saved models are never overwritten.
"""
import json
import logging
import multiprocessing as mp
import re
import uuid
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from src.benchmarks import ResourceMonitor
from src.config import DEFAULTS, LABELS, MODEL_NAMES, TARGETS
from src.evaluate import evaluate_targets
from src.predict import load_bundle, probability_matrix
from src.thresholds import apply_thresholds
from src.train import fit_bundle
from src.utils import read_json, save_json, sha256_file, timestamp

CANDIDATES = MODEL_NAMES + ['word_char_lr_balanced']
RECALL_FLOORS = {label: (0.85 if label == 'threat' else 0.80) for label in TARGETS}
COHORT_PATTERNS = {
    'profanity_cue': r'\b(?:fuck\w*|shit\w*|damn\w*|bastard\w*|asshole\w*|bitch\w*)\b',
    'positive_word_cue': r'\b(?:love|lovely|amazing|excellent|wonderful|fantastic|great|brilliant|awesome|beautiful|enjoy\w*|thank\w*)\b',
    'criticism_cue': r'\b(?:worst|bad|poor|wrong|incorrect|disagree|terrible|awful|boring|useless|mistake\w*|error\w*)\b',
    'identity_mention_cue': r'\b(?:religio\w*|muslim\w*|christian\w*|jew\w*|atheis\w*|gay|lesbian\w*|black|white|women|woman|immigran\w*)\b',
    'threat_cue': r'\b(?:kill\w*|murder\w*|shoot\w*|stab\w*|hurt|attack\w*|beat\w*|die|dead|death)\b',
    'insult_cue': r'\b(?:idiot\w*|stupid|moron\w*|dumb|loser\w*|pathetic|worthless|ignorant)\b',
}


def context_cohorts(texts: pd.Series) -> dict[str, np.ndarray]:
    """Retrieval slices only: cues never alter labels, features or predictions.

    Names intentionally describe word co-occurrence rather than claiming that a
    regex establishes positive intent, discrimination, sarcasm or harmlessness.
    Ground truth is the independently supplied Jigsaw annotation for each row.
    """
    masks = {name: texts.str.contains(pattern, flags=re.I, regex=True, na=False).to_numpy(dtype=bool)
             for name, pattern in COHORT_PATTERNS.items()}
    masks['profanity_and_positive_word_cues'] = masks['profanity_cue'] & masks['positive_word_cue']
    masks['criticism_without_profanity_cues'] = masks['criticism_cue'] & ~masks['profanity_cue']
    return masks


def select_thresholds(truth: pd.DataFrame, probabilities: np.ndarray,
                      floors: dict | None = None) -> tuple[dict, list[dict]]:
    """Minimize calibration false positives subject to each recall floor.

    The highest observed score threshold with sufficient recall is selected.
    This changes only hard decisions. No assessment rows or supplied examples
    participate. Single-class/no-positive calibration targets retain 0.5.
    """
    floors = floors or RECALL_FLOORS
    selected, records = {}, []
    for j, label in enumerate(TARGETS):
        y, p = truth[label].to_numpy(), probabilities[:, j]
        if len(np.unique(y)) < 2:
            selected[label] = 0.5
            records.append(dict(label=label, threshold=0.5, status='single-class calibration; no tuning'))
            continue
        precision, recall, thresholds = precision_recall_curve(y, p)
        feasible = np.flatnonzero(recall[:-1] >= floors[label])
        index = int(feasible[-1])
        selected[label] = float(thresholds[index])
        # Thin only the displayed trade-off table, never the optimization data.
        points = np.unique(np.append(np.linspace(0, len(thresholds)-1, min(101, len(thresholds))).astype(int), index))
        for i in points:
            records.append(dict(label=label, threshold=float(thresholds[i]), precision=float(precision[i]),
                                recall=float(recall[i]), selected=bool(i == index), recall_floor=floors[label]))
    return selected, records


def make_context_split(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """70/15/15 group split of original training source, stratified by any_toxic."""
    groups = frame.groupby('text_group', sort=True).any_toxic.max()
    fit, remaining = train_test_split(groups.index.to_numpy(), test_size=0.30,
                                      stratify=groups, random_state=seed)
    calibration, assessment = train_test_split(remaining, test_size=0.5,
                        stratify=groups.loc[remaining], random_state=seed + 1)
    assignment = pd.Series('fit', index=frame.index)
    assignment.loc[frame.text_group.isin(calibration)] = 'calibration'
    assignment.loc[frame.text_group.isin(assessment)] = 'assessment'
    return frame[['id', 'text_group']].assign(partition=assignment)


def fit_context_candidate(frame: pd.DataFrame, config: dict) -> dict:
    """Six independent binary logistic outputs plus separate any_toxic output.

    Word bigrams add local phrase evidence; character word n-grams add spelling
    and negation-form coverage. This remains a linear lexical model, not a
    Transformer or a reliable estimator of intent.
    """
    vectorizer = FeatureUnion([
        ('word', TfidfVectorizer(max_features=50000, min_df=2, ngram_range=(1, 2),
                    token_pattern=r'(?u)\b\w+\b', sublinear_tf=True, dtype=np.float32)),
        ('character', TfidfVectorizer(analyzer='char_wb', max_features=30000, min_df=3,
                    ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32)),
    ], transformer_weights={'word': 1.0, 'character': 0.5}, n_jobs=1)
    matrix = vectorizer.fit_transform(frame.comment_text).tocsr()
    estimators, fallback = [], []
    for label in TARGETS:
        if frame[label].nunique() < 2:
            model = DummyClassifier(strategy='prior')
            fallback.append(label)
        else:
            model = LogisticRegression(C=1.0, solver='liblinear', max_iter=1000,
                                        class_weight='balanced', random_state=config['seed'])
        model.fit(matrix, frame[label].to_numpy())
        estimators.append(model)
    return dict(model_name='word_char_lr_balanced', targets=TARGETS, vectorizer=vectorizer,
                estimators=estimators, thresholds=dict.fromkeys(TARGETS, 0.5),
                single_class_fallbacks=fallback, configuration=config)


def _score_texts(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    return np.vstack([probability_matrix(bundle, frame.comment_text.iloc[start:start+2048].tolist())
                      for start in range(0, len(frame), 2048)])


def _context_worker(root_string: str, run_id: str, name: str) -> None:
    root = Path(root_string)
    path = root / 'models/contextual' / run_id
    config = read_json(path / 'protocol.json')
    frame = pd.read_parquet(root / 'data/processed/train.parquet')
    assignments = pd.read_parquet(path / 'split_assignments.parquet').set_index('id').partition
    partitions = {key: frame.loc[frame.id.map(assignments).eq(key)].reset_index(drop=True)
                  for key in ['fit', 'calibration', 'assessment']}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        with ResourceMonitor() as fit:
            bundle = (fit_context_candidate(partitions['fit'], config) if name == 'word_char_lr_balanced'
                      else fit_bundle(name, partitions['fit'], {**DEFAULTS, 'seed': config['seed']}))
        with ResourceMonitor() as inference:
            calibration = _score_texts(bundle, partitions['calibration'])
            assessment = _score_texts(bundle, partitions['assessment'])
    thresholds, tradeoffs = select_thresholds(partitions['calibration'][TARGETS], calibration,
                                             config['recall_floors'])
    bundle.update(thresholds=thresholds, run_id=f'{run_id}_{name}', experiment_id=run_id,
                  threshold_source='Calibration partition only; highest threshold satisfying per-label recall floors.')
    artifact = path / f'{name}.joblib'
    joblib.dump(bundle, artifact, compress=3)
    np.savez_compressed(path / f'{name}_scores.npz', calibration=calibration, assessment=assessment)
    pd.DataFrame(tradeoffs).to_csv(path / f'{name}_threshold_tradeoffs.csv', index=False)
    result = dict(model_name=name, artifact=artifact.relative_to(root).as_posix(),
                  artifact_sha256=sha256_file(artifact), thresholds=thresholds,
                  training_seconds=fit.seconds, inference_seconds=inference.seconds,
                  inference_rows=len(calibration)+len(assessment),
                  inference_ms_per_1000=1000 * inference.seconds / (len(calibration)+len(assessment)) * 1000,
                  peak_rss_mb=max(fit.peak_mb,inference.peak_mb),
                  warnings=sorted(set(str(w.message) for w in captured)),
                  single_class_fallbacks=bundle['single_class_fallbacks'],
                  calibration_default=evaluate_targets(partitions['calibration'][TARGETS], calibration),
                  calibration_tuned=evaluate_targets(partitions['calibration'][TARGETS], calibration, thresholds))
    save_json(path / f'{name}.json', result)


def _metric_row(name: str, policy: str, result: dict, resource: dict) -> dict:
    binary = result['per_label']['any_toxic']
    return dict(model=name, policy=policy, mean_auc_six=result['mean_roc_auc_six'],
                macro_ap_six=result['macro_average_precision_six'],
                binary_precision=binary['precision'], binary_recall=binary['recall'], binary_f1=binary['f1'],
                false_positive_rate=binary['false_positive_rate'],
                threat_recall=result['per_label']['threat']['recall'],
                identity_hate_recall=result['per_label']['identity_hate']['recall'],
                inference_ms_per_1000=resource['inference_ms_per_1000'], peak_rss_mb=resource['peak_rss_mb'],
                training_seconds=resource['training_seconds'])


def wilson_interval(success: int, total: int) -> list[float] | None:
    if not total:
        return None
    z, p = 1.95996398454, success / total
    den = 1 + z*z/total
    mid = (p + z*z/(2*total))/den
    radius = z * np.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [float(mid-radius), float(mid+radius)]


def paired_fp_comparison(truth, reference, candidate, seed: int) -> dict:
    mask = np.asarray(truth) == 0
    r, c = np.asarray(reference)[mask], np.asarray(candidate)[mask]
    categories = np.bincount(r.astype(int)*2+c.astype(int), minlength=4)
    draws = np.random.default_rng(seed).multinomial(len(r), categories/len(r), size=10000)
    differences = (draws[:,1]-draws[:,2])/len(r)
    return dict(negative_rows=len(r), avoided_false_positives=int(categories[2]),
                introduced_false_positives=int(categories[1]),
                candidate_minus_reference_fpr=float((c.sum()-r.sum())/len(r)),
                paired_bootstrap_95_interval=np.quantile(differences,[0.025,0.975]).tolist(),
                note='Paired nonparametric bootstrap of the four discordance cells, 10,000 draws. Exploratory study; no multiplicity-adjusted superiority claim.')


def run_context_experiment(root: Path) -> dict:
    """Freeze calibration choice before accessing assessment metrics."""
    run_id = 'context_' + uuid.uuid4().hex[:10]
    model_dir = root / 'models/contextual' / run_id
    output = root / 'results/contextual' / run_id
    model_dir.mkdir(parents=True)
    output.mkdir(parents=True)
    source_manifest = read_json(root / 'data/processed/manifest.json')
    protocol = dict(run_id=run_id, created_at=timestamp(), seed=20261117,
                    fractions={'fit':0.70,'calibration':0.15,'assessment':0.15},
                    candidates=CANDIDATES, recall_floors=RECALL_FLOORS,
                    candidate_configuration='Word TF-IDF 50k unigrams/bigrams + character-word TF-IDF 30k 3-5 grams, weights 1/0.5; balanced independent LR C=1.',
                    selection='Calibration only: lowest any_toxic false-positive rate among candidates within 0.01 mean six-label AUC of the balanced word baseline, with all calibration recall floors satisfied. Tie-break by six-label AUC.',
                    assessment_acceptance={'relative_fpr_reduction_min':0.20,'any_recall_min':0.78,
                       'threat_recall_min':0.80,'identity_hate_recall_min':0.75,'mean_auc_loss_max':0.01,
                       'paired_fpr_interval_upper_below_zero':True},
                    source_training_sha256=source_manifest['files']['train.csv']['sha256'],
                    target_definition='Six independent original binary labels plus separately trained any_toxic=max(six truth labels), including obscene. It does not equal an intent-only abuse target.',
                    independent_labels='Pre-existing Jigsaw human annotations, supplied independently of these model predictions. Regex slices describe lexical cues only and do not relabel intent.',
                    cohorts=COHORT_PATTERNS, supplied_examples='Diagnostic-only. Not used for fitting, thresholds, hyperparameters or model selection.',
                    official_test='Not loaded or evaluated. Previously reported official-test results remain historical.',
                    limitation='New three-way held-out assessment within a historically explored training corpus; not a pristine external benchmark.',
                    artifact_dir=model_dir.relative_to(root).as_posix())
    save_json(model_dir / 'protocol.json', protocol)
    frame = pd.read_parquet(root / 'data/processed/train.parquet')
    assignments = make_context_split(frame, protocol['seed'])
    assignments.to_parquet(model_dir / 'split_assignments.parquet', index=False)
    part = assignments.set_index('id').partition
    calibration = frame.loc[frame.id.map(part).eq('calibration')].reset_index(drop=True)
    assessment = frame.loc[frame.id.map(part).eq('assessment')].reset_index(drop=True)
    # Save independently annotated assessment rows before model scoring.
    assessment[['id','comment_text']+TARGETS].to_parquet(output / 'independently_labeled_assessment.parquet', index=False)
    records = []
    for name in CANDIDATES:
        logging.info('Context experiment: fitting %s', name)
        process = mp.get_context('spawn').Process(target=_context_worker,args=(str(root),run_id,name))
        process.start(); process.join()
        if process.exitcode:
            raise RuntimeError(f'Context candidate {name} failed; prior experiment remains intact')
        records.append(read_json(model_dir / f'{name}.json'))
    reference = next(r for r in records if r['model_name']=='tfidf_lr_balanced')
    auc_floor = reference['calibration_tuned']['mean_roc_auc_six']-0.01
    eligible = [r for r in records if r['calibration_tuned']['mean_roc_auc_six'] >= auc_floor
                and all(r['calibration_tuned']['per_label'][label]['recall'] >= RECALL_FLOORS[label] for label in TARGETS)]
    if not eligible:
        raise ValueError('No candidate meets the calibration recall floors. Supply more labeled positives; no recommendation is published.')
    winner = min(eligible,key=lambda r:(r['calibration_tuned']['per_label']['any_toxic']['false_positive_rate'],
                                      -r['calibration_tuned']['mean_roc_auc_six']))
    frozen = dict(model=winner['model_name'],thresholds=winner['thresholds'],frozen_at=timestamp(),
                  reason=protocol['selection'],assessment_metrics_accessed=False)
    save_json(output / 'decision_before_assessment.json', frozen)
    summaries, per_label, slice_rows, errors = [], [], [], []
    masks = context_cohorts(assessment.comment_text)
    all_results, score_arrays = {}, {}
    for record in records:
        name=record['model_name']
        with np.load(model_dir / f'{name}_scores.npz') as scores:
            p=scores['assessment'].copy()
        score_arrays[name]=p
        results={policy:evaluate_targets(assessment[TARGETS],p,threshold)
                 for policy,threshold in [('default_0.5',0.5),('calibration_thresholds',record['thresholds'])]}
        all_results[name]=results
        for policy, result in results.items():
            summaries.append(_metric_row(name,policy,result,record))
            for label,values in result['per_label'].items():
                per_label.append(dict(model=name,policy=policy,label=label,
                      **{key:values[key] for key in ['roc_auc','average_precision','precision','recall','f1','false_positive_rate','positives','negatives','decision_threshold','confusion_matrix']}))
            predicted = apply_thresholds(p,dict.fromkeys(TARGETS,.5) if policy=='default_0.5' else record['thresholds'])
            for cohort,mask in masks.items():
                if not mask.any(): continue
                binary_y=assessment.any_toxic.to_numpy()[mask]; binary_p=predicted[mask,-1]
                negatives=int((binary_y==0).sum()); positives=int(binary_y.sum())
                fp=int(((binary_y==0)&(binary_p==1)).sum()); fn=int(((binary_y==1)&(binary_p==0)).sum())
                slice_rows.append(dict(model=name,policy=policy,cohort=cohort,rows=int(mask.sum()),
                       human_labeled_positives=positives,human_labeled_negatives=negatives,
                       false_positives=fp,false_negatives=fn,false_positive_rate=fp/negatives if negatives else None,
                       recall=1-fn/positives if positives else None))
            false_positive=np.flatnonzero((assessment.any_toxic.to_numpy()==0)&(predicted[:,-1]==1))
            ranked=false_positive[np.argsort(-p[false_positive,-1])][:25]
            for i in ranked:
                errors.append(dict(model=name,policy=policy,id=str(assessment.iloc[i].id),
                        comment_text=assessment.iloc[i].comment_text,ground_truth_any_toxic=0,
                        any_toxic_probability=float(p[i,-1]),
                        cohorts=','.join(k for k,mask in masks.items() if mask[i])))
    reference_result=all_results['tfidf_lr_balanced']['default_0.5']
    selected_result=all_results[winner['model_name']]['calibration_thresholds']
    rpred=(score_arrays['tfidf_lr_balanced'][:,-1]>=.5)
    cpred=(score_arrays[winner['model_name']][:,-1]>=winner['thresholds']['any_toxic'])
    paired=paired_fp_comparison(assessment.any_toxic.to_numpy(),rpred,cpred,protocol['seed'])
    base_fpr=reference_result['per_label']['any_toxic']['false_positive_rate']
    chosen_fpr=selected_result['per_label']['any_toxic']['false_positive_rate']
    reduction=(base_fpr-chosen_fpr)/base_fpr
    guards = dict(relative_fpr_reduction=reduction>=.20,
                  any_recall=selected_result['per_label']['any_toxic']['recall']>=.78,
                  threat_recall=selected_result['per_label']['threat']['recall']>=.80,
                  identity_hate_recall=selected_result['per_label']['identity_hate']['recall']>=.75,
                  mean_auc=selected_result['mean_roc_auc_six']>=reference_result['mean_roc_auc_six']-.01,
                  paired_fpr=paired['paired_bootstrap_95_interval'][1]<0)
    intervals={label:wilson_interval(selected_result['per_label'][label]['confusion_matrix'][1][1],
                                     selected_result['per_label'][label]['positives'])
               for label in ['any_toxic','threat','identity_hate']}
    study=dict(protocol=protocol,counts=assignments.partition.value_counts().to_dict(),runs=records,
               frozen_selection=frozen,assessment=all_results,
               selected_model=winner['model_name'],acceptance_guards=guards,
               accepted=all(guards.values()),relative_fpr_reduction=float(reduction),paired_false_positives=paired,
               selected_recall_wilson95=intervals,
               conclusion='Conditional operating-point improvement only if every prespecified guard passes. Thresholding cannot establish semantic understanding or fix dataset ontology.')
    pd.DataFrame(summaries).to_csv(output/'assessment_comparison.csv',index=False)
    pd.DataFrame(per_label).to_csv(output/'per_label_assessment.csv',index=False)
    pd.DataFrame(slice_rows).to_csv(output/'contextual_slices.csv',index=False)
    pd.DataFrame(errors).to_csv(output/'false_positive_examples.csv',index=False)
    save_json(output/'study.json',study)
    save_json(root/'results/contextual/latest.json',dict(run_id=run_id,
        study=(output/'study.json').relative_to(root).as_posix(),model_dir=model_dir.relative_to(root).as_posix(),
        training_sha256=protocol['source_training_sha256'],accepted=study['accepted'],selected_model=winner['model_name']))
    logging.info('Context assessment completed: selected %s; guards %s',winner['model_name'],guards)
    return study

"""Verify a saved real run without refitting or repeating official evaluation."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import pandas as pd
from src.config import ROOT, LABELS, MODEL_NAMES, TARGETS
from src.database import connect, db_path, predefined_queries, run_query
from src.evaluate import comparison_frame
from src.utils import read_json, save_json, sha256_file, timestamp


def verify_artifacts(root: Path) -> dict:
    manifest = read_json(root / 'data/processed/manifest.json')
    experiment = read_json(root / 'results/metrics/experiment.json')
    if not manifest or not experiment or manifest['files'] != experiment['dataset_manifest']['files']:
        raise ValueError('Current prepared data and completed experiment must match')
    checks = {}
    for name, source in manifest['files'].items():
        path = root / 'data/raw' / name
        checks[f'raw_hash:{name}'] = sha256_file(path) == source['sha256'] if path.exists() else 'raw source not bundled'
    for run in experiment['runs']:
        checks[f'model_hash:{run["model_name"]}'] = sha256_file(root / run['artifact']) == run['artifact_sha256']
    checks['four_candidates'] = [run['model_name'] for run in experiment['runs']] == MODEL_NAMES
    winner = max(experiment['runs'], key=lambda r: (r['validation']['mean_roc_auc_six'],
                  r['validation']['per_label']['any_toxic']['f1'], -MODEL_NAMES.index(r['model_name'])))
    checks['validation_selection'] = winner['model_name'] == experiment['selected_model']
    assignments = pd.read_parquet(root / experiment['split_assignments'])
    source = pd.read_parquet(root / 'data/processed/train.parquet', columns=['id', 'text_group'])
    merged = source.merge(assignments, on='id', validate='one_to_one')
    checks['split_covers_training'] = len(merged) == len(source)
    checks['duplicate_groups_disjoint'] = set(merged.loc[merged.partition.eq('train'), 'text_group']).isdisjoint(
                                            merged.loc[merged.partition.eq('validation'), 'text_group'])
    test = pd.read_parquet(root / 'data/processed/test.parquet', columns=LABELS)
    eligible = int(test.isin([0, 1]).all(axis=1).sum())
    official = read_json(root / 'results/metrics/official_test' / f"{experiment['experiment_id']}.json")
    checks['official_row_eligibility'] = official['evaluation']['rows'] == eligible
    connection = connect(db_path(root), readonly=True)
    try:
        counts = {name: connection.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
                  for name in ['comments', 'labels', 'model_runs', 'predictions']}
        checks['foreign_keys_valid'] = not connection.execute('PRAGMA foreign_key_check').fetchall()
        checks['sqlite_integrity'] = connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        checks['no_unscored_ground_truth'] = connection.execute(
            "SELECT COUNT(*) FROM labels WHERE label_value NOT IN (0,1)").fetchone()[0] == 0
        expected_predictions = (experiment['split']['validation_rows'] * len(MODEL_NAMES) + eligible) * len(TARGETS)
        checks['prediction_row_count'] = counts['predictions'] == expected_predictions
        checks['label_row_count'] = counts['labels'] == (len(source) + eligible) * len(TARGETS)
    finally:
        connection.close()
    sql_rows = {name: len(run_query(db_path(root), name)) for name in predefined_queries()}
    checks['eight_sql_results'] = len(sql_rows) == 8 and all(value > 0 for value in sql_rows.values())
    if any(value is False for value in checks.values()):
        raise ValueError(f'Artifact verification failed: {checks}')
    metrics = official['evaluation']
    table = pd.DataFrame([{'target': name, **{key: entry[key] for key in
                ['roc_auc', 'average_precision', 'precision', 'recall', 'f1', 'accuracy']}}
                         for name, entry in metrics['per_label'].items()])
    table.to_csv(root / 'results/metrics/official_test/per_label.csv', index=False)
    result = dict(status='passed', verified_at=timestamp(), experiment_id=experiment['experiment_id'],
                  python=platform.python_version(), platform=platform.platform(),
                  package_versions={name: importlib.metadata.version(name) for name in
                    ['pandas', 'numpy', 'scikit-learn', 'scipy', 'streamlit', 'pyarrow', 'joblib', 'python-pptx']},
                  source_comment_rows=sum(s['rows'] for s in manifest['splits'].values()),
                  source_csv_bytes=sum(s['bytes'] for s in manifest['files'].values()),
                  training_rows=experiment['split']['train_rows'], validation_rows=experiment['split']['validation_rows'],
                  official_scored_rows=eligible, official_excluded_rows=len(test)-eligible,
                  database_counts=counts, database_bytes=db_path(root).stat().st_size,
                  sql_result_rows=sql_rows, checks=checks,
                  validation_comparison=comparison_frame(experiment).to_dict(orient='records'),
                  official_test_summary={key: metrics[key] for key in ['mean_roc_auc_six', 'macro_average_precision_six']},
                  official_any_toxic=metrics['per_label']['any_toxic'])
    save_json(root / 'results/artifact_verification.json', result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify_artifacts(args.root)
    print(json.dumps({key: result[key] for key in ['status', 'experiment_id', 'database_counts', 'sql_result_rows']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

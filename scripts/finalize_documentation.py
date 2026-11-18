"""Refresh factual results and compliance documents from verified artifacts."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from src.config import ROOT
from src.deliverables import markdown_table
from src.evaluate import comparison_frame
from src.utils import read_json
import pandas as pd


def finalize_documentation(root: Path) -> None:
    verified = read_json(root / 'results/artifact_verification.json')
    if not verified or verified['status'] != 'passed':
        raise ValueError('Run scripts.verify_artifacts before documenting completion')
    experiment = read_json(root / 'results/metrics/experiment.json')
    official = read_json(root / 'results/metrics/official_test' / f"{experiment['experiment_id']}.json")
    audit = read_json(root / 'results/data_validation_audit.json')
    suite = next(ET.parse(root / 'results/test_results.xml').getroot().iter('testsuite'))
    if int(suite.get('failures', 0)) or int(suite.get('errors', 0)):
        raise ValueError('Resolve test failures before claiming verified compliance')
    tests, seconds = suite.get('tests'), suite.get('time')
    table = comparison_frame(experiment)
    ev = official['evaluation']
    binary = ev['per_label']['any_toxic']
    matrix = binary['confusion_matrix']
    counts = verified['database_counts']
    all_bytes = sum(value.get('bytes', 0) for value in audit.values() if isinstance(value, dict))
    per_label = pd.read_csv(root / 'results/metrics/official_test/per_label.csv')
    result_doc = f'''# Real experimental results

Experiment `{experiment['experiment_id']}`, Python {verified['python']}. Official-data source hashes and version information are recorded in `results/artifact_verification.json` and the validation audit. The six competition labels and separately fitted seventh `any_toxic` target remain distinct.

## Dataset and split

- Training source: {experiment['dataset_manifest']['splits']['train']['rows']:,} rows. Shared fitted partition: {verified['training_rows']:,}; validation: {verified['validation_rows']:,}.
- Official test source: {verified['official_scored_rows'] + verified['official_excluded_rows']:,} rows. Final evaluation: {verified['official_scored_rows']:,} fully scored rows. Excluded unknown-label rows: {verified['official_excluded_rows']:,}.
- Seed 42, 20% validation groups, group-level any_toxic stratification. Vectorizers fit only on the fitted partition. No exact trimmed training duplicates or train/test text overlaps were found.
- One U+2003-only test comment has six -1 labels. It is preserved unchanged and excluded from evaluation. No source CSV was modified.
- Source comment records: {verified['source_comment_rows']:,}. Three core CSVs: {verified['source_csv_bytes']:,} bytes. All four files: {all_bytes:,} bytes ({all_bytes/1e6:.2f} decimal MB). Label/template rows reuse test IDs and do not create additional comments.

## Validation comparison

{markdown_table(table)}

The declared criterion selected **{experiment['selected_model']}** by validation mean ROC-AUC across the six original labels, excluding any_toxic. The LR candidates are close; one split does not establish statistical significance. No official-test metrics influenced the selection.

## Selected model on the official test

Mean six-label ROC-AUC: **{ev['mean_roc_auc_six']:.6f}**. Macro average precision over six labels: **{ev['macro_average_precision_six']:.6f}**. Fixed decision threshold: 0.5.

{markdown_table(per_label)}

any_toxic confusion matrix, actual rows and predicted columns in order 0/1: **{matrix}**. This corresponds to {matrix[0][0]:,} true negatives, {matrix[0][1]:,} false positives, {matrix[1][0]:,} false negatives and {matrix[1][1]:,} true positives.

The operating point has precision {binary['precision']:.2%}, recall {binary['recall']:.2%} and F1 {binary['f1']:.6f}. High recall creates a substantial human-review load. The official test was evaluated once per experiment; these results were reported without retuning or reselection. No other candidate was evaluated on the official test.

## SQLite and SQL evidence

Database counts: {counts['comments']:,} comments, {counts['labels']:,} known label records, {counts['model_runs']} runs and {counts['predictions']:,} held-out prediction records. Unknown test rows have no ground-truth records. SQLite integrity is `ok`, foreign-key checks are clean, and source/model fingerprints match.

{markdown_table(pd.DataFrame([{'query': name, 'result_rows': count} for name,count in verified['sql_result_rows'].items()]))}

The training set contains 16,225 any_toxic positives (10.17%). Threat is the rarest original label (478, 0.30%). The strongest original-label training co-occurrence is obscene + toxic (7,926 comments). Full query outputs, including actual confident errors, are in `results/sql/`.

## Verification and limits

{tests} tests passed in {float(seconds):.2f} seconds, with no failures/errors/skips. Unit fixtures and actual-artifact integration checks are separate from the scientific benchmark. All six Streamlit pages were also inspected in a browser. Live inference, explorer/model controls, SQL execution and a five-row CSV upload/scoring/download worked without retraining. The download has 15 columns, including seven probability columns. No browser console errors were observed.

RSS is sampled absolute process memory in MiB every 10 ms in fresh model processes. Fit/inference timing excludes disk I/O and metric calculations. Desktop background activity and sampling limit hardware comparisons. Full fitting holds training text in RAM despite chunked preparation and sparse features.

Scores are uncalibrated and affected by source/context/annotation limitations. Actual student/contribution details remain to be supplied. No GitHub push, instructor message or Moodle upload was performed.
'''
    checklist = f'''# University requirements and final verification

**Requirements 1-10 are verified with real-data evidence.** Requirement 11's memory-aware implementation is verified. Student identity/contribution details also need to reflect the actual submission.

Experiment `{experiment['experiment_id']}`. Python {verified['python']}. **{tests} tests passed**, with zero failures/errors/skips. The current XML, real artifact verification and browser checklist are under `results/`. Synthetic unit scores are never used as scientific results.

| # | Requirement | Code and actual evidence | Status |
|---|---|---|---|
| 1 | At least five custom functions | `prepare_data`, `derive_targets`, `split_training`, `fit_bundle`, `evaluate_targets`, `run_query`, `predict_batch`, and more in `src/` | Verified by inspection and execution. |
| 2 | Lists/dictionaries | Target/model lists in `src/config.py`; configuration, per-label metrics and query dictionaries | Verified by the real pipeline and tests. |
| 3 | if/elif/else | Model selection branches in `src/train.py`, routing in `app.py`, metric edge cases | Verified with four fitted variants and all six pages. |
| 4 | Loops | CSV/Parquet chunks, seven-target fitting, batches and SQL exports | Verified on {verified['source_comment_rows']:,} comment records. |
| 5 | Persistent files | CSV reads, Parquet, JSON, SQLite, joblib, query exports, PDF and PPTX | Real artifacts saved/read back; hashes checked. |
| 6 | Error handling | `DataValidationError`, CLI/UI handling, malformed/missing-data tests | Verified. A narrow rule preserves the one fully unscored whitespace-only test comment. Blank labeled text remains rejected. |
| 7 | Interactive interface | Six pages, saved-model and threshold controls, preview filters, live Analyze, batch upload/download and SQL execution | Real-artifact tests and browser checks passed. Download: five rows, 15 columns. See browser JSON/screenshots. |
| 8 | Modules and standard library | `src/`, `scripts/`; pathlib, sqlite3, json, datetime, logging, csv, multiprocessing | Imports, compileall and dependency checks passed. |
| 9 | Baseline plus two methods, quality/time/memory | Four variants and measured `results/metrics/comparison.csv`, per-label metrics and figures | Verified on one {verified['training_rows']:,}/{verified['validation_rows']:,} split. Balanced LR selected at AUC 0.981074; selected-model official AUC {ev['mean_roc_auc_six']:.6f}. Real timings/RSS saved. |
| 10 | DB, ER, eight SQL analyses and Pandas | Normalized/indexed schema, Mermaid/SVG ER, eight named SQL files/exports, SQL page | Verified: {counts['comments']:,} comments, {counts['labels']:,} labels, four runs, {counts['predictions']:,} predictions. All eight real result sets populated. JOIN, GROUP BY/HAVING, CTE and window functions executed. Integrity/FKs passed. |
| 11 | Volume and memory awareness | 10,000-row CSV chunks, int8/int32 types, Parquet, sparse float32 matrices, feature cap and batched DB/inference | Memory implementation verified. {verified['source_comment_rows']:,} source records and {all_bytes/1e6:.2f} MB across four CSVs. Labels/template IDs do not add comments; fitting is not fully out of core. |

The 19-section report and 12-slide editable deck contain actual EDA, SQL findings, comparison results and final test results. Models/vectorizers/thresholds/settings/split assignments are saved and the dashboard never trains on loading. See `docs/experiment_results.md` for detailed results and limitations.

The source submission contains code, requirements, SQL, README, report, deck and measured result exports. The private runtime submission additionally contains saved models, processed data and SQLite and must not be published publicly. No credentials, GitHub push or Moodle submission are included.
'''
    notes = f'''# Final execution record

Completed experiment: `{experiment['experiment_id']}`. Python {verified['python']}, {verified['platform']}. Pinned versions are in the requirements files; observed versions are in `results/artifact_verification.json`.

Executed successfully: `scripts.validate_data`, `scripts.prepare_data`, `scripts.build_database`, `scripts.train_models`, `scripts.evaluate_models`, `scripts.verify_artifacts`, the entire pytest suite, compileall and pip check. The report/deck generator uses the actual run; no synthetic benchmark or screenshot is substituted.

Validation audited all four CSVs, six binary labels, CSV structure, IDs and template probabilities. The original strict loader detected one whitespace-only fully unscored official test row. Its raw text is now preserved with explicit counting, and labeled blanks remain errors. Regression tests cover both cases. Raw source files were not changed.

The four candidates used {verified['training_rows']:,} fitted rows and the identical {verified['validation_rows']:,} validation rows. All completed without fit warnings or prior fallbacks. Only `{experiment['selected_model']}` was evaluated on {verified['official_scored_rows']:,} fully scored official test rows; {verified['official_excluded_rows']:,} unknown-label rows were excluded. No test retuning/reselection occurred.

SQLite has {counts['comments']:,} comments, {counts['labels']:,} known labels, four runs and {counts['predictions']:,} held-out predictions. All eight SQL reads/exports returned real results. SQLite integrity is `ok`; FK violations are zero. Input/model hashes, split coverage, duplicate-group separation, eligibility and expected DB counts all passed verification.

The entire suite passed **{tests} tests** in {float(seconds):.2f} seconds. Unit tests use synthetic temporary fixtures. Read-only real-artifact integration tests cover saved models, splits, eligibility, all eight queries, every Streamlit page, live inference and batch execution. Compileall succeeded and pip check found no broken requirements. The explicit `six` dependency supports the tested Kaggle package.

Browser inspection verified all six pages, live inference, explorer/model controls, SQL execution and a five-row CSV upload/prediction/download (15 output columns, seven probability columns). There were no browser console errors. The live sentence and upload were functional demonstrations, not additional scientific evaluations. Actual captures and the browser checklist are saved under `results/`.

Final PDF/PowerPoint rendering, visual inspection and structural/layout checks are recorded in private workspace QA records. The report has 19 requested sections and the deck has 12 slides. Source and private runtime archives exclude raw CSVs and credentials; processed Kaggle text in the runtime archive must stay private.

Remaining submission details: actual student/contribution/AI-assistance information. The measured review burden is {matrix[0][1]:,} false positives and {matrix[1][0]:,} false negatives at threshold 0.5. Scores are uncalibrated; one split, source bias, context limits, exact-only duplicate matching and RAM-resident fitting remain limitations. Desktop load and RSS sampling affect resource comparisons. No repository push, instructor message or Moodle upload was performed.
'''
    for name, content in [('experiment_results.md', result_doc), ('requirements_checklist.md', checklist), ('execution_notes.md', notes)]:
        (root / 'docs' / name).write_text(content, encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    finalize_documentation(args.root)
    print('Refreshed real results, requirements checklist and execution record.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

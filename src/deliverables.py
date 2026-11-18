"""Generate an honest academic report and an editable 12-slide PowerPoint.

The requested Python/python-pptx stack keeps regeneration independent of Codex.
No synthetic test metrics are read by this module.
"""
from html import escape
from pathlib import Path
import json
import textwrap
import xml.etree.ElementTree as ET
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               PageBreak, Image, KeepTogether)
from src.config import ROOT, DEFAULTS, LABELS, TARGETS, directories
from src.database import connect, db_path, export_queries, predefined_queries, run_query
from src.evaluate import comparison_frame
from src.utils import read_json, save_json
from src.visualization import generate_figures
from svglib.svglib import svg2rlg

REFERENCES = [
 ('Jigsaw / Google', 'Toxic Comment Classification Challenge, official data and access terms',
  'https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data'),
 ('scikit-learn', 'Metrics and scoring: quantifying the quality of predictions',
  'https://scikit-learn.org/stable/modules/model_evaluation.html'),
 ('scikit-learn', 'Common pitfalls and recommended practices',
  'https://scikit-learn.org/stable/common_pitfalls.html'),
 ('scikit-learn', 'Feature extraction: text features',
  'https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction'),
 ('SQLite', 'Foreign key support and window functions',
  'https://www.sqlite.org/foreignkeys.html'),
 ('SQLite', 'Window functions', 'https://www.sqlite.org/windowfunctions.html'),
 ('Streamlit', 'Cache model resources without retraining on reruns',
  'https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource'),
 ('pandas', 'Chunked CSV ingestion', 'https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html'),
 ('Apache Arrow', 'Parquet columnar persistence', 'https://arrow.apache.org/docs/python/parquet.html'),
 ('python-pptx', 'Editable PowerPoint generation', 'https://python-pptx.readthedocs.io/'),
]


def markdown_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    """Avoid optional tabulate dependency and escape Markdown cell delimiters."""
    if frame.empty:
        return 'No matching rows (empty result, not an estimated value).'
    data = frame.head(max_rows).copy()
    def fmt(value):
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '/').replace('\n', ' ')[:180]
    lines = ['| ' + ' | '.join(map(str, data.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(data.columns)) + ' |']
    lines += ['| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in data.itertuples(index=False, name=None)]
    if len(frame) > max_rows:
        lines.append(f'First {max_rows} of {len(frame)} rows. Full CSV is in results/sql/.')
    return '\n'.join(lines)


def test_evidence(root: Path) -> str:
    path = root / 'results/test_results.xml'
    if not path.exists():
        return 'No recorded pytest XML is present. Run python -m pytest --junitxml=results/test_results.xml.'
    suite = ET.parse(path).getroot()
    suites = list(suite.iter('testsuite'))
    counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    seconds = sum(float(s.get('time', 0)) for s in suites)
    return (f"Recorded software verification: {counts['tests']} tests, {counts['failures']} failures, "
            f"{counts['errors']} errors, {counts['skipped']} skipped, {seconds:.2f} seconds. "
            'Unit tests use explicitly synthetic fixtures in temporary directories. Read-only integration tests '
            'also check saved real artifacts when available. Test pass counts and demo predictions establish no Kaggle quality metric.')


def report_sections(root: Path) -> tuple[str, list[tuple[str, str]]]:
    paths = directories(root)
    manifest = read_json(paths['processed'] / 'manifest.json')
    experiment = read_json(paths['metrics'] / 'experiment.json')
    status = ('REAL-DATA EXPERIMENT AVAILABLE' if experiment else 'IMPLEMENTATION VERIFIED - REAL-DATA EXPERIMENT PENDING')
    size = ('No official CSVs are available in this environment. Observed dataset row count, byte size, missingness, '
            'prevalence and co-occurrence are therefore unavailable. No training or test values have been invented.')
    eda = ('EDA is implemented but has no observed dataset to summarize. prepare_data validates inputs in chunks '
           'and records exact row totals, bytes, SHA-256 fingerprints, duplicate text counts, mean word/character '
           'counts, length histograms, label prevalence and co-occurrence. Null/blank comments, empty IDs, invalid '
           'labels, malformed CSVs and duplicate IDs stop preparation with an actionable error. Failed input '
           'validation does not publish a new manifest. Identical trimmed texts remain in a single validation partition.')
    results = ('MISSING RESULTS: no Jigsaw experiment has run. Supply the official files and execute prepare_data, '
               'build_database, train_models, evaluate_models and generate_deliverables, in that order. '
               'The generator then replaces this section with actual measured tables and links to actual figures. '
               'No model winner is claimed before a real experiment.')
    resource = ('No Jigsaw quality, runtime or memory comparison is available. The implemented benchmark runs each '
                'candidate in a fresh spawned process, samples absolute RSS every 10 ms, and times fit and inference '
                'separately with perf_counter. Fit includes vectorization and seven estimators. Inference includes '
                'feature transformation. Measurements exclude CSV/Parquet loading, model serialization and metric '
                'calculation. Absolute RSS includes the Python process and loaded data. Sampling can miss short spikes. '
                'A fixed seed does not make timing or memory numerically deterministic.')
    if manifest:
        frame = pd.DataFrame([dict(split=k, rows=v['rows'], scored=v['scored_rows'], unknown=v['unscored_rows'],
                              duplicate_texts=v['duplicate_comments']) for k, v in manifest['splits'].items()])
        size = (markdown_table(frame) + '\n\nObserved source sizes: ' + ', '.join(
                f'{k}: {v["bytes"]:,} bytes' for k, v in manifest['files'].items()) +
                '.\n\nFingerprints and validation metadata: data/processed/manifest.json. '
                f'Exact trimmed-text overlap in official test: {manifest["train_test_duplicate_text_rows"]:,} rows. '
                'Overlap is disclosed and official eligible test rows are not silently removed.')
        stats = manifest['splits']['train']
        eda = (f"Observed training duplicate comments: {stats['duplicate_comments']:,}. "
               f"Mean words: {stats['mean_words']:.2f}. Mean characters: {stats['mean_characters']:.2f}.\n\n" +
               markdown_table(pd.DataFrame({'label': TARGETS, 'positives': [stats['positives'][k] for k in TARGETS],
                               'prevalence': [stats['prevalence'][k] for k in TARGETS]})) +
               '\n\nSee results/figures/label_prevalence.png, cooccurrence.png and comment_lengths.png. '
               'Quality checks found no blank training comments, duplicate IDs or invalid labels. '
               f"The test source contains {manifest['splits'].get('test', {}).get('blank_comments', 0)} whitespace-only "
               'fully unscored comment(s), preserved exactly and excluded from evaluation. '
               'Detailed validation of all four supplied CSVs is in results/data_validation_audit.json.')
    if experiment:
        results = (f"Selected model: {experiment['selected_model']}.\n\n{experiment['selection_criterion']}\n\n" +
                   markdown_table(comparison_frame(experiment)[['model', 'mean_ROC_AUC_6', 'macro_AP_6', 'any_F1',
                                                               'any_precision', 'any_recall']]) +
                   f"\n\nSplit: {json.dumps(experiment['split'])}. Full per-label results: results/metrics/per_label.csv. "
                   'Curves and confusion matrices are saved under results/figures/.')
        skipped = experiment['runs'][0]['validation']['skipped_auc_labels']
        results += f'\n\nOriginal labels with undefined validation ROC-AUC: {skipped or "none"}.'
        official = read_json(paths['metrics'] / 'official_test' / f"{experiment['experiment_id']}.json")
        if official and official.get('status') == 'evaluated':
            result = official['evaluation']
            results += (f"\n\nOfficial held-out test: {result['rows']:,} eligible rows; mean six-label ROC-AUC "
                        f"{result['mean_roc_auc_six']}; any_toxic F1 {result['per_label']['any_toxic']['f1']:.4f}. "
                        'Only the validation-selected model was evaluated. Results are cached once per experiment.')
            label_table = pd.DataFrame([{'target': label,
                **{key: result['per_label'][label][key] for key in
                   ['roc_auc', 'average_precision', 'precision', 'recall', 'f1']}} for label in TARGETS])
            results += '\n\nSelected-model official-test metrics at threshold 0.5:\n\n' + markdown_table(label_table)
            matrix = result['per_label']['any_toxic']['confusion_matrix']
            results += '\n\nOfficial-test any_toxic confusion matrix (rows actual, columns predicted):\n\n' + markdown_table(
                pd.DataFrame({'actual': ['Non-toxic (0)', 'Toxic (1)'],
                              'predicted_0': [matrix[0][0], matrix[1][0]],
                              'predicted_1': [matrix[0][1], matrix[1][1]]}))
            selected_run = next(r for r in experiment['runs'] if r['model_name'] == experiment['selected_model'])
            binary_test = result['per_label']['any_toxic']
            binary_val = selected_run['validation']['per_label']['any_toxic']
            results += (f"\n\nAt the fixed 0.5 threshold, any_toxic F1 changes from {binary_val['f1']:.4f} on validation "
                        f"to {binary_test['f1']:.4f} on official test. Official-test precision is {binary_test['precision']:.2%} "
                        f"and recall is {binary_test['recall']:.2%}. The model raises {matrix[0][1]:,} false positives "
                        f"and misses {matrix[1][0]:,} positives. High recall therefore comes with a substantial review load. "
                        'Class weighting and source differences can affect operating-point behavior, but this single '
                        'experiment does not isolate a cause. The test result is reported without retuning or reselection.')
        else:
            results += '\n\nOfficial test evaluation is pending or unavailable. No test-based model selection occurred.'
        resource = markdown_table(comparison_frame(experiment)[['model', 'training_s', 'inference_s', 'peak_RSS_MiB']])
        resource += '\n\n' + experiment['memory_method']
        resource += (' Resource values are observed desktop wall-clock/RSS measurements, not controlled hardware '
                     'benchmarks. Background applications and sampling may affect them.')
    sql_sections = []
    if db_path(root).exists():
        export_queries(root)
    for index, (name, query) in enumerate(predefined_queries().items(), 1):
        observed = run_query(db_path(root), name) if db_path(root).exists() else pd.DataFrame()
        disclosure = ('Actual dataset analysis.' if manifest else 'Executed on the empty schema only; no dataset conclusions follow.')
        display = observed
        interpretation = ''
        if manifest and not observed.empty:
            train_rows = observed.loc[observed.dataset_split.eq('train')] if 'dataset_split' in observed else observed
            if name == 'split_counts':
                interpretation = f"SQLite contains {int(observed.comments.sum()):,} distinct source comments across {len(observed)} source splits."
            elif name == 'binary_distribution':
                positives = int(train_rows.loc[train_rows.any_toxic.eq(1), 'comments'].sum())
                total = int(train_rows.comments.sum())
                interpretation = f"Training any_toxic prevalence is {positives:,}/{total:,} ({positives/max(total,1):.2%}); the majority class is non-toxic."
            elif name == 'category_prevalence':
                high = train_rows.loc[train_rows.prevalence.idxmax()]
                low = train_rows.loc[train_rows.prevalence.idxmin()]
                interpretation = f"Among original training labels, {high.label_name} is most frequent ({high.prevalence:.2%}) and {low.label_name} least frequent ({low.prevalence:.2%})."
            elif name == 'positive_cooccurrence':
                pair = train_rows.loc[train_rows.both_positive.idxmax()]
                interpretation = f"The largest training co-occurrence is {pair.label_a} + {pair.label_b}: {int(pair.both_positive):,} comments. Category labels overlap."
            elif name == 'length_groups':
                high = train_rows.loc[train_rows.toxic_rate.idxmax()]
                interpretation = f"The highest observed training any_toxic rate is in {high.length_group}: {high.toxic_rate:.2%}. This association does not establish causation."
            elif name == 'label_ranking':
                interpretation = 'Training labels in descending positive-count order: ' + ', '.join(train_rows.sort_values('prevalence_rank').label_name) + '.'
        if name == 'prediction_distribution' and not observed.empty:
            display = observed.loc[observed.label_name.eq('any_toxic'), ['model_name', 'dataset_split', 'label_name', 'predictions',
                                'mean_probability', 'predicted_positives']]
            disclosure += ' any_toxic rows shown; full seven-target outputs and run IDs are retained in the CSV export.'
            interpretation = 'Source train rows here are validation predictions only. Test predictions belong only to the validation-selected model.'
        elif name == 'high_confidence_errors' and not observed.empty:
            display = observed[['model_name', 'label_name', 'label_value', 'probability', 'confidence', 'comment_excerpt']].head(5).copy()
            display['comment_excerpt'] = display.comment_excerpt.str.slice(0, 60)
            disclosure += ' Five example errors and shortened excerpts shown; full results are in the CSV export.'
            interpretation = ('These are actual errors with model confidence at least 0.8. They show why confident '
                              'predictions still require human review. Examples can contain offensive source text.')
        sql_sections.append(f'### {index}. {name}\n\n```sql\n{query}\n```\n\n{disclosure}\n\n' + markdown_table(display) +
                            ('\n\n' + interpretation if interpretation else ''))
    sections = [
      ('1. Title and objective',
       'Toxic Comment Detection: Machine Learning Classification and SQL-Based Data Analysis. '
       'Master of Big Data Analysis, Programming for Data Analysis and Database. '
       'The objective is a reproducible Python application for binary and six-category toxicity classification, '
       'relational data analysis and a measured comparison of CPU models.\n\nStatus: ' + status),
      ('2. Introduction',
       'Online discussion can contain harassment and abusive language. A text classifier can help a human reviewer '
       'prioritize comments, but language depends on context and an automated score should not become a final judgment. '
       'This project connects data validation, exploratory analysis, SQL and machine learning in one auditable workflow [1, 2].'),
      ('3. Problem statement',
       'A comment can express more than one toxicity category at once, so the original problem is multi-label. '
       'The six binary labels are toxic, severe_toxic, obscene, threat, insult and identity_hate. '
       'The seventh target any_toxic equals the maximum of those six known ground-truth labels. '
       'A separate binary estimator learns any_toxic. The six predicted probabilities do not sum to one, '
       'and their maximum need not match the separately trained binary score.'),
      ('4. Data source, schema and observed size',
       'Source: official Jigsaw Toxic Comment Classification Challenge [1]. Kaggle account authentication and '
       'competition rules govern access. The original restricted files are excluded from Git and this package. '
       'Training CSV requires id, comment_text and the six labels. Test text and official test labels join by id '
       'with unique IDs, rather than by row position. sample_submission.csv is validated as an optional prediction '
       'template and is never treated as ground truth.\n\n' + size +
       '\n\nSource-file sizes and distinct comment counts are recorded above.'),
      ('5. Exploratory data analysis', eda),
      ('6. Preprocessing decisions',
       'Text remains largely intact. The vectorizers lowercase text, retain word tokens including one-character tokens, '
       'and use unigrams/bigrams without stop-word removal or stemming. Numbers, quoted language and negation can carry '
       'signal and are retained. Whitespace trimming serves only duplicate grouping. This word-token approach does not '
       'explicitly model punctuation or capitalization and may miss obfuscation. Vectorizers fit on the training partition '
       'only. Default vocabulary cap is 50,000 features and min_df is 2. Count/TF-IDF matrices remain sparse and use float32 [3, 4].'),
      ('7. Class imbalance and mitigation',
       'Rare positive classes can make a high overall accuracy misleading. The balanced logistic regression candidate '
       'uses class_weight=balanced independently for every target, while the unweighted candidate provides a direct '
       'comparison. Stratifying on any_toxic at the duplicate-group level helps the binary distribution but does not '
       'guarantee positives for every rare label. A single-class training label falls back to a prior DummyClassifier '
       'and the run records that fallback. No resampling occurs before splitting. Decision thresholds remain 0.5 '
       'for the recorded experiments. Dashboard thresholds affect interactive decisions only.'),
      ('8. Database design and ER diagram',
       'SQLite stores comments, label_types, labels, model_runs, predictions and an independent metadata table. '
       'Composite primary keys prevent duplicate comment/label and run/comment/label records. Foreign keys are enabled '
       'on every connection, and value checks constrain ground truth and predictions. Unscored official test rows '
       'have no labels records. Indexed split, label and prediction columns support joins and grouped analyses. '
       'Rows insert in batches inside transactions. The schema is sql/schema.sql; the rendered ER diagram and Mermaid '
       'source are docs/er_diagram.svg and docs/er_diagram.md [5, 6].\n\nThe source split train includes the validation '
       'subset; per-experiment split_assignments.parquet records actual training/validation membership. Predictions '
       'persisted for that source are validation-only. The SQL error analysis therefore uses held-out predictions.'),
      ('9. Eight SQL queries and actual outputs', '\n\n'.join(sql_sections)),
      ('10. Methodology and leakage precautions',
       'The fixed seed is 42 and the validation fraction is 0.20 of unique trimmed-text groups, so the row fraction '
       'can differ slightly when duplicates exist. Group labels use maximum any_toxic for stratification. '
       'A deterministic unstratified fallback is documented if too few groups support stratification. All candidates '
       'share the exact saved split and seven targets. Official test labels never enter fitting, tuning or selection. '
       'After validation-based model selection, only the winner is evaluated once per experiment on fully scored '
       'test rows. A row containing any -1 is excluded, including partial -1 rows. No -1 value becomes a negative. '
       'Known train/test duplicate text is counted and disclosed as a limitation, rather than hidden [3].'),
      ('11. Baseline and alternative methods',
       'Baseline: seven DummyClassifier(strategy=prior) predictors learn training priors. BoW/NB: CountVectorizer '
       'with MultinomialNB(alpha=1) uses token frequency evidence. TF-IDF/LR: sublinear TF-IDF with seven independent '
       'LogisticRegression estimators, C=1, liblinear, max_iter=1000. A fourth candidate adds class_weight=balanced. '
       'The candidates compare a prior-only reference with two representations and learning methods, and explicitly '
       'test the imbalance choice. Models, vectorizers, 0.5 thresholds, warnings and configuration persist together '
       'with joblib and strict JSON. Joblib artifacts are trusted local files, never arbitrary uploads.'),
      ('12. Metrics and model selection',
       'Each target receives ROC-AUC, average precision, precision, recall, F1, accuracy and a 2x2 confusion matrix '
       '(rows actual, columns predicted, label order 0/1). The competition-style mean ROC-AUC uses the six original '
       'labels only, excluding any_toxic. Average precision is a non-interpolated PR summary and is not claimed '
       'as trapezoidal area. The selection criterion is highest mean six-label validation ROC-AUC over evaluable '
       'labels, with any_toxic F1 breaking ties, followed by fixed model order. All candidates use the same truth '
       'and therefore the same evaluable-label set. Single-class truth makes ROC-AUC undefined (JSON null); '
       'no positives makes AP undefined. Undefined labels and reasons are saved. Precision uses zero_division=0 '
       'when no positives are predicted. ROC-AUC/AP are threshold-independent; F1 and confusion matrices depend '
       'on the declared 0.5 threshold [2].'),
      ('13. Experimental results', results),
      ('14. Quality, timing and memory comparison', resource),
      ('15. Streamlit application architecture',
       'The six pages are Home / Overview, Live Toxicity Detection, Dataset Explorer, Model Comparison, SQL Analytics '
       'and Batch Prediction. Home shows observed metadata and setup state. Live detection offers saved-model '
       'selection, a text field, Analyze and threshold controls. Explorer filters a bounded preview and plots '
       'full-split statistics. Comparison exposes per-label curves and confusion matrices. SQL Analytics executes '
       'only eight predefined reads through a read-only connection and exports CSV. Batch Prediction validates '
       'UTF-8 CSV uploads and scores sparse feature batches of 2,048 rows, with a 20 MB / 100,000-row upload cap. '
       'The page uses st.cache_resource for saved models and never retrains during loading or reruns [7]. '
       'Missing inputs produce actionable empty states.\n\n' + test_evidence(root)),
      ('16. Discussion, limitations and bias',
       'This is an educational research tool. Training text reflects its source domain and annotation practices. '
       'It can mistake identity mentions, quotations, reclaimed language, satire or context for abuse, and may miss '
       'spelling variations or multilingual abuse. These models have no conversational context and probabilities '
       'have not been calibrated. Separate binary estimates can disagree with the six category estimates. '
       'One validation split is less robust than repeated or grouped cross-validation. Near-duplicates beyond exact '
       'trimmed matches can remain across partitions. Any exact train/test text overlap is disclosed. '
       'The preparation stage streams CSVs and Parquet, but final sparse-model fitting loads the prepared training '
       'text into RAM. This is bounded in vocabulary, not fully out-of-core machine learning. Larger datasets '
       'would require incremental learners or stronger memory limits. Toxicity judgments require human review.'),
      ('17. Conclusion',
       ('The application provides a reproducible comparison with observed results above. The selected model is '
        'justified by the declared validation criterion, subject to the stated limitations.' if experiment else
        'The application, SQL workflow and test suite are implemented. A scientifically valid conclusion about '
        'model quality must wait for authorized access to the official data and execution of the full experiment. '
        'Synthetic unit fixtures support only software correctness. They provide no evidence of real-world quality.') +
       '\n\nFuture work can examine calibrated scores, character n-grams, robust cross-validation, subgroup '
       'error analysis and optional Transformer models.'),
      ('18. References', '\n\n'.join(f'[{i}] {author}. {title}. {url}' for i, (author, title, url) in enumerate(REFERENCES, 1))),
      ('19. Contribution statement',
       'Prepared as an individual submission package unless the student specifies a team. Author names and student IDs '
       'have not been supplied and are intentionally not fabricated. Before submission, '
       'add the actual student identity and accurately disclose team contributions and AI assistance according to '
       'course policy. A team should attribute data preparation, modeling, database work, dashboard implementation '
       'and verification to the people who performed them.'),
    ]
    return status, sections


def _pdf_table(lines: list[str], styles, available_width: float):
    rows = [[cell.strip() for cell in line.strip('|').split('|')] for line in lines if line.startswith('|')]
    rows = [row for row in rows if not all(set(cell) <= set('-: ') for cell in row)]
    if not rows:
        return []
    columns = len(rows[0])
    cellstyle = ParagraphStyle('cell', parent=styles['BodyText'], fontSize=7 if columns > 6 else 8,
                               leading=10, spaceAfter=0)
    data = [[Paragraph(escape(cell), cellstyle) for cell in row] for row in rows]
    table = Table(data, colWidths=[available_width / columns] * columns, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef4')),
                              ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                              ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#ced7e0')),
                              ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                              ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    return [table, Spacer(1, 10)]


def generate_pdf(root: Path, status: str, sections: list[tuple[str, str]]) -> None:
    """Render report via Platypus so text wraps, tables paginate and headers repeat."""
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 9.5
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceAfter = 8
    styles['BodyText'].splitLongWords = True
    styles['Heading1'].textColor = colors.HexColor('#17283d')
    styles['Heading2'].textColor = colors.HexColor('#087f8c')
    styles['Heading1'].keepWithNext = True
    styles['Heading2'].keepWithNext = True
    code_style = ParagraphStyle('Code', fontName='Courier', fontSize=7, leading=10,
                               spaceAfter=3, wordWrap='CJK', keepWithNext=True, textColor=colors.HexColor('#334155'))
    output = root / 'reports/final_report.pdf'
    doc = SimpleDocTemplate(str(output), pagesize=(595.28, 841.89), leftMargin=48, rightMargin=48,
                            topMargin=48, bottomMargin=48, title='Toxic Comment Detection - Final Report',
                            author='Project author to be supplied')
    story = [Paragraph('TOXIC COMMENT<br/>DETECTION', styles['Title']), Spacer(1, 12),
             Paragraph('Machine Learning Classification and<br/>SQL-Based Data Analysis', styles['Heading2']),
             Spacer(1, 14), Paragraph('Programming for Data Analysis and Database<br/>Master of Big Data Analysis', styles['BodyText']),
             Spacer(1, 20), Paragraph(escape(status), styles['Heading2']),
             Paragraph('Software verification and experimental evidence are reported separately. '
                       'No synthetic fixture is presented as a Kaggle result.', styles['BodyText']), Spacer(1, 18)]
    story.append(Paragraph('Report contents', styles['Heading2']))
    for title, _ in sections:
        story.append(Paragraph(escape(title), styles['BodyText']))
    story.append(PageBreak())
    for title, body in sections:
        story.append(Paragraph(escape(title), styles['Heading1']))
        lines = body.split('\n')
        index = 0
        in_code = False
        while index < len(lines):
            line = lines[index]
            if line.startswith('```'):
                in_code = not in_code
            elif in_code:
                for part in textwrap.wrap(line, width=100, replace_whitespace=False) or [' ']:
                    story.append(Paragraph(escape(part).replace(' ', '&#160;'), code_style))
            elif line.startswith('|'):
                table_lines = []
                while index < len(lines) and lines[index].startswith('|'):
                    table_lines.append(lines[index])
                    index += 1
                story.extend(_pdf_table(table_lines, styles, doc.width))
                continue
            elif line.startswith('### '):
                story.append(Paragraph(escape(line[4:]), styles['Heading2']))
            elif line.strip():
                story.append(Paragraph(escape(line), styles['BodyText']))
            index += 1
        if title.startswith('8.'):
            drawing = svg2rlg(str(ROOT / 'docs/er_diagram.svg'))
            factor = doc.width / drawing.width
            drawing.scale(factor, factor)
            drawing.width *= factor
            drawing.height *= factor
            story += [drawing, Spacer(1, 12)]
            diagram_rows = [['Parent', 'Child', 'Relationship'], ['comments', 'labels', '1 to many by comment_id'],
                            ['comments', 'predictions', '1 to many by comment_id'],
                            ['label_types', 'labels / predictions', '1 to many by label_name'],
                            ['model_runs', 'predictions', '1 to many by run_id']]
            table = Table(diagram_rows, colWidths=[110, 145, 240])
            table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eef4')),
                                      ('GRID', (0, 0), (-1, -1), .4, colors.lightgrey),
                                      ('FONTSIZE', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8)]))
            story += [table, Spacer(1, 12)]
        images = []
        if title.startswith('5.') and (root / 'data/processed/manifest.json').exists():
            images = ['label_prevalence.png', 'cooccurrence.png']
        elif title.startswith('13.') and (root / 'results/metrics/experiment.json').exists():
            images = ['model_comparison.png', 'confusion_matrices.png', 'official_test_confusion_matrices.png']
        for name in images:
            path = root / 'results/figures' / name
            if path.exists():
                from PIL import Image as PILImage
                with PILImage.open(path) as image:
                    width, height = image.size
                story += [Image(str(path), width=doc.width, height=doc.width * height / width), Spacer(1, 10)]
        if title.startswith('15.'):
            proof = read_json(root / 'results/browser_verification.json', {})
            current = read_json(root / 'results/metrics/experiment.json', {})
            screenshot = root / 'results/screenshots/live_detection.jpg'
            if screenshot.exists() and proof.get('experiment_id') == current.get('experiment_id') and current:
                from PIL import Image as PILImage
                with PILImage.open(screenshot) as image:
                    width, height = image.size
                story += [Image(str(screenshot), width=doc.width, height=doc.width * height / width),
                          Paragraph('Actual dashboard capture using the saved selected model. The entered sentence '
                                    'is a functional demonstration, not a quality-evaluation sample.', styles['BodyText'])]
    def footer(canvas, document):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#62748a'))
        canvas.drawString(48, 28, 'Toxic Comment Detection - academic project')
        canvas.drawRightString(547, 28, f'{document.page}')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def presentation_content(root: Path) -> list[dict]:
    manifest = read_json(root / 'data/processed/manifest.json')
    experiment = read_json(root / 'results/metrics/experiment.json')
    counts = ('Official data pending. No row counts or prevalence claimed.' if not manifest else
              ', '.join(f'{key}: {value["rows"]:,} rows' for key, value in manifest['splits'].items()))
    result = ('Real-data training is pending. No model winner or performance claim is available.' if not experiment else
              f"Validation-selected model: {experiment['selected_model']}. Full results accompany this deck.")
    return [
     {'title': 'Toxic Comment Detection', 'subtitle': 'Machine learning classification and SQL-based data analysis',
      'body': ['Programming for Data Analysis and Database', 'Master of Big Data Analysis'], 'kind': 'cover'},
     {'title': 'Problem and objectives', 'body': ['Identify comments that may need human review.',
       'Predict six overlapping toxicity categories and a separate any_toxic target.',
       'Compare measured model quality, runtime and memory. Analyze data with SQL.']},
     {'title': 'Jigsaw data and seven targets', 'body': [counts,
       'Original labels: toxic, severe_toxic, obscene, threat, insult, identity_hate.',
       'Derived target: any_toxic = maximum of the six known labels.',
       'Official test labels join by id. Any row containing -1 is unscored.']},
     {'title': 'System architecture', 'kind': 'architecture', 'body': [
       'Chunked validation produces Parquet, quality summaries and provenance.',
       'SQLite stores comments, labels, model runs and held-out predictions.',
       'Saved models power a six-page Streamlit dashboard. Startup never trains.']},
     {'title': 'Exploration and class imbalance', 'body': [
       'Summaries cover length, duplicates, prevalence and positive-label co-occurrence.',
       'Rare labels require recall and average precision alongside ROC-AUC.',
       'Balanced logistic regression tests the impact of class weighting.'], 'kind': 'eda'},
     {'title': 'SQLite design and SQL evidence', 'kind': 'er', 'body': [
       'Composite keys and foreign keys enforce consistent labels and predictions.',
       'Eight versioned queries use joins, grouping, CTEs and window functions.']},
     {'title': 'Baseline and alternative models', 'kind': 'methods', 'body': [
       'DummyClassifier: seven predictors based on training priors.',
       'Bag of Words + Multinomial Naive Bayes.',
       'TF-IDF + Logistic Regression, with unweighted and balanced candidates.',
       'All feature matrices remain sparse. Default vocabulary cap: 50,000.']},
     {'title': 'Evaluation protocol', 'body': [
       'Seed 42. Split training text groups 80/20 and stratify any_toxic when feasible.',
       'Fit vectorizers only on training. Keep duplicate comments in one partition.',
       'Select by validation mean ROC-AUC across the six original labels.',
       'Evaluate only the winner once per experiment on eligible official test rows.']},
     {'title': 'Observed comparison and resources', 'body': [result,
       'Track ROC-AUC, average precision, precision, recall, F1 and confusion matrices.',
       'Fresh model processes measure fit/inference time and sampled peak RSS.',
       'Synthetic unit fixtures verify code only. They are excluded from this evidence.'], 'kind': 'results'},
     {'title': 'Streamlit demonstration', 'kind': 'demo', 'body': [
       'Home and Dataset Explorer show provenance, quality checks and class imbalance.',
       'Live Detection analyzes text with model and threshold controls.',
       'Model Comparison shows per-label curves and confusion matrices.',
       'SQL Analytics runs eight queries. Batch Prediction exports scored CSVs.']},
     {'title': 'Limitations and conclusions', 'body': [
       'Context, identity mentions and source-domain bias can produce errors.',
       'Exact duplicate grouping does not eliminate near-duplicate leakage.',
       'Review dataset provenance and interpret predictions with human oversight.',
       (f"Validation selected {experiment['selected_model']}. Human review remains necessary." if experiment else
        'Complete the real-data experiment before making a scientific quality claim.')]},
     {'title': 'References and questions', 'body': [
       'Jigsaw Toxic Comment Classification Challenge: official Kaggle data.',
       'scikit-learn: text feature extraction, model evaluation and leakage guidance.',
       'SQLite: foreign keys and window functions. Streamlit: cached model resources.',
       'Full URLs and software versions appear in the report and README.'], 'kind': 'references'},
    ]


def generate_presentation(root: Path) -> None:
    """Use the explicitly requested Python/python-pptx stack for portability."""
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    navy, teal, grey = '17283D', '087F8C', '526579'
    content = presentation_content(root)
    experiment = read_json(root / 'results/metrics/experiment.json')
    manifest = read_json(root / 'data/processed/manifest.json')

    def textbox(slide, x, y, width, height, text, size=22, color=navy, bold=False):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
        frame = shape.text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = Inches(0.02)
        frame.margin_top = Inches(0.04)
        for index, line in enumerate(text.split('\n')):
            paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            paragraph.text = line
            paragraph.font.name = 'Arial'
            paragraph.font.size = Pt(size)
            paragraph.font.bold = bold
            paragraph.font.color.rgb = RGBColor.from_string(color)
            paragraph.space_after = Pt(12)
        return shape

    def entity(slide, x, y, width, height, title, fields):
        # Requested editable ER and architecture diagrams, not decorative graphics.
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(width), Inches(height))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(255, 255, 255)
        shape.line.color.rgb = RGBColor.from_string(teal)
        textbox(slide, x + .12, y + .08, width - .24, .38, title, 20, teal, True)
        textbox(slide, x + .12, y + .52, width - .24, height - .6, '\n'.join(fields), 17, navy)

    for index, entry in enumerate(content, 1):
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor.from_string('F6F8FB')
        if entry.get('kind') == 'cover':
            textbox(slide, .8, 1.05, 11.8, 1.0, entry['title'], 44, navy, True)
            textbox(slide, .8, 2.25, 10.7, 1.1, entry['subtitle'], 27, teal)
            textbox(slide, .8, 4.2, 11.5, 1.4, '\n'.join(entry['body']), 22, grey)
        else:
            textbox(slide, .65, .45, 12, .9, entry['title'], 33, navy, True)
            kind = entry.get('kind')
            if kind == 'er':
                entity(slide, .7, 1.65, 3.0, 1.9, 'comments', ['PK comment_id', 'text, split, lengths'])
                entity(slide, 5.1, 1.65, 3.0, 1.9, 'labels', ['PK/FK comment_id', 'PK/FK label_name, value'])
                entity(slide, 9.5, 1.65, 3.0, 1.5, 'label_types', ['PK label_name'])
                entity(slide, .7, 4.2, 3.0, 1.6, 'model_runs', ['PK run_id, config', 'fit time, peak RSS'])
                entity(slide, 5.1, 4.2, 3.9, 1.9, 'predictions', ['PK/FK run_id, comment_id, label_name', 'probability, predicted_label'])
                for x1, y1, x2, y2 in [(3.7, 2.5, 5.1, 2.5), (9.5, 2.5, 8.1, 2.5),
                                       (3.7, 4.9, 5.1, 4.9), (2.2, 3.55, 5.1, 4.2), (10.9, 3.15, 9.0, 4.65)]:
                    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
                    connector.line.color.rgb = RGBColor.from_string(teal)
                sql_caption = 'Each line represents one parent to many child rows. metadata independently stores the manifest.'
                if manifest:
                    pairs = run_query(db_path(root), 'positive_cooccurrence')
                    pairs = pairs.loc[pairs.dataset_split.eq('train')]
                    if not pairs.empty:
                        pair = pairs.loc[pairs.both_positive.idxmax()]
                        sql_caption = (f"{sum(v['rows'] for v in manifest['splits'].values()):,} SQLite comments. "
                                       f"Top training pair: {pair.label_a} + {pair.label_b} ({int(pair.both_positive):,}). Eight queries executed.")
                textbox(slide, .7, 6.35, 11.9, .55, sql_caption, 17, grey)
            elif kind == 'architecture':
                steps = [('CSV inputs', ['train / test / labels']), ('Prepare + SQLite', ['validate, summarize, query']),
                         ('Train + evaluate', ['saved models and metrics']), ('Streamlit', ['interactive analysis'])]
                for number, (title, fields) in enumerate(steps):
                    x = .7 + number * 3.15
                    entity(slide, x, 1.8, 2.65, 1.65, title, fields)
                    if number < 3:
                        line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x + 2.65), Inches(2.6), Inches(x + 3.15), Inches(2.6))
                        line.line.color.rgb = RGBColor.from_string(teal)
                for number, line in enumerate(entry['body']):
                    textbox(slide, .8, 4.0 + number * .73, 11.7, .7, line, 22)
            elif kind == 'results' and experiment:
                table = comparison_frame(experiment)
                cols = ['model', 'mean_ROC_AUC_6', 'any_F1', 'training_s', 'peak_RSS_MiB']
                shape = slide.shapes.add_table(len(table) + 1, len(cols), Inches(.7), Inches(1.7), Inches(11.9), Inches(3.5))
                headers = ['Model', 'Mean AUC (6)', 'any F1', 'Fit seconds', 'Peak RSS MiB']
                for column, name in enumerate(headers):
                    shape.table.cell(0, column).text = name
                for row, values in enumerate(table[cols].itertuples(index=False, name=None), 1):
                    for column, value in enumerate(values):
                        if isinstance(value, float):
                            text = f'{value:.6f}' if column == 1 else f'{value:.3f}'
                        else:
                            text = str(value)
                        shape.table.cell(row, column).text = text
                for row in shape.table.rows:
                    for cell in row.cells:
                        for paragraph in cell.text_frame.paragraphs:
                            paragraph.font.name = 'Arial'
                            paragraph.font.size = Pt(17)
                summary = f"Selected on validation: {experiment['selected_model']}."
                official = read_json(root / 'results/metrics/official_test' / f"{experiment['experiment_id']}.json")
                if official and official.get('status') == 'evaluated':
                    ev = official['evaluation']
                    summary += (f" Official test ({ev['rows']:,} rows): mean AUC {ev['mean_roc_auc_six']:.4f}, "
                                f"any_toxic F1 {ev['per_label']['any_toxic']['f1']:.4f}.")
                summary += '\nPeak RSS uses 10 ms sampling in separate model processes.'
                textbox(slide, .8, 5.55, 11.7, 1.2, summary, 20, teal)
            elif kind == 'eda' and manifest:
                # Native editable data chart, never a decorative reconstruction.
                from pptx.chart.data import CategoryChartData
                from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
                data = CategoryChartData()
                data.categories = LABELS
                stats = manifest['splits']['train']
                data.add_series('Training prevalence', [stats['prevalence'][k] for k in LABELS])
                chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(.8), Inches(1.6),
                                               Inches(11.6), Inches(3.8), data).chart
                chart.has_legend = False
                chart.font.name = 'Arial'
                chart.font.size = Pt(17)
                chart.has_title = True
                chart.chart_title.text_frame.text = 'Observed training label prevalence'
                for paragraph in chart.chart_title.text_frame.paragraphs:
                    paragraph.font.name = 'Arial'
                    paragraph.font.size = Pt(18)
                chart.category_axis.tick_labels.font.name = 'Arial'
                chart.category_axis.tick_labels.font.size = Pt(16)
                chart.value_axis.tick_labels.font.name = 'Arial'
                chart.value_axis.tick_labels.font.size = Pt(16)
                chart.value_axis.tick_labels.number_format = '0%'
                # python-pptx's template can emit signed axis IDs. OOXML axis IDs
                # are unsigned; normalize both definitions and cross-references.
                for element in chart._chartSpace.xpath('.//c:axId | .//c:crossAx'):
                    value = int(element.get('val'))
                    if value < 0:
                        element.set('val', str(value + 2**32))
                textbox(slide, .8, 5.65, 11.5, 1.0, 'Observed training prevalence. Balanced logistic regression tests '
                        'the effect of rare-label weighting.', 22)
            elif kind == 'demo' and experiment and (root / 'results/screenshots/live_detection.jpg').exists() and \
                    read_json(root / 'results/browser_verification.json', {}).get('experiment_id') == experiment['experiment_id']:
                textbox(slide, .8, 1.7, 4.25, 1.0, 'Live detection loads a saved model.', 25, teal, True)
                textbox(slide, .8, 3.0, 4.25, 1.3, 'Model and threshold controls work without retraining.', 23)
                textbox(slide, .8, 4.6, 4.25, 1.4, 'CSV upload, prediction download and SQL execution were verified in the browser.', 22)
                slide.shapes.add_picture(str(root / 'results/screenshots/live_detection.jpg'),
                                         Inches(6.2), Inches(1.5), height=Inches(5.1))
                textbox(slide, .8, 6.65, 10.5, .4, 'Actual application capture. Functional demo only, excluded from benchmark metrics.', 16, grey)
            else:
                y = 1.7
                for line in entry['body']:
                    height = .95 if len(line) > 100 else .8
                    textbox(slide, .8, y, 11.6, height, line, 24 if len(entry['body']) <= 3 else 22)
                    y += height + .25
            textbox(slide, 12.1, 6.98, .6, .28, f'{index:02d}', 12, grey)
        notes = 'Project source: CODEX_PROMPT.md and PROJECT_REQUIREMENTS.md. '
        if index in [3, 5, 9, 11]:
            notes += 'Real-data status is derived only from data/processed/manifest.json and results/metrics/experiment.json. '
        notes += '\n'.join(f'{author}: {url}' for author, _, url in REFERENCES)
        slide.notes_slide.notes_text_frame.text = notes
    target = root / 'presentation/final_presentation.pptx'
    deck.save(target)
    save_json(root / 'presentation/slide_content.json', content)


def generate_deliverables(root: Path) -> dict:
    paths = directories(root)
    manifest = read_json(paths['processed'] / 'manifest.json')
    experiment = read_json(paths['metrics'] / 'experiment.json')
    if experiment and (not manifest or manifest['files'] != experiment['dataset_manifest']['files']):
        raise ValueError('Saved experiment does not match prepared inputs. Rebuild the database and train a new experiment first.')
    if manifest:
        if not db_path(root).exists():
            raise ValueError('Build the dataset database before generating data-backed reports: python -m scripts.build_database')
        connection = connect(db_path(root), readonly=True)
        try:
            stored = connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        finally:
            connection.close()
        if stored is None or json.loads(stored[0])['files'] != manifest['files']:
            raise ValueError('Database does not match prepared inputs. Run python -m scripts.build_database, then retrain if needed.')
    figures = generate_figures(root)
    status, sections = report_sections(root)
    markdown = '# Toxic Comment Detection\n\n' + '\n\n'.join(f'## {title}\n\n{body}' for title, body in sections)
    (paths['reports'] / 'final_report.md').write_text(markdown + '\n', encoding='utf-8')
    generate_pdf(root, status, sections)
    generate_presentation(root)
    result = dict(status=status, figures=figures, report_sections=len(sections), slides=12,
                  software_verification=test_evidence(root), real_data_available=(paths['processed'] / 'manifest.json').exists())
    save_json(root / 'results/deliverable_status.json', result)
    return result

"""Six-page dashboard. Startup reads artifacts and never fits a model."""
import io
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.config import ROOT, LABELS, TARGETS
from src.data_loader import preview_data
from src.database import db_path, predefined_queries, run_query
from src.evaluate import comparison_frame
from src.predict import load_bundle, predict_batch
from src.utils import csv_bytes, read_json
from src.thresholds import resolve_thresholds
from src.contextual_ui import contextual_catalog

PROJECT = Path(os.getenv('TOXIC_PROJECT_ROOT', str(ROOT))).resolve()
PAGES = ['Home / Overview', 'Live Toxicity Detection', 'Dataset Explorer',
         'Model Comparison', 'SQL Analytics', 'Batch Prediction']


@st.cache_resource(max_entries=8)
def cached_model(path: str, modified_ns: int):
    return load_bundle(Path(path))


def artifact_state() -> tuple[dict, dict, dict]:
    manifest = read_json(PROJECT / 'data/processed/manifest.json', {})
    experiment = read_json(PROJECT / 'results/metrics/experiment.json', {})
    available = {}
    if experiment and (not manifest or manifest['files'] != experiment['dataset_manifest']['files']):
        return manifest, {}, available
    if manifest and experiment and manifest['files'] == experiment['dataset_manifest']['files']:
        available = {r['model_name']: PROJECT / r['artifact'] for r in experiment['runs']
                     if (PROJECT / r['artifact']).is_file()}
    return manifest, experiment, available


def choose_model(available: dict, experiment: dict, key: str):
    if not available:
        st.info('No fitted models for the current data. Run: python -m scripts.train_models')
        return None
    names = list(available)
    chosen = st.selectbox('Model', names, index=names.index(experiment['selected_model'])
                          if experiment.get('selected_model') in names else 0, key=key)
    path = available[chosen]
    return cached_model(str(path), path.stat().st_mtime_ns)


def home(manifest: dict, experiment: dict, available: dict) -> None:
    st.title('Toxic Comment Detection')
    st.markdown('Machine learning classification and SQL-based data analysis. '
                'Explore the dataset, compare CPU models, and inspect predictions.')
    st.link_button('Official Jigsaw competition',
                   'https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data')
    if not manifest:
        st.warning('Dataset not prepared. This package contains no Kaggle data or invented experimental results.')
        st.code('python -m scripts.download_data\npython -m scripts.prepare_data\n'
                'python -m scripts.build_database\npython -m scripts.train_models\n'
                'python -m scripts.evaluate_models\npython -m scripts.generate_deliverables')
        st.write('You can also manually place train.csv, test.csv and test_labels.csv in data/raw/. '
                 'Kaggle authentication and acceptance of competition rules belong to your account.')
    else:
        columns = st.columns(3)
        columns[0].metric('Training comments', f"{manifest['splits']['train']['rows']:,}")
        columns[1].metric('Official test comments', f"{manifest['splits'].get('test', {}).get('rows', 0):,}")
        columns[2].metric('Available models', len(available))
        st.subheader('Data quality and provenance')
        st.dataframe(pd.DataFrame(manifest['splits']).T.drop(columns=['cooccurrence', 'positives', 'prevalence',
                     'length_bins', 'length_counts'], errors='ignore'), width='stretch')
        st.json({k: v for k, v in manifest.items() if k not in {'splits', 'instructor_approval'}}, expanded=False)
    st.subheader('Model status')
    st.write(f"Selected model: {experiment.get('selected_model', 'Pending a real-data experiment')}")
    contextual, contextual_models = contextual_catalog(PROJECT)
    if contextual_models:
        st.caption(f"{len(contextual_models)} additional refitted contextual-study candidates are available in Live Detection and Batch Prediction. "
                   'The original model default and historical test results remain unchanged. The study is an internal assessment, not proof of intent understanding.')
    st.caption('any_toxic is a separately trained seventh target. It is the maximum of the six ground-truth '
               'labels during training, and is distinct from the original toxic label.')
    st.write('Use the sidebar to navigate. Predictions are estimates and require human review.')


def live(experiment: dict, available: dict) -> None:
    st.title('Live Toxicity Detection')
    st.caption('Language may be offensive. Scores reflect patterns learned from the training data.')
    study, extra = contextual_catalog(PROJECT)
    live_available = {**available, **extra}
    bundle = choose_model(live_available, experiment, 'live_model')
    text = st.text_area('Comment', max_chars=100000, placeholder='Enter a comment to analyze')
    threshold = st.slider('Decision threshold', 0.0, 1.0, 0.5, 0.01, key='live_threshold')
    policy = st.radio('Threshold policy', ['Global override', 'Saved per-label thresholds', 'Custom per-label'],
                       horizontal=True, key='live_policy')
    custom = None
    if policy == 'Custom per-label' and bundle is not None:
        custom = {}
        defaults = resolve_thresholds(bundle)
        for label in TARGETS:
            custom[label] = st.slider(f'{label} threshold', 0.0, 1.0, defaults[label], 0.001, key=f'custom_{label}')
    st.caption('Global override uses the slider for every label. Saved thresholds use each model\'s own '
               'calibration values. Custom values apply to all compared models. All policies leave probability outputs unchanged.')
    defaults = [experiment['selected_model']] if experiment.get('selected_model') in live_available else []
    if extra and f'Study: {study.get("selected_model")}' in extra:
        defaults.append(f'Study: {study["selected_model"]}')
    comparisons = st.multiselect('Compare models side by side', list(live_available), default=defaults)
    if study:
        st.info('Study models were refitted on a new shared split. Choose Saved per-label thresholds to apply '
                'the calibration policies. ' + ('The frozen operating-point recommendation passed its assessment guards.'
                 if study.get('accepted') else 'The frozen recommendation did not pass every assessment guard; no superiority is claimed.'))
    if st.button('Analyze', type='primary', disabled=bundle is None):
        chosen_thresholds = (None if policy == 'Saved per-label thresholds' else custom if custom is not None else threshold)
        output = predict_batch(bundle, pd.DataFrame({'comment_text': [text]}), chosen_thresholds).iloc[0]
        toxic = bool(output.any_toxic_prediction)
        (st.warning if toxic else st.success)('Toxic / flagged under the dataset any-label target' if toxic else 'Non-toxic under the dataset any-label target')
        st.metric('any_toxic probability', f'{output.any_toxic_probability:.1%}')
        if bool(output.threat_prediction) or bool(output.identity_hate_prediction):
            st.warning('A threat or identity-hate category is flagged. Review it independently of the binary any_toxic decision.')
        frame = pd.DataFrame({'label': LABELS, 'probability': [output[f'{k}_probability'] for k in LABELS]})
        chart = px.bar(frame, x='probability', y='label', orientation='h', range_x=[0, 1])
        if policy == 'Global override':
            chart.add_vline(x=threshold, line_dash='dash')
        st.plotly_chart(chart, width='stretch')
        active_thresholds = resolve_thresholds(bundle, chosen_thresholds)
        decisions = pd.DataFrame({'target': TARGETS,
                    'probability': [float(output[f'{label}_probability']) for label in TARGETS],
                    'threshold': [active_thresholds[label] for label in TARGETS],
                    'flagged': [bool(output[f'{label}_prediction']) for label in TARGETS]})
        st.dataframe(decisions, width='stretch')
        probability_columns = {'target': TARGETS}
        comparison_decisions = []
        for name in comparisons:
            path = live_available[name]
            other = cached_model(str(path), path.stat().st_mtime_ns)
            scored = predict_batch(other, pd.DataFrame({'comment_text': [text]}), chosen_thresholds).iloc[0]
            probability_columns[name] = [float(scored[f'{label}_probability']) for label in TARGETS]
            values = resolve_thresholds(other, chosen_thresholds)
            comparison_decisions.append(dict(model=name, any_toxic_score=float(scored.any_toxic_probability),
                        any_toxic_threshold=values['any_toxic'], any_toxic_flag=bool(scored.any_toxic_prediction),
                        threat_flag=bool(scored.threat_prediction), identity_hate_flag=bool(scored.identity_hate_prediction)))
        if comparisons:
            st.subheader('Side-by-side model scores')
            st.dataframe(pd.DataFrame(probability_columns), width='stretch')
            st.dataframe(pd.DataFrame(comparison_decisions), width='stretch')
        st.info('Label meanings: obscene concerns profanity/sexual language, including language used positively. '
                'Negative criticism can be non-abusive. Insults target people; threats express harm; identity-based hate '
                'targets people because of identity. These lexical models do not reliably determine intent or context. '
                'any_toxic includes obscene and therefore is not an intent-only abuse score. Review threat/hate flags independently.')
        st.caption('The binary result uses the separately trained any_toxic model. '
                   'The six label probabilities need not agree with its score or sum to one. '
                   'Scores are uncalibrated estimates. Threshold changes do not alter saved evaluation metrics.')


def explorer(manifest: dict) -> None:
    st.title('Dataset Explorer')
    if not manifest:
        st.info('Run python -m scripts.prepare_data to validate and summarize your official CSVs.')
        return
    split = st.selectbox('Dataset split', list(manifest['splits']))
    limit = st.slider('Preview row limit', 20, 2000, 200, 20)
    frame = preview_data(PROJECT, split, limit)
    label_filter = st.selectbox('Preview label filter', ['All', 'any_toxic'] + LABELS)
    if label_filter != 'All':
        value = st.selectbox('Ground-truth value', [1, 0, -1], format_func=lambda x: {1: 'Positive', 0: 'Negative', -1: 'Unknown'}[x])
        frame = frame.loc[frame[label_filter] == value]
    max_chars = st.number_input('Maximum characters in preview', min_value=1, value=100000)
    frame = frame.loc[frame.character_count <= max_chars]
    st.caption('Filters apply to the first preview rows. Charts below summarize the entire prepared split.')
    st.dataframe(frame.drop(columns=['text_group'], errors='ignore'), width='stretch')
    stats = manifest['splits'][split]
    st.subheader('Quality checks')
    st.json({key: stats[key] for key in ['rows', 'scored_rows', 'unscored_rows', 'duplicate_comments', 'blank_comments',
                                        'null_comments', 'null_ids', 'duplicate_ids', 'mean_words', 'mean_characters']})
    prevalence = pd.DataFrame({'label': TARGETS, 'positive_fraction': [stats['prevalence'][k] for k in TARGETS]})
    st.plotly_chart(px.bar(prevalence, x='label', y='positive_fraction', range_y=[0, 1]), width='stretch')
    st.caption('Prevalence and co-occurrence use only fully scored rows. Unknown rows contribute to length counts.')
    st.plotly_chart(px.imshow(stats['cooccurrence'], x=LABELS, y=LABELS, text_auto=True,
                              title='Positive-label co-occurrence'), width='stretch')
    st.plotly_chart(px.bar(x=stats['length_bins'], y=stats['length_counts'],
                           labels={'x': 'Characters', 'y': 'Comments'}, title='Comment lengths'), width='stretch')


def comparison(experiment: dict) -> None:
    st.title('Model Comparison')
    if not experiment:
        st.info('No real-data experiment is available. Run python -m scripts.train_models.')
        return
    st.success(f"Selected on validation: {experiment['selected_model']}")
    st.write(experiment['selection_criterion'])
    st.dataframe(comparison_frame(experiment), width='stretch')
    st.caption('ROC-AUC and average precision assess ranking. Precision, recall, F1, accuracy and confusion '
               'matrices use threshold 0.5. The six-label mean excludes any_toxic and reports undefined labels.')
    label = st.selectbox('Target', TARGETS)
    model = st.selectbox('Confusion matrix model', [r['model_name'] for r in experiment['runs']])
    selected = next(r for r in experiment['runs'] if r['model_name'] == model)
    entry = selected['validation']['per_label'][label]
    st.dataframe(pd.DataFrame([{'model': r['model_name'],
                    **{key: r['validation']['per_label'][label][key] for key in
                       ['roc_auc', 'average_precision', 'precision', 'recall', 'f1', 'accuracy', 'positives', 'negatives']}}
                    for r in experiment['runs']]), width='stretch')
    for kind, x, y in [('roc', 'fpr', 'tpr'), ('pr', 'recall', 'precision')]:
        fig = go.Figure()
        for run in experiment['runs']:
            curve = run['validation']['per_label'][label][kind]
            if curve:
                fig.add_trace(go.Scatter(x=curve[x], y=curve[y], mode='lines', name=run['model_name']))
        fig.update_layout(title=f'{label}: {kind.upper()}', xaxis_title=x, yaxis_title=y,
                          xaxis_range=[0, 1], yaxis_range=[0, 1.02])
        st.plotly_chart(fig, width='stretch')
    st.plotly_chart(px.imshow(entry['confusion_matrix'], x=['Predicted 0', 'Predicted 1'],
                    y=['Actual 0', 'Actual 1'], text_auto=True, title=f'{model}: {label}'), width='stretch')
    for note in entry['notes']:
        st.info(note)
    table = comparison_frame(experiment)
    for metric in ['training_s', 'inference_s', 'peak_RSS_MiB']:
        st.plotly_chart(px.bar(table, x='model', y=metric), width='stretch')
    st.caption(experiment['memory_method'])
    official = read_json(PROJECT / 'results/metrics/official_test' / f"{experiment['experiment_id']}.json")
    if official:
        st.subheader('Held-out official test: selected model only')
        ev = official['evaluation']
        st.caption(f"{ev['rows']:,} fully scored test rows; every -1 row excluded. Fixed threshold: {ev['threshold']}.")
        st.dataframe(pd.DataFrame([{'target': target,
                     **{key: ev['per_label'][target][key] for key in
                        ['roc_auc', 'average_precision', 'precision', 'recall', 'f1', 'accuracy']}}
                     for target in TARGETS]), width='stretch')
        st.plotly_chart(px.imshow(ev['per_label'][label]['confusion_matrix'],
                    x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'], text_auto=True,
                    title=f'Official test: {experiment["selected_model"]}, {label}'), width='stretch')
        st.json(official, expanded=False)
    else:
        st.info('Official test results are pending. Run python -m scripts.evaluate_models after selection.')
    study, _ = contextual_catalog(PROJECT)
    if study:
        st.subheader('Contextual operating-point study')
        directory = (PROJECT / study['study']).parent
        st.write('Five candidates were refitted on a shared 70/15/15 fit/calibration/assessment split. '
                 'This table uses assessment rows only. Probabilities and AUC do not change when thresholds change. '
                 'Historical official-test results above were not reused for this study.')
        st.dataframe(pd.read_csv(directory / 'assessment_comparison.csv'), width='stretch')
        st.dataframe(pd.read_csv(directory / 'per_label_assessment.csv'), width='stretch')
        st.caption('Contextual slices retain pre-existing Jigsaw annotations; lexical cue matches are not independent intent labels.')


def sql_analytics() -> None:
    st.title('SQL Analytics')
    st.image(str(ROOT / 'docs/er_diagram.svg'), caption='Normalized SQLite schema')
    queries = predefined_queries()
    name = st.selectbox('Analysis', list(queries))
    st.code(queries[name], language='sql')
    if not db_path(PROJECT).exists():
        st.info('Run python -m scripts.build_database. An empty schema can be created with --empty.')
        return
    if st.button('Run query', type='primary'):
        frame = run_query(db_path(PROJECT), name)
        st.dataframe(frame, width='stretch')
        if frame.empty:
            st.info('No matching rows. Data, scored labels, fitted-model predictions, or qualifying errors may be absent.')
        else:
            numeric = list(frame.select_dtypes(include='number'))
            categories = [column for column in frame.columns if column not in numeric]
            if numeric and categories and len(frame) <= 100:
                st.plotly_chart(px.bar(frame, x=categories[0], y=numeric[-1]), width='stretch')
        st.download_button('Download query CSV', csv_bytes(frame), f'{name}.csv', 'text/csv')


def batch(experiment: dict, available: dict) -> None:
    st.title('Batch Prediction')
    st.write('Upload a CSV with a nonempty comment_text column. Limit: 20 MB and 100,000 rows. '
             'Feature matrices are processed in batches of 2,048 rows.')
    _, extra = contextual_catalog(PROJECT)
    bundle = choose_model({**available, **extra}, experiment, 'batch_model')
    upload = st.file_uploader('Comments CSV', type=['csv'])
    threshold = st.slider('Decision threshold', 0.0, 1.0, 0.5, 0.01, key='batch_threshold')
    saved_thresholds = st.checkbox('Use saved per-label thresholds', value=False, key='batch_saved_thresholds')
    if st.button('Run batch predictions', disabled=bundle is None or upload is None, type='primary'):
        if upload.size > 20 * 1024**2:
            raise ValueError('Upload is larger than 20 MB')
        from src.data_loader import DataValidationError
        import csv
        contents = upload.getvalue().decode('utf-8-sig')
        reader = csv.reader(io.StringIO(contents), strict=True)
        header = next(reader, [])
        if len(header) != len(set(header)) or 'comment_text' not in header:
            raise DataValidationError('CSV must contain unique columns including comment_text')
        if any(len(row) != len(header) for row in reader):
            raise DataValidationError('Malformed CSV: every record must match the header width')
        frame = pd.read_csv(io.StringIO(contents), nrows=100001, keep_default_na=False)
        with st.spinner('Scoring comments...'):
            output = predict_batch(bundle, frame, None if saved_thresholds else threshold)
        st.dataframe(output.head(100), width='stretch')
        st.download_button('Download predictions', csv_bytes(output), 'toxicity_predictions.csv', 'text/csv')
        st.caption('Predictions do not constitute ground truth or an automatic moderation recommendation.')


def main() -> None:
    st.set_page_config(page_title='Toxic Comment Detection', page_icon='💬', layout='wide')
    st.sidebar.title('Toxic Comment Detection')
    page = st.sidebar.radio('Navigation', PAGES)
    st.sidebar.caption('Programming for Data Analysis and Database\nBig Data Analysis')
    try:
        manifest, experiment, available = artifact_state()
        if page == PAGES[0]:
            home(manifest, experiment, available)
        elif page == PAGES[1]:
            live(experiment, available)
        elif page == PAGES[2]:
            explorer(manifest)
        elif page == PAGES[3]:
            comparison(experiment)
        elif page == PAGES[4]:
            sql_analytics()
        else:
            batch(experiment, available)
    except (ValueError, OSError, KeyError, pd.errors.ParserError, UnicodeError) as exc:
        st.error(str(exc))
        st.caption('Check the input file and README setup commands. No training runs during page loading.')


if __name__ == '__main__':
    main()

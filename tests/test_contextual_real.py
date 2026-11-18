"""Read-only contextual-artifact checks; no real-model fitting or test-set access."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from src.config import ROOT, TARGETS
from src.predict import load_bundle, predict_batch
from src.utils import read_json, sha256_file

pytestmark=pytest.mark.skipif(not (ROOT/'results/contextual/latest.json').exists(),reason='Contextual study not installed')


def test_frozen_choice_and_partition_provenance():
    latest=read_json(ROOT/'results/contextual/latest.json'); study=read_json(ROOT/latest['study'])
    parts=pd.read_parquet(ROOT/latest['model_dir']/'split_assignments.parquet')
    assert parts.groupby('text_group').partition.nunique().max()==1
    assert set(parts.partition)=={'fit','calibration','assessment'}
    frozen=read_json((ROOT/latest['study']).parent/'decision_before_assessment.json')
    assert frozen['assessment_metrics_accessed'] is False
    baseline=next(r for r in study['runs'] if r['model_name']=='tfidf_lr_balanced')
    eligible=[r for r in study['runs'] if r['calibration_tuned']['mean_roc_auc_six']>=baseline['calibration_tuned']['mean_roc_auc_six']-.01
              and all(r['calibration_tuned']['per_label'][label]['recall']>=study['protocol']['recall_floors'][label] for label in TARGETS)]
    chosen=min(eligible,key=lambda r:(r['calibration_tuned']['per_label']['any_toxic']['false_positive_rate'],-r['calibration_tuned']['mean_roc_auc_six']))
    assert chosen['model_name']==frozen['model']
    for run in study['runs']:
        assert study['assessment'][run['model_name']]['default_0.5']['mean_roc_auc_six']==study['assessment'][run['model_name']]['calibration_thresholds']['mean_roc_auc_six']


def test_original_outputs_and_artifacts_preserved():
    protected=read_json(ROOT/'results/contextual/preserved_artifacts.json')
    assert all(sha256_file(ROOT/path)==value for path,value in protected.items())
    old=read_json(ROOT/'results/metrics/experiment.json')
    expected=pd.read_csv(ROOT/'results/contextual/reported_cases_original_models.csv')
    for run in old['runs']:
        rows=expected.loc[expected.model.eq(run['model_name'])].sort_values('case')
        bundle=load_bundle(ROOT/run['artifact'])
        a=predict_batch(bundle,rows[['comment_text']].reset_index(drop=True),.5)
        b=predict_batch(bundle,rows[['comment_text']].reset_index(drop=True),dict.fromkeys(TARGETS,.9))
        np.testing.assert_allclose(a[[f'{label}_probability' for label in TARGETS]],rows[TARGETS],atol=1e-14)
        np.testing.assert_array_equal(a[[f'{label}_probability' for label in TARGETS]],b[[f'{label}_probability' for label in TARGETS]])


def test_live_saved_thresholds_and_side_by_side(monkeypatch):
    monkeypatch.setenv('TOXIC_PROJECT_ROOT',str(ROOT))
    latest=read_json(ROOT/'results/contextual/latest.json')
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    app.sidebar.radio[0].set_value('Live Toxicity Detection').run()
    app.selectbox[0].set_value('Study: '+latest['selected_model']).run()
    policy=next(widget for widget in app.radio if widget.label=='Threshold policy')
    policy.set_value('Saved per-label thresholds').run()
    app.text_area[0].set_value('Thank you for explaining this so clearly.')
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.dataframe)>=3
    bundle=load_bundle(ROOT/latest['model_dir']/f"{latest['selected_model']}.joblib")
    assert app.dataframe[0].value.set_index('target').loc['any_toxic','threshold']==bundle['thresholds']['any_toxic']

"""Read-only integration checks against saved real artifacts, with no refitting."""
import io
import json
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from src.config import ROOT, MODEL_NAMES, TARGETS
from src.database import db_path, predefined_queries, run_query
from src.predict import load_bundle, predict_batch
from src.utils import read_json, sha256_file

AVAILABLE = all(path.exists() for path in [ROOT / 'results/metrics/experiment.json', db_path(ROOT),
                    ROOT / 'data/processed/train.parquet', ROOT / 'data/processed/test.parquet'])
pytestmark = pytest.mark.skipif(not AVAILABLE, reason='Real-data runtime artifacts are not installed')


def test_real_split_and_unknown_label_exclusion():
    experiment = read_json(ROOT / 'results/metrics/experiment.json')
    assignments = pd.read_parquet(ROOT / experiment['split_assignments'])
    train = pd.read_parquet(ROOT / 'data/processed/train.parquet', columns=['id', 'text_group'])
    merged = train.merge(assignments, on='id', validate='one_to_one')
    assert len(merged) == len(train)
    assert set(merged.loc[merged.partition.eq('train'), 'text_group']).isdisjoint(
               merged.loc[merged.partition.eq('validation'), 'text_group'])
    official = read_json(ROOT / 'results/metrics/official_test' / f"{experiment['experiment_id']}.json")
    test = pd.read_parquet(ROOT / 'data/processed/test.parquet', columns=TARGETS)
    assert official['evaluation']['rows'] == int(test[TARGETS].isin([0, 1]).all(axis=1).sum())


@pytest.mark.parametrize('name', MODEL_NAMES)
def test_real_model_saved_and_loadable(name):
    experiment = read_json(ROOT / 'results/metrics/experiment.json')
    run = next(r for r in experiment['runs'] if r['model_name'] == name)
    artifact = ROOT / run['artifact']
    assert sha256_file(artifact) == run['artifact_sha256']
    sample = pd.read_parquet(ROOT / 'data/processed/train.parquet', columns=['comment_text']).head(3)
    scored = predict_batch(load_bundle(artifact), sample, batch_size=2)
    probabilities = scored[[f'{label}_probability' for label in TARGETS]].to_numpy()
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()


@pytest.mark.parametrize('name', list(predefined_queries()))
def test_real_sql_outputs(name):
    assert not run_query(db_path(ROOT), name).empty


@pytest.mark.parametrize('page', ['Home / Overview', 'Live Toxicity Detection', 'Dataset Explorer',
                                  'Model Comparison', 'SQL Analytics', 'Batch Prediction'])
def test_real_dashboard_pages(monkeypatch, page):
    monkeypatch.setenv('TOXIC_PROJECT_ROOT', str(ROOT))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    app.sidebar.radio[0].set_value(page).run()
    if page == 'SQL Analytics':
        app.button[0].click().run()
    assert not app.exception and not app.error


def test_real_live_analyze(monkeypatch):
    monkeypatch.setenv('TOXIC_PROJECT_ROOT', str(ROOT))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    app.sidebar.radio[0].set_value('Live Toxicity Detection').run()
    app.text_area[0].set_value('Thank you for explaining this so clearly.')
    app.button[0].click().run()
    assert not app.exception and not app.error and len(app.metric) == 1


def test_real_batch_page_and_csv(monkeypatch):
    monkeypatch.setenv('TOXIC_PROJECT_ROOT', str(ROOT))
    sample = pd.read_parquet(ROOT / 'data/processed/train.parquet', columns=['comment_text']).head(5)
    contents = sample.to_csv(index=False).encode('utf-8')
    upload = io.BytesIO(contents)
    upload.size = len(contents)
    upload.name = 'real_comment_sample.csv'
    monkeypatch.setattr(st, 'file_uploader', lambda *args, **kwargs: upload)
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    app.sidebar.radio[0].set_value('Batch Prediction').run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.dataframe) == 1 and len(app.dataframe[0].value) == 5

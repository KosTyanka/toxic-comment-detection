"""Streamlit controls execute without requiring a browser or official data."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from src.config import ROOT


@pytest.mark.parametrize('page', ['Home / Overview', 'Live Toxicity Detection', 'Dataset Explorer',
                                  'Model Comparison', 'SQL Analytics', 'Batch Prediction'])
def test_dashboard_missing_data(tmp_path, monkeypatch, page):
    monkeypatch.setenv('TOXIC_PROJECT_ROOT', str(tmp_path))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert not app.error


@pytest.mark.parametrize('page', ['Home / Overview', 'Dataset Explorer', 'Model Comparison', 'SQL Analytics'])
def test_dashboard_artifacts(pipeline_root, monkeypatch, page):
    monkeypatch.setenv('TOXIC_PROJECT_ROOT', str(pipeline_root))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    if page == 'SQL Analytics':
        app.button[0].click().run()
    assert not app.exception
    assert not app.error


def test_live_detection_control(pipeline_root, monkeypatch):
    monkeypatch.setenv('TOXIC_PROJECT_ROOT', str(pipeline_root))
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    app.sidebar.radio[0].set_value('Live Toxicity Detection').run()
    app.text_area[0].set_value('friendly discussion')
    app.slider[0].set_value(0.4)
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.metric) == 1

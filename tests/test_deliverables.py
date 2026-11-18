"""Artifact generators must work with both absent and observed input artifacts."""
from pptx import Presentation
from src.config import ROOT
from src.database import db_path, initialize
from src.deliverables import generate_deliverables
from src.utils import save_json
import pytest


def test_missing_results_deliverables(tmp_path):
    initialize(db_path(tmp_path))
    status = generate_deliverables(tmp_path)
    assert status['real_data_available'] is False
    assert status['figures'] == []
    markdown = (tmp_path / 'reports/final_report.md').read_text(encoding='utf-8')
    assert 'MISSING RESULTS' in markdown
    assert 'empty schema only' in markdown
    assert markdown.count('\n## ') == 19
    pdf = tmp_path / 'reports/final_report.pdf'
    assert pdf.read_bytes().startswith(b'%PDF') and pdf.stat().st_size > 5000
    slides = Presentation(tmp_path / 'presentation/final_presentation.pptx')
    assert len(slides.slides) == 12
    text = '\n'.join(shape.text for slide in slides.slides for shape in slide.shapes if shape.has_text_frame)
    assert 'Real-data training is pending' in text


def test_generators_with_temporary_fixture_artifacts(pipeline_root):
    # These generated fixture documents remain inside pytest's temporary folder.
    # They are never copied to the distributed reports or described as Kaggle results.
    status = generate_deliverables(pipeline_root)
    assert len(status['figures']) >= 12
    deck = Presentation(pipeline_root / 'presentation/final_presentation.pptx')
    assert len(deck.slides) == 12
    assert any(shape.has_chart for shape in deck.slides[4].shapes)
    assert any(shape.has_table for shape in deck.slides[8].shapes)
    chart = next(shape.chart for shape in deck.slides[4].shapes if shape.has_chart)
    assert all(0 <= int(element.get('val')) < 2**32
               for element in chart._chartSpace.xpath('.//c:axId | .//c:crossAx'))


def test_stale_experiment_cannot_enter_report(tmp_path):
    save_json(tmp_path / 'results/metrics/experiment.json', {'dataset_manifest': {'files': {'old': 'fingerprint'}}})
    with pytest.raises(ValueError, match='does not match'):
        generate_deliverables(tmp_path)

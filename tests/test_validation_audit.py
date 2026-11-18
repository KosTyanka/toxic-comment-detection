"""Validate the optional template as probabilities, never as ground truth."""
import pandas as pd
import pytest
from src.config import LABELS
from src.data_loader import DataValidationError
from scripts.validate_data import validate_dataset_files


def test_validation_audit_with_template(raw_root):
    test = pd.read_csv(raw_root / 'data/raw/test.csv')
    template = pd.DataFrame({'id': test.id, **{label: 0.5 for label in LABELS}})
    template.to_csv(raw_root / 'data/raw/sample_submission.csv', index=False)
    audit = validate_dataset_files(raw_root)
    assert audit['test_and_labels_identical_ids']
    assert audit['test_and_submission_identical_ids']
    assert audit['sample_submission.csv']['label_values']['toxic'] == {'0.5': 12}


def test_template_id_mismatch_rejected(raw_root):
    template = pd.DataFrame({'id': ['wrong-id'], **{label: [0.5] for label in LABELS}})
    template.to_csv(raw_root / 'data/raw/sample_submission.csv', index=False)
    with pytest.raises(DataValidationError, match='Submission-template IDs'):
        validate_dataset_files(raw_root)

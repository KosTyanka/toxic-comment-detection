"""Model tests use exclusively the labeled artificial pytest fixtures."""
import numpy as np
import pandas as pd
import pytest
from src.config import DEFAULTS, LABELS, MODEL_NAMES, TARGETS
from src.evaluate import evaluate_targets
from src.predict import load_bundle, predict_batch, probability_matrix, validate_batch
from src.train import evaluate_official, fit_bundle, split_training
from src.utils import read_json


def test_split_reproducible_and_duplicate_aware(pipeline_root):
    frame = pd.read_parquet(pipeline_root / 'data/processed/train.parquet')
    a, b, _ = split_training(frame, DEFAULTS)
    c, d, _ = split_training(frame, DEFAULTS)
    assert a.id.tolist() == c.id.tolist() and b.id.tolist() == d.id.tolist()
    assert set(a.text_group).isdisjoint(b.text_group)
    assert set(a.id).isdisjoint(b.id)
    assert len(a) + len(b) == len(frame)


@pytest.mark.parametrize('name', MODEL_NAMES)
def test_fitted_probabilities_and_sparse_fit(pipeline_root, name):
    experiment = read_json(pipeline_root / 'results/metrics/experiment.json')
    run = next(r for r in experiment['runs'] if r['model_name'] == name)
    bundle = load_bundle(pipeline_root / run['artifact'])
    p = probability_matrix(bundle, ['friendly discussion', 'hostile rude words'])
    assert p.shape == (2, 7)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    if bundle['vectorizer'] is not None:
        assert hasattr(bundle['vectorizer'].transform(['text']), 'tocsr')
    assert run['training_time'] >= 0 and run['peak_memory_mb'] > 0


def test_vectorizer_does_not_fit_validation(pipeline_root):
    experiment = read_json(pipeline_root / 'results/metrics/experiment.json')
    run = next(r for r in experiment['runs'] if r['model_name'] == 'bow_nb')
    bundle = load_bundle(pipeline_root / run['artifact'])
    assignments = pd.read_parquet(pipeline_root / experiment['split_assignments'])
    frame = pd.read_parquet(pipeline_root / 'data/processed/train.parquet')
    validation = frame.loc[frame.id.isin(assignments.loc[assignments.partition.eq('validation'), 'id'])]
    train = frame.loc[~frame.id.isin(validation.id)]
    validation_tokens = set(validation.comment_text.str.split().str[-1]) - set(train.comment_text.str.split().str[-1])
    assert validation_tokens
    assert all(token not in bundle['vectorizer'].vocabulary_ for token in validation_tokens)


def test_single_class_training_and_undefined_metrics():
    frame = pd.DataFrame({'comment_text': ['friendly sample', 'helpful sample'], **dict.fromkeys(TARGETS, [0, 0])})
    bundle = fit_bundle('tfidf_lr', frame, {**DEFAULTS, 'min_df': 1})
    p = probability_matrix(bundle, ['new discussion'])
    assert np.array_equal(p, np.zeros((1, 7)))
    result = evaluate_targets(frame.iloc[:1][TARGETS], p)
    assert result['mean_roc_auc_six'] is None
    assert result['per_label']['threat']['average_precision'] is None
    assert len(result['skipped_auc_labels']) == 6


def test_all_positive_probability_handling():
    frame = pd.DataFrame({'comment_text': ['hostile sample', 'rude sample'], **dict.fromkeys(TARGETS, [1, 1])})
    bundle = fit_bundle('bow_nb', frame, {**DEFAULTS, 'min_df': 1})
    assert np.array_equal(probability_matrix(bundle, ['text']), np.ones((1, 7)))


def test_official_evaluation_once_and_unknown_exclusion(pipeline_root):
    first = evaluate_official(pipeline_root)
    second = evaluate_official(pipeline_root)
    assert first == second
    assert first['evaluation']['rows'] == 10
    experiment = read_json(pipeline_root / 'results/metrics/experiment.json')
    assert first['selected_model'] == experiment['selected_model']


def test_six_label_mean_excludes_derived_target():
    truth = pd.DataFrame({label: [0, 1, 0, 1] for label in TARGETS})
    scores = np.array([[0.1] * 6 + [0.9], [0.9] * 6 + [0.1],
                       [0.2] * 6 + [0.8], [0.8] * 6 + [0.2]])
    result = evaluate_targets(truth, scores)
    assert result['mean_roc_auc_six'] == 1
    assert result['per_label']['any_toxic']['roc_auc'] == 0


def test_unknown_truth_never_negative():
    truth = pd.DataFrame({label: [-1] for label in TARGETS})
    with pytest.raises(ValueError, match='exclude -1'):
        evaluate_targets(truth, np.zeros((1, 7)))


@pytest.mark.parametrize('frame', [pd.DataFrame({'wrong': ['x']}), pd.DataFrame({'comment_text': ['']}),
                                   pd.DataFrame({'comment_text': [None]}), pd.DataFrame({'comment_text': []})])
def test_invalid_batch(frame):
    with pytest.raises(ValueError):
        validate_batch(frame)


def test_batch_consistency(pipeline_root):
    experiment = read_json(pipeline_root / 'results/metrics/experiment.json')
    bundle = load_bundle(pipeline_root / experiment['runs'][1]['artifact'])
    frame = pd.DataFrame({'comment_text': ['friendly discussion', 'hostile nasty rude'] * 3})
    a = predict_batch(bundle, frame, batch_size=1)
    b = predict_batch(bundle, frame, batch_size=6)
    pd.testing.assert_frame_equal(a, b)
    assert len(a.columns) == 15

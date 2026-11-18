"""Threshold, split and vocabulary invariants using explicit synthetic fixtures."""
import numpy as np
import pandas as pd
import pytest
from src.config import TARGETS, DEFAULTS
from src.contextual import (make_context_split, select_thresholds, context_cohorts,
                            paired_fp_comparison, fit_context_candidate)
from src.evaluate import evaluate_targets
from src.predict import predict_batch, probability_matrix
from src.preprocessing import text_group
from src.thresholds import resolve_thresholds, apply_thresholds
from src.train import fit_bundle


def _fixture():
    return pd.DataFrame({'id':[f'fixture-{i}' for i in range(100)],
         'comment_text':[f'SYNTHETIC fixture {i}' for i in range(100)],
         'text_group':[text_group(f'SYNTHETIC fixture {i}') for i in range(100)],
         **{label:[i%2 for i in range(100)] for label in TARGETS}})


def test_saved_thresholds_change_decisions_not_scores():
    frame=pd.DataFrame({'comment_text':['friendly helpful sample','hostile rude sample']*8,
                         **{label:[0,1]*8 for label in TARGETS}})
    bundle=fit_bundle('tfidf_lr',frame,{**DEFAULTS,'min_df':1})
    texts=pd.DataFrame({'comment_text':['hostile sample','friendly sample']})
    before=probability_matrix(bundle,texts.comment_text.tolist())
    bundle['thresholds']={label:1.0 if j%2 else 0.0 for j,label in enumerate(TARGETS)}
    scored=predict_batch(bundle,texts)
    after=scored[[f'{label}_probability' for label in TARGETS]].to_numpy()
    np.testing.assert_array_equal(before,after)
    assert scored.toxic_prediction.tolist()==[1,1]
    assert scored.severe_toxic_prediction.tolist()==[0,0]
    scalar=predict_batch(bundle,texts,threshold=.5)
    assert np.array_equal(scalar.toxic_prediction,(before[:,0]>=.5).astype(int))


@pytest.mark.parametrize('override',[{'unknown':.4},{'threat':float('nan')},{'any_toxic':1.1},float('inf'),-.1])
def test_invalid_thresholds_rejected(override):
    with pytest.raises(ValueError): resolve_thresholds(override=override)


def test_partial_threshold_override():
    values=resolve_thresholds({'thresholds':dict.fromkeys(TARGETS,.6)}, {'threat':.2})
    assert values['threat']==.2 and values['toxic']==.6


def test_threshold_evaluation_auc_is_invariant():
    truth=pd.DataFrame({label:[0,0,1,1] for label in TARGETS})
    p=np.tile(np.array([.1,.4,.5,.9])[:,None],(1,7))
    a=evaluate_targets(truth,p,.5)
    b=evaluate_targets(truth,p,dict.fromkeys(TARGETS,.8))
    assert a['mean_roc_auc_six']==b['mean_roc_auc_six']
    assert a['per_label']['threat']['recall']>b['per_label']['threat']['recall']
    assert a['per_label']['threat']['false_positive_rate']==0


def test_threshold_selection_recall_constraint():
    truth=pd.DataFrame({label:[0,0,0,1,1] for label in TARGETS})
    p=np.tile(np.array([.1,.2,.7,.6,.9])[:,None],(1,7))
    thresholds,_=select_thresholds(truth,p,dict.fromkeys(TARGETS,1.))
    assert all(value==.6 for value in thresholds.values())
    result=evaluate_targets(truth,p,thresholds)
    assert result['per_label']['any_toxic']['recall']==1


def test_three_way_split_is_disjoint_and_reproducible():
    frame=_fixture(); a=make_context_split(frame,4); b=make_context_split(frame,4)
    pd.testing.assert_frame_equal(a,b)
    assert a.partition.value_counts().to_dict()=={'fit':70,'calibration':15,'assessment':15}
    assert a.groupby('text_group').partition.nunique().max()==1


def test_context_slices_do_not_relabel_truth():
    texts=pd.Series(['I love this damn example','An incorrect explanation','A kind message'])
    masks=context_cohorts(texts)
    assert masks['profanity_and_positive_word_cues'].tolist()==[True,False,False]
    assert masks['criticism_without_profanity_cues'].tolist()==[False,True,False]


def test_paired_false_positive_comparison():
    result=paired_fp_comparison([0]*10,[1]*10,[0]*10,42)
    assert result['avoided_false_positives']==10
    assert result['candidate_minus_reference_fpr']==-1


def test_word_character_candidate_multilabel_sparse_outputs():
    frame=pd.DataFrame({'comment_text':['SYNTHETIC friendly discussion','SYNTHETIC hostile abuse']*8,
                         **{label:[0,1]*8 for label in TARGETS}})
    bundle=fit_context_candidate(frame,{'seed':42})
    assert len(bundle['estimators'])==7 and bundle['targets']==TARGETS
    matrix=bundle['vectorizer'].transform(['SYNTHETIC new text'])
    assert hasattr(matrix,'tocsr')
    probabilities=probability_matrix(bundle,['SYNTHETIC new text'])
    assert probabilities.shape==(1,7) and np.isfinite(probabilities).all()

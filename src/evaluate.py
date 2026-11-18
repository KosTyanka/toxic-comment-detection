"""Seven-target metrics, with competition averages restricted to six labels."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve, precision_recall_curve)
from src.config import LABELS, TARGETS
from src.thresholds import resolve_thresholds


def _thin(*values, limit: int = 400):
    indices = np.unique(np.linspace(0, len(values[0]) - 1, min(limit, len(values[0]))).astype(int))
    return [np.asarray(value)[indices].tolist() for value in values]


def evaluate_targets(truth: pd.DataFrame, probabilities: np.ndarray,
                     threshold: float | dict = 0.5) -> dict:
    """Return JSON-safe real metrics; single-class AUC is explicitly undefined."""
    if probabilities.shape != (len(truth), len(TARGETS)):
        raise ValueError('Probability array does not match truth rows and seven targets')
    if len(truth) == 0 or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Require nonempty truth and finite probabilities within [0,1]')
    if not truth[TARGETS].isin([0, 1]).all().all():
        raise ValueError('Evaluation requires fully known binary ground truth; exclude -1 rows first')
    thresholds = resolve_thresholds(override=threshold)
    output = {}
    for i, label in enumerate(TARGETS):
        y = truth[label].to_numpy(dtype='int8')
        p = probabilities[:, i]
        predicted = (p >= thresholds[label]).astype('int8')
        precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, average='binary', zero_division=0)
        both_classes = len(np.unique(y)) == 2
        has_positive = bool(y.sum())
        notes = []
        if not both_classes:
            notes.append('ROC-AUC/ROC curve undefined: evaluation truth has one class.')
        if not has_positive:
            notes.append('Average precision/PR curve undefined: no positive ground truth.')
        if not predicted.sum():
            notes.append('Precision uses zero_division=0 because no positives were predicted.')
        roc = pr = None
        if both_classes:
            fpr, tpr, _ = roc_curve(y, p)
            roc = dict(zip(['fpr', 'tpr'], _thin(fpr, tpr)))
        if has_positive:
            pre, rec, _ = precision_recall_curve(y, p)
            pr = dict(zip(['precision', 'recall'], _thin(pre, rec)))
        matrix = confusion_matrix(y, predicted, labels=[0, 1])
        negative_count = int((y == 0).sum())
        output[label] = dict(roc_auc=float(roc_auc_score(y, p)) if both_classes else None,
                             average_precision=float(average_precision_score(y, p)) if has_positive else None,
                             precision=float(precision), recall=float(recall), f1=float(f1),
                             accuracy=float(accuracy_score(y, predicted)),
                             positives=int(y.sum()), negatives=int(len(y) - y.sum()),
                             false_positive_rate=float(matrix[0, 1] / negative_count) if negative_count else None,
                             decision_threshold=thresholds[label],
                             confusion_matrix=matrix.tolist(),
                             roc=roc, pr=pr, notes=notes)
    auc_values = [output[k]['roc_auc'] for k in LABELS if output[k]['roc_auc'] is not None]
    ap_values = [output[k]['average_precision'] for k in LABELS if output[k]['average_precision'] is not None]
    return dict(rows=len(truth), threshold=threshold, per_label=output,
                mean_roc_auc_six=float(np.mean(auc_values)) if auc_values else None,
                macro_average_precision_six=float(np.mean(ap_values)) if ap_values else None,
                roc_auc_labels=len(auc_values), skipped_auc_labels=[k for k in LABELS if output[k]['roc_auc'] is None],
                metric_note='Average precision is the non-interpolated PR summary, not trapezoidal PR area. '
                            'ROC/PR display curves may be downsampled; scores use all observations.')


def comparison_frame(experiment: dict) -> pd.DataFrame:
    rows = []
    for run in experiment.get('runs', []):
        metrics = run['validation']
        binary = metrics['per_label']['any_toxic']
        rows.append(dict(model=run['model_name'], mean_ROC_AUC_6=metrics['mean_roc_auc_six'],
                         macro_AP_6=metrics['macro_average_precision_six'], any_F1=binary['f1'],
                         any_precision=binary['precision'], any_recall=binary['recall'],
                         training_s=run['training_time'], inference_s=run['inference_time'],
                         peak_RSS_MiB=run['peak_memory_mb']))
    return pd.DataFrame(rows)

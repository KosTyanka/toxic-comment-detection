"""Shared batch and interactive inference from trusted local joblib artifacts."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from src.config import TARGETS
from src.thresholds import resolve_thresholds


def probability_matrix(bundle: dict, texts: list[str]) -> np.ndarray:
    """Transform once, then score each binary model without densifying features."""
    matrix = (bundle['vectorizer'].transform(texts) if bundle['vectorizer'] is not None
              else sparse.csr_matrix((len(texts), 1)))
    columns = []
    for predictor in bundle['estimators']:
        probabilities = predictor.predict_proba(matrix)
        classes = list(predictor.classes_)
        columns.append(probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(texts)))
    return np.column_stack(columns)


def load_bundle(path: Path) -> dict:
    """Load only project-produced artifacts; pickle/joblib files must be trusted."""
    bundle = joblib.load(path)
    if bundle.get('targets') != TARGETS:
        raise ValueError('Incompatible model target schema; retrain using this project')
    return bundle


def validate_batch(frame: pd.DataFrame, max_rows: int = 100000) -> None:
    if 'comment_text' not in frame:
        raise ValueError('CSV needs a comment_text column')
    if frame.empty or len(frame) > max_rows:
        raise ValueError(f'Upload 1 to {max_rows:,} rows')
    if frame.comment_text.isna().any() or not frame.comment_text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError('Every comment_text must be a nonempty string')
    if frame.comment_text.str.len().max() > 100000:
        raise ValueError('Each comment must contain at most 100,000 characters')


def predict_batch(bundle: dict, frame: pd.DataFrame, threshold: float | dict | None = None,
                  batch_size: int = 2048) -> pd.DataFrame:
    validate_batch(frame)
    thresholds = resolve_thresholds(bundle, threshold)
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    parts = []
    for start in range(0, len(frame), batch_size):
        parts.append(probability_matrix(bundle, frame.comment_text.iloc[start:start + batch_size].tolist()))
    probabilities = np.vstack(parts)
    result = frame.copy()
    for i, label in enumerate(TARGETS):
        result[f'{label}_probability'] = probabilities[:, i]
        result[f'{label}_prediction'] = (probabilities[:, i] >= thresholds[label]).astype('int8')
    return result

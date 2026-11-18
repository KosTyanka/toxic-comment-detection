"""Conservative text preparation and explicit target definitions."""
import hashlib
import numpy as np
import pandas as pd
from src.config import LABELS


def text_group(text: str) -> str:
    """Group identical whitespace-trimmed comments across the split."""
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result['character_count'] = result.comment_text.str.len().astype('int32')
    result['word_count'] = result.comment_text.str.split().str.len().astype('int32')
    result['text_group'] = result.comment_text.map(text_group)
    return result


def derive_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Only fully scored rows receive the derived seventh label."""
    result = frame.copy()
    values = result[LABELS].to_numpy()
    known = np.isin(values, [0, 1]).all(axis=1)
    result['any_toxic'] = np.where(known, values.max(axis=1), -1).astype('int8')
    return result

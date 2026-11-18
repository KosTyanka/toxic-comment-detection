"""Decision thresholds are independent of unchanged probability estimates."""
from collections.abc import Mapping
import numpy as np
from src.config import TARGETS


def resolve_thresholds(bundle: dict | None = None, override=None) -> dict[str, float]:
    """Use saved per-target values unless an explicit scalar/dict override is given.

    Partial dictionaries override only named targets; unknown targets and nonfinite
    values are rejected. Older bundles without thresholds retain the 0.5 default.
    """
    values = {label: 0.5 for label in TARGETS}
    saved = (bundle or {}).get('thresholds', {})
    if not isinstance(saved, Mapping):
        raise ValueError('Saved thresholds must be a target mapping')
    for item in [saved, override]:
        if item is None:
            continue
        if isinstance(item, Mapping):
            if set(item) - set(TARGETS):
                raise ValueError('Unknown target in threshold mapping')
            values.update(item)
        else:
            values = dict.fromkeys(TARGETS, item)
    for label, value in values.items():
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid threshold for {label}') from exc
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Thresholds must be finite values in [0,1]')
        values[label] = value
    return values


def apply_thresholds(probabilities: np.ndarray, thresholds: dict) -> np.ndarray:
    """Return decisions without editing probabilities or reordering the targets."""
    values = resolve_thresholds(override=thresholds)
    return (probabilities >= np.array([values[k] for k in TARGETS])).astype('int8')

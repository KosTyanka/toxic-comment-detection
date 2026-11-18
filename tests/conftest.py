"""Explicitly synthetic fixtures for software verification, never research evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.config import LABELS
from src.data_loader import prepare_data
from src.database import build_database
from src.train import train_models


def synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate tiny artificial rows solely inside pytest temporary directories."""
    rows = []
    for i in range(120):
        values = [int(i % 2 == 1 and (i // 2) % (j + 2) == 0) for j in range(6)]
        words = 'hostile nasty rude' if any(values) else 'friendly helpful discussion'
        rows.append(dict(id=f'fixture-train-{i}', comment_text=f'SYNTHETIC UNIT FIXTURE {words} token{i}',
                         **dict(zip(LABELS, values))))
    train = pd.DataFrame(rows)
    # One exact duplicate comment under a distinct ID tests grouped splitting.
    train.loc[119, 'comment_text'] = train.loc[117, 'comment_text']
    train.loc[119, LABELS] = train.loc[117, LABELS].to_numpy()
    test = pd.DataFrame({'id': [f'fixture-test-{i}' for i in range(12)],
                         'comment_text': [f'SYNTHETIC HELD OUT FIXTURE discussion sample{i}' for i in range(12)]})
    labels = pd.DataFrame({'id': test.id, **{name: [int((i + j) % 3 == 0) for i in range(12)]
                                           for j, name in enumerate(LABELS)}})
    labels.loc[0, LABELS] = -1
    labels.loc[1, LABELS[0]] = -1
    # Deliberately shuffle labels to verify ID-based joining rather than row order.
    return train, test, labels.sample(frac=1, random_state=42)


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    raw = tmp_path / 'data/raw'
    raw.mkdir(parents=True)
    for name, frame in zip(['train', 'test', 'test_labels'], synthetic_frames()):
        frame.to_csv(raw / f'{name}.csv', index=False)
    return tmp_path


@pytest.fixture(scope='session')
def pipeline_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp('SYNTHETIC_UNIT_ONLY')
    raw = root / 'data/raw'
    raw.mkdir(parents=True)
    for name, frame in zip(['train', 'test', 'test_labels'], synthetic_frames()):
        frame.to_csv(raw / f'{name}.csv', index=False)
    prepare_data(root, chunksize=17)
    build_database(root)
    train_models(root, {'max_features': 300, 'min_df': 1})
    return root

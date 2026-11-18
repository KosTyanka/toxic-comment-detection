"""Load additive contextual-study artifacts without changing the original run."""
from pathlib import Path
from src.config import MODEL_NAMES
from src.utils import read_json


def contextual_catalog(root: Path) -> tuple[dict, dict]:
    latest = read_json(root / 'results/contextual/latest.json', {})
    manifest = read_json(root / 'data/processed/manifest.json', {})
    if not latest or latest.get('training_sha256') != manifest.get('files', {}).get('train.csv', {}).get('sha256'):
        return {}, {}
    directory = root / latest['model_dir']
    available = {f'Study: {name}': directory / f'{name}.joblib'
                 for name in MODEL_NAMES + ['word_char_lr_balanced'] if (directory / f'{name}.joblib').is_file()}
    return latest, available

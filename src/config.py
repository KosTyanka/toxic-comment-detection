"""Paths and experiment defaults, independent of the calling directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
TARGETS = LABELS + ['any_toxic']
SEED = 42
MODEL_NAMES = ['dummy', 'bow_nb', 'tfidf_lr', 'tfidf_lr_balanced']
DEFAULTS = dict(seed=SEED, validation_size=0.2, max_features=50000,
                ngram_range=[1, 2], min_df=2, batch_size=2048, threshold=0.5)


def directories(root: Path = ROOT) -> dict[str, Path]:
    """Create and return project storage locations."""
    paths = {name: root / rel for name, rel in {
        'raw': 'data/raw', 'processed': 'data/processed', 'models': 'models',
        'metrics': 'results/metrics', 'figures': 'results/figures',
        'sql_results': 'results/sql', 'reports': 'reports',
        'presentation': 'presentation'}.items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

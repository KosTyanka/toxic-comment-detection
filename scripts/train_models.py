"""Compare baseline, BoW/NB, TF-IDF/LR and balanced TF-IDF/LR."""
import argparse
import logging
from pathlib import Path
from src.config import ROOT
from src.train import train_models
from src.utils import configure_logging
from src.visualization import generate_figures


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--max-features', type=int, default=50000)
    parser.add_argument('--min-df', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-size', type=float, default=0.2)
    args = parser.parse_args()
    try:
        result = train_models(args.root.resolve(), dict(max_features=args.max_features, min_df=args.min_df,
                              seed=args.seed, validation_size=args.validation_size))
        generate_figures(args.root)
        logging.info('Selected on validation: %s', result['selected_model'])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logging.error('%s', exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

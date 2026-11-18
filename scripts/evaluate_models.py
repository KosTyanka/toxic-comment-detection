"""Evaluate selected model once on eligible official test rows and export SQL."""
import argparse
import logging
from pathlib import Path
from src.config import ROOT
from src.database import db_path, export_queries
from src.train import evaluate_official
from src.utils import configure_logging


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = evaluate_official(args.root)
        logging.info('Official test: %s', result)
        if db_path(args.root).exists():
            export_queries(args.root)
    except (FileNotFoundError, ValueError) as exc:
        logging.error('%s', exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Validate raw CSVs and create compact Parquet plus an observed-data manifest."""
import argparse
import logging
from pathlib import Path
from src.config import ROOT
from src.data_loader import DataValidationError, prepare_data
from src.utils import configure_logging


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--chunksize', type=int, default=10000)
    args = parser.parse_args()
    try:
        prepare_data(args.root.resolve(), args.chunksize)
    except (FileNotFoundError, DataValidationError, ValueError) as exc:
        logging.error('%s', exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

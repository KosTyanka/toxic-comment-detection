"""Build SQLite and execute/export all eight versioned queries."""
import argparse
import logging
from pathlib import Path
from src.config import ROOT
from src.database import build_database, db_path, export_queries, initialize
from src.utils import configure_logging


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--empty', action='store_true', help='Initialize schema only, with no invented comments')
    args = parser.parse_args()
    try:
        if args.empty:
            initialize(db_path(args.root))
        else:
            build_database(args.root)
        logging.info('SQL result row counts: %s', export_queries(args.root))
    except (FileNotFoundError, ValueError) as exc:
        logging.error('%s', exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

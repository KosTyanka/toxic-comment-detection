"""Regenerate charts, SQL exports, report PDF/Markdown and the 12-slide PowerPoint."""
import argparse
import logging
from pathlib import Path
from src.config import ROOT
from src.deliverables import generate_deliverables
from src.utils import configure_logging


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    logging.info('Deliverables: %s', generate_deliverables(args.root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

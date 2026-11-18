"""Run the independent fit/calibration/assessment contextual study."""
import argparse
from pathlib import Path
from src.config import ROOT
from src.contextual import run_context_experiment
from src.utils import configure_logging


def main() -> int:
    configure_logging()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    run_context_experiment(args.root.resolve())
    return 0


if __name__=='__main__':
    raise SystemExit(main())

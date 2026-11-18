"""Generate a versioned contextual report, figures and presentation supplement."""
import argparse
from pathlib import Path
from src.config import ROOT
from src.contextual_deliverables import generate_contextual_deliverables


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();print(generate_contextual_deliverables(args.root));return 0


if __name__=='__main__':raise SystemExit(main())

"""Official Kaggle downloader; never reads or embeds credential contents."""
import argparse
import logging
import subprocess
import sys
import zipfile
from pathlib import Path
from src.config import ROOT, directories
from src.utils import configure_logging


def extract_archives(destination: Path) -> None:
    """Extract nested CSV archives with traversal and symlink protection."""
    for _ in range(4):
        archives = sorted(destination.rglob('*.zip'))
        if not archives:
            break
        for archive in archives:
            with zipfile.ZipFile(archive) as container:
                for member in container.infolist():
                    target = (archive.parent / member.filename).resolve()
                    if not target.is_relative_to(destination.resolve()) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError(f'Unsafe archive member: {member.filename}')
                container.extractall(archive.parent)
            archive.unlink()


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    raw = directories(args.root)['raw']
    try:
        subprocess.run([sys.executable, '-m', 'kaggle', 'competitions', 'download', '-c',
                        'jigsaw-toxic-comment-classification-challenge', '-p', str(raw)], check=True)
        extract_archives(raw)
        missing = {'train.csv', 'test.csv', 'test_labels.csv'} - {p.name for p in raw.glob('*.csv')}
        if missing:
            raise FileNotFoundError(f'Missing extracted files: {sorted(missing)}')
    except (subprocess.CalledProcessError, OSError, ValueError, zipfile.BadZipFile) as exc:
        logging.error('Download unavailable: %s. Configure your Kaggle authentication and accept '
                      'competition access rules, or place CSVs manually in %s.', exc, raw)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

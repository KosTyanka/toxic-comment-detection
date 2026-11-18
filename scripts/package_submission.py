"""Create a source submission or a private ready-to-run runtime archive."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from src.config import ROOT
from src.utils import read_json, sha256_file


def create_package(root: Path, output: Path, include_runtime: bool = False) -> dict:
    """Stream files into ZIP; raw CSVs, credentials and caches are never bundled."""
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Choose an archive destination outside the project directory')
    experiment = read_json(root / 'results/metrics/experiment.json')
    verified = read_json(root / 'results/artifact_verification.json')
    if not experiment or not verified or verified.get('status') != 'passed' or verified['experiment_id'] != experiment['experiment_id']:
        raise ValueError('A completed, matching real experiment and artifact verification are required')
    blocked_parts = {'__pycache__', '.pytest_cache', '.venv', 'venv', '.git', 'work', 'archive'}
    blocked_names = {'kaggle.json', 'secrets.toml', '.env', 'file_manifest.json'}
    files = []
    contextual = read_json(root / 'results/contextual/latest.json', {})
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in blocked_parts for part in relative.parts) or path.name in blocked_names:
            continue
        if path.suffix in {'.pyc', '.zip', '.tmp'} or path.name.startswith('.env.'):
            continue
        if relative.parts[:2] == ('data', 'raw') and path.name != 'README.md':
            continue
        if not include_runtime and relative.parts[0] in {'data', 'models'} and path.name != 'README.md':
            continue
        if not include_runtime and (path.suffix in {'.parquet', '.npz', '.joblib', '.db'} or path.name == 'false_positive_examples.csv'):
            continue
        if relative.parts[0] == 'models' and include_runtime and len(relative.parts) > 2:
            if relative.parts[1] == 'contextual':
                if relative.parts[2] != contextual.get('run_id'):
                    continue
            elif relative.parts[1] != experiment['experiment_id']:
                continue
        files.append(path)
    inventory = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                  'sha256': sha256_file(p)} for p in files]
    manifest = dict(mode='private-runtime' if include_runtime else 'source-only',
                    experiment_id=experiment['experiment_id'], selected_model=experiment['selected_model'],
                    note='Raw CSVs and credentials excluded. Runtime mode contains processed Kaggle text and must remain private.',
                    files=inventory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, root.name + '/' + path.relative_to(root).as_posix())
        archive.writestr(root.name + '/submission_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC verification failed')
        names = archive.namelist()
        if any('/data/raw/' in name and not name.endswith('/README.md') for name in names):
            raise ValueError('Unexpected raw input in package')
    return dict(path=str(output), mode=manifest['mode'], files=len(files) + 1,
                bytes=output.stat().st_size, sha256=sha256_file(output))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-runtime', action='store_true', help='Include private SQLite/Parquet/models; never publish this ZIP')
    args = parser.parse_args()
    print(json.dumps(create_package(args.root, args.output, args.include_runtime), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

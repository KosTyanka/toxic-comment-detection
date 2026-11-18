"""Audit every supplied Jigsaw CSV, including the optional submission template."""
import argparse
from pathlib import Path
import pandas as pd
from src.config import ROOT, LABELS
from src.data_loader import DataValidationError, csv_chunks, validate_ids, validate_labels
from src.utils import save_json, sha256_file


def validate_dataset_files(root: Path) -> dict:
    """Record exact counts, hashes and label values without altering raw files."""
    audit, id_sets, blanks = {}, {}, []
    for name in ['train.csv', 'test.csv', 'test_labels.csv', 'sample_submission.csv']:
        path = root / 'data/raw' / name
        if not path.exists():
            if name != 'sample_submission.csv':
                raise FileNotFoundError(f'Missing official dataset input: {path}')
            audit[name] = {'exists': False}
            continue
        text_file = name in ['train.csv', 'test.csv']
        required = ['id'] + (['comment_text'] if text_file else []) + (LABELS if name != 'test.csv' else [])
        seen = set()
        info = dict(exists=True, bytes=path.stat().st_size, sha256=sha256_file(path),
                    rows=0, duplicate_ids=0, blank_ids=0, blank_comments=0,
                    malformed_width_rows=0, label_values={})
        for chunk in csv_chunks(path, required):
            info.setdefault('columns', list(chunk.columns))
            validate_ids(chunk, seen, name)
            info['rows'] += len(chunk)
            if name in ['train.csv', 'test_labels.csv']:
                validate_labels(chunk, allow_unknown=name == 'test_labels.csv')
            if text_file:
                empty = chunk.loc[chunk.comment_text.str.strip().eq('')]
                info['blank_comments'] += len(empty)
                blanks.extend(dict(id=str(r.id), source=name, characters=len(r.comment_text),
                                   whitespace_repr=repr(r.comment_text)) for r in empty.itertuples())
            for label in LABELS:
                if label not in chunk:
                    continue
                values = pd.to_numeric(chunk[label], errors='raise')
                if name == 'sample_submission.csv' and not values.between(0, 1).all():
                    raise DataValidationError('Submission-template scores must be finite and within [0,1]')
                counts = info['label_values'].setdefault(label, {})
                for value, count in values.value_counts(dropna=False).items():
                    counts[str(value)] = counts.get(str(value), 0) + int(count)
        audit[name], id_sets[name] = info, seen
    if id_sets['train.csv'] & id_sets['test.csv']:
        raise DataValidationError('Train and test IDs overlap')
    if id_sets['test.csv'] != id_sets['test_labels.csv']:
        raise DataValidationError('test.csv and test_labels.csv must have the same ID set for this final audit')
    if 'sample_submission.csv' in id_sets and id_sets['test.csv'] != id_sets['sample_submission.csv']:
        raise DataValidationError('Submission-template IDs differ from the test IDs')
    official = pd.read_csv(root / 'data/raw/test_labels.csv', dtype={'id': 'string'}).set_index('id')
    for entry in blanks:
        if entry['source'] == 'test.csv':
            entry['official_labels'] = {label: int(official.loc[entry['id'], label]) for label in LABELS}
        if entry['source'] == 'train.csv' or any(v != -1 for v in entry.get('official_labels', {}).values()):
            raise DataValidationError('Blank text occurs in a labeled row')
    audit.update(blank_comment_details=blanks, train_test_id_overlap=0,
                 test_and_labels_identical_ids=True,
                 test_and_submission_identical_ids=id_sets.get('sample_submission.csv') == id_sets['test.csv'])
    save_json(root / 'results/data_validation_audit.json', audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        audit = validate_dataset_files(args.root)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Validation failed: {exc}\n')
    for name, values in audit.items():
        if isinstance(values, dict) and 'rows' in values:
            print(f'{name}: {values["rows"]:,} rows; blank text={values["blank_comments"]}; valid schema/IDs/values')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

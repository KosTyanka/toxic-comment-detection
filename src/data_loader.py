"""Chunked validation and atomic Parquet preparation for official Jigsaw CSVs."""
import csv
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.config import LABELS, directories
from src.eda import Summary
from src.preprocessing import add_features, derive_targets
from src.utils import save_json, sha256_file, timestamp


class DataValidationError(ValueError):
    """An input file violates the documented schema or split protocol."""


def csv_chunks(path: Path, required: list[str], chunksize: int = 10000):
    """Validate CSV header and every row, including over-wide CSV records."""
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Place official CSVs in data/raw or run download_data.')
    try:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, [])
            if len(header) != len(set(header)) or not set(required).issubset(header):
                raise DataValidationError(f'{path.name}: require unique columns {required}')
            for row_number, row in enumerate(reader, 2):
                if len(row) != len(header):
                    raise DataValidationError(f'{path.name}: malformed record {row_number}')
        for chunk in pd.read_csv(path, chunksize=chunksize, dtype={'id': 'string', 'comment_text': 'string'},
                                 keep_default_na=False, encoding='utf-8-sig', on_bad_lines='error'):
            yield chunk
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError, csv.Error) as exc:
        raise DataValidationError(f'{path.name}: invalid UTF-8 CSV: {exc}') from exc


def validate_ids(chunk: pd.DataFrame, seen: set[str], source: str) -> None:
    ids = chunk.id
    if ids.isna().any() or ids.str.strip().eq('').any():
        raise DataValidationError(f'{source}: empty ID')
    if ids.duplicated().any() or set(ids) & seen:
        raise DataValidationError(f'{source}: duplicate or cross-split ID')
    seen.update(ids.tolist())


def validate_labels(chunk: pd.DataFrame, allow_unknown: bool = False) -> pd.DataFrame:
    result = chunk.copy()
    allowed = [-1, 0, 1] if allow_unknown else [0, 1]
    for label in LABELS:
        try:
            values = pd.to_numeric(result[label], errors='raise')
        except (ValueError, TypeError) as exc:
            raise DataValidationError(f'Invalid numeric label {label}') from exc
        if not values.isin(allowed).all():
            raise DataValidationError(f'{label}: expected {allowed}, with no nulls')
        result[label] = values.astype('int8')
    return result


def prepare_data(root: Path, chunksize: int = 10000) -> dict:
    """Publish processed files only after every supplied input has validated."""
    paths = directories(root)
    raw, processed = paths['raw'], paths['processed']
    if not (raw / 'train.csv').exists():
        raise FileNotFoundError('Missing data/raw/train.csv. See GET_DATASET.md; no synthetic research data is substituted.')
    test_path, label_path = raw / 'test.csv', raw / 'test_labels.csv'
    if label_path.exists() and not test_path.exists():
        raise DataValidationError('test_labels.csv requires test.csv')
    test_labels = None
    label_ids: set[str] = set()
    if label_path.exists():
        parts = []
        for chunk in csv_chunks(label_path, ['id'] + LABELS, chunksize):
            validate_ids(chunk, label_ids, label_path.name)
            parts.append(validate_labels(chunk, True)[['id'] + LABELS])
        if not parts:
            raise DataValidationError('test_labels.csv has no rows')
        test_labels = pd.concat(parts, ignore_index=True).set_index('id')
    seen_ids: set[str] = set()
    test_ids: set[str] = set()
    summaries = {}
    with TemporaryDirectory(dir=processed, prefix='prepare_') as temporary:
        staging = Path(temporary)
        for split in ['train', 'test']:
            source = raw / f'{split}.csv'
            if split == 'test' and not source.exists():
                continue
            summary = Summary()
            writer = None
            try:
                required = ['id', 'comment_text'] + (LABELS if split == 'train' else [])
                for chunk in csv_chunks(source, required, chunksize):
                    validate_ids(chunk, seen_ids, source.name)
                    if chunk.comment_text.isna().any():
                        raise DataValidationError(f'{source.name}: missing comment_text; repair input explicitly')
                    if split == 'train':
                        chunk = validate_labels(chunk)
                    else:
                        test_ids.update(chunk.id.tolist())
                        chunk = chunk[['id', 'comment_text']].copy()
                        if test_labels is not None:
                            joined = test_labels.reindex(chunk.id)
                            joined.index = chunk.index
                            for label in LABELS:
                                chunk[label] = joined[label].fillna(-1).astype('int8')
                        else:
                            for label in LABELS:
                                chunk[label] = -1
                    blank = chunk.comment_text.str.strip().eq('')
                    fully_unscored = chunk[LABELS].eq(-1).all(axis=1)
                    invalid_blank = blank & ~fully_unscored if split == 'test' else blank
                    if invalid_blank.any():
                        raise DataValidationError(f'{source.name}: blank comment_text in labeled data; repair input explicitly')
                    chunk = add_features(derive_targets(chunk))
                    chunk['dataset_split'] = split
                    chunk = chunk[['id', 'comment_text'] + LABELS + ['any_toxic', 'character_count',
                                   'word_count', 'text_group', 'dataset_split']]
                    summary.update(chunk)
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(staging / f'{split}.parquet', table.schema, compression='zstd')
                    writer.write_table(table)
            finally:
                if writer is not None:
                    writer.close()
            if summary.rows == 0:
                raise DataValidationError(f'{source.name}: no data rows')
            summaries[split] = summary.result()
        if label_ids - test_ids:
            raise DataValidationError('test_labels.csv contains IDs absent from test.csv')
        train_groups = set()
        for chunk in parquet_chunks(staging / 'train.parquet', columns=['text_group']):
            train_groups.update(chunk.text_group)
        overlap = 0
        if (staging / 'test.parquet').exists():
            for chunk in parquet_chunks(staging / 'test.parquet', columns=['text_group']):
                overlap += int(chunk.text_group.isin(train_groups).sum())
        sources = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                   for p in [raw / 'train.csv', test_path, label_path] if p.exists()}
        manifest = dict(created_at=timestamp(), source='User-supplied CSVs in the official Jigsaw schema; authenticity not independently verified',
                        splits=summaries, files=sources, train_test_duplicate_text_rows=overlap,
                        test_rows_missing_label_id=len(test_ids - label_ids),
                        quality_policy='Reject blank labeled comments, null/invalid labels, malformed rows and duplicate IDs. '
                        'Preserve whitespace-only fully unscored test rows exactly as supplied; count them separately. '
                        'Keep duplicate texts together in train/validation. Exclude any -1 row from official evaluation.')
        for split in ['train', 'test']:
            target = processed / f'{split}.parquet'
            candidate = staging / target.name
            if candidate.exists():
                candidate.replace(target)
            elif target.exists():
                target.unlink()
        save_json(processed / 'manifest.json', manifest)
    logging.info('Prepared %s', {key: val['rows'] for key, val in summaries.items()})
    return manifest


def parquet_chunks(path: Path, batch_size: int = 10000, columns=None):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def preview_data(root: Path, split: str, limit: int = 200) -> pd.DataFrame:
    path = root / 'data/processed' / f'{split}.parquet'
    if not path.exists():
        return pd.DataFrame()
    first = next(parquet_chunks(path, max(limit, 1)), pd.DataFrame())
    return first.head(limit)

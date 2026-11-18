"""Normalized SQLite persistence, batched writes, and predefined read-only SQL."""
import json
import sqlite3
from pathlib import Path
import pandas as pd
from src.config import ROOT, LABELS, TARGETS, directories
from src.data_loader import parquet_chunks
from src.utils import csv_bytes, read_json, save_json


def connect(path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path)
    connection.execute('PRAGMA foreign_keys=ON')
    return connection


def db_path(root: Path) -> Path:
    return root / 'data/processed/toxicity.db'


def initialize(path: Path) -> None:
    with connect(path) as connection:
        connection.executescript((ROOT / 'sql/schema.sql').read_text(encoding='utf-8'))


def build_database(root: Path) -> Path:
    """Build in a staging file so failed imports do not destroy the previous DB."""
    paths = directories(root)
    manifest = read_json(paths['processed'] / 'manifest.json')
    if not manifest:
        raise FileNotFoundError('Run python -m scripts.prepare_data first.')
    target = db_path(root)
    staging = target.with_suffix('.building.db')
    if staging.exists():
        staging.unlink()
    initialize(staging)
    connection = connect(staging)
    try:
        with connection:
            for split in manifest['splits']:
                for chunk in parquet_chunks(paths['processed'] / f'{split}.parquet'):
                    rows = [(r.id, r.comment_text, split, int(r.character_count), int(r.word_count))
                            for r in chunk.itertuples()]
                    connection.executemany('INSERT INTO comments VALUES (?,?,?,?,?)', rows)
                    labels = []
                    for row in chunk.itertuples():
                        # A partially scored official row is ineligible as a whole.
                        if all(getattr(row, label) >= 0 for label in LABELS):
                            labels.extend((row.id, label, int(getattr(row, label))) for label in TARGETS)
                    connection.executemany('INSERT INTO labels VALUES (?,?,?)', labels)
            connection.execute('INSERT INTO metadata VALUES (?,?)', ('manifest', json.dumps(manifest)))
    finally:
        connection.close()
    staging.replace(target)
    return target


def predefined_queries() -> dict[str, str]:
    source = (ROOT / 'sql/queries.sql').read_text(encoding='utf-8')
    queries = {}
    for block in source.split('-- name: ')[1:]:
        title, statement = block.split('\n', 1)
        queries[title.strip()] = statement.strip()
    return queries


def run_query(path: Path, name: str) -> pd.DataFrame:
    """Only allow versioned statements, using a read-only connection."""
    queries = predefined_queries()
    if name not in queries:
        raise ValueError('Choose one of the eight predefined SQL analyses.')
    connection = connect(path, readonly=True)
    try:
        return pd.read_sql_query(queries[name], connection)
    finally:
        connection.close()


def export_queries(root: Path) -> dict:
    paths = directories(root)
    counts = {}
    for index, name in enumerate(predefined_queries(), 1):
        frame = run_query(db_path(root), name)
        (paths['sql_results'] / f'{index:02d}_{name}.csv').write_bytes(csv_bytes(frame))
        counts[name] = len(frame)
    save_json(paths['sql_results'] / 'execution.json', counts)
    return counts


def record_predictions(root: Path, run: dict, frame: pd.DataFrame, probabilities) -> None:
    """Persist measured run metadata and genuine validation/test predictions."""
    path = db_path(root)
    if not path.exists():
        return
    connection = connect(path)
    try:
        with connection:
            connection.execute('INSERT OR IGNORE INTO model_runs VALUES (?,?,?,?,?,?)',
                               (run['run_id'], run['model_name'], json.dumps(run['configuration']),
                                run['training_time'], run['peak_memory_mb'], run['created_at']))
            for start in range(0, len(frame), 2048):
                batch = []
                for i in range(start, min(start + 2048, len(frame))):
                    batch.extend((run['run_id'], str(frame.iloc[i].id), label,
                                  float(probabilities[i, j]), int(probabilities[i, j] >= 0.5))
                                 for j, label in enumerate(TARGETS))
                connection.executemany('INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?)', batch)
    finally:
        connection.close()

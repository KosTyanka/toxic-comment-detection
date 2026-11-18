"""Executed SQL and relational constraints using temporary artificial data."""
import sqlite3
import pandas as pd
import pytest
from src.database import connect, db_path, export_queries, predefined_queries, run_query
from src.utils import csv_bytes


def test_foreign_keys_and_unknown_labels(pipeline_root):
    connection = connect(db_path(pipeline_root))
    try:
        assert connection.execute('PRAGMA foreign_keys').fetchone()[0] == 1
        assert not connection.execute('PRAGMA foreign_key_check').fetchall()
        assert connection.execute("SELECT COUNT(*) FROM labels WHERE comment_id IN ('fixture-test-0','fixture-test-1')").fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO labels VALUES ('nonexistent','toxic',1)")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO labels VALUES ('fixture-test-0','toxic',-1)")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO labels VALUES ('fixture-train-0','toxic',0)")
    finally:
        connection.close()


@pytest.mark.parametrize('name', list(predefined_queries()))
def test_all_eight_queries(pipeline_root, name):
    result = run_query(db_path(pipeline_root), name)
    assert isinstance(result, pd.DataFrame)
    if name not in ['high_confidence_errors']:
        assert not result.empty


def test_sql_exports_and_readonly(pipeline_root):
    counts = export_queries(pipeline_root)
    assert len(counts) == 8
    files = sorted((pipeline_root / 'results/sql').glob('*.csv'))
    assert len(files) == 8
    for path in files:
        assert isinstance(pd.read_csv(path), pd.DataFrame)
    connection = connect(db_path(pipeline_root), readonly=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute('DELETE FROM comments')
    finally:
        connection.close()
    with pytest.raises(ValueError):
        run_query(db_path(pipeline_root), 'DROP TABLE comments')


def test_csv_export_roundtrip_and_formula_safety():
    original = pd.DataFrame({'comment_text': ['=1+1', 'normal,comma', 'line\nbreak', '@SUM(1)'], 'probability': [0.1, 0.2, 0.3, 0.4]})
    import io
    result = pd.read_csv(io.BytesIO(csv_bytes(original)))
    assert result.comment_text.tolist() == ["'=1+1", 'normal,comma', 'line\nbreak', "'@SUM(1)"]
    assert result.probability.tolist() == original.probability.tolist()

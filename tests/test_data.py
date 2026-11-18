"""Schema, target definition, malformed input and unknown-label regression tests."""
import pandas as pd
import pytest
from src.config import LABELS
from src.data_loader import DataValidationError, csv_chunks, prepare_data
from src.preprocessing import derive_targets


def test_any_toxic_is_separate_from_toxic():
    frame = pd.DataFrame([{**dict.fromkeys(LABELS, 0), 'insult': 1}, dict.fromkeys(LABELS, -1)])
    result = derive_targets(frame)
    assert result.toxic.tolist() == [0, -1]
    assert result.any_toxic.tolist() == [1, -1]


def test_chunked_prepare_and_id_join(raw_root):
    manifest = prepare_data(raw_root, chunksize=7)
    assert manifest['splits']['train']['rows'] == 120
    assert manifest['splits']['train']['duplicate_comments'] == 1
    assert manifest['splits']['test']['scored_rows'] == 10
    frame = pd.read_parquet(raw_root / 'data/processed/test.parquet').set_index('id')
    assert frame.loc['fixture-test-0', 'any_toxic'] == -1
    assert frame.loc['fixture-test-1', 'any_toxic'] == -1
    assert frame.loc['fixture-test-3', 'toxic'] == 1


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match='train.csv'):
        prepare_data(tmp_path)


def test_unscored_blank_test_text_is_preserved(raw_root):
    path = raw_root / 'data/raw/test.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'comment_text'] = '\u2003'
    frame.to_csv(path, index=False)
    manifest = prepare_data(raw_root)
    assert manifest['splits']['test']['blank_comments'] == 1
    prepared = pd.read_parquet(raw_root / 'data/processed/test.parquet')
    row = prepared.loc[prepared.id.eq('fixture-test-0')].iloc[0]
    assert row.comment_text == '\u2003' and row.any_toxic == -1


def test_scored_blank_test_text_is_rejected(raw_root):
    path = raw_root / 'data/raw/test.csv'
    frame = pd.read_csv(path)
    frame.loc[3, 'comment_text'] = '   '
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match='blank comment_text'):
        prepare_data(raw_root)


@pytest.mark.parametrize('contents', [
    'wrong,columns\na,b\n',
    'id,comment_text,id\na,text,a\n',
    'id,comment_text\na,text,extra\n',
    'id,comment_text\na\n',
    'id,comment_text\na,"unfinished\n'])
def test_malformed_csv(tmp_path, contents):
    source = tmp_path / 'bad.csv'
    source.write_text(contents, encoding='utf-8')
    with pytest.raises(DataValidationError):
        list(csv_chunks(source, ['id', 'comment_text']))


@pytest.mark.parametrize('column,value', [('id', ''), ('comment_text', ''), ('toxic', 2), ('threat', ''), ('toxic', -1)])
def test_invalid_values_rejected(raw_root, column, value):
    path = raw_root / 'data/raw/train.csv'
    frame = pd.read_csv(path)
    frame[column] = frame[column].astype(object)
    frame.loc[3, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        prepare_data(raw_root, 3)
    assert not (raw_root / 'data/processed/manifest.json').exists()


def test_duplicate_id_across_chunks(raw_root):
    path = raw_root / 'data/raw/train.csv'
    frame = pd.read_csv(path)
    frame.loc[119, 'id'] = frame.loc[0, 'id']
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match='duplicate'):
        prepare_data(raw_root, 10)


def test_cross_split_id_rejected(raw_root):
    path = raw_root / 'data/raw/test.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'id'] = 'fixture-train-0'
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        prepare_data(raw_root)


def test_unmatched_label_id_rejected(raw_root):
    path = raw_root / 'data/raw/test_labels.csv'
    frame = pd.read_csv(path)
    frame.loc[0, 'id'] = 'nonexistent'
    frame.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match='absent'):
        prepare_data(raw_root)

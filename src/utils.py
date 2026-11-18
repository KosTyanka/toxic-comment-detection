"""Safe persistent metadata and deterministic input fingerprints."""
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, value: Any) -> None:
    """Write strict JSON atomically; undefined metrics must use None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def csv_bytes(frame: Any) -> bytes:
    """Export spreadsheet-safe text cells, retaining numeric values."""
    copy = frame.copy()
    for col in copy.select_dtypes(include=['object', 'str', 'string']).columns:
        copy[col] = copy[col].map(lambda x: "'" + x if isinstance(x, str)
                                and x.lstrip().startswith(('=', '+', '-', '@')) else x)
    return copy.to_csv(index=False).encode('utf-8-sig')

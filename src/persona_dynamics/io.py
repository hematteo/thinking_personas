"""Small atomic artifacts, hashes, and append-only resumable transcript caches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for record in records:
            f.write(json.dumps(record, default=json_default, allow_nan=False) + "\n")
    os.replace(tmp, path)


def stable_seed(seed: int, *parts) -> int:
    """Stable across Python processes, batch order, and missing cache entries."""
    return int(digest([seed, *parts])[:8], 16) % (2**31 - 1)

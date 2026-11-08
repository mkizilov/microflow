"""Hashing helpers shared by provenance and batch state."""

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK_BYTES = 1 << 20


def digest(payload: Any) -> str:
    """SHA-256 of canonical JSON (sorted keys, no spaces), so equal content gives equal digests."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

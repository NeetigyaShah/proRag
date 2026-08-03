"""Blob persistence + content-hash dedupe. Infrastructure layer: writes raw
uploaded bytes to disk under settings.blob_dir, keyed by sha256, and removes
them when ingestion fails. Runs inside ingest_bytes() (ingest/core.py) on
every upload — write_blob()/blob_path_for() during ingest, delete_blob() when
a later step fails so no orphan blob survives on disk."""

import hashlib
from pathlib import Path

from prorag.settings import settings


def sha256_hex(data: bytes) -> str:
    """When called: ingest_bytes() first thing, to compute the dedupe key.
    What: sha256 digest of the file bytes. Returns: the hex digest string."""
    return hashlib.sha256(data).hexdigest()


def blob_path_for(sha256: str, filename: str) -> Path:
    """When called: ingest_bytes(), before writing the blob. What: on-disk
    path under settings.blob_dir, content-hash keyed with the original file's
    extension (or .bin). Returns: the Path to write to."""
    ext = Path(filename).suffix or ".bin"
    return Path(settings.blob_dir) / f"{sha256}{ext}"


def write_blob(data: bytes, path: Path) -> None:
    """When called: ingest_bytes(), after dedupe, to persist the upload. What:
    writes the bytes, creating parent directories; skips the write when the
    path already exists (a concurrent ingest of the same bytes won the race).
    Returns: None."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def delete_blob(path: Path) -> None:
    """Remove a blob whose ingestion failed, so a rejected upload doesn't leave
    an orphan on disk. Missing file is not an error."""
    path.unlink(missing_ok=True)

"""Blob write + sha256 dedupe."""

import hashlib
from pathlib import Path

from prorag.settings import settings


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob_path_for(sha256: str, filename: str) -> Path:
    ext = Path(filename).suffix or ".bin"
    return Path(settings.blob_dir) / f"{sha256}{ext}"


def write_blob(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def delete_blob(path: Path) -> None:
    """Remove a blob whose ingestion failed, so a rejected upload doesn't leave
    an orphan on disk. Missing file is not an error."""
    path.unlink(missing_ok=True)

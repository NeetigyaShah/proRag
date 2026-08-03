"""POST /ingest — synchronous in Phase 1 (no worker/job queue yet).

Phase 3: PDF/DOCX/PPTX route through Docling (prose vs Table elements, §3.1);
CSV/XLSX/TSV route through pandas straight to table artifacts (§3.3); .txt/.md
keep the plain fixed-chunk path. Docling absent or failing falls back to the
Phase 1 PyMuPDF text path automatically.

The actual parse->chunk->embed->store pipeline lives in ingest/core.py's
ingest_bytes() (#22) — this handler keeps only the HTTP trust-boundary
concerns on the untrusted multipart upload: streaming read with a size
ceiling, the suffix allowlist, and collection-name validation. #22's S3
connector calls ingest_bytes() directly with bytes it already has, skipping
this layer (a source object's key isn't the same kind of untrusted input as
an anonymous multipart filename).

# ponytail: ingestion runs inline in the request handler. Structured so the
# parse->chunk->embed->store steps can be lifted into a `jobs` table + worker
# loop later (§3, Phase 1 note) without touching the API contract.
"""

import re

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.db import get_session
from prorag.ingest.core import ingest_bytes
from prorag.schemas import IngestResponse
from prorag.settings import settings

router = APIRouter()

ALLOWED_SUFFIXES = (
    ".pdf", ".txt", ".md", ".docx", ".pptx", ".csv", ".xlsx", ".tsv",
    ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp",
)
_COLLECTION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(
    file: UploadFile = File(...),
    collection: str = Form("default"),
    session: AsyncSession = Depends(get_session),
):
    """When called: every POST /ingest request. What: the HTTP trust boundary —
    suffix allowlist, collection-name validation, and a streaming read capped
    at the upload size ceiling — then hands the bytes to ingest_bytes() for
    the parse→chunk→embed→store pipeline. Returns: IngestResponse (202) or an
    HTTPException for unsupported, oversized, or empty uploads."""
    filename = (file.filename or "upload")[:255]
    if not filename.lower().endswith(ALLOWED_SUFFIXES):
        raise HTTPException(400, f"unsupported file type; allowed: {ALLOWED_SUFFIXES}")
    if not _COLLECTION_RE.match(collection):
        raise HTTPException(400, "invalid collection name (use letters, digits, _ or -, max 64 chars)")

    # Read in chunks so an oversized upload is rejected as soon as it crosses
    # the limit, instead of after it's already resident in memory.
    buf = bytearray()
    while chunk := await file.read(1024 * 1024):
        buf.extend(chunk)
        if len(buf) > settings.max_upload_bytes:
            raise HTTPException(413, f"file exceeds {settings.max_upload_bytes // (1024 * 1024)} MB limit")
    data = bytes(buf)
    if not data:
        raise HTTPException(400, "empty file")

    return await ingest_bytes(session, data, filename, collection)

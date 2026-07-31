"""POST /ingest — synchronous in Phase 1 (no worker/job queue yet).

Phase 3: PDF/DOCX/PPTX route through Docling (prose vs Table elements, §3.1);
CSV/XLSX/TSV route through pandas straight to table artifacts (§3.3); .txt/.md
keep the plain fixed-chunk path. Docling absent or failing falls back to the
Phase 1 PyMuPDF text path automatically.

# ponytail: ingestion runs inline in the request handler. Structured so the
# parse->chunk->embed->store steps can be lifted into a `jobs` table + worker
# loop later (§3, Phase 1 note) without touching the API contract.
"""

import asyncio
import logging
import re
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.db import get_session
from prorag.ingest.chunk import chunk_elements, chunk_pages
from prorag.ingest.parse import (
    DOCLING_MIMES,
    STRUCTURED_MIMES,
    ParsedTable,
    guess_mime,
    parse_structured,
    parse_to_pages,
    sniff_mime,
    try_docling_parse,
)
from prorag.ingest.store import blob_path_for, delete_blob, sha256_hex, write_blob
from prorag.ingest.tables import build_table_artifacts
from prorag.llm import embed_texts_batched
from prorag.models import Chunk, Document, Table, TableRow
from prorag.retrieve.crop import normalize_title
from prorag.schemas import IngestResponse
from prorag.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_SUFFIXES = (".pdf", ".txt", ".md", ".docx", ".pptx", ".csv", ".xlsx", ".tsv")
_COLLECTION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# Deterministic failures (bad payload, unknown model, auth) won't fix themselves;
# retrying them just burns time and quota. Retry transient ones only.
_TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "rate limit",
    "ratelimit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "connection",
    "temporarily",
    "overloaded",
    "unavailable",
)


def _is_transient(exc: Exception) -> bool:
    return any(m in f"{type(exc).__name__} {exc}".lower() for m in _TRANSIENT_ERROR_MARKERS)


def _safe_error(exc: Exception) -> str:
    """Store a short, sanitised reason — provider errors can carry URLs, keys
    and payload fragments we don't want in the DB or the API response."""
    return f"{type(exc).__name__}: {str(exc)[:200]}"


async def _embed_with_retry(texts: list[str], session: AsyncSession, attempts: int = 3) -> list[list[float]]:
    """Retry-with-backoff around the embed step (§8 Phase 5 job retries) — the
    one step in inline ingestion that talks to a network provider and can
    transiently fail. Raises the last error after `attempts` tries so the
    caller can mark the document status='failed' instead of 500ing."""
    delay = 1.0
    for attempt in range(attempts):
        try:
            return await embed_texts_batched(texts, session=session)
        except Exception as exc:
            if attempt == attempts - 1 or not _is_transient(exc):
                raise
            logger.warning("embed attempt %d failed (%s); retrying", attempt + 1, _safe_error(exc))
            await asyncio.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")  # pragma: no cover


async def _store_table(
    session: AsyncSession, doc_id: uuid.UUID, parsed: ParsedTable
) -> tuple[Table, list[Chunk], list[str]]:
    """Insert Table + table_rows, build the three artifacts (§3.3), return the
    new (unembedded) Chunk rows alongside the texts to embed."""
    table = Table(
        doc_id=doc_id,
        caption=parsed.caption,
        columns=parsed.columns,
        row_count=len(parsed.rows),
        page_no=parsed.page_no,
        bbox=list(parsed.bbox) if parsed.bbox else None,
    )
    session.add(table)
    await session.flush()

    # parsed.rows is already list[dict] (Docling export_to_dataframe / pandas to_dict).
    # build_table_rows() (ingest/tables.py) exists for the list-of-lists shape; unused here.
    row_dicts = parsed.rows
    for i, row in enumerate(row_dicts):
        session.add(TableRow(table_id=table.id, row_no=i, data=row))

    artifacts = build_table_artifacts(parsed.caption, parsed.columns, row_dicts)
    chunks = [
        Chunk(
            doc_id=doc_id,
            kind=a.kind,
            text=a.text,
            embed_text=a.text,
            page_start=parsed.page_no,
            page_end=parsed.page_no,
            bbox=list(parsed.bbox) if parsed.bbox else None,
            table_id=table.id,
            token_count=len(a.text.split()),
        )
        for a in artifacts
    ]
    return table, chunks, [a.text for a in artifacts]


@router.post("/ingest", response_model=IngestResponse, status_code=202)
async def ingest(
    file: UploadFile = File(...),
    collection: str = Form("default"),
    session: AsyncSession = Depends(get_session),
):
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

    digest = sha256_hex(data)

    # Dedupe BEFORE writing the blob — re-uploading a known file shouldn't cost
    # disk I/O or leave a redundant copy behind.
    existing = (await session.execute(select(Document).where(Document.sha256 == digest))).scalar_one_or_none()
    if existing is not None:
        return IngestResponse(doc_id=existing.id, status=existing.status, duplicate_of=existing.id)

    mime = guess_mime(filename)
    sniffed = sniff_mime(data)
    if sniffed and sniffed != mime:
        # Extension and content disagree — refuse rather than hand a mislabelled
        # file to a parser that assumes otherwise.
        raise HTTPException(400, f"file content ({sniffed}) does not match its extension ({mime})")

    path = blob_path_for(digest, filename)
    write_blob(data, path)

    prose_chunks: list = []  # ingest.chunk.Chunk | StructuredChunk
    parsed_tables: list[ParsedTable] = []
    page_count: int | None = None
    used_structured_chunking = False
    parser_used = "pymupdf"

    try:
        if mime in STRUCTURED_MIMES:
            parser_used = "pandas"
            parsed_tables = parse_structured(data, mime)
        elif mime in DOCLING_MIMES:
            docling_doc = try_docling_parse(data, mime, filename)
            if docling_doc is not None:
                parser_used = "docling"
                prose_chunks = chunk_elements(
                    docling_doc.prose, settings.chunk_target_tokens, settings.chunk_overlap_tokens
                )
                parsed_tables = docling_doc.tables
                page_count = docling_doc.page_count
                used_structured_chunking = True
            else:
                # Visible degradation: no tables and no bbox means citations can
                # open the right page but can't highlight the passage.
                logger.info("docling unavailable/failed for %s; using pymupdf text path", filename)
                pages = parse_to_pages(data, mime)
                prose_chunks = chunk_pages(pages, settings.chunk_target_tokens, settings.chunk_overlap_tokens)
                page_count = len(pages)
        else:
            pages = parse_to_pages(data, mime)
            prose_chunks = chunk_pages(pages, settings.chunk_target_tokens, settings.chunk_overlap_tokens)
            page_count = len(pages)
    except HTTPException:
        raise
    except Exception as exc:
        # Parsing blew up before any DB row existed — don't leave the blob behind.
        logger.exception("parse failed for %s", filename)
        delete_blob(path)
        raise HTTPException(422, f"could not parse this file ({_safe_error(exc)})") from exc

    if not prose_chunks and not parsed_tables:
        delete_blob(path)
        raise HTTPException(422, "no extractable text found (scanned image PDF or empty file?)")

    if len(prose_chunks) > settings.max_chunks_per_document:
        delete_blob(path)
        raise HTTPException(413, f"document produced more than {settings.max_chunks_per_document} chunks")

    doc = Document(
        id=uuid.uuid4(),
        sha256=digest,
        filename=filename,
        mime=mime,
        blob_path=str(path),
        page_count=page_count,
        title=filename,
        title_norm=normalize_title(filename),
        collection=collection,
        # Only flipped to "ready" once chunks + embeddings are committed, so a
        # half-written document is never retrievable (retrieval joins on it).
        status="processing",
    )
    session.add(doc)
    try:
        await session.flush()
    except IntegrityError:
        # Lost a race with a concurrent upload of the same bytes — the unique
        # constraint on sha256 is the real guarantee; the earlier SELECT is just
        # a fast path. Resolve to the winner instead of failing.
        await session.rollback()
        existing = (await session.execute(select(Document).where(Document.sha256 == digest))).scalar_one_or_none()
        if existing is not None:
            return IngestResponse(doc_id=existing.id, status=existing.status, duplicate_of=existing.id)
        raise

    try:
        table_chunks: list[Chunk] = []
        table_texts: list[str] = []
        for parsed in parsed_tables:
            _table, chunks, texts = await _store_table(session, doc.id, parsed)
            table_chunks.extend(chunks)
            table_texts.extend(texts)

        prose_texts = [getattr(c, "embed_text", None) or c.text for c in prose_chunks]
        # Hard ceiling on the whole embed phase so a hung provider can't pin the
        # request open indefinitely.
        all_embeddings = await asyncio.wait_for(
            _embed_with_retry([*prose_texts, *table_texts], session),
            timeout=settings.embed_timeout_seconds,
        )

        prose_embeddings = all_embeddings[: len(prose_texts)]
        table_embeddings = all_embeddings[len(prose_texts) :]

        new_chunks = [
            Chunk(
                doc_id=doc.id,
                ord=ord_,
                kind="prose",
                text=c.text,
                embed_text=c.embed_text if used_structured_chunking else c.text,
                heading_path=getattr(c, "heading_path", None),
                page_start=c.page_start,
                page_end=c.page_end,
                bbox=getattr(c, "bbox", None),
                token_count=c.token_count,
                embedding=emb,
            )
            for ord_, (c, emb) in enumerate(zip(prose_chunks, prose_embeddings, strict=True))
        ]

        base_ord = len(prose_chunks)
        for i, (chunk, emb) in enumerate(zip(table_chunks, table_embeddings, strict=True)):
            chunk.ord = base_ord + i
            chunk.embedding = emb
            new_chunks.append(chunk)

        session.add_all(new_chunks)  # one flush instead of per-row round trips
        doc.status = "ready"
        await session.commit()
    except Exception as exc:
        # Any failure after the document row exists: roll back the partial work,
        # then record the failure so the upload is diagnosable and retryable.
        is_timeout = isinstance(exc, TimeoutError | asyncio.TimeoutError)
        logger.exception("ingest failed for %s", filename)
        await session.rollback()
        doc_row = (await session.execute(select(Document).where(Document.sha256 == digest))).scalar_one_or_none()
        if doc_row is not None:
            doc_row.status = "failed"
            doc_row.error = "embedding timed out" if is_timeout else _safe_error(exc)
            await session.commit()
            return IngestResponse(doc_id=doc_row.id, status="failed")
        delete_blob(path)
        raise HTTPException(500, "ingestion failed") from exc

    logger.info(
        "ingested %s: parser=%s pages=%s prose_chunks=%d tables=%d",
        filename,
        parser_used,
        page_count,
        len(prose_chunks),
        len(parsed_tables),
    )
    return IngestResponse(doc_id=doc.id, status="ready")

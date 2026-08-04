"""The ingest core: dedup, parse routing, table artifacts, embed-with-retry,
status transitions, blob write/delete (§3, §8 Phase 5 job retries). Factored
out of ingest/router.py (#22) so the S3 connector's sync engine can drive the
exact same parse->chunk->embed->store path from bytes it already has, without
going through an HTTP upload. `ingest_bytes()` is everything ingest/router.py's
POST /ingest handler used to do from `digest = sha256_hex(data)` onward —
upload-specific trust-boundary checks (streaming/size limit, suffix allowlist,
collection name validation) stay in the HTTP handler, since a connector's
`filename` (from a source key, not an attacker-controlled multipart field)
isn't the same kind of untrusted input.

# ponytail: ingestion runs inline, not through a `jobs` table + worker loop —
# same deviation ingest/router.py's docstring already documents.
"""

import asyncio
import logging
import uuid

from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.ingest.chunk import chunk_elements, chunk_pages
from prorag.ingest.parse import (
    DOCLING_MIMES,
    STRUCTURED_MIMES,
    ParsedTable,
    _pdf_has_text_layer,
    _pdf_page_count,
    guess_mime,
    ocr_pages,
    parse_structured,
    parse_to_pages,
    sniff_mime,
    try_docling_parse,
)
from prorag.ingest.store import blob_path_for, delete_blob, sha256_hex, write_blob
from prorag.ingest.tables import build_table_artifacts
from prorag.llm import embed_texts_batched
from prorag.models import AccessRule, Chunk, Document, Table, TableRow
from prorag.retrieve.crop import normalize_title
from prorag.schemas import IngestResponse
from prorag.settings import settings

logger = logging.getLogger(__name__)

# Deterministic failures (bad payload, unknown model, auth) won't fix themselves;
# retrying them just burns time and quota. Retry transient ones only.
_TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "rate limit",
    "ratelimit",
    "429",
    # OpenRouter free-tier embed models report overload as 422 Unprocessable —
    # retrying genuinely helps (verified: HTTP-level retries succeed).
    "422",
    "unprocessable",
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
    """When called: inside _embed_with_retry() after a failed embed call, to
    decide whether retrying can help. What: matches the exception text against
    known transient markers (timeout, rate limit, 5xx, connection, …).
    Returns: True to retry, False to give up immediately."""
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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """When called: _apply_confirmed_rules(), once per (rule, chunk) pair.
    What: cosine similarity between two embedding vectors; 0.0 when either is
    zero-length. Returns: a float in [0, 1]."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _apply_confirmed_rules(session: AsyncSession, doc: Document, chunks: list[Chunk]) -> None:
    """Auto-admission (#4, #24): check a just-ingested document against every
    confirmed rule's *stored* embedding — one cosine comparison per
    (rule, chunk) pair against embeddings the ingest pipeline already
    computed, so this costs zero extra LLM calls. Matches
    admin/router.py's _rule_candidates() floor semantics but in Python
    instead of SQL, since these embeddings aren't committed yet.

    A rule-check failure must never fail the ingest that triggered it — log
    and move on.

    # ponytail: O(confirmed_rules * chunks) in Python, fine at this scale (a
    # handful of rules, a few hundred chunks per doc). Upgrade path if either
    # grows: batch this as one SQL query per rule against the just-flushed
    # chunk rows (same GROUP BY/HAVING shape as _rule_candidates), like a
    # single-document version of confirm's candidate search.
    """
    try:
        rules = (
            (
                await session.execute(
                    select(AccessRule).where(AccessRule.state == "confirmed", AccessRule.query_embedding.is_not(None))
                )
            )
            .scalars()
            .all()
        )
        if not rules:
            return
        for rule in rules:
            matched = any(
                _cosine_similarity(c.embedding, rule.query_embedding) >= settings.rule_similarity_floor for c in chunks
            )
            if not matched:
                continue
            await session.execute(
                text(
                    """
                    INSERT INTO document_acl (doc_id, principal_type, principal_id, source)
                    SELECT :doc_id, 'group', :group_id, CAST(:src AS text)
                    WHERE NOT EXISTS (
                        SELECT 1 FROM document_acl
                        WHERE doc_id = :doc_id AND principal_type = 'group'
                          AND principal_id = :group_id AND source = CAST(:src AS text)
                    )
                    """
                ),
                {"doc_id": doc.id, "group_id": rule.group_id, "src": f"rule:{rule.id}"},
            )
    except Exception:
        logger.exception("auto-admission rule check failed for doc %s; continuing without it", doc.id)


async def ingest_bytes(
    session: AsyncSession, data: bytes, filename: str, collection: str = "default"
) -> IngestResponse:
    """Dedup, parse, chunk, embed, store — the whole document lifecycle past
    the upload trust boundary. Same function whether the bytes came from a
    multipart upload (ingest/router.py) or a connector's fetch() (#22)."""
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
        elif mime.startswith("image/") or (
            mime == "application/pdf" and not _pdf_has_text_layer(data)
        ):
            # Raster uploads and scanned PDFs: transcribe via the OpenRouter
            # vision model (no local OCR), then take the plain-text chunk
            # path. No bboxes — citations open the page without highlight.
            parser_used = "ocr"
            pages = await ocr_pages(data, mime, _pdf_page_count(data, mime))
            prose_chunks = chunk_pages(pages, settings.chunk_target_tokens, settings.chunk_overlap_tokens)
            page_count = len(pages)
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
                if mime.startswith("image/"):
                    # OCR is the only parse path for raster uploads; a failed
                    # docling parse must not fall through to the raw-text path
                    # (that would ingest binary garbage as text).
                    delete_blob(path)
                    raise HTTPException(422, "could not extract text from this image (OCR unavailable?)")
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
        await _apply_confirmed_rules(session, doc, new_chunks)
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

"""Retrieval arms: vector (Phase 1), keyword/BM25 (Phase 2), structured (Phase 3).

Each arm takes an optional `user` (#18): None keeps today's unfiltered
behaviour (auth disabled / legacy unscoped key / admin), a real User adds the
document_acl visibility predicate before LIMIT so ranking and downstream
fusion/crop only ever see a legal candidate pool.

Runs during every retrieval: operations/retrieval.gather_hits() (shared by
POST /chat, POST /chat/stream, GET /search and the eval runner) runs the
vector arm per query embedding and the keyword arm per planner query, plus
the structured arm when the planner asked for a table, then RRF-fuses the
arms' hits.
"""

import logging

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import Text, and_, cast, func, literal, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.models import Chunk, Document, TableRow, User
from prorag.retrieve.visibility import visibility_clause
from prorag.settings import settings

logger = logging.getLogger(__name__)


def _distance_expr(query_embedding: list[float]):
    """Cosine distance, cast to match whichever index 0001_initial.py built:
    above pgvector's 2000-dim hnsw cap it indexes embedding::halfvec(dim), so
    the query must cast both sides the same way or the planner Seq Scans (#16)."""
    if settings.embed_dim > 2000:
        halfvec = HALFVEC(settings.embed_dim)
        return cast(Chunk.embedding, halfvec).cosine_distance(cast(query_embedding, halfvec))
    return Chunk.embedding.cosine_distance(query_embedding)


async def vector_search(
    session: AsyncSession, query_embedding: list[float], k: int, user: User | None = None
) -> list[dict]:
    """When called: by gather_hits() (operations/retrieval.py) once per
    planner query embedding, on every answer/search request. What: k-NN
    cosine search over chunk embeddings joined to ready documents, ACL
    predicate applied before LIMIT. Returns: best-first hit dicts
    {chunk_id, doc_id, text, page, bbox, kind, title, score}, where score is
    cosine similarity (1 - distance)."""
    stmt = (
        select(
            Chunk.id,
            Chunk.doc_id,
            Chunk.text,
            Chunk.page_start,
            Chunk.bbox,
            Chunk.kind,
            Document.title,
            Document.filename,
            _distance_expr(query_embedding).label("distance"),
        )
        .join(Document, Chunk.doc_id == Document.id)
        .where(Document.status == "ready")
        .order_by("distance")
        .limit(k)
    )
    clause = visibility_clause(user)
    if clause is not None:
        stmt = stmt.where(clause)
    rows = (await session.execute(stmt)).all()
    return [
        {
            "chunk_id": r.id,
            "doc_id": r.doc_id,
            "text": r.text,
            "page": r.page_start,
            "bbox": r.bbox,
            "kind": r.kind,
            "title": r.title or r.filename,
            "score": 1 - r.distance,  # cosine similarity
        }
        for r in rows
    ]


def _visibility_sql(user: User | None) -> tuple[str, dict]:
    """Raw-SQL twin of visibility_clause(), for bm25_search's text() query.
    Returns ("", {}) when no filtering applies (None/admin), else an
    `AND EXISTS (...)` fragment with bound params only."""
    if user is None or user.is_admin:
        return "", {}
    return (
        """AND EXISTS (
            SELECT 1 FROM document_acl a
            WHERE a.doc_id = c.doc_id
              AND (a.principal_type = 'public'
                   OR (a.principal_type = 'user' AND a.principal_id = :vis_user_id)
                   OR (a.principal_type = 'group' AND a.principal_id IN (
                       SELECT group_id FROM user_groups WHERE user_id = :vis_user_id
                   )))
        )""",
        {"vis_user_id": user.id},
    )


_BM25_AVAILABLE: bool | None = None  # probed once per process


async def _bm25_available(session: AsyncSession) -> bool:
    """When called: by keyword_search() on every keyword lookup; the probe
    query itself runs only once per process (cached in _BM25_AVAILABLE).
    What: checks whether the pg_search BM25 index (ix_chunks_bm25) exists.
    Returns: True/False."""
    global _BM25_AVAILABLE
    if _BM25_AVAILABLE is None:
        row = await session.execute(text("SELECT to_regclass('ix_chunks_bm25') IS NOT NULL"))
        _BM25_AVAILABLE = bool(row.scalar_one())
    return _BM25_AVAILABLE


async def bm25_search(session: AsyncSession, query: str, k: int, user: User | None = None) -> list[dict]:
    """Real BM25 via pg_search (ParadeDB). `@@@` matches against the bm25 index
    and paradedb.score() returns the BM25 score for the row.

    This is raw SQL text, not an ORM select, so it can't reuse
    visibility_clause() directly — instead of a second bound-param dialect,
    the same EXISTS shape is inlined as a string fragment (params still
    bound, never interpolated: injection-safe and index-friendly)."""
    acl_sql, acl_params = _visibility_sql(user)
    stmt = text(
        f"""
        SELECT c.id, c.doc_id, c.text, c.page_start, c.bbox, c.kind,
               d.title, d.filename, paradedb.score(c.id) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        WHERE c.text @@@ :q AND d.status = 'ready' {acl_sql}
        ORDER BY score DESC
        LIMIT :k
        """
    )
    rows = (await session.execute(stmt, {"q": query, "k": k, **acl_params})).all()
    return [
        {
            "chunk_id": r.id,
            "doc_id": r.doc_id,
            "text": r.text,
            "page": r.page_start,
            "bbox": r.bbox,
            "kind": r.kind,
            "title": r.title or r.filename,
            "score": float(r.score),
        }
        for r in rows
    ]


async def keyword_search(session: AsyncSession, query: str, k: int, user: User | None = None) -> list[dict]:
    """Keyword arm: BM25 when pg_search is installed, tsvector ranking otherwise.
    Both feed the same RRF fusion, which ranks by position rather than raw score."""
    if await _bm25_available(session):
        try:
            return await bm25_search(session, query, k, user=user)
        except Exception:
            logger.warning("bm25 query failed; falling back to tsvector", exc_info=True)
    return await fts_search(session, query, k, user=user)


async def fts_search(session: AsyncSession, query: str, k: int, user: User | None = None) -> list[dict]:
    """websearch_to_tsquery + ts_rank_cd — fallback keyword arm (§4.2)."""
    tsq = func.websearch_to_tsquery("english", literal(query))
    rank = func.ts_rank_cd(Chunk.tsv, tsq).label("rank")
    stmt = (
        select(
            Chunk.id,
            Chunk.doc_id,
            Chunk.text,
            Chunk.page_start,
            Chunk.bbox,
            Chunk.kind,
            Document.title,
            Document.filename,
            rank,
        )
        .join(Document, Chunk.doc_id == Document.id)
        .where(Document.status == "ready")
        .where(Chunk.tsv.op("@@")(tsq))
        .order_by(rank.desc())
        .limit(k)
    )
    clause = visibility_clause(user)
    if clause is not None:
        stmt = stmt.where(clause)
    rows = (await session.execute(stmt)).all()
    return [
        {
            "chunk_id": r.id,
            "doc_id": r.doc_id,
            "text": r.text,
            "page": r.page_start,
            "bbox": r.bbox,
            "kind": r.kind,
            "title": r.title or r.filename,
            "score": float(r.rank),
        }
        for r in rows
    ]


def build_structured_query(
    query: str, k: int, field_filters: dict[str, str] | None = None, user: User | None = None
):
    """Build the JSONB row-search statement (§4.2): matches `query` against
    table_rows.data values (cast to text — no tsvector on JSONB rows), plus
    optional allow-listed field=value equality filters. Returns the owning
    row/window/summary chunk so results fuse with the other arms on chunk_id.

    Split out from structured_search() so query construction is testable
    (via str(stmt) / compile()) without a DB connection.
    """
    conditions = [cast(TableRow.data, Text).ilike(f"%{query}%")]
    if field_filters:
        for field, value in field_filters.items():
            conditions.append(TableRow.data[field].astext == value)

    stmt = (
        select(
            Chunk.id.label("chunk_id"),
            Chunk.doc_id,
            Chunk.text,
            Chunk.page_start,
            Document.title,
            Document.filename,
        )
        .join(Document, Chunk.doc_id == Document.id)
        .where(Document.status == "ready")
        .join(TableRow, TableRow.table_id == Chunk.table_id)
        .where(Chunk.kind.in_(["row", "table_window", "table_summary"]))
        .where(and_(*conditions))
        .distinct()
        .limit(k)
    )
    clause = visibility_clause(user)
    if clause is not None:
        stmt = stmt.where(clause)
    return stmt


async def structured_search(
    session: AsyncSession,
    query: str,
    k: int,
    field_filters: dict[str, str] | None = None,
    user: User | None = None,
) -> list[dict]:
    """Structured arm (§4.2): only meaningful when the planner set
    `wants_table` or emitted a field filter mapping to a known column."""
    stmt = build_structured_query(query, k, field_filters, user=user)
    rows = (await session.execute(stmt)).all()
    return [
        {
            "chunk_id": r.chunk_id,
            "doc_id": r.doc_id,
            "text": r.text,
            "page": r.page_start,
            "title": r.title or r.filename,
            "score": 1.0,
        }
        for r in rows
    ]

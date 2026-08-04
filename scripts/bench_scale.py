"""Throwaway scale-measurement script for GitHub issue #11 — NOT part of the
app, NOT meant to be kept clean forever. Answers "where does the pipeline
break" with numbers; fixes nothing.

Generates synthetic chunks/documents (filename prefix 'bench_', collection
'bench_scale') at increasing tiers directly via asyncpg + COPY (bulk-insert,
matching how the HNSW index would see real growth), measuring:

  1. insert throughput + HNSW/BM25/GIN index size as chunks grows
  2. vector top-40 query latency (the exact query shape prorag.retrieve.arms
     .vector_search emits)
  3. BM25 (pg_search @@@) query latency
  4. ingestion throughput through the real inline /ingest path, embeddings
     stubbed
  5. rerank throughput (CrossEncoder), if the model is reachable
  6. pool-ceiling arithmetic from the measured per-stage numbers (not
     brute-forced)

Run: .venv/Scripts/python.exe scripts/bench_scale.py
Cleans up every 'bench_%' row it created; verifies baseline counts after.
"""

import asyncio
import io
import random
import statistics
import time
import uuid

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from prorag.settings import settings

random.seed(42)
np.random.seed(42)

DSN = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
DIM = settings.embed_dim
COLLECTION = "bench_scale"
FILENAME_PREFIX = "bench_"
CHUNKS_PER_DOC = 200
COPY_BATCH = 2000
N_QUERIES = 20
TOP_K = 40
TIER_TIME_BUDGET_S = 20 * 60  # per-tier hard stop per issue #11 instructions

# ponytail: a fixed pool of pre-built paragraph texts, reused (with a unique
# per-row suffix) across rows instead of generating unique prose per row —
# this bench is about DB-side insert/query/index behaviour, not text realism,
# and generating 100k+ unique ~200-word paragraphs in Python would itself
# dominate wall time. Vocabulary is domain-flavoured so BM25/tsvector ranking
# still has real lexical variety to chew on.
_VOCAB = (
    "vessel ship crew bridge deck engine cargo hold ballast anchor mooring lifeboat "
    "extinguisher alarm emergency checklist audit compliance standard code annex chapter "
    "section clause fire drill muster station requirement frequency training certificate "
    "inspection maintenance procedure regulation SOLAS ISM safety officer master watchkeeping "
    "navigation weather route port harbour pilot tug berth cargo manifest survey class "
    "notation flag state port state control deficiency corrective action non-conformity "
    "risk assessment permit work hot cold enclosed space entry gas free ventilation "
    "life jacket immersion suit rescue boat davit winch hoist load test rigging "
    "communication radio distress signal SART EPIRB GMDSS log book record retention "
    "review approval revision amendment manual policy objective target indicator"
).split()


def _random_paragraph(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choices(_VOCAB, k=n_words))


def _build_text_pool(size: int, rng: random.Random) -> list[str]:
    return [_random_paragraph(rng, rng.randint(150, 300)) for _ in range(size)]


def _unit_vectors(n: int, dim: int) -> np.ndarray:
    v = np.random.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


def _vector_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v.tolist()) + "]"


async def get_baseline_counts(conn: asyncpg.Connection) -> tuple[int, int]:
    docs = await conn.fetchval("SELECT count(*) FROM documents")
    chunks = await conn.fetchval("SELECT count(*) FROM chunks")
    return docs, chunks


async def cleanup_bench_rows(conn: asyncpg.Connection) -> None:
    await conn.execute("DELETE FROM documents WHERE filename LIKE $1", f"{FILENAME_PREFIX}%")
    await conn.execute("VACUUM ANALYZE chunks")
    await conn.execute("VACUUM ANALYZE documents")


async def index_sizes(conn: asyncpg.Connection) -> dict[str, int]:
    rows = await conn.fetch(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'"
    )
    out = {}
    for r in rows:
        out[r["indexname"]] = await conn.fetchval("SELECT pg_relation_size($1)", r["indexname"])
    out["chunks_table"] = await conn.fetchval("SELECT pg_relation_size('chunks')")
    return out


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


async def insert_tier(
    conn: asyncpg.Connection,
    text_pool: list[str],
    rng: random.Random,
    target_total_new: int,
    already_inserted: int,
    time_budget_s: float,
) -> dict:
    """Inserts (target_total_new - already_inserted) new bench chunks via COPY,
    in COPY_BATCH-row batches, grouping CHUNKS_PER_DOC chunks per synthetic
    document. Stops early if time_budget_s is exceeded; returns what happened."""
    to_insert = target_total_new - already_inserted
    inserted = 0
    batch_rates: list[float] = []
    t_start = time.time()
    doc_id = None
    doc_chunk_count = CHUNKS_PER_DOC  # force new doc on first batch

    while inserted < to_insert:
        if time.time() - t_start > time_budget_s:
            break
        n = min(COPY_BATCH, to_insert - inserted)
        vecs = _unit_vectors(n, DIM)
        records = []
        for i in range(n):
            if doc_chunk_count >= CHUNKS_PER_DOC:
                doc_id = uuid.uuid4()
                await conn.execute(
                    """INSERT INTO documents
                       (id, sha256, filename, mime, blob_path, page_count, title,
                        title_norm, collection, meta, status, error)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11,$12)""",
                    doc_id,
                    uuid.uuid4().hex,
                    f"{FILENAME_PREFIX}doc_{doc_id}.txt",
                    "text/plain",
                    "bench",
                    1,
                    f"bench doc {doc_id}",
                    f"bench doc {doc_id}",
                    COLLECTION,
                    "{}",
                    "ready",
                    None,
                )
                doc_chunk_count = 0
            text = rng.choice(text_pool) + f" ref{already_inserted + inserted + i}"
            records.append(
                (doc_id, doc_chunk_count, "prose", text, text, 1, 1, len(text.split()), vecs[i])
            )
            doc_chunk_count += 1

        t0 = time.time()
        await conn.copy_records_to_table(
            "chunks",
            records=records,
            columns=[
                "doc_id", "ord", "kind", "text", "embed_text",
                "page_start", "page_end", "token_count", "embedding",
            ],
        )
        dt = time.time() - t0
        inserted += n
        batch_rates.append(n / dt)

    elapsed = time.time() - t_start
    return {
        "inserted": inserted,
        "elapsed_s": elapsed,
        "rows_per_sec_overall": inserted / elapsed if elapsed > 0 else float("nan"),
        "rows_per_sec_first_batch": batch_rates[0] if batch_rates else float("nan"),
        "rows_per_sec_last_batch": batch_rates[-1] if batch_rates else float("nan"),
        "hit_time_budget": (target_total_new - already_inserted) > inserted,
    }


async def bench_vector_query(conn: asyncpg.Connection, n_queries: int, k: int) -> list[float]:
    """Exact query shape of prorag.retrieve.arms.vector_search — no halfvec
    cast, matching what the app actually issues."""
    latencies = []
    for _ in range(n_queries):
        v = _unit_vectors(1, DIM)[0]
        lit = _vector_literal(v)
        t0 = time.time()
        await conn.fetch(
            f"""
            SELECT c.id, c.embedding <=> '{lit}'::vector AS distance
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE d.status = 'ready'
            ORDER BY distance
            LIMIT {k}
            """
        )
        latencies.append(time.time() - t0)
    return latencies


async def bench_vector_query_halfvec_cast(conn: asyncpg.Connection, n_queries: int, k: int) -> list[float]:
    """Diagnostic-only variant: casts to halfvec so the query actually matches
    the HNSW index's expression. Not what the app runs today — included to
    show the gap between 'index exists' and 'index gets used'."""
    latencies = []
    for _ in range(n_queries):
        v = _unit_vectors(1, DIM)[0]
        lit = _vector_literal(v)
        t0 = time.time()
        await conn.fetch(
            f"""
            SELECT c.id, (c.embedding::halfvec({DIM})) <=> '{lit}'::halfvec({DIM}) AS distance
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE d.status = 'ready'
            ORDER BY distance
            LIMIT {k}
            """
        )
        latencies.append(time.time() - t0)
    return latencies


async def bench_bm25_query(conn: asyncpg.Connection, n_queries: int, k: int, rng: random.Random) -> list[float]:
    latencies = []
    for _ in range(n_queries):
        q = " ".join(rng.sample(_VOCAB, 3))
        t0 = time.time()
        await conn.fetch(
            """
            SELECT c.id, paradedb.score(c.id) AS score
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE c.text @@@ $1 AND d.status = 'ready'
            ORDER BY score DESC
            LIMIT $2
            """,
            q,
            k,
        )
        latencies.append(time.time() - t0)
    return latencies


def p(latencies: list[float], pct: float) -> float:
    if not latencies:
        return float("nan")
    return statistics.quantiles(latencies, n=100, method="inclusive")[int(pct) - 1] if len(latencies) > 1 else latencies[0]


async def explain_seq_or_index(conn: asyncpg.Connection) -> str:
    v = _unit_vectors(1, DIM)[0]
    lit = _vector_literal(v)
    rows = await conn.fetch(
        f"""
        EXPLAIN (FORMAT TEXT)
        SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE d.status = 'ready'
        ORDER BY c.embedding <=> '{lit}'::vector
        LIMIT {TOP_K}
        """
    )
    plan = "\n".join(r["QUERY PLAN"] for r in rows)
    return "Seq Scan" if "Seq Scan" in plan else ("Index Scan" if "Index Scan" in plan else "?")


async def get_bench_chunk_count(conn: asyncpg.Connection) -> int:
    return await conn.fetchval(
        "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE d.filename LIKE $1",
        f"{FILENAME_PREFIX}%",
    )


# ponytail: the full 10k+100k run originally lived in one long-running
# coroutine; on this host it's slow enough (HNSW insert-bound) to blow past
# any single foreground command's timeout, and a background run with nobody
# watching its output is worse than useless. Split into small CLI verbs
# (insert / measure / cleanup / ingest / rerank) so each invocation is a
# short, foreground, individually-verifiable step; `insert` is resumable
# because it reads the current bench-row count from the DB instead of
# trusting an in-memory counter across process runs.
async def cmd_insert(add_rows: int, budget_s: float) -> None:
    conn = await asyncpg.connect(DSN)
    await register_vector(conn)
    await conn.execute("SET max_parallel_maintenance_workers = 0")  # avoid the shm blowup seen during calibration

    already = await get_bench_chunk_count(conn)
    rng = random.Random(42)
    text_pool = _build_text_pool(3000, rng)

    print(f"already {already} bench chunks; inserting {add_rows} more (budget {budget_s:.0f}s)...")
    stats = await insert_tier(conn, text_pool, rng, already + add_rows, already, budget_s)
    print(f"inserted {stats['inserted']} rows in {stats['elapsed_s']:.1f}s "
          f"({stats['rows_per_sec_overall']:.1f} rows/s overall, "
          f"first batch {stats['rows_per_sec_first_batch']:.1f} rows/s, "
          f"last batch {stats['rows_per_sec_last_batch']:.1f} rows/s, "
          f"hit_time_budget={stats['hit_time_budget']})")
    total = await get_bench_chunk_count(conn)
    print(f"total bench chunks now: {total}")
    await conn.close()


async def cmd_measure(label: str) -> None:
    """Runs VACUUM + size + vector/bm25 latency measurements against whatever
    is in the table right now (baseline + however many bench rows `insert`
    has accumulated so far). Safe to call repeatedly."""
    conn = await asyncpg.connect(DSN)
    await register_vector(conn)

    print("VACUUM ANALYZE chunks...")
    await conn.execute("VACUUM ANALYZE chunks")
    await conn.execute("VACUUM ANALYZE documents")

    total_chunks = await conn.fetchval("SELECT count(*) FROM chunks")
    bench_chunks = await get_bench_chunk_count(conn)
    sizes = await index_sizes(conn)
    plan_kind = await explain_seq_or_index(conn)
    rng = random.Random(43)

    print(f"total_chunks={total_chunks} bench_chunks={bench_chunks}")
    print("sizes:", {k: fmt_bytes(v) for k, v in sizes.items()})
    print("query plan for the app's actual vector_search shape:", plan_kind)

    print(f"running {N_QUERIES} vector queries (app's real, uncast shape)...")
    vec_lat = await bench_vector_query(conn, N_QUERIES, TOP_K)
    print(f"running {N_QUERIES} vector queries (halfvec-cast, index-matching shape, diagnostic only)...")
    vec_lat_cast = await bench_vector_query_halfvec_cast(conn, N_QUERIES, TOP_K)
    print(f"running {N_QUERIES} bm25 queries...")
    bm25_lat = await bench_bm25_query(conn, N_QUERIES, TOP_K, rng)

    tier_result = {
        "label": label,
        "total_chunks": total_chunks,
        "bench_chunks": bench_chunks,
        "sizes": {k: v for k, v in sizes.items()},
        "query_plan": plan_kind,
        "vector_p50_ms": p(vec_lat, 50) * 1000,
        "vector_p95_ms": p(vec_lat, 95) * 1000,
        "vector_cast_p50_ms": p(vec_lat_cast, 50) * 1000,
        "vector_cast_p95_ms": p(vec_lat_cast, 95) * 1000,
        "bm25_p50_ms": p(bm25_lat, 50) * 1000,
        "bm25_p95_ms": p(bm25_lat, 95) * 1000,
    }
    print("RESULT_JSON", __import__("json").dumps(tier_result, default=str))
    await conn.close()


async def cmd_cleanup() -> None:
    conn = await asyncpg.connect(DSN)
    before = await get_baseline_counts(conn)
    print(f"before cleanup: {before[0]} documents, {before[1]} chunks")
    await cleanup_bench_rows(conn)
    after = await get_baseline_counts(conn)
    print(f"after cleanup: {after[0]} documents, {after[1]} chunks")
    await conn.close()


async def cmd_ingest() -> None:
    """Times the real inline /ingest path end to end for a synthetic ~50-page
    .txt document, with embed_texts_batched monkeypatched to random vectors.
    Never raises -- any failure is reported and the partial doc is still
    cleaned up, so this can't leave the DB dirty or crash the whole run."""
    from sqlalchemy import select

    import prorag.ingest.router as ingest_router
    from prorag.db import SessionLocal
    from prorag.models import Chunk, Document

    async def fake_embed(texts, session=None):
        await asyncio.sleep(0)  # keep it a real coroutine hop
        return _unit_vectors(len(texts), DIM).tolist()

    orig_embed = ingest_router.embed_texts_batched
    ingest_router.embed_texts_batched = fake_embed

    rng = random.Random(7)
    # ~50 pages @ ~500 words/page
    body = " ".join(_random_paragraph(rng, 500) for _ in range(50))
    data = body.encode("utf-8")

    from starlette.datastructures import UploadFile

    upload = UploadFile(file=io.BytesIO(data), filename=f"{FILENAME_PREFIX}50page.txt")

    try:
        try:
            async with SessionLocal() as session:
                t0 = time.time()
                resp = await ingest_router.ingest(file=upload, collection=COLLECTION, session=session)
                elapsed = time.time() - t0
                doc_id = resp.doc_id
                print(f"ingest: status={resp.status} doc_id={doc_id} elapsed={elapsed:.2f}s")
        finally:
            ingest_router.embed_texts_batched = orig_embed

        async with SessionLocal() as session:
            doc = (await session.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
            n_chunks = 0
            if doc is not None:
                rows = (await session.execute(select(Chunk).where(Chunk.doc_id == doc_id))).scalars().all()
                n_chunks = len(rows)

        ceiling = 3600 / elapsed if elapsed > 0 else float("nan")
        print(f"chunks produced: {n_chunks}; elapsed={elapsed:.2f}s; implied ceiling: {ceiling:.1f} docs/hour "
              "(single synchronous request path, one worker)")
        print("RESULT_JSON", __import__("json").dumps(
            {"elapsed_s": elapsed, "status": resp.status, "chunks_produced": n_chunks, "docs_per_hour_ceiling": ceiling}
        ))
    except Exception as exc:
        print(f"INGEST BENCH FAILED: {type(exc).__name__}: {exc}")
        raise
    finally:
        conn = await asyncpg.connect(DSN)
        await conn.execute("DELETE FROM documents WHERE filename LIKE $1", f"{FILENAME_PREFIX}%")
        await conn.close()


async def cmd_rerank() -> None:
    """Times one OpenRouter hosted rerank call for TOP_K docs and reports
    the billed cost — the local CrossEncoder bench is gone (no offline
    models; laptop thermals)."""
    import httpx

    api_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("rerank: OPENROUTER_API_KEY unset; skipping")
        return
    rng = random.Random(3)
    texts = [_random_paragraph(rng, 200) for _ in range(TOP_K)]
    t0 = time.time()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": settings.rerank_api_model,
                "query": "fire drill frequency requirement",
                "documents": texts,
            },
        )
        resp.raise_for_status()
        usage = resp.json().get("usage", {})
    s = time.time() - t0
    print(f"rerank: {settings.rerank_api_model} {TOP_K} docs in {s:.2f}s, usage={usage}")
    print("RESULT_JSON", __import__("json").dumps({
        "loaded": True,
        "single_query_40hits_s": s,
        "usage": usage,
    }))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_insert = sub.add_parser("insert", help="insert N more bench chunks (resumable, reads current count from DB)")
    p_insert.add_argument("--rows", type=int, required=True)
    p_insert.add_argument("--budget", type=float, default=TIER_TIME_BUDGET_S)

    p_measure = sub.add_parser("measure", help="vacuum + measure sizes/latencies at current row count")
    p_measure.add_argument("--label", default="")

    sub.add_parser("cleanup", help="delete all bench_ rows, verify baseline restored")
    sub.add_parser("ingest", help="time the real inline /ingest path once")
    sub.add_parser("rerank", help="load the CrossEncoder and time a 40-hit rerank (cold+warm)")

    args = parser.parse_args()
    if args.cmd == "insert":
        asyncio.run(cmd_insert(args.rows, args.budget))
    elif args.cmd == "measure":
        asyncio.run(cmd_measure(args.label))
    elif args.cmd == "cleanup":
        asyncio.run(cmd_cleanup())
    elif args.cmd == "ingest":
        asyncio.run(cmd_ingest())
    elif args.cmd == "rerank":
        asyncio.run(cmd_rerank())


if __name__ == "__main__":
    main()

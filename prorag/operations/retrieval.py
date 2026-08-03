"""Operations layer — hybrid retrieval pipeline (§4, §5).

Per the three-layer architecture (Routers -> Operations -> Database), this
module holds the *business logic* of retrieval and prompt building, shared by
every caller that needs context for an answer:

- POST /chat and POST /chat/stream (chat/router.py) — the chat flow
- GET /search (retrieve/router.py) — the debug endpoint, which reuses
  gather_hits() so the hybrid pipeline can't drift between the two surfaces
- /eval/run (eval/runner.py) — the golden-set runner

Routers import from here; this module never imports a router. It depends only
on the retrieve arms (prorag/retrieve/*), the LLM service (prorag/llm.py) and
persistence models — never on HTTP concerns.

The pipeline: planner -> embed both queries -> gather every arm sequentially
(vector x2, fts x2[, structured x2]) -> RRF fuse -> cross-encoder rerank ->
adaptive crop -> commit the read transaction so the caller's long LLM call
doesn't sit on a pool connection.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from prorag.llm import embed_texts
from prorag.models import User
from prorag.retrieve.arms import keyword_search, structured_search, vector_search
from prorag.retrieve.crop import crop_context
from prorag.retrieve.fuse import rrf_fuse
from prorag.retrieve.plan import plan
from prorag.retrieve.prefill import prefill
from prorag.retrieve.rerank import rerank
from prorag.schemas import Source
from prorag.settings import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers strictly from the numbered context "
    "blocks below. Cite every claim inline immediately after the sentence it "
    "supports using bracketed numbers like [S1], [S2] — never a separate "
    "'Sources' section, never markdown links. If the context does not contain "
    "the answer, say so plainly."
)


async def gather_hits(
    session: AsyncSession,
    plan_result: dict,
    embeddings: list[list[float]],
    *,
    user: User | None = None,
) -> tuple[list[str], list[list[dict]], list[list[dict]], list[list[dict]], list[dict]]:
    """Run every retrieval arm against one session and RRF-fuse the results.

    Shared by retrieve() and GET /search so the hybrid pipeline is defined in
    exactly one place. Returns (queries, vector_lists, fts_lists,
    structured_lists, fused).

    Sequential, not asyncio.gather: an AsyncSession is a single connection and
    is not concurrency-safe. Gathering these raised InvalidRequestError
    ("session is provisioning a new connection") whenever the session arrived
    cold — which is exactly how GET /search calls this, since plan() and
    embed_texts() only stage usage rows and never execute a query. /chat only
    escaped it because check_daily_cap() happens to run a SELECT first.
    # ponytail: gathering bought nothing anyway — measured 0.81s either way,
    # because one connection serializes them regardless. Real parallelism needs
    # a session per arm, which would take ~6 pool connections per request and
    # re-create the starvation problem the commit below fixes.
    """
    queries = plan_result["queries"]
    wants_table = plan_result.get("mode") == "table"

    vector_lists = [await vector_search(session, e, settings.rerank_top_n, user=user) for e in embeddings]
    fts_lists = [await keyword_search(session, q, settings.rerank_top_n, user=user) for q in queries]
    structured_lists = (
        [await structured_search(session, q, settings.rerank_top_n, user=user) for q in queries]
        if wants_table
        else []
    )

    ranked_lists = [*vector_lists, *fts_lists, *structured_lists]
    weights = [1.0] * len(vector_lists) + [1.0] * len(fts_lists) + [settings.structured_weight] * len(structured_lists)
    fused = rrf_fuse(ranked_lists, weights=weights)
    return queries, vector_lists, fts_lists, structured_lists, fused


async def retrieve(session: AsyncSession, message: str, user: User | None = None) -> tuple[list[dict], str]:
    """Prefill -> planner -> embed both queries -> gather_hits -> rerank
    top-N -> adaptive crop. The retrieval half of every answer request, shared
    by /chat, /chat/stream and /eval/run.

    Returns (hits, cleaned): `cleaned` is the prefill-refined prompt (== the
    raw message when prefill is disabled, failed, or judged already-clean).
    Callers must use `cleaned` — not the raw message — for build_prompt() so
    the answer model sees the same question retrieval searched for.

    `user` (#18) is threaded into every arm so the ACL predicate applies
    before LIMIT/rerank/crop ever see a row — None (auth disabled, legacy
    unscoped key, or the eval runner's service-user — see eval/runner.py)
    means unfiltered, same as today."""
    user_id = user.id if user is not None else None
    cleaned = message
    if settings.prefill_enabled:
        cleaned = await prefill(message, session=session, user_id=user_id)
    plan_result = await plan(cleaned, session=session, user_id=user_id)
    embeddings = await embed_texts(plan_result["queries"], session=session, user_id=user_id)

    queries, _, _, _, fused = await gather_hits(session, plan_result, embeddings, user=user)
    reranked = await rerank(queries[0], fused[: settings.rerank_top_n])
    cropped = crop_context(
        reranked,
        min_docs=settings.crop_min_docs,
        max_docs=settings.crop_max_docs,
        max_chunks_per_doc=settings.crop_max_chunks_per_doc,
        score_floor=settings.crop_score_floor,
        token_budget=settings.crop_token_budget,
    )
    # Retrieval is read-only, but the first query opened a transaction and
    # SQLAlchemy holds the connection until *something* ends it. Every caller
    # then spends seconds on an LLM call before writing, so without this the
    # connection sits `idle in transaction` for the whole answer — verified
    # against pg_stat_activity. At pool_size=10 that's ~10 concurrent chats
    # before the 11th blocks on pool_timeout. Committing here also durably
    # records the planner/embedding cost track_usage() staged during retrieval,
    # which a failed answer would otherwise roll back.
    await session.commit()
    return cropped, cleaned


def to_source(n: int, hit: dict) -> Source:
    """Shape a retrieval hit dict into the API's Source schema (chat/router.py
    and the stream's `sources` event both render these)."""
    page = hit.get("page")
    return Source(
        n=n,
        doc_id=hit["doc_id"],
        page=page,
        file_url=f"/files/{hit['doc_id']}/original" + (f"#page={page}" if page else ""),
        snippet=hit["text"][:300],
        score=hit["score"],
        title=hit.get("title"),
        kind=hit.get("kind"),
        bbox=hit.get("bbox"),
    )


def build_prompt(message: str, hits: list[dict]) -> str:
    """Assemble the [Sn]-numbered context block (citations.py) plus the user's
    question into the answerer's prompt. Runs on every chat/eval turn."""
    from prorag.chat.citations import build_context_block

    context_block = build_context_block(hits)
    return f"Context:\n{context_block}\n\nQuestion: {message}"

"""GET /search — retrieval debug endpoint: per-arm results + fused + reranked
lists, so hybrid retrieval tuning has something to look at without going
through the LLM (§6)."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.db import get_session
from prorag.llm import embed_texts
from prorag.retrieve.arms import keyword_search, structured_search, vector_search
from prorag.retrieve.crop import crop_context
from prorag.retrieve.fuse import rrf_fuse
from prorag.retrieve.plan import plan
from prorag.retrieve.rerank import rerank
from prorag.settings import settings

router = APIRouter()


@router.get("/search")
async def search(q: str, k: int = 10, session: AsyncSession = Depends(get_session)):
    plan_result = await plan(q, session=session)
    queries = plan_result["queries"]

    embeddings = await embed_texts(queries, session=session)
    # Sequential for the same reason as chat/router.py's retrieve(): one
    # AsyncSession is one connection and forbids overlapping execute() calls.
    # This endpoint is where that actually bit — it hands retrieve's arms a cold
    # session, so the gather raised InvalidRequestError rather than just
    # serializing.
    vector_lists = [await vector_search(session, e, settings.rerank_top_n) for e in embeddings]
    fts_lists = [await keyword_search(session, query, settings.rerank_top_n) for query in queries]
    wants_table = plan_result.get("mode") == "table"
    structured_lists = (
        [await structured_search(session, query, settings.rerank_top_n) for query in queries] if wants_table else []
    )

    ranked_lists = [*vector_lists, *fts_lists, *structured_lists]
    weights = [1.0] * len(vector_lists) + [1.0] * len(fts_lists) + [settings.structured_weight] * len(structured_lists)
    fused = rrf_fuse(ranked_lists, weights=weights)
    reranked = await rerank(queries[0], fused[: settings.rerank_top_n])
    cropped = crop_context(
        reranked,
        min_docs=settings.crop_min_docs,
        max_docs=settings.crop_max_docs,
        score_gap=settings.crop_score_gap,
        score_floor=settings.crop_score_floor,
        token_budget=settings.crop_token_budget,
    )

    return {
        "plan": plan_result,
        "arms": {
            "vector": vector_lists,
            "fts": fts_lists,
            "structured": structured_lists,
        },
        "fused": fused[:k],
        "reranked": reranked[:k],
        "cropped": cropped,
    }

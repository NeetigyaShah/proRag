"""GET /search — retrieval debug endpoint: per-arm results + fused + reranked
lists, so hybrid retrieval tuning has something to look at without going
through the LLM (§6).

The pipeline itself is shared with the chat flow (operations/retrieval.py) —
this handler only shapes the per-stage breakdown for display.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.auth import current_user
from prorag.db import get_session
from prorag.llm import embed_texts
from prorag.models import User
from prorag.operations.retrieval import gather_hits
from prorag.retrieve.crop import crop_context
from prorag.retrieve.plan import plan
from prorag.retrieve.rerank import rerank
from prorag.settings import settings

router = APIRouter()


@router.get("/search")
async def search(
    q: str, k: int = 10, session: AsyncSession = Depends(get_session), user: User | None = Depends(current_user)
):
    plan_result = await plan(q, session=session)
    queries = plan_result["queries"]

    embeddings = await embed_texts(queries, session=session)
    # Sequential for the same reason as operations/retrieval.py's gather_hits():
    # one AsyncSession is one connection and forbids overlapping execute() calls.
    # This endpoint is where that actually bit — it hands retrieve's arms a cold
    # session, so the gather raised InvalidRequestError rather than just
    # serializing.
    queries, vector_lists, fts_lists, structured_lists, fused = await gather_hits(
        session, plan_result, embeddings, user=user
    )
    reranked = await rerank(queries[0], fused[: settings.rerank_top_n])
    cropped = crop_context(
        reranked,
        min_docs=settings.crop_min_docs,
        max_docs=settings.crop_max_docs,
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

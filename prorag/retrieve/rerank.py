"""Rerank — OpenRouter's hosted cross-encoder endpoint.

POST https://openrouter.ai/api/v1/rerank with the fused chunks; a real
cross-encoder (voyageai/rerank-2.5-lite by default) scores every (query,
chunk) pair in one call — ~2s and ~$0.000002 for 40 chunks, zero local
compute (the sentence-transformers fallback was removed: laptop thermals).

Degrades gracefully: any failure returns the input order unchanged (no local
model to fall back to). The flatness guard (ADR 0002) keeps the pre-rerank
fused order when the reranker's top scores can't discriminate.
"""

import logging
import os

import httpx

from prorag.settings import settings

logger = logging.getLogger(__name__)

# OpenRouter's rerank endpoint mirrors the provider-standard shape: a POST with
# {model, query, documents} returning {"results": [{index, relevance_score}]}.
RERANK_URL = "https://openrouter.ai/api/v1/rerank"


def _flat_guard(hits: list[dict], input_scores: list[float]) -> list[dict] | None:
    """Flatness guard: if the reranker's five best chunks (by score) are all
    within settings.rerank_flat_spread, its re-ordering is noise — the fused
    (input) order is the order of record. The real score MULTISET is still
    attached, but assigned in descending order down the fused order:
    crop_context and the UI both sort by score (stably), so the scores must
    encode the fused order or the sort would silently undo it. Measured on
    the top-5 BY SCORE: that is the band the crop and the answer actually
    use, and an outlier (e.g. a junk doc scoring 0.010) must not inflate the
    spread and mask a flat top. Returns None when the scores discriminate
    (keep the rerank order)."""
    top = sorted(input_scores, reverse=True)[:5]
    if len(top) >= 2 and (max(top) - min(top)) < settings.rerank_flat_spread:
        logger.info(
            "rerank scores flat (top-5 spread %.4f) — keeping fused order",
            max(top) - min(top),
        )
        ordered = sorted(input_scores, reverse=True)
        return [{**h, "score": ordered[i]} for i, h in enumerate(hits)]
    return None


async def _rerank_api(query: str, hits: list[dict]) -> list[dict]:
    """When called: by rerank() on every retrieval. What: one hosted
    cross-encoder call scoring all pairs; re-sorts by relevance_score desc
    (scores already 0..1), unless the flatness guard fires. Returns: hits
    unchanged on any failure — no key, HTTP error, timeout, malformed body."""
    api_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY is not set — rerank API unavailable")
        return hits
    try:
        async with httpx.AsyncClient(timeout=settings.rerank_api_timeout) as client:
            resp = await client.post(
                RERANK_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": settings.rerank_api_model,
                    "query": query,
                    "documents": [h["text"] for h in hits],
                },
            )
            resp.raise_for_status()
            results = (resp.json().get("results") or [])
        valid = [r for r in results if 0 <= r.get("index", -1) < len(hits)]
        if not valid:
            logger.warning("rerank API returned no results; degrading")
            return hits
        input_scores = [0.0] * len(hits)
        for r in valid:
            input_scores[r["index"]] = float(r["relevance_score"])
        guarded = _flat_guard(hits, input_scores)
        if guarded is not None:
            return guarded
        scored = [{**hits[r["index"]], "score": float(r["relevance_score"])} for r in valid]
        # The API returns best-first, but the defensive sort keeps the
        # contract even if a provider ever ignores ordering.
        scored.sort(key=lambda h: h["score"], reverse=True)
        return scored
    except Exception as exc:
        logger.warning("rerank API failed (%s); degrading", exc)
        return hits


async def rerank(query: str, hits: list[dict]) -> list[dict]:
    """Reranks `hits` (already trimmed to top-N by the caller). No-op
    (returns hits unchanged) if rerank is disabled, the API fails, or the
    flatness guard keeps the fused order."""
    if not settings.rerank_enabled or not hits:
        return hits
    return await _rerank_api(query, hits)

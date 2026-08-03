"""Rerank — OpenRouter's hosted cross-encoder endpoint by default, with a
local sentence-transformers fallback.

`rerank_backend == "api"` (default): POST https://openrouter.ai/api/v1/rerank
with the fused chunks; a real cross-encoder (cohere/rerank-v3.5) scores every
(query, chunk) pair in one call — ~1.5s and ~$0.001 for 40 chunks, zero local
compute. Scores come back already 0..1 so downstream crop thresholds and the
UI's score meters keep working.

`rerank_backend == "local"`: sentence-transformers CrossEncoder on a dedicated
single-thread ThreadPoolExecutor (§4.4). `OMP_NUM_THREADS=1` is set before the
model is ever touched: the thread-in-thread contention between torch's own
thread pool and the asyncio executor was a measured, hard-won finding in the
ancestor project — running the cross-encoder on a pool with more than one
worker (or without pinning OMP threads) reintroduces it.

Degrades gracefully in both backends: any failure returns the input order
unchanged, and when `rerank_api_fallback_to_local` is set an API failure
drops to the local cross-encoder first. If sentence-transformers isn't
installed or the model can't be downloaded, `get_model()` returns None, a
warning is logged once, and the local path becomes a no-op. This must stay
import-safe with no network access and no local model weights.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import httpx

from prorag.settings import settings

os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker")

# OpenRouter's rerank endpoint mirrors the provider-standard shape: a POST with
# {model, query, documents} returning {"results": [{index, relevance_score}]}.
RERANK_URL = "https://openrouter.ai/api/v1/rerank"


async def _rerank_api(query: str, hits: list[dict]) -> list[dict]:
    """When called: by rerank() when settings.rerank_backend == "api".
    What: one hosted cross-encoder call scoring all pairs; re-sorts by
    relevance_score desc (scores already 0..1). Returns: hits unchanged on
    any failure — no key, HTTP error, timeout, or malformed body."""
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
        scored = [
            {**hits[r["index"]], "score": float(r["relevance_score"])}
            for r in results
            if 0 <= r.get("index", -1) < len(hits)
        ]
        if not scored:
            logger.warning("rerank API returned no results; degrading")
            return hits
        # The API returns best-first, but the defensive sort keeps the
        # contract even if a provider ever ignores ordering.
        scored.sort(key=lambda h: h["score"], reverse=True)
        return scored
    except Exception as exc:
        logger.warning("rerank API failed (%s); degrading", exc)
        return hits


@lru_cache(maxsize=1)
def get_model():
    """Lazily loaded singleton for the LOCAL backend. Returns None (never
    raises) if the model can't be loaded — missing dependency, no network,
    whatever."""
    if not settings.rerank_enabled:
        return None
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning("sentence-transformers not installed; rerank disabled")
        return None
    try:
        try:
            return CrossEncoder(settings.rerank_model, backend="onnx")
        except Exception:
            return CrossEncoder(settings.rerank_model)  # fall back to default backend
    except Exception:
        logger.warning("could not load reranker model %s; rerank disabled", settings.rerank_model)
        return None


def _rerank_sync(query: str, hits: list[dict]) -> list[dict]:
    """When called: by rerank() inside the dedicated single-thread executor
    (local backend). What: loads the model (if available), scores every
    (query, hit text) pair, and re-sorts hits by score descending. Returns:
    hits unchanged when no model or no hits, otherwise the reranked list."""
    model = get_model()
    if model is None or not hits:
        return hits
    pairs = [(query, h["text"]) for h in hits]
    scores = model.predict(pairs)
    reranked = [{**h, "score": float(s)} for h, s in zip(hits, scores, strict=True)]
    reranked.sort(key=lambda h: h["score"], reverse=True)
    return reranked


async def rerank(query: str, hits: list[dict]) -> list[dict]:
    """Reranks `hits` (already trimmed to top-N by the caller). No-op
    (returns hits unchanged) if rerank is disabled or every backend fails.
    Backend is settings.rerank_backend: "api" (OpenRouter's hosted
    cross-encoder, with an automatic degrade to the local cross-encoder when
    the API fails) or "local" (sentence-transformers in the dedicated
    executor)."""
    if not settings.rerank_enabled or not hits:
        return hits
    if settings.rerank_backend == "api":
        scored = await _rerank_api(query, hits)
        if scored is not hits or not settings.rerank_api_fallback_to_local:
            return scored
        # API failed (network, provider outage) — degrade to local.
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _rerank_sync, query, hits)

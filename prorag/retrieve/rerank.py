"""Cross-encoder rerank — bge-reranker-v2-m3 via sentence-transformers
CrossEncoder, on a dedicated single-thread ThreadPoolExecutor (§4.4).

`OMP_NUM_THREADS=1` is set before the model is ever touched: the
thread-in-thread contention between torch's own thread pool and the asyncio
executor was a measured, hard-won finding in the ancestor project — running
the cross-encoder on a pool with more than one worker (or without pinning
OMP threads) reintroduces it.

Degrades gracefully: if sentence-transformers isn't installed or the model
can't be downloaded, `get_model()` returns None, a warning is logged once,
and `rerank()` becomes a no-op that returns the input order unchanged. This
must stay import-safe with no network access and no local model weights.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from prorag.settings import settings

os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reranker")


@lru_cache(maxsize=1)
def get_model():
    """Lazily loaded singleton. Returns None (never raises) if the model
    can't be loaded — missing dependency, no network, whatever."""
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
    model = get_model()
    if model is None or not hits:
        return hits
    pairs = [(query, h["text"]) for h in hits]
    scores = model.predict(pairs)
    reranked = [{**h, "score": float(s)} for h, s in zip(hits, scores, strict=True)]
    reranked.sort(key=lambda h: h["score"], reverse=True)
    return reranked


async def rerank(query: str, hits: list[dict]) -> list[dict]:
    """Reranks `hits` (already trimmed to top-N by the caller) in the
    dedicated executor. No-op (returns hits unchanged) if rerank is disabled
    or the model failed to load."""
    if not settings.rerank_enabled or not hits:
        return hits
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _rerank_sync, query, hits)

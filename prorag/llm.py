"""LiteLLM wrapper: cheap/strong tiers, embeddings, and cost tracking (§5.4,
§8 Phase 5) — every call records prompt/completion tokens to `usage` when a
DB session is passed in, via prorag.cost.track_usage().

# ponytail: retries/semaphores per operation (architecture §llm.py) still
# aren't here — add when a provider actually rate-limits this app.
"""

import asyncio

import litellm

from prorag.cost import track_usage
from prorag.settings import settings


def _usage_tokens(resp) -> tuple[int, int]:
    usage = resp.get("usage") or {}
    return usage.get("prompt_tokens", 0) or 0, usage.get("completion_tokens", 0) or 0


def _reasoning_kwargs(model: str) -> dict:
    """OpenRouter models accept {"reasoning": {"enabled": bool}}; passing it to
    other providers would error, so only attach it for openrouter/ models."""
    if model.startswith("openrouter/") and not settings.reasoning_enabled:
        return {"reasoning": {"enabled": False}}
    return {}


def estimate_tokens(model: str, text: str) -> int:
    """Token estimate for the streaming path, where litellm doesn't hand back
    a usage object mid-stream. Falls back to a whitespace split if the
    model's tokenizer isn't known to litellm."""
    try:
        return litellm.token_counter(model=model, text=text)
    except Exception:
        return len(text.split())


_local_embedder = None


def _get_local_embedder(model_name: str):
    """Free local embeddings (EMBED_MODEL=local/<hf-model>) — needed because
    OpenRouter has no embeddings API. Lazy singleton, CPU."""
    global _local_embedder
    if _local_embedder is None:
        from sentence_transformers import SentenceTransformer

        _local_embedder = SentenceTransformer(model_name)
    return _local_embedder


async def embed_texts_batched(texts: list[str], session=None) -> list[list[float]]:
    """Embed in bounded batches with bounded concurrency, preserving order.

    One giant request risks provider payload/rate limits and makes any failure
    retry everything; batching keeps each call small and lets several run at
    once. Order is preserved because results are placed back by index.
    """
    if not texts:
        return []

    size = max(1, settings.embed_batch_size)
    batches = [texts[i : i + size] for i in range(0, len(texts), size)]
    if len(batches) == 1:
        return await embed_texts(batches[0], session=session)

    sem = asyncio.Semaphore(max(1, settings.embed_batch_concurrency))
    results: list[list[list[float]]] = [[] for _ in batches]

    async def run(idx: int, batch: list[str]) -> None:
        async with sem:
            results[idx] = await embed_texts(batch, session=session)

    await asyncio.gather(*(run(i, b) for i, b in enumerate(batches)))
    return [vec for batch_result in results for vec in batch_result]


async def embed_texts(texts: list[str], session=None, user_id=None) -> list[list[float]]:
    if not texts:
        return []
    if settings.embed_model.startswith("openrouter-embed/"):
        # OpenRouter's /embeddings endpoint isn't in litellm's provider map —
        # call it directly.
        import os

        import httpx

        model_name = settings.embed_model.removeprefix("openrouter-embed/")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                json={"model": model_name, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        return [item["embedding"] for item in data]
    if settings.embed_model.startswith("local/"):
        import asyncio

        model = _get_local_embedder(settings.embed_model.removeprefix("local/"))
        vectors = await asyncio.get_running_loop().run_in_executor(
            None, lambda: model.encode(texts, normalize_embeddings=True)
        )
        return [v.tolist() for v in vectors]
    resp = await litellm.aembedding(model=settings.embed_model, input=texts)
    prompt_tokens, completion_tokens = _usage_tokens(resp)
    track_usage(session, settings.embed_model, prompt_tokens, completion_tokens, user_id=user_id)
    return [item["embedding"] for item in resp["data"]]


async def answer(system: str, user: str, session=None, message_id=None, user_id=None) -> str:
    resp = await litellm.acompletion(
        model=settings.answer_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **_reasoning_kwargs(settings.answer_model),
    )
    prompt_tokens, completion_tokens = _usage_tokens(resp)
    track_usage(session, settings.answer_model, prompt_tokens, completion_tokens, message_id=message_id, user_id=user_id)
    return resp["choices"][0]["message"]["content"]


async def answer_stream(system: str, user: str):
    """LiteLLM streaming variant of answer() — yields text deltas as they
    arrive, for /chat/stream (§5.2, Phase 4). stream_options include_usage
    makes the final chunk carry real token counts, yielded as ("usage", ...);
    that chunk's `choices` list is typically empty, so it's guarded rather
    than indexed."""
    response = await litellm.acompletion(
        model=settings.answer_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        stream=True,
        stream_options={"include_usage": True},
        **_reasoning_kwargs(settings.answer_model),
    )
    async for chunk in response:
        usage = chunk.get("usage")
        if usage:
            prompt_tokens, completion_tokens = _usage_tokens(chunk)
            yield ("usage", {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens})
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta_obj = choices[0]["delta"]
        # Reasoning models (e.g. Ling via OpenRouter) stream their thinking as
        # `reasoning_content`/`reasoning` deltas before the answer content.
        reasoning = (
            getattr(delta_obj, "reasoning_content", None)
            or delta_obj.get("reasoning_content")
            or delta_obj.get("reasoning")
        )
        if reasoning:
            yield ("thinking", reasoning)
        delta = delta_obj.get("content")
        if delta:
            yield ("answer", delta)


async def plan_completion(system: str, user: str, session=None, user_id=None) -> str:
    """Cheap-tier call for the planner (§4.1). Separate from answer() so the
    two tiers can point at different models via settings."""
    resp = await litellm.acompletion(
        model=settings.planner_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **_reasoning_kwargs(settings.planner_model),
    )
    prompt_tokens, completion_tokens = _usage_tokens(resp)
    track_usage(session, settings.planner_model, prompt_tokens, completion_tokens, user_id=user_id)
    return resp["choices"][0]["message"]["content"]

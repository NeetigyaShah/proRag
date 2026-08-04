"""LiteLLM wrapper: cheap/strong tiers, embeddings, and cost tracking (§5.4,
§8 Phase 5) — every call records prompt/completion tokens to `usage` when a
DB session is passed in, via prorag.cost.track_usage().

# ponytail: retries/semaphores per operation (architecture §llm.py) still
# aren't here — add when a provider actually rate-limits this app.
"""

import asyncio
import os

import litellm

from prorag.cost import track_usage
from prorag.settings import settings

# litellm authenticates providers via os.environ — a key that lives only in
# .env (visible to Settings, not to os.environ) would fail every litellm path
# (answer, answer_stream, plan_completion, aembedding) from a bare shell.
# Export it once at import; a real env var (higher-priority) wins untouched.
if settings.openrouter_api_key and not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key


def _usage_tokens(resp) -> tuple[int, int]:
    """When called: by answer(), answer_stream() and embed_texts() to read
    token counts off a litellm response. What: pulls prompt/completion token
    counts, defaulting to 0 when absent. Returns: (prompt_tokens,
    completion_tokens)."""
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
    """When called: by every embedding consumer — query planning
    (operations/retrieval.py), GET /search (retrieve/router.py), rule
    preview/confirm (admin/router.py), doctor's check_embed, and
    embed_texts_batched(). What: embeds all texts via the configured
    embed_model (OpenRouter direct call for openrouter-embed/* models, or
    litellm) and records usage when a session is given. Returns: one vector
    per input text, in order."""
    if not texts:
        return []
    if settings.embed_model.startswith("openrouter-embed/"):
        # OpenRouter's /embeddings endpoint isn't in litellm's provider map —
        # call it directly.
        import httpx

        api_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            # A clear error beats the old raw KeyError: this branch is reached
            # whenever EMBED_MODEL is an openrouter-embed/* model, and the key
            # can live in .env (settings field) or the process env.
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set — openrouter-embed models need it (set it in .env)"
            )
        model_name = settings.embed_model.removeprefix("openrouter-embed/")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model_name, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        return [item["embedding"] for item in data]
    resp = await litellm.aembedding(model=settings.embed_model, input=texts)
    prompt_tokens, completion_tokens = _usage_tokens(resp)
    track_usage(session, settings.embed_model, prompt_tokens, completion_tokens, user_id=user_id)
    return [item["embedding"] for item in resp["data"]]


async def answer(system: str, user: str, session=None, message_id=None, user_id=None) -> str:
    """When called: by POST /chat (chat/router.py) and /eval/run
    (eval/runner.py) for the full-strength answer. What: one non-streaming
    completion via settings.answer_model, recording prompt/completion tokens
    to usage when a session is given. Returns: the model's text response."""
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


async def prefill_completion(system: str, user: str, session=None, user_id=None) -> str:
    """Cheapest-tier call for the prefill agent (query refinement). Deliberately
    hard-capped: a bounded max_tokens keeps the output tiny (the contract is
    one JSON object) and the timeout keeps a slow free-tier model from adding
    more than a few seconds to the retrieval path. Callers must never rely on
    this succeeding — retrieve() falls back to the raw prompt on any error."""
    resp = await litellm.acompletion(
        model=settings.prefill_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=settings.prefill_max_tokens,
        timeout=settings.prefill_timeout,
    )
    prompt_tokens, completion_tokens = _usage_tokens(resp)
    track_usage(session, settings.prefill_model, prompt_tokens, completion_tokens, user_id=user_id)
    return resp["choices"][0]["message"]["content"]

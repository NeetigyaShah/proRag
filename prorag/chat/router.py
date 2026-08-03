"""POST /chat, /chat/stream — HTTP layer for the chat surface.

Routers are the composition root in this three-layer design (Routers ->
Operations -> Database): they parse requests, inject dependencies (session,
current_user), call the operations layer, and shape responses. All business
logic here lives in prorag/operations/:

- retrieval + prompt building: operations/retrieval.py
- cost-cap policy:             operations/budget.py
- chat persistence:            operations/chat.py

Both endpoints share one retrieval+prompt path (`retrieve()` /
`build_prompt()`) and one persistence path (`persist_exchange()`);
/chat/stream additionally follows §5's trust rule — `sources` is *every*
chunk the crop selected (context == sources), not just the ones the model
happened to cite.
"""

import asyncio
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.auth import current_user
from prorag.chat.citations import extract_cited_indices, normalize_citations, resolve_citations
from prorag.chat.stream import HEARTBEAT_INTERVAL_SECONDS, TokenGuard, sse_comment, sse_event, sse_retry
from prorag.cost import track_usage
from prorag.db import get_session
from prorag.llm import answer, answer_stream, estimate_tokens
from prorag.models import Feedback, User
from prorag.operations.budget import check_daily_cap
from prorag.operations.chat import persist_exchange
from prorag.operations.retrieval import SYSTEM_PROMPT, build_prompt, retrieve, to_source
from prorag.schemas import ChatRequest, ChatResponse, FeedbackRequest
from prorag.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest, session: AsyncSession = Depends(get_session), user: User | None = Depends(current_user)
):
    """Synchronous one-shot chat: budget gate -> retrieve -> answer -> persist."""
    budget_warning = await check_daily_cap(session, user)

    hits, cleaned = await retrieve(session, req.message, user=user)

    user_prompt = build_prompt(cleaned, hits)
    raw_answer = await answer(
        SYSTEM_PROMPT, user_prompt, session=session, user_id=(user.id if user is not None else None)
    )

    normalized = normalize_citations(raw_answer)
    resolved = resolve_citations(normalized, hits)
    sources = [to_source(r["n"], r) for r in resolved]

    _chat_id, message_id = await persist_exchange(session, req.chat_id, req.message, normalized, sources)
    return ChatResponse(answer=normalized, sources=sources, message_id=message_id, budget_warning=budget_warning)


@router.post("/feedback")
async def feedback(req: FeedbackRequest, session: AsyncSession = Depends(get_session)):
    """Like/dislike toggling on a message (§6). Posting the same rating again
    removes it, matching QDMS-AI's toggle behaviour."""
    existing = (
        await session.execute(select(Feedback).where(Feedback.message_id == req.message_id))
    ).scalar_one_or_none()

    if existing is not None:
        if existing.rating == req.rating:
            await session.delete(existing)
            await session.commit()
            return {"ok": True, "rating": None}
        existing.rating = req.rating
        existing.comment = req.comment
        await session.commit()
        return {"ok": True, "rating": existing.rating}

    session.add(Feedback(message_id=req.message_id, rating=req.rating, comment=req.comment))
    await session.commit()
    return {"ok": True, "rating": req.rating}


@router.post("/chat/stream")
async def chat_stream(
    req: ChatRequest, session: AsyncSession = Depends(get_session), user: User | None = Depends(current_user)
):
    """SSE per §5.2: sources[] before the first token (trust rule — every
    chunk the crop selected, not just what got cited), then token/citation
    events as they arrive, then meta + done."""
    # Budget check runs before the LLM is ever touched, not inside the generator.
    budget_warning = await check_daily_cap(session, user)

    async def event_source():
        yield sse_retry()

        try:
            async for frame in _stream_frames():
                yield frame
        except Exception:
            logger.exception("chat stream failed")
            yield sse_event(
                "error",
                {
                    "message": "The answer could not be generated. Check the server logs (is an LLM API key configured?)."
                },
            )
            yield sse_event("done", {})

    async def _stream_frames():
        hits, cleaned = await retrieve(session, req.message, user=user)
        all_sources = [to_source(i, h) for i, h in enumerate(hits, start=1)]
        yield sse_event("sources", [s.model_dump(mode="json") for s in all_sources])
        # Tell the client when the prefill agent rewrote the prompt, so the
        # thinking drawer can show what retrieval actually searched for.
        if cleaned != req.message:
            yield sse_event("prefill", {"cleaned": cleaned})
        if budget_warning is not None:
            yield sse_event("budget", {"warning": budget_warning})

        user_prompt = build_prompt(cleaned, hits)
        guard = TokenGuard()
        raw_answer = ""
        cited_seen: list[int] = []

        source_stream = answer_stream(SYSTEM_PROMPT, user_prompt).__aiter__()
        usage = None
        while True:
            try:
                kind, delta = await asyncio.wait_for(source_stream.__anext__(), timeout=HEARTBEAT_INTERVAL_SECONDS)
            except TimeoutError:
                yield sse_comment()
                continue
            except StopAsyncIteration:
                break

            if kind == "usage":
                usage = delta
                continue

            if kind == "thinking":
                # Reasoning deltas are dropped (user preference: answer only).
                # They still reset the heartbeat timer, so long reasoning
                # stretches never look like a dead connection.
                continue

            if not isinstance(delta, str):
                # Narrow the union pyright can't see across the await: after
                # the usage/thinking branches, only "answer" chunks reach
                # here. An unexpected chunk shape is skipped, not fatal — the
                # stream (and its heartbeat) survives instead of crashing.
                continue

            raw_answer += delta
            emitted = guard.feed(delta)
            if emitted:
                yield sse_event("token", {"t": emitted})

            for n in extract_cited_indices(normalize_citations(raw_answer)):
                if n not in cited_seen:
                    cited_seen.append(n)
                    yield sse_event("citation", {"n": n})

            if guard.done:
                break

        tail = guard.flush()
        if tail:
            yield sse_event("token", {"t": tail})

        normalized = normalize_citations(raw_answer)

        # Real usage from litellm's final stream chunk (§llm.py); estimate_tokens()
        # is only a fallback for providers that don't return one (§5.4).
        if usage:
            prompt_tokens, completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
        else:
            prompt_tokens = estimate_tokens(settings.answer_model, SYSTEM_PROMPT + user_prompt)
            completion_tokens = estimate_tokens(settings.answer_model, raw_answer)
        track_usage(
            session,
            settings.answer_model,
            prompt_tokens,
            completion_tokens,
            user_id=(user.id if user is not None else None),
        )

        cited_sources = [s for s in all_sources if s.n in cited_seen]
        chat_id, message_id = await persist_exchange(session, req.chat_id, req.message, normalized, cited_sources)

        yield sse_event("meta", {"message_id": str(message_id), "chat_id": str(chat_id), "cited": cited_seen})
        yield sse_event("done", {})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

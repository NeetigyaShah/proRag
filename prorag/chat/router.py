"""POST /chat, /chat/stream — hybrid retrieval (vector+FTS, both queries) → RRF
→ rerank → adaptive crop → [Sn] citations (§4, §5, §8 Phase 2/4).

Both endpoints share one retrieval+prompt path (`retrieve()` /
`build_prompt()`) and one persistence path (`persist_exchange()`); /chat/stream
additionally follows §5's trust rule — `sources` is *every* chunk the crop
selected (context == sources), not just the ones the model happened to cite.
"""

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.auth import current_user
from prorag.chat.citations import build_context_block, extract_cited_indices, normalize_citations, resolve_citations
from prorag.chat.stream import HEARTBEAT_INTERVAL_SECONDS, TokenGuard, sse_comment, sse_event, sse_retry
from prorag.cost import budget_decision, over_daily_cap, today_cost_usd, today_user_cost_usd, track_usage
from prorag.db import get_session
from prorag.llm import answer, answer_stream, embed_texts, estimate_tokens
from prorag.models import Chat, Citation, Feedback, Message, User
from prorag.retrieve.arms import keyword_search, structured_search, vector_search
from prorag.retrieve.crop import crop_context
from prorag.retrieve.fuse import rrf_fuse
from prorag.retrieve.plan import plan
from prorag.retrieve.rerank import rerank
from prorag.schemas import ChatRequest, ChatResponse, FeedbackRequest, Source
from prorag.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers strictly from the numbered context "
    "blocks below. Cite every claim inline immediately after the sentence it "
    "supports using bracketed numbers like [S1], [S2] — never a separate "
    "'Sources' section, never markdown links. If the context does not contain "
    "the answer, say so plainly."
)


async def retrieve(session: AsyncSession, message: str, user: User | None = None) -> list[dict]:
    """Planner → embed both queries → gather(vector×2, fts×2[, structured×2])
    → RRF → rerank top-N → adaptive crop. Shared by /chat and /search.

    `user` (#18) is threaded into every arm so the ACL predicate applies
    before LIMIT/rerank/crop ever see a row — None (auth disabled, legacy
    unscoped key, or the eval runner's service-user — see eval/runner.py)
    means unfiltered, same as today."""
    user_id = user.id if user is not None else None
    plan_result = await plan(message, session=session, user_id=user_id)
    queries = plan_result["queries"]  # [primary, alternative]
    wants_table = plan_result.get("mode") == "table"

    embeddings = await embed_texts(queries, session=session, user_id=user_id)
    # Sequential, not asyncio.gather: an AsyncSession is a single connection and
    # is not concurrency-safe. Gathering these raised InvalidRequestError
    # ("session is provisioning a new connection") whenever the session arrived
    # cold — which is exactly how GET /search calls this, since plan() and
    # embed_texts() only stage usage rows and never execute a query. /chat only
    # escaped it because check_daily_cap() happens to run a SELECT first.
    # ponytail: gathering bought nothing anyway — measured 0.81s either way,
    # because one connection serializes them regardless. Real parallelism needs
    # a session per arm, which would take ~6 pool connections per request and
    # re-create the starvation problem the commit below fixes.
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
    reranked = await rerank(queries[0], fused[: settings.rerank_top_n])
    cropped = crop_context(
        reranked,
        min_docs=settings.crop_min_docs,
        max_docs=settings.crop_max_docs,
        score_gap=settings.crop_score_gap,
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
    return cropped


def _to_source(n: int, hit: dict) -> Source:
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
    context_block = build_context_block(hits)
    return f"Context:\n{context_block}\n\nQuestion: {message}"


async def persist_exchange(
    session: AsyncSession,
    chat_id: uuid.UUID | None,
    user_message: str,
    answer_text: str,
    cited_sources: list[Source],
) -> tuple[uuid.UUID, uuid.UUID]:
    """Persists the user turn + assistant turn + one citations row per
    actually-cited source (§5.3 — the writer QDMS-AI's `cited_sources`
    column never had). Returns (chat_id, assistant_message_id)."""
    if chat_id is None:
        chat = Chat()
        session.add(chat)
        await session.flush()
        chat_id = chat.id

    session.add(Message(chat_id=chat_id, role="user", content=user_message))

    assistant_message = Message(chat_id=chat_id, role="assistant", content=answer_text)
    session.add(assistant_message)
    await session.flush()

    for s in cited_sources:
        session.add(Citation(message_id=assistant_message.id, n=s.n, doc_id=s.doc_id, page=s.page, bbox=s.bbox))

    await session.commit()
    return chat_id, assistant_message.id


async def check_daily_cap(session: AsyncSession, user: User | None = None) -> str | None:
    """Runs BEFORE any LLM call (§5.4, §8 Phase 5, #21) — install-wide check
    first, unchanged semantics: a day over budget returns 429 without spending
    another token. Shared with /eval/run, which spends one answer call per
    golden entry — and, like /chat, can overshoot the cap by exactly one call
    since the check runs before spend, not after (accepted per #9.4, not
    reservation machinery).

    Then the per-user soft/hard cap (#9's resolution, #21): `user=None` (auth
    disabled, legacy unscoped key, /eval/run's service work) skips it entirely
    — only the install-wide check above applies. Otherwise returns a warning
    string when the user is over their soft cap but under the hard multiplier
    (the caller threads it onto the response/SSE stream), or raises 429 once
    the hard cap is reached."""
    spent = await today_cost_usd(session)
    if over_daily_cap(spent, settings.daily_cost_cap_usd):
        raise HTTPException(
            429, f"daily cost cap of ${settings.daily_cost_cap_usd:.2f} reached (${spent:.2f} spent today)"
        )

    if user is None:
        return None

    cap = user.daily_cap_usd_override if user.daily_cap_usd_override is not None else settings.user_daily_cap_usd
    user_spent = await today_user_cost_usd(session, user.id)
    decision = budget_decision(user_spent, cap, settings.user_hard_cap_multiplier)

    if decision == "block":
        hard_cap = cap * settings.user_hard_cap_multiplier
        raise HTTPException(429, f"you have used ${user_spent:.2f} of ${hard_cap:.2f} today, resets at midnight UTC")
    if decision == "warn":
        return f"you have used ${user_spent:.2f} of ${cap:.2f} today, resets at midnight UTC"
    return None


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest, session: AsyncSession = Depends(get_session), user: User | None = Depends(current_user)
):
    budget_warning = await check_daily_cap(session, user)

    hits = await retrieve(session, req.message, user=user)

    user_prompt = build_prompt(req.message, hits)
    raw_answer = await answer(
        SYSTEM_PROMPT, user_prompt, session=session, user_id=(user.id if user is not None else None)
    )

    normalized = normalize_citations(raw_answer)
    resolved = resolve_citations(normalized, hits)
    sources = [_to_source(r["n"], r) for r in resolved]

    chat_id, message_id = await persist_exchange(session, req.chat_id, req.message, normalized, sources)
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
    budget_warning = await check_daily_cap(session, user)  # before the LLM is ever touched, not inside the generator

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
        hits = await retrieve(session, req.message, user=user)
        all_sources = [_to_source(i, h) for i, h in enumerate(hits, start=1)]
        yield sse_event("sources", [s.model_dump(mode="json") for s in all_sources])
        if budget_warning is not None:
            yield sse_event("budget", {"warning": budget_warning})

        user_prompt = build_prompt(req.message, hits)
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

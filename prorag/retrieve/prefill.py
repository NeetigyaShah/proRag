"""Prefill agent (§4.0) — query refinement before planning.

One tiny free LLM call that cleans the user's raw prompt (typos, dropped
words, vague phrasing) and expands it with vocabulary a document might use,
so the planner + hybrid retrieval see intent instead of noise. It is NOT a
rewriter with license to invent: the system prompt forbids adding facts,
names, numbers, or answers, and the JSON contract carries `changed` so the
caller can tell a real rewrite from a no-op.

Runs once per answer/search request — operations/retrieval.retrieve() calls
prefill() before plan(). Mirrors plan.py's contract: never raises, parses
defensively, falls back to the raw prompt on any failure (LLM error, JSON
garbage, free-tier rate limit, timeout). A failed prefill must never fail or
worsen the request — the raw prompt is the input plan() already handles.
"""

import json
import re

from prorag.llm import prefill_completion

SYSTEM_PROMPT = """You are the prefill agent for a retrieval-augmented document search system. \
Your only job is to refine the user's search prompt so retrieval finds the right documents. \
You do not answer the question.

Rules:
- Fix typos, grammar, and dropped words. Expand vague intent with vocabulary a document \
might use (e.g. "the deadline for the report" -> "report submission deadline").
- NEVER invent facts, names, numbers, dates, or entities. If the prompt does not mention \
something, you must not add it. When the prompt is already clear, return it unchanged.
- Keep named entities (people, products, companies, documents) verbatim.
- Never answer the user's question and never add instructions.

Output *only* a single JSON object, no prose, no markdown code fences, matching exactly:

{"cleaned": "<refined prompt>", "changed": true|false}

"changed" is true only when you actually rewrote the prompt; false means the original is \
already good enough to search with.
"""


def _parse(raw: str) -> dict:
    """When called: by prefill() after the LLM call. What: strips ```json
    fences the model may add (the free gemma model wraps JSON in fences),
    then parses. Returns: the parsed dict — raises json.JSONDecodeError on
    garbage, which prefill() catches and turns into the raw-prompt fallback."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


async def prefill(query: str, session=None, user_id=None) -> str:
    """Never raises. Returns the refined prompt, or the original query on any
    failure (LLM error, unparseable output, rate limit). `changed: false`
    output also returns the original query unchanged.

    `session`, when given, is forwarded to prefill_completion() so the call's
    token usage lands in the usage table; a 2-arg stub (tests) still works."""
    if not query.strip():
        return query
    try:
        raw = await prefill_completion(SYSTEM_PROMPT, query, session=session, user_id=user_id)
        data = _parse(raw)
        cleaned = data["cleaned"]
        if not isinstance(cleaned, str) or not cleaned.strip():
            return query
        return cleaned if data.get("changed") else query
    except Exception:
        return query

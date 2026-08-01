"""Two-query planner (§4.1) — one cheap-tier LLM call, JSON out.

Prompt-cache-aware split: the rulebook lives in a static SYSTEM message (never
changes between calls, so provider-side prefix caching can kick in); only
{query} varies in the USER message. Filters/is_follow_up from the full
architecture are not implemented yet (Phase 3 needs the filter compiler to
land first) — this returns just {search_needed, queries, mode}.

Parsing is defensive on purpose: planner output is LLM JSON, which means code
fences, stray prose, or occasional garbage. A parse failure must never fail
the request — it falls back to running the raw user query verbatim as both
queries.
"""

import json
import re

from prorag.llm import plan_completion

SYSTEM_PROMPT = """You are the query planner for a retrieval-augmented search system. \
Your only job is to turn a user's chat message into a JSON retrieval plan. You do not \
answer the question yourself.

Output *only* a single JSON object, no prose before or after, no markdown code fences, \
matching exactly this shape:

{"search_needed": true, "queries": ["<primary query>", "<alternative query>"], "mode": "default"}

Rules:
- "search_needed" is false only for pure greetings ("hi", "thanks") or a follow-up that is \
fully answerable from the chat history already shown to you, with no new lookup needed. \
When in doubt, set it to true — a wasted search is cheap, a missed one is not.
- "queries" always has exactly two entries, even when search_needed is false (repeat the \
original message if nothing better applies). The two entries must be genuinely different \
search strategies for the same underlying need — vocabulary-disjoint, not paraphrases. \
Example: for "how often are fire drills required", a good pair is \
["fire drill frequency requirement", "muster exercise interval SOLAS"], not \
["fire drill frequency", "frequency of fire drills"].
- "mode" is "default" unless the user is clearly asking for tabular/numeric data (counts, \
sums, rows of a table), in which case it is "table".
- Never invent facts, never answer the user's question, never add keys beyond the three shown.
"""


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    # strip ```json ... ``` or ``` ... ``` fences if the model added them anyway
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def _fallback(query: str) -> dict:
    return {"search_needed": True, "queries": [query, query], "mode": "default"}


async def plan(query: str, session=None, user_id=None) -> dict:
    """Never raises. On any planner or parse failure, degrades to running the
    raw query verbatim (search_needed=True) rather than failing the request.

    `session`, when given, is forwarded to plan_completion() so the planner
    call's cost gets tracked (§5.4), same for `user_id` (#21). Left out of the
    call entirely when session is None so tests that monkeypatch
    plan_completion with a 2-arg stub keep working."""
    try:
        if session is not None:
            raw = await plan_completion(SYSTEM_PROMPT, query, session=session, user_id=user_id)
        else:
            raw = await plan_completion(SYSTEM_PROMPT, query)
    except Exception:
        return _fallback(query)

    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, TypeError):
        return _fallback(query)

    if not isinstance(data, dict) or "queries" not in data:
        return _fallback(query)

    queries = data.get("queries") or []
    if not isinstance(queries, list) or not queries:
        return _fallback(query)
    # pad/truncate defensively to exactly two, in case the model miscounts
    queries = [str(q) for q in queries][:2]
    if len(queries) < 2:
        queries.append(queries[0])

    return {
        "search_needed": bool(data.get("search_needed", True)),
        "queries": queries,
        "mode": data.get("mode") if data.get("mode") in ("default", "table") else "default",
    }

"""Cost tracking (§5.4, §8 Phase 5) — wired into every planner/answer/embedding
call via prorag/llm.py, not the dead module QDMS-AI shipped. `litellm.completion_cost()`
first; a flat per-1k-token price fallback when the model has no litellm price entry
(local ONNX embeddings, custom LiteLLM proxy names).
"""

from datetime import UTC, datetime

import litellm
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.models import Usage
from prorag.settings import settings


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """When called: by track_usage() for every planner/answer/embedding call.
    What: prices a call via litellm.completion_cost() when the model is known
    to litellm, else the flat per-1k-token settings.fallback_price_per_1k_usd.
    Returns: the cost in USD."""
    try:
        cost = litellm.completion_cost(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        if cost is not None:
            return float(cost)
    except Exception:
        pass
    return (prompt_tokens + completion_tokens) / 1000 * settings.fallback_price_per_1k_usd


def track_usage(
    session: AsyncSession | None,
    model: str,
    prompt_tokens: int,
    completion_tokens: int = 0,
    message_id=None,
    user_id=None,
) -> float | None:
    """Adds a `usage` row to `session` (caller commits). Returns the computed
    cost, or None if no session was given — call sites without a session yet
    (e.g. a bare script) simply skip tracking instead of erroring.

    `user_id` (#21) attributes the spend for the per-user budget query below;
    None (the default) is system/service work — planner+embed calls with no
    session yet, ingest embeds, /eval/run — same "no attribution" meaning
    unscoped usage rows already had pre-#21."""
    if session is None:
        return None
    cost = compute_cost(model, prompt_tokens, completion_tokens)
    session.add(
        Usage(
            message_id=message_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            user_id=user_id,
        )
    )
    return cost


def _utc_day_start(now: datetime | None = None) -> datetime:
    """UTC midnight for `now` (defaults to current UTC time), tz-aware — split
    out so the window logic is testable without a DB (§8 Phase 5 tests)."""
    now = now or datetime.now(UTC)
    return now.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


async def today_cost_usd(session: AsyncSession) -> float:
    """Sum of usage.cost_usd since UTC midnight — a cheap query at this scale (§5.4).
    Usage.created_at is a tz-aware timestamptz (func.now()), so the comparison
    stays tz-aware end to end rather than mixing naive local time in."""
    start = _utc_day_start()
    total = (
        await session.execute(select(func.coalesce(func.sum(Usage.cost_usd), 0.0)).where(Usage.created_at >= start))
    ).scalar_one()
    return float(total)


async def today_user_cost_usd(session: AsyncSession, user_id) -> float:
    """Same UTC-window shape as today_cost_usd(), scoped to one user (#21) —
    backs the per-user soft/hard cap. Uses the same _utc_day_start() so the
    two windows can never drift apart."""
    start = _utc_day_start()
    total = (
        await session.execute(
            select(func.coalesce(func.sum(Usage.cost_usd), 0.0)).where(
                Usage.user_id == user_id, Usage.created_at >= start
            )
        )
    ).scalar_one()
    return float(total)


def over_daily_cap(today_total: float, cap: float) -> bool:
    """Pure decision logic, split out from today_cost_usd() so it's testable
    without a DB (§8 Phase 5 tests)."""
    return today_total >= cap


def budget_decision(spent: float, cap: float, hard_multiplier: float) -> str:
    """Pure per-user soft/hard cap decision (#9's resolution, #21), split out
    the same way over_daily_cap() is so it's testable without a DB. `cap` is
    already resolved (user.daily_cap_usd_override or settings.user_daily_cap_usd
    — check_daily_cap()'s job, not this function's).

    - spent < cap            -> 'ok'    (under budget)
    - cap <= spent < cap*mult -> 'warn'  (request runs, response carries a warning)
    - spent >= cap*mult       -> 'block' (429 — the accepted overshoot per #9.4
      is one answer at the soft cap, not an unbounded one past the hard cap)
    """
    if spent >= cap * hard_multiplier:
        return "block"
    if spent >= cap:
        return "warn"
    return "ok"

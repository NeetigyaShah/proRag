"""Cost tracking (§5.4, §8 Phase 5) — wired into every planner/answer/embedding
call via prorag/llm.py, not the dead module QDMS-AI shipped. `litellm.completion_cost()`
first; a flat per-1k-token price fallback when the model has no litellm price entry
(local ONNX embeddings, custom LiteLLM proxy names).
"""

from datetime import datetime, timezone

import litellm
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.models import Usage
from prorag.settings import settings


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
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
) -> float | None:
    """Adds a `usage` row to `session` (caller commits). Returns the computed
    cost, or None if no session was given — call sites without a session yet
    (e.g. a bare script) simply skip tracking instead of erroring."""
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
        )
    )
    return cost


def _utc_day_start(now: datetime | None = None) -> datetime:
    """UTC midnight for `now` (defaults to current UTC time), tz-aware — split
    out so the window logic is testable without a DB (§8 Phase 5 tests)."""
    now = now or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


async def today_cost_usd(session: AsyncSession) -> float:
    """Sum of usage.cost_usd since UTC midnight — a cheap query at this scale (§5.4).
    Usage.created_at is a tz-aware timestamptz (func.now()), so the comparison
    stays tz-aware end to end rather than mixing naive local time in."""
    start = _utc_day_start()
    total = (
        await session.execute(select(func.coalesce(func.sum(Usage.cost_usd), 0.0)).where(Usage.created_at >= start))
    ).scalar_one()
    return float(total)


def over_daily_cap(today_total: float, cap: float) -> bool:
    """Pure decision logic, split out from today_cost_usd() so it's testable
    without a DB (§8 Phase 5 tests)."""
    return today_total >= cap

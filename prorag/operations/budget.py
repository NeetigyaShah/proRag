"""Operations layer — budget / cost-cap policy (§5.4, §8 Phase 5, #21).

check_daily_cap() is the gate every spendy endpoint runs BEFORE touching the
LLM: install-wide hard cap first, then the per-user soft/hard cap. Shared by
POST /chat, POST /chat/stream and POST /eval/run — it used to live in
chat/router.py, which made the eval router depend on the chat router. The
pure decision helpers it calls (budget_decision, over_daily_cap, ...) stay in
prorag/cost.py where the cost tracking lives.
"""

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.cost import budget_decision, over_daily_cap, today_cost_usd, today_user_cost_usd
from prorag.models import User
from prorag.settings import settings


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

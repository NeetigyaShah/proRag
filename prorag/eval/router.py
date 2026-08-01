"""POST /eval/run, GET /eval/runs/{id} — golden-set evaluation (§6, §8 Phase 6).

Runs synchronously and returns the aggregate directly: there is no jobs queue
in this codebase yet (see the ponytail note in models.py), and a golden set of
~8-50 questions against a local reranker is fast enough not to need one.
Add a `jobs` row + 202 if the golden set grows large enough that this blocks
the request thread uncomfortably.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.chat.router import check_daily_cap
from prorag.db import get_session
from prorag.eval.runner import run_eval
from prorag.models import EvalRun
from prorag.schemas import EvalRunDetail, EvalRunResponse
from prorag.settings import settings

router = APIRouter()


@router.post("/eval/run", response_model=EvalRunResponse)
async def eval_run(session: AsyncSession = Depends(get_session)):
    # One eval = one planner + one embed + one answer call *per golden entry*.
    # /chat checks the daily cap before spending a token; this endpoint spends
    # N times as much and must too.
    # ponytail: checked once up front, not per question — a single run can
    # overshoot the cap by one golden set, same way /chat can overshoot by one
    # answer. Move the check inside run_eval's loop if that ever matters.
    # user=None (#21): this is service work, not a per-user request — install-
    # wide semantics only, same as before per-user budgets existed.
    await check_daily_cap(session)
    try:
        async with asyncio.timeout(settings.eval_timeout_seconds):
            result = await run_eval(session)
    except TimeoutError as exc:
        raise HTTPException(504, "eval run timed out — shrink golden.jsonl or raise eval_timeout_seconds") from exc
    return EvalRunResponse(**result)


@router.get("/eval/runs/{run_id}", response_model=EvalRunDetail)
async def get_eval_run(run_id: int, session: AsyncSession = Depends(get_session)):
    run = await session.get(EvalRun, run_id)
    if run is None:
        raise HTTPException(404, "eval run not found")
    return {
        "run_id": run.id,
        "created_at": run.created_at,
        "aggregate": run.aggregate,
        "questions": run.questions,
    }

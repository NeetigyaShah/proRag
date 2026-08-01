"""Pure-function tests for Phase 5 operations: key hashing/verification, cost
computation fallback, and the daily-cap decision logic. Mostly no DB, no LLM,
no network — the DB sum itself (today_cost_usd) is mocked out, only the pure
decision functions (over_daily_cap, budget_decision) are exercised directly,
per §8 Phase 5. The one exception is the today_user_cost_usd windowing test
(#21), which is DB-backed the same way tests/test_identity_schema.py is —
skips cleanly if the DB is unreachable."""

import asyncio
import time
import uuid
from datetime import UTC, datetime, timedelta, timezone

import litellm
import pytest
from pydantic import ValidationError
from sqlalchemy import delete, text

from prorag.auth import hash_key, new_api_key
from prorag.cost import _utc_day_start, budget_decision, compute_cost, over_daily_cap, today_user_cost_usd, track_usage
from prorag.db import SessionLocal, engine
from prorag.models import Usage, User
from prorag.schemas import ChatRequest, FeedbackRequest, IngestResponse
from prorag.settings import settings

# ---- API key hashing ---------------------------------------------------------


def test_new_api_key_is_url_safe_and_long():
    key = new_api_key()
    assert len(key) >= 32
    assert " " not in key


def test_new_api_key_is_unique_per_call():
    assert new_api_key() != new_api_key()


def test_hash_key_deterministic():
    key = "some-raw-key"
    assert hash_key(key) == hash_key(key)


def test_hash_key_differs_for_different_keys():
    assert hash_key("key-a") != hash_key("key-b")


def test_hash_key_never_returns_the_raw_key():
    key = "super-secret-value"
    assert hash_key(key) != key


# ---- cost computation fallback -----------------------------------------------


def test_compute_cost_uses_litellm_when_available(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0042)
    assert compute_cost("gpt-4o-mini", 100, 50) == 0.0042


def test_compute_cost_falls_back_when_litellm_raises(monkeypatch):
    def boom(**kw):
        raise Exception("no price entry for this model")

    monkeypatch.setattr(litellm, "completion_cost", boom)
    from prorag.settings import settings

    cost = compute_cost("local/bge-m3", 1000, 0)
    assert cost == pytest.approx(1000 / 1000 * settings.fallback_price_per_1k_usd)


def test_compute_cost_falls_back_when_litellm_returns_none(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: None)
    from prorag.settings import settings

    cost = compute_cost("local/bge-m3", 500, 500)
    assert cost == pytest.approx(1000 / 1000 * settings.fallback_price_per_1k_usd)


def test_track_usage_noop_without_session():
    assert track_usage(None, "gpt-4o-mini", 10, 5) is None


class _FakeSession:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)


def test_track_usage_adds_usage_row_with_computed_cost(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.01)
    session = _FakeSession()
    cost = track_usage(session, "gpt-4o-mini", 100, 50, message_id="msg-1")
    assert cost == 0.01
    assert len(session.added) == 1
    row = session.added[0]
    assert isinstance(row, Usage)
    assert row.model == "gpt-4o-mini"
    assert row.prompt_tokens == 100
    assert row.completion_tokens == 50
    assert row.cost_usd == 0.01
    assert row.message_id == "msg-1"


# ---- UTC cost window (issue #13: date.today() used the server's local date) --


def test_utc_day_start_is_midnight_utc_not_local():
    # 11pm on Jan 1 at UTC-5 is already Jan 2 in UTC — date.today() (local)
    # would wrongly anchor the window to Jan 1.
    now = datetime(2024, 1, 1, 23, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert _utc_day_start(now) == datetime(2024, 1, 2, 0, 0, tzinfo=UTC)


def test_utc_day_start_is_tz_aware():
    assert _utc_day_start().tzinfo is not None


# ---- daily cap decision logic -------------------------------------------------


def test_over_daily_cap_false_when_under_budget():
    assert over_daily_cap(1.23, cap=5.0) is False


def test_over_daily_cap_true_when_at_cap():
    assert over_daily_cap(5.0, cap=5.0) is True


def test_over_daily_cap_true_when_over_budget():
    assert over_daily_cap(9.99, cap=5.0) is True


def test_over_daily_cap_false_at_zero_spend():
    assert over_daily_cap(0.0, cap=5.0) is False


# ---- per-user soft/hard cap decision (#21) -------------------------------------
# `cap` here is whatever check_daily_cap() already resolved (user override or
# settings.user_daily_cap_usd) — budget_decision() itself doesn't know or care
# which; testing it at a couple of different cap values covers "override
# present" (e.g. cap=2.5) vs "override absent" (the settings default, 1.0) the
# same way, since the override is resolved before this function ever runs.


@pytest.mark.parametrize(
    "spent,cap,multiplier,expected",
    [
        (0.0, 1.0, 2.0, "ok"),  # zero spend
        (0.99, 1.0, 2.0, "ok"),  # under the soft cap
        (1.0, 1.0, 2.0, "warn"),  # exactly at the soft cap
        (1.5, 1.0, 2.0, "warn"),  # between soft and hard
        (1.99, 1.0, 2.0, "warn"),  # just under the hard cap
        (2.0, 1.0, 2.0, "block"),  # exactly at the hard cap (cap * multiplier)
        (5.0, 1.0, 2.0, "block"),  # well past the hard cap
        # same matrix again at a per-user override cap, not the settings default
        (2.49, 2.5, 2.0, "ok"),
        (2.5, 2.5, 2.0, "warn"),
        (5.0, 2.5, 2.0, "block"),
    ],
)
def test_budget_decision_matrix(spent, cap, multiplier, expected):
    assert budget_decision(spent, cap, multiplier) == expected


# ---- today_user_cost_usd UTC-midnight windowing (#21, DB-backed) --------------
# Inserts rows with an explicit created_at straddling _utc_day_start() and
# checks only the row on the "today" side of the boundary is summed. Skips
# cleanly if the DB is unreachable, same pattern as test_identity_schema.py.


async def _get_db_session():
    try:
        session = SessionLocal()
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    return session


async def test_today_user_cost_usd_only_sums_rows_since_utc_midnight():
    session = await _get_db_session()
    tag = uuid.uuid4().hex[:8]

    try:
        async with session:
            user = User(email=f"{tag}@example.com")
            session.add(user)
            await session.flush()

            start = _utc_day_start()
            just_inside = start + timedelta(seconds=1)
            just_outside = start - timedelta(seconds=1)

            session.add_all(
                [
                    Usage(
                        model="gpt-4o-mini",
                        prompt_tokens=10,
                        completion_tokens=5,
                        cost_usd=1.23,
                        user_id=user.id,
                        created_at=just_inside,
                    ),
                    Usage(
                        model="gpt-4o-mini",
                        prompt_tokens=10,
                        completion_tokens=5,
                        cost_usd=9.99,
                        user_id=user.id,
                        created_at=just_outside,
                    ),
                ]
            )
            await session.commit()

            try:
                total = await today_user_cost_usd(session, user.id)
                assert total == pytest.approx(1.23)
            finally:
                await session.execute(delete(Usage).where(Usage.user_id == user.id))
                await session.execute(delete(User).where(User.id == user.id))
                await session.commit()
    finally:
        # pytest-asyncio gives each test its own event loop; pooled connections
        # left open here would stay bound to this (now-closing) loop and blow up
        # the next DB test's checkout. Dispose so the next test gets fresh ones.
        await engine.dispose()


# ---- request schema validation ------------------------------------------------


@pytest.mark.parametrize("blank", ["   ", "\n\t ", "\xa0"])
def test_chat_request_rejects_whitespace_only_message(blank):
    """The web UI trims, but the API is the trust boundary — a direct POST of
    "   " would otherwise be embedded, sent to the LLM and billed."""
    with pytest.raises(ValidationError):
        ChatRequest(message=blank)


def test_chat_request_strips_surrounding_whitespace():
    assert ChatRequest(message="  what is the drill interval?  ").message == "what is the drill interval?"


def test_chat_request_strip_lets_padded_collection_match_the_pattern():
    # Without stripping, " docs " fails the ^[A-Za-z0-9_-]+$ pattern with a 422.
    assert ChatRequest(message="q", collection=" docs ").collection == "docs"


def test_feedback_comment_is_stripped():
    fb = FeedbackRequest(message_id=uuid.uuid4(), rating="up", comment="  helpful  ")
    assert fb.comment == "helpful"


def test_ingest_status_literal_covers_every_value_the_code_can_produce():
    """IngestResponse is a RESPONSE model, so a status outside the Literal is a
    500, not a 422. `pending` is models.py's column default and is reachable by
    any insert that doesn't set status explicitly — it must stay in the set."""
    for status in ("pending", "processing", "ready", "failed"):
        assert IngestResponse(doc_id=uuid.uuid4(), status=status).status == status


def test_ingest_status_rejects_an_unknown_value():
    with pytest.raises(ValidationError):
        IngestResponse(doc_id=uuid.uuid4(), status="halfway")


# ---- readiness probe ----------------------------------------------------------
# A saturated pool blocks the connection checkout itself for db_pool_timeout
# (30s). Without its own shorter deadline, /readyz *hangs* rather than answering,
# and an orchestrator that gives up at 5-10s sees a dead endpoint instead of a
# clean 503. The DB is faked here; only the timeout behaviour is under test.


@pytest.fixture
def readyz_client(monkeypatch):
    from fastapi.testclient import TestClient

    from prorag.db import get_session
    from prorag.main import app

    monkeypatch.setattr(settings, "readyz_timeout_seconds", 0.05)
    yield TestClient(app, raise_server_exceptions=False), app, get_session
    app.dependency_overrides.clear()


class _SlowSession:
    """Stands in for a session whose connection checkout never completes."""

    async def execute(self, *_a, **_kw):
        await asyncio.sleep(30)


class _DeadSession:
    async def execute(self, *_a, **_kw):
        raise OSError("connection refused to postgres://user:pw@host/db")


def test_readyz_returns_503_instead_of_hanging_on_a_saturated_pool(readyz_client):
    client, app, get_session = readyz_client
    app.dependency_overrides[get_session] = lambda: _SlowSession()

    started = time.monotonic()
    resp = client.get("/readyz")

    assert resp.status_code == 503
    assert resp.json() == {"status": "not ready"}
    assert time.monotonic() - started < 5, "must fail fast, not wait out db_pool_timeout"


def test_readyz_never_echoes_the_connection_string(readyz_client):
    """The probe is unauthenticated and asyncpg errors embed the DSN."""
    client, app, get_session = readyz_client
    app.dependency_overrides[get_session] = lambda: _DeadSession()

    resp = client.get("/readyz")

    assert resp.status_code == 503
    assert resp.json() == {"status": "not ready"}
    assert "postgres" not in resp.text and "pw@host" not in resp.text

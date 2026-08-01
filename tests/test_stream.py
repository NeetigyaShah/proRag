"""Pure-function tests for SSE framing + the streaming guards (§5.2, Phase 4).
No DB, no LLM, no network. One endpoint-level exception at the bottom (#21):
retrieve/check_daily_cap/answer_stream/persist_exchange are stubbed the same
way tests/test_files_endpoints.py stubs router internals, so it still touches
no DB or LLM — only /chat/stream's own event-ordering wiring is under test."""

import uuid

import pytest
from fastapi.testclient import TestClient

import prorag.chat.router as router_mod
from prorag.auth import current_user
from prorag.chat.stream import (
    TRUNCATION_NOTICE,
    TokenGuard,
    sse_comment,
    sse_event,
    sse_retry,
)
from prorag.db import get_session
from prorag.main import app
from prorag.models import User

# ---- SSE framing ------------------------------------------------------------


def test_sse_event_framing():
    frame = sse_event("token", {"t": "hi"})
    assert frame == 'event: token\ndata: {"t": "hi"}\n\n'


def test_sse_event_list_payload():
    frame = sse_event("sources", [{"n": 1}])
    assert frame.startswith("event: sources\ndata: [")
    assert frame.endswith("\n\n")


def test_sse_retry_default():
    assert sse_retry() == "retry: 3000\n\n"


def test_sse_comment_is_heartbeat():
    assert sse_comment() == ": ping\n\n"


# ---- TokenGuard: markdown-table-row buffer ----------------------------------


def test_plain_text_streams_through_immediately():
    guard = TokenGuard()
    out = "".join(guard.feed(tok) for tok in ["hel", "lo ", "world"])
    assert out == "hello world"


def test_table_row_held_until_newline():
    guard = TokenGuard()
    held = guard.feed("| a | b")
    assert held == ""  # nothing emitted yet — mid table row, no newline
    released = guard.feed(" |\n")
    assert released == "| a | b |\n"


def test_table_row_split_across_many_feeds():
    guard = TokenGuard()
    chunks = ["|", " col1", " | col2", " |", "\n"]
    out = "".join(guard.feed(c) for c in chunks)
    assert out == "| col1 | col2 |\n"


def test_non_table_line_after_table_row_streams_normally():
    guard = TokenGuard()
    out = guard.feed("| a | b |\nplain text")
    assert out == "| a | b |\nplain text"


def test_flush_emits_unterminated_table_row():
    guard = TokenGuard()
    guard.feed("| unfinished row")
    assert guard.flush() == "| unfinished row"
    assert guard.flush() == ""  # idempotent after flush


# ---- TokenGuard: whitespace-runaway abort -----------------------------------


def test_whitespace_runaway_truncates():
    guard = TokenGuard(whitespace_limit=10)
    out = guard.feed("hello" + " " * 20)
    assert guard.done is True
    assert out.startswith("hello")
    assert TRUNCATION_NOTICE in out


def test_whitespace_run_resets_on_non_whitespace():
    guard = TokenGuard(whitespace_limit=5)
    guard.feed("a   b   c   d")  # never 5 consecutive whitespace chars
    assert guard.done is False


def test_guard_stops_emitting_after_done():
    guard = TokenGuard(whitespace_limit=3)
    guard.feed("   ")  # trips the guard
    assert guard.done is True
    assert guard.feed("more text") == ""


def test_whitespace_run_spans_across_feed_calls():
    guard = TokenGuard(whitespace_limit=6)
    guard.feed("  ")
    guard.feed("  ")
    out = guard.feed("  ")  # 6th consecutive space across three feed() calls
    assert guard.done is True
    assert TRUNCATION_NOTICE in out


# ---- /chat/stream: budget event ordering (#21) --------------------------------
# The warning has to land after `sources` (the trust-rule event the UI renders
# first) and before the first `token`, so a client can show it alongside the
# sources it's already committed to displaying rather than mid-answer.


class _NoopSession:
    """Nothing under test ever needs a real row — persist_exchange and
    track_usage's caller (chat_stream) are exercised, but both DB-touching
    calls are stubbed out below."""

    async def commit(self):
        pass

    def add(self, _obj):
        pass


async def _stub_retrieve(_session, _message, user=None):
    return [
        {
            "doc_id": uuid.uuid4(),
            "text": "hello world",
            "score": 1.0,
            "page": None,
            "title": None,
            "kind": None,
            "bbox": None,
        }
    ]


async def _stub_cap_with_warning(_session, _user=None):
    return "you have used $1.50 of $1.00 today, resets at midnight UTC"


async def _stub_answer_stream(_system, _user):
    yield ("answer", "hi")


async def _stub_persist_exchange(_session, _chat_id, _user_message, _answer_text, _cited_sources):
    return uuid.uuid4(), uuid.uuid4()


@pytest.fixture
def stream_client(monkeypatch):
    monkeypatch.setattr(router_mod, "retrieve", _stub_retrieve)
    monkeypatch.setattr(router_mod, "check_daily_cap", _stub_cap_with_warning)
    monkeypatch.setattr(router_mod, "answer_stream", _stub_answer_stream)
    monkeypatch.setattr(router_mod, "persist_exchange", _stub_persist_exchange)

    app.dependency_overrides[get_session] = lambda: _NoopSession()
    app.dependency_overrides[current_user] = lambda: User(id=uuid.uuid4(), email="u@example.com")
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def test_chat_stream_emits_budget_event_after_sources_before_first_token(stream_client):
    event_names = []
    with stream_client.stream("POST", "/chat/stream", json={"message": "hi"}) as resp:
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event_names.append(line.removeprefix("event: "))

    assert event_names.index("budget") == event_names.index("sources") + 1
    assert event_names.index("budget") < event_names.index("token")

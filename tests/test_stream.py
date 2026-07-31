"""Pure-function tests for SSE framing + the streaming guards (§5.2, Phase 4).
No DB, no LLM, no network."""

from prorag.chat.stream import (
    TRUNCATION_NOTICE,
    TokenGuard,
    sse_comment,
    sse_event,
    sse_retry,
)

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

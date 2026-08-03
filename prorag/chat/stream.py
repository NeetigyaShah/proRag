"""SSE framing + streaming hardening (§5.2, Phase 4).

Pure, no I/O: `TokenGuard` is fed raw text deltas from the LLM stream and
decides what's safe to emit right now. Two guards, carried over verbatim
from the ancestor project:

- markdown-table-row buffer: a line starting with `|` is held (nothing
  emitted) until its terminating `\\n`, so the UI never renders a half-drawn
  table row.
- whitespace-runaway abort: 200 *consecutive* whitespace chars (spans lines/
  buffers) truncates the stream with a notice instead of hanging on a
  degenerate completion.

`sse_event`/`sse_comment` do the actual `text/event-stream` framing.
"""

import json

WHITESPACE_RUNAWAY_LIMIT = 200
HEARTBEAT_INTERVAL_SECONDS = 15
TRUNCATION_NOTICE = "\n\n_[response truncated: repeated whitespace]_"


def sse_event(event: str, data: dict | list) -> str:
    """When called: by chat/router.py's /chat/stream generator for every SSE
    frame it emits — sources, token, citation, budget, meta, error, done.
    What: frames an event in text/event-stream format with JSON-encoded data.
    Returns: the raw "event: <name>\ndata: <json>\n\n" payload string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sse_retry(ms: int = 3000) -> str:
    """When called: once at the top of the /chat/stream response, before any
    event, telling the browser how long to wait before reconnecting after a
    dropped connection. What: emits the SSE retry directive. Returns: the
    "retry: <ms>\n\n" frame string."""
    return f"retry: {ms}\n\n"


def sse_comment(text: str = "ping") -> str:
    """SSE comment line — used for the heartbeat. Ignored by EventSource,
    keeps idle proxies from closing the connection."""
    return f": {text}\n\n"


class TokenGuard:
    """Feed raw text deltas in; get back the text that's safe to flush now."""

    def __init__(self, whitespace_limit: int = WHITESPACE_RUNAWAY_LIMIT):
        self.whitespace_limit = whitespace_limit
        self.pending = ""  # unflushed chars of the current line
        self.ws_run = 0
        self.done = False  # True once the whitespace-runaway guard fired

    def feed(self, token: str) -> str:
        """When called: once per raw text delta from the LLM stream, in order,
        by chat/router.py's stream loop. What: holds a line while it looks
        like an in-progress markdown table row (flushed whole at its '\n'),
        and truncates the stream with a notice after 200 consecutive
        whitespace chars; anything else is emitted immediately. Returns: the
        substring of `token` safe to flush now ("" when all of it is still
        buffered)."""
        if self.done or not token:
            return ""
        out = []
        for ch in token:
            self.pending += ch
            self.ws_run = self.ws_run + 1 if ch.isspace() else 0

            if self.ws_run >= self.whitespace_limit:
                out.append(self.pending)
                out.append(TRUNCATION_NOTICE)
                self.pending = ""
                self.done = True
                break

            if ch == "\n":
                out.append(self.pending)
                self.pending = ""
            elif not self.pending.lstrip().startswith("|"):
                # Not (or not yet) a table row — safe to stream immediately.
                out.append(self.pending)
                self.pending = ""
            # else: mid table-row, hold — flushed whole on the next '\n'.
        return "".join(out)

    def flush(self) -> str:
        """Call once the source stream ends, to emit any row still held."""
        rem, self.pending = self.pending, ""
        return rem

"""Pure-ASGI middleware (§7, §9): gzip that skips `text/event-stream`, and a
request-timing header. Written at the raw ASGI level on purpose —
`BaseHTTPMiddleware`-based gzip genuinely buffers a streaming response to
completion before the client sees a byte, which would defeat SSE entirely.
"""

import gzip
import logging
import time

logger = logging.getLogger("prorag.request")


class GZipSkipSSEMiddleware:
    """Buffers and gzips the response body in one shot (fine at this app's
    response sizes) unless the response is `text/event-stream` or the client
    didn't send `Accept-Encoding: gzip` — those pass through untouched."""

    def __init__(self, app, minimum_size: int = 500):
        self.app = app
        self.minimum_size = minimum_size

    async def __call__(self, scope, receive, send):
        """When called: by the ASGI server on every request through this
        middleware. What: passes non-HTTP scopes and non-gzip clients straight
        through; otherwise buffers the response and gzips it (skipping
        `text/event-stream`, see class docstring)."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        accept_encoding = next((v for k, v in scope.get("headers", []) if k == b"accept-encoding"), b"")
        if b"gzip" not in accept_encoding:
            await self.app(scope, receive, send)
            return

        start_message = {}
        body_chunks: list[bytes] = []
        skip = False

        async def send_wrapper(message):
            nonlocal start_message, skip
            if message["type"] == "http.response.start":
                content_type = next((v for k, v in message.get("headers", []) if k == b"content-type"), b"")
                if content_type.startswith(b"text/event-stream"):
                    skip = True
                    await send(message)
                    return
                start_message = message
                return  # held until the body is fully assembled below
            if skip:
                await send(message)
                return

            body_chunks.append(message.get("body", b""))
            if message.get("more_body", False):
                return

            body = b"".join(body_chunks)
            headers = [(k, v) for k, v in start_message.get("headers", []) if k.lower() != b"content-length"]
            if len(body) >= self.minimum_size:
                body = gzip.compress(body)
                headers.append((b"content-encoding", b"gzip"))
            headers.append((b"content-length", str(len(body)).encode()))
            await send({**start_message, "headers": headers})
            await send({"type": "http.response.body", "body": body, "more_body": False})

        await self.app(scope, receive, send_wrapper)


class RequestTimingMiddleware:
    """Adds `X-Response-Time-Ms` to every response and logs one line per
    request (method, path, status, duration) — the request-logging half of
    §8 Phase 5, piggybacked on the timing middleware that already wraps
    every response (§7)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """When called: by the ASGI server on every request through this
        middleware. What: times each HTTP request, stamps the response with
        `X-Response-Time-Ms`, and logs one line per request; non-HTTP scopes
        pass through."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.monotonic()
        status_holder = {}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                elapsed_ms = (time.monotonic() - start) * 1000
                status_holder["status"] = message["status"]
                status_holder["elapsed_ms"] = elapsed_ms
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-response-time-ms", f"{elapsed_ms:.1f}".encode()),
                ]
            await send(message)

        await self.app(scope, receive, send_wrapper)
        logger.info(
            "%s %s %s %.1fms",
            scope.get("method"),
            scope.get("path"),
            status_holder.get("status"),
            status_holder.get("elapsed_ms", 0.0),
        )

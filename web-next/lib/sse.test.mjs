// node --test --experimental-strip-types lib/sse.test.mjs   (run from web-next/)
// Covers the framing subset and, mainly, the idle-timeout path: a stream that
// goes quiet must reject instead of hanging the UI forever.
import assert from "node:assert/strict";
import test from "node:test";

import { IDLE_TIMEOUT_MS, SseStalledError, parseFrame, streamSse } from "./sse.ts";

const IDLE_MS = 60; // test-only deadline; production default is IDLE_TIMEOUT_MS

// Mimics real fetch: aborting the signal errors the response body, so a pending
// reader.read() rejects. A mock that leaves the body detached from the signal
// would hang here and prove nothing.
function sseResponse(chunks, { stallForever = false, signal } = {}) {
  return {
    ok: true,
    status: 200,
    body: new ReadableStream({
      start(controller) {
        for (const c of chunks) controller.enqueue(new TextEncoder().encode(c));
        // stallForever: never close — the dead-connection case.
        if (!stallForever) return controller.close();
        signal?.addEventListener("abort", () => {
          try {
            controller.error(signal.reason);
          } catch {
            /* already closed */
          }
        });
      },
    }),
  };
}

test("production deadline is a multiple of the server's 15s heartbeat", () => {
  assert.equal(IDLE_TIMEOUT_MS, 45_000);
});

test("parses event + json data frames", () => {
  assert.deepEqual(parseFrame('event: token\ndata: {"t":"hi"}'), { event: "token", data: { t: "hi" } });
});

test("heartbeat comments and retry lines carry no data", () => {
  assert.equal(parseFrame(": ping").data, null);
  assert.equal(parseFrame("retry: 3000").data, null);
});

test("delivers frames split across chunk boundaries", async () => {
  globalThis.fetch = async () =>
    sseResponse(['event: token\ndata: {"t":"a', '"}\n\nevent: done\ndata: {}\n\n']);
  const seen = [];
  await streamSse("/x", {}, (f) => seen.push(f));
  assert.deepEqual(seen, [
    { event: "token", data: { t: "a" } },
    { event: "done", data: {} },
  ]);
});

test("a stream that goes quiet rejects instead of hanging", async () => {
  const seen = [];
  globalThis.fetch = async (_u, init) =>
    sseResponse(['event: token\ndata: {"t":"a"}\n\n'], { stallForever: true, signal: init.signal });
  await assert.rejects(streamSse("/x", {}, (f) => seen.push(f), undefined, IDLE_MS), SseStalledError);
  // The frames that did arrive were still delivered before the deadline fired.
  assert.deepEqual(seen, [{ event: "token", data: { t: "a" } }]);
});

test("a backend that never sends headers also rejects", async () => {
  globalThis.fetch = async (_u, init) =>
    new Promise((_resolve, reject) => init.signal.addEventListener("abort", () => reject(init.signal.reason)));
  await assert.rejects(streamSse("/x", {}, () => {}, undefined, IDLE_MS), SseStalledError);
});

test("user abort stays an AbortError, not a stall", async () => {
  const ctl = new AbortController();
  globalThis.fetch = async (_u, init) =>
    new Promise((_resolve, reject) => init.signal.addEventListener("abort", () => reject(init.signal.reason)));
  const p = streamSse("/x", {}, () => {}, ctl.signal, IDLE_MS);
  ctl.abort();
  await assert.rejects(p, (e) => e.name === "AbortError");
});

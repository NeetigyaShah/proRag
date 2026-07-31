// Minimal SSE frame parser + streaming reader for POST /api/chat/stream.
//
// Deliberately a *subset* of the SSE spec, matched to the only server it ever
// talks to: prorag/chat/stream.py's sse_event() emits
// "event: <name>\ndata: <json.dumps(...)>\n\n". json.dumps escapes newlines, so
// `data` is always exactly one line and always JSON — multi-line `data:` frames
// and \r\n terminators can't reach us. Comment (":") and "retry:" lines are
// skipped. If the backend ever streams non-JSON or multi-line data, this parser
// needs the full field-accumulation rules first.

export interface SseFrame {
  event: string;
  data: unknown;
}

// The server heartbeats a ":" comment every 15s (HEARTBEAT_INTERVAL_SECONDS)
// precisely so a dead connection is detectable. Without a client-side deadline
// that heartbeat does nothing: a silently dropped TCP connection (sleeping
// laptop, proxy idle-kill, killed backend) leaves reader.read() pending
// forever, and the UI sits in "streaming" with no error and no completion.
// 3 missed heartbeats = dead.
export const IDLE_TIMEOUT_MS = 45_000;

export class SseStalledError extends Error {
  constructor(ms: number) {
    super(`No data received for ${ms / 1000}s`);
    this.name = "SseStalledError";
  }
}

export function parseFrame(frame: string): SseFrame {
  let event = "message";
  let data: unknown = null;
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) {
      try {
        data = JSON.parse(line.slice(5).trim());
      } catch {
        // ignore malformed data lines
      }
    }
  }
  return { event, data };
}

export async function streamSse(
  url: string,
  body: unknown,
  onFrame: (frame: SseFrame) => void,
  signal?: AbortSignal,
  idleTimeoutMs: number = IDLE_TIMEOUT_MS, // overridable so tests don't wait 45s
) {
  const idle = new AbortController();
  let idleTimer = setTimeout(() => idle.abort(), idleTimeoutMs);
  const bump = () => {
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => idle.abort(), idleTimeoutMs);
  };

  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      // Covers the pre-headers stall too: a backend that accepts the socket and
      // never replies would otherwise hang here, before any frame exists.
      signal: signal ? AbortSignal.any([signal, idle.signal]) : idle.signal,
    });
    if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

    reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bump();
      buf += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const parsed = parseFrame(frame);
        if (parsed.data === null) continue;
        onFrame(parsed);
      }
    }
  } catch (err) {
    // The idle abort surfaces as a generic AbortError, which the caller would
    // otherwise read as "user pressed Stop" and show as a clean stop.
    if (idle.signal.aborted && !signal?.aborted) throw new SseStalledError(idleTimeoutMs);
    throw err;
  } finally {
    clearTimeout(idleTimer);
    // cancel(), not releaseLock() — releasing the lock on a still-flowing body
    // leaves the connection open. Matters when onFrame throws mid-stream.
    reader?.cancel().catch(() => {});
  }
}

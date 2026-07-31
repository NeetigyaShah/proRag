"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TopBar } from "@/components/top-bar";
import { MessageList } from "@/components/message-list";
import { ChatComposer } from "@/components/chat-composer";
import { PdfViewer } from "@/components/pdf-viewer";
import { SseStalledError, streamSse } from "@/lib/sse";
import { cn } from "@/lib/utils";
import type { ChatMessage, Source, Stats } from "@/lib/types";

let idCounter = 0;
const nextId = () => `m${++idCounter}`;

// --- Typewriter pacing (characters per second) -------------------------------
// Rate scales with how long the ANSWER is, not with how fast the network
// delivered it: an answer up to TYPE_SHORT_CHARS reveals at a steady
// TYPE_MIN_CPS, and anything longer eases up toward TYPE_LONG_CPS by
// TYPE_LONG_CHARS so a wall of text doesn't crawl. TYPE_RUNAWAY_CHARS is a
// safety valve for very large backlogs only — it must not fire on short
// answers, or it would defeat the 20 cps floor.
const TYPE_MIN_CPS = 20;
const TYPE_LONG_CPS = 110;
const TYPE_MAX_CPS = 400;
const TYPE_SHORT_CHARS = 400;
const TYPE_LONG_CHARS = 2500;
const TYPE_RUNAWAY_CHARS = 1500;
const TYPE_RUNAWAY_SECONDS = 4;

export default function Home() {
  const [documents, setDocuments] = useState<number | null>(null);
  const [ingesting, setIngesting] = useState<string | null>(null);
  const [lastIngestResult, setLastIngestResult] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [viewerSource, setViewerSource] = useState<Source | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const chatIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottomRef = useRef(true);

  // Auto-scroll: follow the stream while the user is at (or near) the bottom;
  // stop following the moment they scroll up to read, resume when they return.
  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }, []);

  useEffect(() => {
    if (stickToBottomRef.current) {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
    }
  }, [messages]);

  // Typewriter: SSE tokens land in a target buffer; a rAF loop reveals them at
  // a chars-per-SECOND rate (the previous version advanced per frame, which
  // made the real speed depend on the display's refresh rate).
  const targetRef = useRef("");
  const shownRef = useRef(0); // float — fractional chars carry across frames
  const lastTsRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const patchRef = useRef<((fn: (m: ChatMessage) => ChatMessage) => void) | null>(null);
  const doneResolverRef = useRef<(() => void) | null>(null);
  const ingestTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ingestingRef = useRef(false);

  // Unmount cleanup: cancel the rAF loop, abort any live stream, drop the
  // ingest-message timer, and release a pending typewriter waiter. Without
  // this, navigating away leaves work running that then sets state on a dead
  // component.
  useEffect(
    () => () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      if (ingestTimerRef.current) clearTimeout(ingestTimerRef.current);
      abortRef.current?.abort();
      doneResolverRef.current?.();
      doneResolverRef.current = null;
    },
    [],
  );

  const drain = useCallback((ts: number) => {
    rafRef.current = null;
    const target = targetRef.current;
    const shown = shownRef.current;
    if (shown >= target.length) {
      lastTsRef.current = 0;
      doneResolverRef.current?.();
      doneResolverRef.current = null;
      return;
    }

    // dt is clamped so a backgrounded tab doesn't dump the whole answer at once.
    const dt = lastTsRef.current ? Math.min((ts - lastTsRef.current) / 1000, 0.25) : 0;
    lastTsRef.current = ts;

    const backlog = target.length - shown;
    // Scale on the answer's own length: <=TYPE_SHORT_CHARS stays at the 20 cps
    // floor, then eases toward TYPE_LONG_CPS as the answer gets longer.
    const ramp = Math.min(
      1,
      Math.max(0, target.length - TYPE_SHORT_CHARS) / (TYPE_LONG_CHARS - TYPE_SHORT_CHARS),
    );
    let cps = TYPE_MIN_CPS + ramp * (TYPE_LONG_CPS - TYPE_MIN_CPS);
    // Safety valve for a genuinely huge backlog only (never on short answers).
    if (backlog > TYPE_RUNAWAY_CHARS) {
      cps = Math.max(cps, backlog / TYPE_RUNAWAY_SECONDS);
    }
    cps = Math.min(cps, TYPE_MAX_CPS);

    const next = Math.min(target.length, shown + cps * dt);
    shownRef.current = next;
    patchRef.current?.((m) => ({
      ...m,
      content: target.slice(0, Math.floor(next)),
      status: "streaming",
    }));
    rafRef.current = requestAnimationFrame(drain);
  }, []);

  const pushText = useCallback(
    (t: string) => {
      targetRef.current += t;
      if (rafRef.current === null) {
        lastTsRef.current = 0; // restarting: don't count idle time as elapsed
        rafRef.current = requestAnimationFrame(drain);
      }
    },
    [drain],
  );

  /** Resolves once every buffered character has been revealed. Uses an explicit
   *  resolver that drain() fires — a polling loop would spin forever if the
   *  invariant ever broke or the component unmounted mid-stream. */
  const waitForTypewriter = useCallback(
    () =>
      new Promise<void>((resolve) => {
        if (shownRef.current >= targetRef.current.length) {
          resolve();
          return;
        }
        doneResolverRef.current = resolve;
      }),
    [],
  );

  /** Skip the animation and show everything now (stop button / errors). */
  const finishTypewriter = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    lastTsRef.current = 0;
    shownRef.current = targetRef.current.length;
    doneResolverRef.current?.(); // release any waiter, or it hangs forever
    doneResolverRef.current = null;
  }, []);

  const refreshStats = useCallback(async () => {
    try {
      const resp = await fetch("/api/stats");
      if (resp.ok) {
        const data: Stats = await resp.json();
        setDocuments(data.documents);
      }
    } catch {
      // stats are non-critical; leave last known value in place
    }
  }, []);

  useEffect(() => {
    refreshStats();
  }, [refreshStats]);

  const ingest = useCallback(
    async (file: File) => {
      // Guarded here, not on the Upload button: drag-and-drop is a second
      // entry point, so a `disabled` button alone still lets you drop a file
      // mid-ingest. A ref, not the `ingesting` state — state updates are
      // async, so two calls in the same tick would both read the old value.
      if (ingestingRef.current) return;
      ingestingRef.current = true;
      setIngesting(file.name);
      setLastIngestResult(null);
      const form = new FormData();
      form.append("file", file);
      try {
        const resp = await fetch("/api/ingest", { method: "POST", body: form });
        setLastIngestResult(resp.ok ? `Added ${file.name}` : `Could not ingest ${file.name}`);
        if (resp.ok) refreshStats();
      } catch {
        setLastIngestResult(`Could not ingest ${file.name}`);
      } finally {
        ingestingRef.current = false;
        setIngesting(null);
        // Track the timer so a second upload doesn't have its message cleared
        // early by the first upload's pending timeout.
        if (ingestTimerRef.current) clearTimeout(ingestTimerRef.current);
        ingestTimerRef.current = setTimeout(() => setLastIngestResult(null), 4000);
      }
    },
    [refreshStats],
  );

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || streaming) return;
    setInput("");

    const userMsg: ChatMessage = { id: nextId(), role: "user", content: text };
    const assistantMsg: ChatMessage = {
      id: nextId(),
      role: "assistant",
      content: "Reading your documents",
      status: "thinking",
      sources: [],
      citedNs: [],
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;
    let raw = "";
    let sources: Source[] = [];
    const citedNs: number[] = [];
    let failed = false;

    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantMsg.id ? fn(m) : m)));
    patchRef.current = patch;
    targetRef.current = "";
    shownRef.current = 0;
    stickToBottomRef.current = true;

    try {
      await streamSse(
        "/api/chat/stream",
        { message: text, chat_id: chatIdRef.current },
        ({ event, data }) => {
          if (event === "sources") {
            sources = data as Source[];
            patch((m) => ({ ...m, sources }));
          } else if (event === "token") {
            raw += (data as { t: string }).t;
            pushText((data as { t: string }).t);
          } else if (event === "citation") {
            const n = (data as { n: number }).n;
            if (!citedNs.includes(n)) citedNs.push(n);
            patch((m) => ({ ...m, citedNs: [...citedNs] }));
          } else if (event === "error") {
            failed = true;
            patch((m) => ({ ...m, status: "error", content: (data as { message: string }).message }));
          } else if (event === "meta") {
            chatIdRef.current = (data as { chat_id: string }).chat_id;
          }
        },
        controller.signal,
      );
      // The network stream is done, but the typewriter may still be revealing
      // buffered text — let it finish at its own pace instead of dumping the
      // rest at once, then mark the message done.
      if (failed) {
        finishTypewriter();
      } else {
        await waitForTypewriter();
        // A stream can close having sent sources but no tokens — say so rather
        // than leaving an empty assistant bubble.
        patch((m) => ({
          ...m,
          content: raw || "No answer was returned for that question.",
          status: raw ? "done" : "error",
        }));
      }
    } catch (err) {
      finishTypewriter();
      const aborted = err instanceof DOMException && err.name === "AbortError";
      const stalled = err instanceof SseStalledError;
      // Raw provider/transport errors can carry URLs and internals — log them,
      // show the user something plain.
      if (!aborted) console.error("chat stream failed", err);
      patch((m) => ({
        ...m,
        status: aborted ? "done" : "error",
        content: aborted
          ? raw || "Stopped."
          : stalled
            ? "The connection went quiet. Try asking again."
            : "Something went wrong. Try asking again.",
      }));
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }, [input, streaming, pushText, finishTypewriter, waitForTypewriter]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const hasStarted = messages.length > 0;

  return (
    <div
      className="flex flex-1 flex-col overflow-hidden"
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer.files?.[0];
        if (file) ingest(file);
      }}
    >
      <TopBar
        documents={documents}
        ingesting={ingesting}
        lastIngestResult={lastIngestResult}
        onUpload={ingest}
      />

      <div className="flex flex-1 overflow-hidden relative">
        {dragOver && (
          <div className="absolute inset-0 z-10 flex items-center justify-center border-2 border-dashed border-amber/50 bg-background/80 backdrop-blur-sm">
            <p className="text-sm font-medium text-amber">Drop to ingest</p>
          </div>
        )}

        {/* When a source is open, the PDF takes the main stage and the whole
            chat column (messages + composer) docks to the right. */}
        {viewerSource && (
          <div className="flex-1 min-w-0 hidden md:block">
            <PdfViewer
              // Key on the DOCUMENT only: keying on page/citation remounted the
              // viewer (and re-downloaded + re-parsed the whole PDF) every time
              // you clicked a different citation in the same file.
              key={viewerSource.doc_id}
              source={viewerSource}
              onClose={() => setViewerSource(null)}
            />
          </div>
        )}

        <div
          className={cn(
            "flex flex-col overflow-hidden transition-[width] duration-300",
            viewerSource ? "w-full md:w-[26rem] md:shrink-0 md:border-l md:border-border" : "flex-1",
          )}
        >
          {hasStarted && (
            <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto scrollbar-thin">
              <MessageList messages={messages} onCite={setViewerSource} />
            </div>
          )}

          <ChatComposer
            value={input}
            onChange={setInput}
            onSubmit={sendMessage}
            loading={streaming}
            onStop={stop}
            docked={hasStarted}
          />
        </div>
      </div>
    </div>
  );
}

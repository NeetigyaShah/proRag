"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { TopBar } from "@/components/top-bar";
import { MessageList } from "@/components/message-list";
import { ChatComposer } from "@/components/chat-composer";
import { PdfViewer } from "@/components/pdf-viewer";
import { RagMonsterModal } from "@/components/rag-monster-modal";
import { SseStalledError, streamSse } from "@/lib/sse";
import { fxDuration, useFxSettings } from "@/lib/fx-settings";
import { cn } from "@/lib/utils";
import {
  THINKING_PHASES,
  type ChatMessage,
  type MonsterIngestState,
  type Source,
  type Stats,
  type ThinkingMeta,
  type ViewerTarget,
} from "@/lib/types";

let idCounter = 0;
const nextId = () => `m${++idCounter}`;

// Named handles so refs don't publish ReturnType<typeof setTimeout/setInterval>
// contracts at each use site.
type TimerHandle = ReturnType<typeof setTimeout>;
type IntervalHandle = ReturnType<typeof setInterval>;

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

// --- Arcade monster ingestion pacing ----------------------------------------
// Base time per chomped page at 1x; scaled by the shared FX speed setting.
const PAGE_CHOMP_MS = 900;
// Beat between the `sources` SSE event landing (hybrid search done) and the
// sonar reranking phase completing — gives the drawer a believable cadence.
const RERANK_PHASE_MS = 650;

export default function Home() {
  const [documents, setDocuments] = useState<number | null>(null);
  const [ingesting, setIngesting] = useState<string | null>(null);
  const [lastIngestResult, setLastIngestResult] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [viewer, setViewer] = useState<ViewerTarget | null>(null);
  const [dragOver, setDragOver] = useState(false);
  // Like/dislike per message id, mirroring the backend's toggle semantics:
  // null = no vote, otherwise the active vote (posting it again removes it).
  const [ratings, setRatings] = useState<Record<string, "up" | "down" | null>>({});
  // Arcade monster ingestion overlay (PDF chomping). null = hidden.
  const [monster, setMonster] = useState<MonsterIngestState | null>(null);
  const { speed: fxSpeed, setSpeed: setFxSpeed, muted, setMuted } = useFxSettings();

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
  const ingestTimerRef = useRef<TimerHandle | undefined>(undefined);
  const ingestingRef = useRef(false);
  const ingestAbortRef = useRef<AbortController | null>(null);
  const chompTimerRef = useRef<IntervalHandle | undefined>(undefined);
  // Phase-2 (reranking) beat after `sources` lands; cleared on abort/unmount.
  const rerankTimerRef = useRef<TimerHandle | undefined>(undefined);
  const elapsedTimerRef = useRef<IntervalHandle | undefined>(undefined);
  const elapsedStartRef = useRef(0);

  // Unmount cleanup: cancel the rAF loop, abort any live stream, drop the
  // ingest-message timer, and release a pending typewriter waiter. Without
  // this, navigating away leaves work running that then sets state on a dead
  // component.
  useEffect(
    () => () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      clearTimeout(ingestTimerRef.current);
      clearTimeout(rerankTimerRef.current);
      clearInterval(elapsedTimerRef.current);
      clearInterval(chompTimerRef.current);
      abortRef.current?.abort();
      ingestAbortRef.current?.abort();
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

  /** Page count for the monster conveyor. PDFs are parsed client-side with
   *  pdf.js (same worker wiring as PdfViewer); anything else chomps a single
   *  stylized page. If pdf.js is slow or wedged (dev-bundle chunk hiccups),
   *  fall back to a cheap /Count scan of the PDF header, then to 1 page —
   *  the monster must never block the upload on telemetry. */
  const pdfPageCount = useCallback(async (file: File): Promise<number> => {
    if (!file.name.toLowerCase().endsWith(".pdf")) return 1;
    try {
      // Lazy import on purpose: pdf.js is ~1MB and only PDF uploads need page
      // counts — same bundle-weight call PdfViewer makes when opening a file.
      // The whole pdf.js path (import + getDocument + worker spin-up) is raced
      // against a timeout: in dev a wedged chunk or worker can hang any stage,
      // and the monster must never block the upload on telemetry.
      const pages = await Promise.race([
        (async () => {
          const pdfjsLib = await import("pdfjs-dist");
          pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
            "pdfjs-dist/build/pdf.worker.min.mjs",
            import.meta.url,
          ).toString();
          const doc = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
          return doc.numPages;
        })(),
        new Promise<never>((_, reject) => setTimeout(() => reject(new Error("pdfjs parse timed out")), 4000)),
      ]);
      return pages;
    } catch {
      try {
        // Header scan: the Pages dict's /Count sits in the first KBs of most
        // PDFs — good enough to feed the conveyor when pdf.js is unavailable.
        const head = await file.slice(0, 8192).text();
        const m = head.match(/\/Count\s+(\d+)/);
        if (m) return Math.max(1, Number(m[1]));
      } catch {
        /* fall through to the single-page default */
      }
      return 1;
    }
  }, []);

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
      const totalPages = await pdfPageCount(file);
      setMonster({ filename: file.name, totalPages, currentPage: 0, status: "crunching", speed: fxSpeed });
      const form = new FormData();
      form.append("file", file);
      const controller = new AbortController();
      ingestAbortRef.current = controller;
      try {
        const resp = await fetch("/api/ingest", { method: "POST", body: form, signal: controller.signal });
        if (resp.ok) {
          setLastIngestResult(`Added ${file.name}`);
          refreshStats();
          setMonster((m) => (m ? { ...m, currentPage: m.totalPages, status: "done" } : m));
        } else {
          // Backend rejected the file (bad PDF, size limit…): drop the modal,
          // the top-bar badge carries the error.
          setLastIngestResult(`Could not ingest ${file.name}`);
          setMonster(null);
        }
      } catch (err) {
        const aborted = err instanceof DOMException && err.name === "AbortError";
        if (aborted) {
          setLastIngestResult(`Cancelled ${file.name}`);
          setMonster((m) => (m ? { ...m, status: "cancelled" } : m));
        } else {
          // Real failure: drop the modal (the top-bar badge shows the error)
          // instead of leaving the monster in a misleading "cancelled" state.
          setLastIngestResult(`Could not ingest ${file.name}`);
          setMonster(null);
        }
      } finally {
        ingestAbortRef.current = null;
        // Done/cancelled monsters bow out after their victory/defeat beat.
        setTimeout(() => setMonster(null), 1400);
        ingestingRef.current = false;
        setIngesting(null);
        // Track the timer so a second upload doesn't have its message cleared
        // early by the first upload's pending timeout.
        clearTimeout(ingestTimerRef.current);
        ingestTimerRef.current = setTimeout(() => setLastIngestResult(null), 4000);
      }
    },
    [refreshStats, fxSpeed, pdfPageCount],
  );

  /** Chomp timer: advances the monster one page per interval. Speed changes
   *  (slider anywhere in the UI) restart it. The interval self-guards on
   *  status via the functional update — no state is set in the effect body,
   *  and "instant" collapses the period to the 16ms floor (a visible turbo
   *  cascade instead of a jump). */
  useEffect(() => {
    clearInterval(chompTimerRef.current);
    chompTimerRef.current = setInterval(() => {
      setMonster((m) =>
        m && m.status === "crunching" && m.currentPage < m.totalPages
          ? { ...m, currentPage: m.currentPage + 1 }
          : m,
      );
    }, Math.max(16, fxDuration(fxSpeed, PAGE_CHOMP_MS)));
    return () => {
      clearInterval(chompTimerRef.current);
      chompTimerRef.current = undefined;
    };
  }, [fxSpeed]);

  const sendMessage = useCallback(async (override?: string) => {
    // `override` is used by starter-prompt chips and Regenerate; the composer
    // calls this with no argument and falls back to the input state.
    const text = (override ?? input).trim();
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
      thinkingMeta: {
        phases: [...THINKING_PHASES],
        currentPhase: 0,
        topSources: [],
        elapsedMs: 0,
        isExpanded: false,
      },
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setStreaming(true);
    elapsedStartRef.current = performance.now();
    clearInterval(elapsedTimerRef.current);
    // One tick per second on the thinking drawer's elapsed timer; the patch is
    // cheap and only touches the streaming message.
    elapsedTimerRef.current = setInterval(() => {
      const elapsedMs = Math.round(performance.now() - elapsedStartRef.current);
      patchThinking(assistantMsg.id, (meta) => ({ ...meta, elapsedMs }));
    }, 1000);

    const controller = new AbortController();
    abortRef.current = controller;
    let raw = "";
    let sources: Source[] = [];
    const citedNs: number[] = [];
    let failed = false;

    const patch = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantMsg.id ? fn(m) : m)));
    const patchThinking = (id: string, fn: (meta: ThinkingMeta) => ThinkingMeta) =>
      setMessages((prev) =>
        prev.map((m) => (m.id === id && m.thinkingMeta ? { ...m, thinkingMeta: fn(m.thinkingMeta) } : m)),
      );
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
            // The backend emits hits once, already reranked best-first. Rank
            // by score defensively and keep the top 5 for the radar drawer.
            const ranked = [...sources]
              .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
              .slice(0, 5);
            patchThinking(assistantMsg.id, (meta) => ({
              ...meta,
              topSources: ranked,
              currentPhase: 1, // hybrid search done…
            }));
            // …sonar reranking "sweeps" for a beat, then locks the top-5.
            clearTimeout(rerankTimerRef.current);
            rerankTimerRef.current = setTimeout(() => {
              patchThinking(assistantMsg.id, (meta) => ({
                ...meta,
                currentPhase: Math.max(meta.currentPhase, 2),
              }));
            }, RERANK_PHASE_MS);
          } else if (event === "token") {
            raw += (data as { t: string }).t;
            pushText((data as { t: string }).t);
            // First token means the prompt + context were assembled: the
            // "Context Cropped & Ranked" step is locked.
            patchThinking(assistantMsg.id, (meta) => ({
              ...meta,
              currentPhase: Math.max(meta.currentPhase, 3),
            }));
          } else if (event === "prefill") {
            // The prefill agent rewrote the prompt — show what retrieval
            // actually searched for.
            patchThinking(assistantMsg.id, (meta) => ({
              ...meta,
              refinedPrompt: (data as { cleaned: string }).cleaned,
            }));
          } else if (event === "citation") {
            const n = (data as { n: number }).n;
            if (!citedNs.includes(n)) citedNs.push(n);
            patch((m) => ({ ...m, citedNs: [...citedNs] }));
          } else if (event === "error") {
            failed = true;
            patch((m) => ({ ...m, status: "error", content: (data as { message: string }).message }));
          } else if (event === "meta") {
            const meta = data as { chat_id: string; message_id?: string };
            chatIdRef.current = meta.chat_id;
            // The persisted assistant turn's UUID — /feedback is keyed on it.
            if (meta.message_id) patch((m) => ({ ...m, message_id: meta.message_id }));
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
      clearInterval(elapsedTimerRef.current);
      elapsedTimerRef.current = undefined;
      clearTimeout(rerankTimerRef.current);
      rerankTimerRef.current = undefined;
    }
  }, [input, streaming, pushText, finishTypewriter, waitForTypewriter]);

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  // Mirror of `ratings` for read-modify-write in the click handler: state
  // updaters must be pure (StrictMode double-invokes them in dev, which would
  // duplicate the POST if the fetch lived inside), and the click needs the
  // previous vote to compute the toggle.
  const ratingsRef = useRef<Record<string, "up" | "down" | null>>({});

  /** Like/dislike toggle. Optimistic: the backend's toggle rule (same rating
   *  again removes it, other rating switches) matches the local transition,
   *  so a failed request reverts the button to its previous state.
   *  `messageKey` is the frontend bubble id (state key); `messageUuid` is the
   *  backend message id the /feedback payload requires. */
  const submitFeedback = useCallback(
    (messageKey: string, messageUuid: string, rating: "up" | "down") => {
      const was = ratingsRef.current[messageKey] ?? null;
      const next = was === rating ? null : rating;
      ratingsRef.current = { ...ratingsRef.current, [messageKey]: next };
      setRatings(ratingsRef.current);
      void (async () => {
        try {
          const resp = await fetch("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message_id: messageUuid, rating }),
          });
          if (!resp.ok) throw new Error(`feedback ${resp.status}`);
        } catch {
          ratingsRef.current = { ...ratingsRef.current, [messageKey]: was };
          setRatings(ratingsRef.current);
        }
      })();
    },
    [],
  );

  /** Re-runs the previous user prompt: drop the old user+assistant pair, then
   *  stream a fresh answer for the same prompt. */
  const regenerate = useCallback(
    (messageId: string) => {
      if (streaming) return;
      const idx = messages.findIndex((m) => m.id === messageId);
      if (idx <= 0) return;
      const prevUser = messages[idx - 1];
      if (!prevUser || prevUser.role !== "user") return;
      setMessages((prev) => prev.filter((m) => m.id !== messageId && m.id !== prevUser.id));
      const nextRatings = { ...ratingsRef.current };
      delete nextRatings[messageId];
      ratingsRef.current = nextRatings;
      setRatings(nextRatings);
      void sendMessage(prevUser.content);
    },
    [messages, streaming, sendMessage],
  );

  /** Opens the PDF viewer on a document, highlighting every chunk in
   *  `sources`. Re-opening the same document merges (union by chunk number)
   *  so highlights accumulate — clicking a drawer card for a chunk of an
   *  already-open document adds its box instead of replacing the others. */
  const openViewer = useCallback((sources: Source[]) => {
    const first = sources[0];
    if (!first) return;
    setViewer((prev) => {
      if (prev?.doc_id !== first.doc_id) {
        return { doc_id: first.doc_id, title: first.title, sources };
      }
      const seen = new Set(prev.sources.map((s) => s.n));
      const merged = [...prev.sources];
      for (const s of sources) {
        if (!seen.has(s.n)) merged.push(s);
      }
      return { ...prev, sources: merged };
    });
  }, []);

  const clearChat = useCallback(() => {
    abortRef.current?.abort();
    finishTypewriter();
    setMessages([]);
    setRatings({});
    ratingsRef.current = {};
    chatIdRef.current = null;
  }, [finishTypewriter]);

  /** Flips a message's thinking drawer. Must stay referentially stable —
   *  MessageItem is memo()'d and re-renders ~600x per answer. */
  const toggleThinking = useCallback((id: string) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === id && m.thinkingMeta
          ? { ...m, thinkingMeta: { ...m.thinkingMeta, isExpanded: !m.thinkingMeta.isExpanded } }
          : m,
      ),
    );
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
        onClear={clearChat}
        canClear={hasStarted}
        fxSpeed={fxSpeed}
        onFxSpeedChange={setFxSpeed}
        fxMuted={muted}
        onFxMutedChange={setMuted}
      />

      {/* Arcade monster ingestion overlay — always mounted so its exit
          animation plays; hidden via AnimatePresence when monster is null. */}
      <RagMonsterModal
        ingest={monster}
        onCancel={() => {
          // Done/cancelled monsters just close; anything else aborts the
          // upload (which flips status to cancelled).
          if (monster?.status === "done" || monster?.status === "cancelled") {
            setMonster(null);
          } else {
            ingestAbortRef.current?.abort();
          }
        }}
        onSkip={() =>
          setMonster((m) => (m && m.status === "crunching" ? { ...m, currentPage: m.totalPages } : m))
        }
        speed={fxSpeed}
        onSpeedChange={setFxSpeed}
        muted={muted}
        onMutedChange={setMuted}
      />

      <div className="flex flex-1 overflow-hidden relative">
        {dragOver && (
          <div className="absolute inset-0 z-10 flex items-center justify-center border-2 border-dashed border-amber/50 bg-background/80 backdrop-blur-sm">
            <p className="text-sm font-medium text-amber">Drop to ingest</p>
          </div>
        )}

        {/* When a source is open, the PDF takes the main stage and the whole
            chat column (messages + composer) docks to the right. One viewer
            instance serves both layouts: a fixed slide-over sheet below md,
            the static side panel from md up. */}
        {viewer && (
          <>
            {/* Mobile backdrop */}
            <div
              className="fixed inset-0 z-40 bg-slate-900/40 backdrop-blur-sm animate-[fade-in_0.2s_ease-out] md:hidden"
              onClick={() => setViewer(null)}
              aria-hidden="true"
            />
            {/* Mobile sheet + desktop panel */}
            <div className="fixed inset-x-0 bottom-0 z-50 h-[88dvh] rounded-t-2xl shadow-2xl overflow-hidden animate-[sheet-up_0.25s_ease-out] md:static md:z-auto md:h-full md:min-w-0 md:flex-1 md:rounded-none md:shadow-none md:animate-none">
              {/* Grabber affordance, mobile only */}
              <div className="absolute left-1/2 top-2 z-10 h-1 w-10 -translate-x-1/2 rounded-full bg-slate-300 pointer-events-none md:hidden" />
              <PdfViewer
                // Key on the DOCUMENT only: keying on page/citation remounted
                // the viewer (and re-downloaded + re-parsed the whole PDF)
                // every time you clicked a different citation in the same file.
                key={viewer.doc_id}
                target={viewer}
                onClose={() => setViewer(null)}
              />
            </div>
          </>
        )}

        <div
          className={cn(
            "flex flex-col overflow-hidden transition-[width] duration-300",
            viewer ? "w-full md:w-[26rem] md:shrink-0 md:border-l md:border-border" : "flex-1",
          )}
        >
          {hasStarted && (
            <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto scrollbar-thin">
              <MessageList
                messages={messages}
                onCite={openViewer}
                ratings={ratings}
                streaming={streaming}
                onFeedback={submitFeedback}
                onRegenerate={regenerate}
                fxSpeed={fxSpeed}
                onFxSpeedChange={setFxSpeed}
                onToggleThinking={toggleThinking}
              />
            </div>
          )}

          <ChatComposer
            value={input}
            onChange={setInput}
            onSubmit={sendMessage}
            loading={streaming}
            onStop={stop}
            docked={hasStarted}
            onPrompt={(prompt) => void sendMessage(prompt)}
          />
        </div>
      </div>
    </div>
  );
}

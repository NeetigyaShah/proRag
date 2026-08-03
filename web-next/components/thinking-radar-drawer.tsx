"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronDown, Clock, PenLine, Sparkles } from "lucide-react";
import { SpeedControl } from "@/components/speed-control";
import type { FxSpeed, Source } from "@/lib/types";
import { cn } from "@/lib/utils";

// Collapsible "Thinking about your request" container — ported from the Open
// Design professional redesign (rag-components-redesign.html, v2). The sonar
// radar is gone; a refined 4-step pipeline, an indeterminate streaming bar and
// ranked source cards carry the story. Props are unchanged from v1.

const formatTime = (ms: number) => {
  const total = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
};

const PIPELINE_STEPS = [
  { title: "Query Planning & Embedding", sub: "Vector generated" },
  { title: "Hybrid Search", sub: "Vector + FTS candidates retrieved" },
  { title: "Cross-Encoder Reranking", sub: "Cross-encoder scoring top chunks" },
  { title: "Context Cropped & Ranked", sub: "Top 5 receipts locked" },
];

function Pipeline({ currentPhase, status }: { currentPhase: number; status: "thinking" | "streaming" | "done" }) {
  const done = status === "done";
  return (
    <div>
      {PIPELINE_STEPS.map((step, i) => {
        const isDone = done || i < currentPhase;
        const isActive = !done && i === currentPhase;
        return (
          <div key={step.title} className="relative flex gap-3 pb-3.5 last:pb-0">
            {i < PIPELINE_STEPS.length - 1 && (
              <div className="absolute bottom-0 left-[11px] top-6 w-px" style={{ background: isDone ? "#fcd34d" : "#e2e8f0" }} />
            )}
            <div className="flex h-6 w-6 shrink-0 items-center justify-center">
              {isDone ? (
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-amber-500 text-white">
                  <Check className="h-3.5 w-3.5" />
                </span>
              ) : isActive ? (
                <span className="rag-ring flex h-6 w-6 items-center justify-center rounded-full border-2 border-amber-500 bg-white">
                  <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
                </span>
              ) : (
                <span className="flex h-6 w-6 items-center justify-center rounded-full border border-slate-200 bg-white">
                  <span className="h-1.5 w-1.5 rounded-full bg-slate-300" />
                </span>
              )}
            </div>
            <div className="min-w-0 pt-0.5">
              <p className={cn("text-[13px] font-medium leading-snug", (isDone || isActive) && "text-slate-800", !isDone && !isActive && "text-slate-400")}>
                {step.title}
              </p>
              <p className={cn("mt-0.5 text-xs leading-snug", isActive ? "text-amber-700" : isDone ? "text-slate-500" : "text-slate-400")}>
                {step.sub}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/** Slim indeterminate streaming shimmer while the pipeline runs. */
function ActivityBar() {
  return (
    <div className="relative h-[3px] overflow-hidden rounded-full bg-slate-100">
      <div className="rag-shimmer absolute top-0 h-full w-2/5 rounded-full bg-gradient-to-r from-transparent via-amber-400 to-transparent" />
    </div>
  );
}

function SourceCard({ source, rank, onSelect }: { source: Source; rank: number; onSelect: (s: Source) => void }) {
  // Cross-encoder logits can exceed [0,1] — clamp for the meter.
  const score = source.score == null ? null : Math.max(0, Math.min(1, source.score));
  const pct = score == null ? null : Math.round(score * 100);
  const isTop = rank === 1;
  return (
    <button
      type="button"
      onClick={() => onSelect(source)}
      title={source.title || source.doc_id}
      className={cn(
        "flex w-full items-start gap-2.5 rounded-xl border p-2.5 text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-200",
        isTop ? "border-amber-200 bg-amber-50/40 hover:border-amber-300" : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/60",
      )}
    >
      <span
        className={cn(
          "mt-px flex h-5 min-w-5 shrink-0 items-center justify-center rounded-md px-1 text-[11px] font-semibold tabular-nums",
          isTop ? "bg-amber-100 text-amber-700" : "bg-slate-100 text-slate-500",
        )}
      >
        {rank}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[13px] font-medium text-slate-800">{source.title || source.doc_id}</span>
        {source.snippet && <span className="mt-0.5 block truncate text-xs text-slate-500">{source.snippet}</span>}
      </span>
      <span className="flex shrink-0 flex-col items-end gap-1.5">
        {source.page != null && (
          <span className="rounded-md border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-medium tabular-nums text-slate-500">
            p.{source.page}
          </span>
        )}
        {pct != null && (
          <span className="flex items-center gap-1.5">
            <span className="relative h-[3px] w-14 overflow-hidden rounded-full bg-slate-100">
              <span className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-amber-300 to-amber-600" style={{ width: `${pct}%` }} />
            </span>
            <span className="w-8 text-right text-[10px] font-medium tabular-nums text-slate-500">{pct}%</span>
          </span>
        )}
      </span>
    </button>
  );
}

function SourceCards({ topSources, onSelect }: { topSources: Source[]; onSelect: (s: Source) => void }) {
  return (
    <div className="space-y-1.5">
      <p className="px-0.5 text-[11px] font-medium uppercase tracking-wide text-slate-400">Top sources</p>
      {topSources.map((source, i) => (
        // n is the per-context chunk number — unique even when one document
        // contributes several ranked chunks (doc_id repeats across them).
        <SourceCard key={source.n} source={source} rank={i + 1} onSelect={onSelect} />
      ))}
    </div>
  );
}

export function ThinkingRadarDrawer({
  open,
  onOpenChange,
  elapsedMs,
  currentPhase,
  topSources,
  status,
  speed,
  onSpeedChange,
  onSelect,
  refinedPrompt,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  elapsedMs: number;
  currentPhase: number;
  topSources: Source[];
  status: "thinking" | "streaming" | "done";
  speed: FxSpeed;
  onSpeedChange: (speed: FxSpeed) => void;
  onSelect: (source: Source) => void;
  refinedPrompt?: string;
}) {
  const done = status === "done";
  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-[0_1px_2px_0_rgb(15_23_42/0.05),0_4px_16px_-4px_rgb(15_23_42/0.08)]">
      <button
        type="button"
        onClick={() => onOpenChange(!open)}
        aria-expanded={open}
        className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left transition-colors hover:bg-slate-50"
      >
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md border border-amber-100 bg-amber-50 text-amber-600">
          {done ? <Check className="h-3.5 w-3.5" /> : <Sparkles className="rag-sparkle h-3.5 w-3.5" />}
        </span>
        <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-slate-700">
          {done ? "Thought about your request" : "Thinking about your request"}
        </span>
        {!done && (
          <span className="flex shrink-0 items-center gap-[3px]" aria-hidden="true">
            <span className="rag-dot h-[3px] w-[3px] rounded-full bg-slate-400" />
            <span className="rag-dot h-[3px] w-[3px] rounded-full bg-slate-400" />
            <span className="rag-dot h-[3px] w-[3px] rounded-full bg-slate-400" />
          </span>
        )}
        <span className="shrink-0 text-xs tabular-nums text-slate-400">{formatTime(elapsedMs)}</span>
        <ChevronDown className={cn("h-3.5 w-3.5 shrink-0 text-slate-400 transition-transform duration-200", open && "rotate-180")} />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="drawer-content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28, ease: [0.4, 0, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="space-y-4 px-3.5 pb-3.5">
              {refinedPrompt && (
                <p className="flex items-start gap-1.5 rounded-lg bg-slate-50 px-2.5 py-2 text-xs text-slate-500 ring-1 ring-inset ring-slate-200/60">
                  <PenLine className="mt-px h-3.5 w-3.5 shrink-0 text-amber-600" />
                  <span>
                    <span className="font-medium text-slate-600">Refined:</span> “{refinedPrompt}”
                  </span>
                </p>
              )}
              <Pipeline currentPhase={currentPhase} status={status} />
              {!done && <ActivityBar />}
              <SourceCards topSources={topSources} onSelect={onSelect} />
              <div className="flex items-center justify-between border-t border-slate-100 pt-3">
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-medium uppercase tracking-wide text-slate-400">Speed</span>
                  <SpeedControl speed={speed} onSpeedChange={onSpeedChange} onSkip={() => onSpeedChange("instant")} />
                </div>
                <div className="flex items-center gap-1.5 text-xs tabular-nums text-slate-400">
                  <Clock className="h-3.5 w-3.5" />
                  {formatTime(elapsedMs)}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

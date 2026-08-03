"use client";

import { useEffect, useRef, type CSSProperties } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, FileText, SkipForward, Volume2, VolumeX } from "lucide-react";
import { playScanTick } from "@/lib/chomp-audio";
import { fxDuration } from "@/lib/fx-settings";
import type { FxSpeed, MonsterIngestState } from "@/lib/types";
import { cn } from "@/lib/utils";

// PDF ingestion overlay with the "Scan Bed" animation (Open Design concept,
// rebuilt fully in SVG): a document sheet sits on a scan bed, an amber light
// bar sweeps it once per page, an amber tint fills the scanned region behind
// the bar, a checkmark pops as the sweep completes, and finished sheets
// stack in a tray that counts up. Everything animates inside ONE <svg> via
// CSS keyframes on SVG groups (transform-box: fill-box) — compositor-friendly,
// no DOM churn. Visual only: page.tsx owns pacing, cancel and skip.
//
// Pacing: every keyframe derives from --cycle, set inline below to
// fxDuration(speed, 1000) ms; the bar's sweep distance comes from --sweep-y.
// "instant" collapses to a 16ms floor.

const INSTANT_FACTOR = 1 / 50;

// Scan-bed geometry (viewBox 0 0 520 250): bed and sheet centered with room
// to breathe; the output bin sits below the bed with a clear gap; the sweep
// runs down the sheet's inner area (--sweep-y).
const SHEET = { x: 199, y: 40, w: 122, h: 122 };
const BIN = { x: 185, y: 188, w: 150, h: 38 };
const SWEEP_Y = 114;

/** The sheet's faux text lines (static SVG rects). */
function SheetLines() {
  const widths = [100, 84, 92, 60, 78, 88, 45, 70];
  return (
    <g>
      {widths.map((w, i) => (
        <rect key={i} x={207} y={50 + i * 11} width={w} height={2.5} rx={1.25} fill="#e2e8f0" />
      ))}
    </g>
  );
}

function Tray({ scanned, total, done }: { scanned: number; total: number; done: boolean }) {
  const slips = done ? total : Math.max(0, scanned);
  // Output bin: scanned pages sink DOWN into a recessed slot below the bed and
  // stack bottom-up inside it. The stack compresses to the bin's inner height
  // no matter how large the document is, so it never rises over the scanning
  // design; the counter (beside the bin) keeps the true count.
  const STACK_H = BIN.h - 4;
  const spacing = Math.min(3.5, STACK_H / Math.max(1, total));
  const slipH = Math.max(1.4, spacing * 0.75);
  return (
    <g>
      {/* glow over the bin — pulses when done */}
      {done && (
        <rect className="scan-tray-glow" x={BIN.x} y={BIN.y} width={BIN.w} height={BIN.h} rx={8} fill="url(#trayGlow)" />
      )}
      {/* recessed bin slot */}
      <rect x={BIN.x} y={BIN.y} width={BIN.w} height={BIN.h} rx={8} fill="url(#binGrad)" stroke="#cbd5e1" strokeWidth={1} />
      <rect x={BIN.x + 1} y={BIN.y + 1} width={BIN.w - 2} height={4} rx={2} fill="#94a3b8" opacity={0.18} />
      <rect x={BIN.x + 1} y={BIN.y + BIN.h - 5} width={BIN.w - 2} height={4} rx={2} fill="#64748b" opacity={0.12} />
      {/* slips, bottom-up, clipped to the bin */}
      <g clipPath="url(#binClip)">
        {Array.from({ length: slips }, (_, i) => (
          <rect
            key={i}
            className="scan-slip"
            x={BIN.x + 12}
            y={BIN.y + BIN.h - 2 - i * spacing}
            width={BIN.w - 24}
            height={slipH}
            rx={Math.max(0.4, slipH / 2)}
            fill="#fef3c7"
            stroke="#fcd34d"
            strokeWidth={0.5}
          />
        ))}
      </g>
      {/* counter beside the bin, vertically centered */}
      <text x={BIN.x + BIN.w + 12} y={BIN.y + BIN.h / 2 + 3.5} fontSize={11} fontWeight={600} fill="#64748b" style={{ fontVariantNumeric: "tabular-nums" }}>
        {slips} / {total} pages
      </text>
    </g>
  );
}

function ScanStage({
  page,
  totalPages,
  status,
}: {
  page: number;
  totalPages: number;
  status: MonsterIngestState["status"];
}) {
  // Remount per page so the sweep replays for every scanned page.
  const key = `${page}:${status}`;
  const scanning = status === "crunching" || status === "paused";
  const done = status === "done";
  return (
    <div key={key} className={`scan-stage absolute inset-0 state-${status}`}>
      <svg viewBox="0 0 520 250" className="h-full w-full" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
        <defs>
          <linearGradient id="barGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#d97706" stopOpacity="0" />
            <stop offset="20%" stopColor="#d97706" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="80%" stopColor="#d97706" />
            <stop offset="100%" stopColor="#d97706" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="tintGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#fef3c7" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#fbbf24" stopOpacity="0.12" />
          </linearGradient>
          <radialGradient id="trayGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#f59e0b" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="binGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f8fafc" />
            <stop offset="100%" stopColor="#e2e8f0" />
          </linearGradient>
          <filter id="bedShadow" x="-20%" y="-20%" width="140%" height="150%">
            <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="#0f172a" floodOpacity="0.08" />
          </filter>
          <clipPath id="binClip">
            <rect x={BIN.x} y={BIN.y} width={BIN.w} height={BIN.h} rx={8} />
          </clipPath>
        </defs>

        {/* scan bed */}
        <g filter="url(#bedShadow)">
          <rect x={185} y={26} width={150} height={150} rx={10} fill="#ffffff" stroke="#e2e8f0" strokeWidth={1.5} />
        </g>

        {/* document sheet */}
        <rect x={SHEET.x} y={SHEET.y} width={SHEET.w} height={SHEET.h} rx={4} fill="#ffffff" stroke="#f1f5f9" strokeWidth={1} />
        <SheetLines />

        {/* amber tint filling the scanned region behind the bar */}
        {scanning && (
          <rect
            className="scan-tint"
            x={SHEET.x}
            y={SHEET.y}
            width={SHEET.w}
            height={SHEET.h}
            rx={4}
            fill="url(#tintGrad)"
          />
        )}

        {/* light bar: glow + core, sweeping down the sheet */}
        {scanning && (
          <g className="scan-bar" style={{ "--sweep-y": `${SWEEP_Y}px` } as CSSProperties}>
            <rect x={200} y={SHEET.y + 2} width={120} height={6} rx={3} fill="#f59e0b" opacity={0.12} />
            <rect x={203} y={SHEET.y + 4} width={114} height={3} rx={1.5} fill="url(#barGrad)" />
          </g>
        )}

        {/* checkmark — pops at sweep end; static when done */}
        <g
          className={scanning ? "scan-check" : undefined}
          style={done ? { opacity: 1 } : undefined}
        >
          <path
            d="M 248 100 l 6 6 l 12 -14"
            fill="none"
            stroke="#d97706"
            strokeWidth={4.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </g>

        {/* output bin */}
        <Tray scanned={page - 1} total={totalPages} done={done} />
      </svg>
    </div>
  );
}

export function RagMonsterModal({
  ingest,
  onCancel,
  onSkip,
  speed,
  onSpeedChange,
  muted,
  onMutedChange,
}: {
  ingest: MonsterIngestState | null;
  onCancel: () => void;
  onSkip: () => void;
  speed: FxSpeed;
  onSpeedChange: (speed: FxSpeed) => void;
  muted: boolean;
  onMutedChange: (muted: boolean) => void;
}) {
  const open = ingest !== null;
  const filename = ingest?.filename ?? "";
  const totalPages = ingest?.totalPages ?? 1;
  const status = ingest?.status ?? "crunching";
  const currentPage = Math.min(totalPages, Math.max(1, ingest?.currentPage ?? 1));
  const isDone = status === "done";
  const isCrunching = status === "crunching";
  const isPaused = status === "paused";
  const progress = Math.round((currentPage / totalPages) * 100);
  // One pacing var for the whole sweep timeline; instant hits the 16ms floor.
  const cycle = Math.max(16, fxDuration(speed, 1000));
  const speedLabel = speed === "instant" ? "Instant" : `${speed}x`;
  const prevPageRef = useRef(1);

  // Soft tick when the sweep completes (checkmark moment ~95% of the cycle).
  useEffect(() => {
    if (!open || status !== "crunching") return;
    if (currentPage <= prevPageRef.current) return;
    prevPageRef.current = currentPage;
    if (muted) return;
    const t = setTimeout(() => playScanTick(), cycle * 0.95);
    return () => clearTimeout(t);
  }, [open, status, currentPage, muted, cycle]);

  // Lock body scroll and close on Escape while open.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && status !== "done") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener("keydown", onKey);
    };
  }, [open, status, onCancel]);

  const headerTitle = isDone
    ? `Done — ${totalPages} pages scanned`
    : status === "paused"
      ? "Scanning paused"
      : status === "cancelled"
        ? "Scanning cancelled"
        : "PDF ingestion in progress";

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
        >
          <div className="absolute inset-0 bg-slate-900/25 backdrop-blur-sm" onClick={onCancel} aria-hidden="true" />

          <motion.div
            role="dialog"
            aria-modal="true"
            aria-label="Scanning PDF"
            className="relative w-full max-w-lg overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-[0_2px_4px_0_rgb(15_23_42/0.06),0_16px_48px_-16px_rgb(15_23_42/0.18)]"
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 360, damping: 30 }}
          >
            <div className="px-5 pt-5">
              <div className="flex items-start justify-between gap-3">
                <div className="flex min-w-0 items-center gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-200 bg-white shadow-sm">
                    <FileText className="h-4 w-4 text-slate-500" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-sm font-semibold text-slate-800">{headerTitle}</p>
                    <p className="mt-0.5 max-w-[240px] truncate text-xs text-slate-500">{filename}</p>
                  </div>
                </div>
                <span className="shrink-0 text-[11px] tabular-nums text-slate-400">{progress}%</span>
              </div>

              <div className="mt-4 h-1 w-full overflow-hidden rounded-full bg-slate-100">
                <div className="h-full rounded-full bg-slate-500 transition-[width] duration-300 ease-out" style={{ width: `${progress}%` }} />
              </div>
            </div>

            {/* Scan-bed stage (single SVG) */}
            <div
              className="relative mx-5 mt-4 h-[240px] select-none overflow-hidden rounded-2xl border border-slate-100"
              style={{ "--cycle": `${cycle}ms`, "--speed-factor": typeof speed === "number" ? 1 / speed : INSTANT_FACTOR } as CSSProperties}
            >
              <ScanStage page={currentPage} totalPages={totalPages} status={status} />
            </div>

            <div className="flex items-center justify-between px-5 pb-2 pt-3">
              {isDone ? (
                <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-[11px] font-medium text-amber-700">
                  <Check className="h-3 w-3" /> Done — {totalPages} pages scanned
                </span>
              ) : (
                <span className="text-xs text-slate-500">
                  {isCrunching && (
                    <>
                      Scanning page <span className="font-medium tabular-nums text-slate-700">{currentPage}</span> of {totalPages}
                    </>
                  )}
                  {isPaused && "Paused"}
                  {status === "cancelled" && "Cancelled"}
                </span>
              )}
              <span className="text-[11px] tabular-nums text-slate-400">{speedLabel}</span>
            </div>

            <div className="mt-2 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 px-5 py-4">
              <div className="flex flex-wrap items-center gap-2.5">
                <button
                  type="button"
                  aria-label={muted ? "Unmute sound" : "Mute sound"}
                  onClick={() => onMutedChange(!muted)}
                  className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-500 transition-colors hover:bg-slate-50 hover:text-slate-700"
                >
                  {muted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                </button>
                {/* No speed slider: for long files the animation is either run
                    fast at 3x or skipped outright (Instant). */}
                <button
                  type="button"
                  aria-pressed={speed === 3}
                  title="Fast animation (3x)"
                  onClick={() => onSpeedChange(3)}
                  className={cn(
                    "inline-flex h-8 items-center rounded-lg border px-2.5 text-[11px] font-medium transition-colors",
                    speed === 3
                      ? "border-amber-300 bg-amber-100 text-amber-700"
                      : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-700",
                  )}
                >
                  3x
                </button>
                {isDone ? (
                  <button
                    type="button"
                    onClick={onCancel}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-amber-600 px-3.5 py-1.5 text-xs font-medium text-white transition-colors hover:bg-amber-700"
                  >
                    Done
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={onSkip}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-700 transition-colors hover:bg-amber-100"
                  >
                    <SkipForward className="h-3.5 w-3.5" /> Instant
                  </button>
                )}
              </div>
              <button
                type="button"
                onClick={onCancel}
                className="rounded-lg px-3 py-1.5 text-xs font-medium text-red-600 transition-colors hover:bg-red-50"
              >
                Cancel
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

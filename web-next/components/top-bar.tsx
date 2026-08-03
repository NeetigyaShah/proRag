"use client";

import { Button } from "@/components/ui/button";
import { AlertCircle, CheckCircle2, FileText, FileUp, Gauge, Loader2, RotateCcw, Volume2, VolumeX, Zap } from "lucide-react";
import { useRef } from "react";
import { FX_SPEEDS } from "@/lib/fx-settings";
import type { FxSpeed } from "@/lib/types";
import { cn } from "@/lib/utils";

interface TopBarProps {
  documents: number | null;
  ingesting: string | null;
  lastIngestResult: string | null;
  onUpload: (file: File) => void;
  onClear: () => void;
  canClear: boolean;
  fxSpeed: FxSpeed;
  onFxSpeedChange: (speed: FxSpeed) => void;
  fxMuted: boolean;
  onFxMutedChange: (muted: boolean) => void;
}

export function TopBar({
  documents,
  ingesting,
  lastIngestResult,
  onUpload,
  onClear,
  canClear,
  fxSpeed,
  onFxSpeedChange,
  fxMuted,
  onFxMutedChange,
}: TopBarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const failed = lastIngestResult !== null && lastIngestResult.startsWith("Could not");

  return (
    <header className="flex items-center justify-between gap-3 bg-card px-5 py-4 border-b border-border shrink-0">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-sm font-semibold tracking-tight text-foreground">
          Pro<span className="text-amber">Rag</span>
        </span>
        <span className="text-xs text-muted-foreground hidden sm:inline">
          answers with receipts
        </span>
      </div>

      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        {/* FX controls: shared animation speed + sound mute (persisted in
            localStorage via useFxSettings). Compact on mobile. */}
        <div
          className="hidden items-center gap-1 rounded-full border border-border bg-card px-1.5 py-1 shadow-sm shadow-slate-200/50 sm:flex"
          role="group"
          aria-label="Animation speed"
        >
          <Gauge className="size-3.5 text-muted-foreground" />
          {FX_SPEEDS.map((s) => (
            <button
              key={s}
              type="button"
              aria-pressed={fxSpeed === s}
              title={`Animation speed ${s === "instant" ? "Instant" : `${s}x`}`}
              onClick={() => onFxSpeedChange(s)}
              className={cn(
                "rounded-full px-1.5 py-0.5 text-[0.65rem] font-semibold tabular-nums transition-colors",
                fxSpeed === s
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
            >
              {s === "instant" ? <Zap className="size-2.5" /> : `${s}x`}
            </button>
          ))}
        </div>
        <button
          type="button"
          aria-pressed={!fxMuted}
          title={fxMuted ? "Unmute chomp sounds" : "Mute chomp sounds"}
          onClick={() => onFxMutedChange(!fxMuted)}
          className={cn(
            "inline-flex size-6 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-sm shadow-slate-200/50 transition-colors hover:bg-accent hover:text-foreground",
          )}
        >
          {fxMuted ? <VolumeX className="size-3.5" /> : <Volume2 className="size-3.5" />}
        </button>
        {/* Upload progress/result lands here with no focus change, so a screen
            reader would otherwise never announce it. */}
        <span className="tabular-nums shrink-0" aria-live="polite">
          {ingesting ? (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-amber/30 bg-amber-light px-2.5 py-1 text-amber-foreground">
              <Loader2 className="size-3.5 animate-spin" />
              Ingesting {ingesting}…
            </span>
          ) : lastIngestResult ? (
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1",
                failed
                  ? "border-red-200 bg-red-50 text-destructive"
                  : "border-emerald-200 bg-emerald-50 text-emerald-700",
              )}
            >
              {failed ? <AlertCircle className="size-3.5" /> : <CheckCircle2 className="size-3.5" />}
              {lastIngestResult}
            </span>
          ) : documents === null ? (
            "…"
          ) : (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-card px-2.5 py-1 text-muted-foreground shadow-sm shadow-slate-200/50">
              <FileText className="size-3.5" />
              {documents} {documents === 1 ? "document" : "documents"}
            </span>
          )}
        </span>
        <input
          ref={fileInputRef}
          type="file"
          // Mirrors ALLOWED_SUFFIXES in prorag/ingest/router.py — .txt/.md/.tsv
          // were accepted by the backend but hidden by the file picker.
          accept=".pdf,.txt,.md,.docx,.pptx,.csv,.xlsx,.tsv,.png,.jpg,.jpeg,.tiff,.tif,.webp"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file);
            e.target.value = "";
          }}
        />
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="gap-1.5"
          disabled={!canClear}
          onClick={onClear}
          title="Clear chat history"
        >
          <RotateCcw className="size-3.5" />
          <span className="hidden sm:inline">Clear Chat</span>
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="gap-1.5"
          // ingest() is the real guard (drag-and-drop bypasses this button);
          // disabling here just stops the picker opening to no effect.
          disabled={!!ingesting}
          onClick={() => fileInputRef.current?.click()}
        >
          <FileUp className="size-3.5" />
          Upload
        </Button>
      </div>
    </header>
  );
}

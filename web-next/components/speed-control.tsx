"use client";

import { Zap } from "lucide-react";
import { FX_SPEEDS } from "@/lib/fx-settings";
import type { FxSpeed } from "@/lib/types";
import { cn } from "@/lib/utils";

// Shared speed control for the radar drawer (and formerly the monster modal):
// a slim segmented selector plus an amber Instant button. Segments come from
// FX_SPEEDS (all numeric presets) so the control stays in sync with the
// global preset list; "instant" lights the Instant button.

const SPEEDS = FX_SPEEDS.filter((s): s is Exclude<FxSpeed, "instant"> => typeof s === "number");

export function SpeedControl({
  speed,
  onSpeedChange,
  onSkip,
  className,
  disabled = false,
}: {
  speed: FxSpeed;
  onSpeedChange: (speed: FxSpeed) => void;
  onSkip: () => void;
  className?: string;
  disabled?: boolean;
}) {
  return (
    <div className={cn("flex items-center gap-2", disabled && "pointer-events-none opacity-40", className)}>
      <div role="group" aria-label="Playback speed" className="inline-flex items-center gap-0.5 rounded-lg border border-slate-200 bg-slate-50 p-0.5">
        {SPEEDS.map((s) => {
          const active = s === speed;
          return (
            <button
              key={s}
              type="button"
              aria-pressed={active}
              onClick={() => onSpeedChange(s)}
              className={cn(
                "rounded-md px-2 py-1 text-[11px] font-medium tabular-nums transition-colors",
                active ? "bg-white text-slate-800 shadow-sm ring-1 ring-slate-200" : "text-slate-500 hover:text-slate-700",
              )}
            >
              {s}x
            </button>
          );
        })}
      </div>
      <button
        type="button"
        onClick={onSkip}
        aria-pressed={speed === "instant"}
        title="Skip animation"
        className={cn(
          "inline-flex items-center gap-1 rounded-lg border px-2.5 py-1 text-[11px] font-medium transition-colors",
          speed === "instant"
            ? "border-amber-300 bg-amber-100 text-amber-700"
            : "border-amber-200 bg-amber-50 text-amber-700 hover:bg-amber-100",
        )}
      >
        <Zap className="h-3 w-3" />
        Instant
      </button>
    </div>
  );
}

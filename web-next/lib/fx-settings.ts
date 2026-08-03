"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { FxSpeed } from "@/lib/types";

// Persistent animation controls shared by the arcade monster modal and the
// sonar radar drawer: animation speed (0.5x–5x + instant skip) and audio
// mute. Both survive reloads via localStorage (keys mirror the old web/chat.js
// convention: prorag_fx_speed, prorag_fx_muted).

export const FX_SPEEDS: readonly FxSpeed[] = [0.5, 1, 2, 3, 5, "instant"] as const;

const SPEED_KEY = "prorag_fx_speed";
const MUTED_KEY = "prorag_fx_muted";

export function isFxSpeed(v: unknown): v is FxSpeed {
  return v === 0.5 || v === 1 || v === 2 || v === 3 || v === 5 || v === "instant";
}

function readSpeed(): FxSpeed {
  if (typeof window === "undefined") return 1;
  try {
    const raw = window.localStorage.getItem(SPEED_KEY);
    return raw === null ? 1 : JSON.parse(raw);
  } catch {
    return 1;
  }
}

function readMuted(): boolean {
  if (typeof window === "undefined") return true;
  try {
    // Sound is opt-in: muted unless the user explicitly unmuted.
    return window.localStorage.getItem(MUTED_KEY) !== "0";
  } catch {
    return true;
  }
}

/** How long a base animation (authored at 1x) runs at the given speed.
 *  "instant" collapses it to zero — callers skip the animation entirely. */
export function fxDuration(speed: FxSpeed, baseMs: number): number {
  return speed === "instant" ? 0 : baseMs / speed;
}

export function useFxSettings() {
  // Server-consistent defaults: reading localStorage in the initializer would
  // make client hydration mismatch the SSR HTML (React logs "attributes didn't
  // match" and leaves the wrong button pressed). The mount effect flips to the
  // persisted values once, after hydration.
  const [speed, setSpeedState] = useState<FxSpeed>(1);
  const [muted, setMutedState] = useState<boolean>(true);
  const hydrated = useRef(false);

  useEffect(() => {
    if (hydrated.current) return;
    hydrated.current = true;
    setSpeedState(readSpeed());
    setMutedState(readMuted());
  }, []);

  // Persist on change. localStorage access is guarded — a blocked/disabled
  // store must never crash the chat.
  useEffect(() => {
    try {
      window.localStorage.setItem(SPEED_KEY, JSON.stringify(speed));
    } catch {
      /* storage unavailable — settings stay in-memory */
    }
  }, [speed]);

  useEffect(() => {
    try {
      window.localStorage.setItem(MUTED_KEY, muted ? "1" : "0");
    } catch {
      /* storage unavailable — settings stay in-memory */
    }
  }, [muted]);

  const setSpeed = useCallback((s: FxSpeed) => setSpeedState(s), []);
  const setMuted = useCallback((m: boolean) => setMutedState(m), []);

  return { speed, setSpeed, muted, setMuted };
}

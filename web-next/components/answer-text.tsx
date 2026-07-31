"use client";

import { useMemo } from "react";
import { isCitationMarker, markerIndex, splitOnCitations } from "@/lib/citations";
import type { Source } from "@/lib/types";

// Splits streamed answer text on citation markers and renders them as
// clickable amber superscript pills. Ported from renderAnswer() in web/chat.js.
export function AnswerText({
  text,
  sources,
  onCite,
}: {
  text: string;
  sources: Source[];
  onCite: (source: Source) => void;
}) {
  const parts = useMemo(() => splitOnCitations(text), [text]);
  const byN = useMemo(() => new Map(sources.map((s) => [s.n, s])), [sources]);

  return (
    <p className="answer-serif text-[1.05rem] leading-relaxed text-foreground whitespace-pre-wrap">
      {parts.map((part, i) => {
        if (isCitationMarker(part)) {
          const n = markerIndex(part);
          const src = byN.get(n);
          return (
            <sup key={i} className="mx-0.5">
              <button
                type="button"
                title={src?.snippet || "Open source"}
                aria-label={src ? `Open source ${n}` : `Source ${n} unavailable`}
                disabled={!src}
                onClick={() => src && onCite(src)}
                className="inline-flex items-center justify-center rounded-full bg-amber/20 px-1.5 py-0.5 text-[0.65rem] font-sans font-semibold text-amber transition-colors hover:bg-amber/30 disabled:opacity-50 disabled:hover:bg-amber/20 focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring/70"
              >
                S{n}
              </button>
            </sup>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </p>
  );
}

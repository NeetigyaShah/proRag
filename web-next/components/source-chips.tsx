"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import type { Source } from "@/lib/types";

// Memoized: `sources` and `citedNs` change only on the sources/citation SSE
// events, not on every token, so this can skip the ~600 typewriter renders.
export const SourceChips = memo(function SourceChips({
  sources,
  citedNs,
  onSelect,
}: {
  sources: Source[];
  citedNs: number[];
  onSelect: (source: Source) => void;
}) {
  if (!sources.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {sources.map((s) => {
        const cited = citedNs.includes(s.n);
        return (
          <button
            key={s.n}
            type="button"
            title={s.snippet || ""}
            onClick={() => onSelect(s)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors",
              cited
                ? "border-amber/40 bg-amber/10 text-amber hover:bg-amber/20"
                : "border-border bg-secondary text-muted-foreground hover:bg-accent hover:text-foreground",
            )}
          >
            <span className="font-semibold">S{s.n}</span>
            <span className="max-w-[14rem] truncate">{s.title || s.doc_id}</span>
            {/* `!= null`, not truthiness: JSX renders a falsy NUMBER as a text
                node, so `s.page && ...` would print a bare "0" for page 0. */}
            {s.page != null && <span className="text-[0.65rem] opacity-70">p.{s.page}</span>}
          </button>
        );
      })}
    </div>
  );
});

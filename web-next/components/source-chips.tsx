"use client";

import { memo } from "react";
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
  onSelect: (sources: Source[]) => void;
}) {
  // Receipts = what the answer actually used: only cited chunks, one chip per
  // document (the same PDF often contributes several chunks — S2 and S5 can
  // both be Shah_Resume.pdf — and a single chip per doc keeps the list honest
  // about "which documents were used" without repeating the title).
  const cited = sources.filter((s) => citedNs.includes(s.n));
  if (!cited.length) return null;
  const groups = new Map<string, Source[]>();
  for (const s of cited) {
    const group = groups.get(s.doc_id) ?? [];
    group.push(s);
    groups.set(s.doc_id, group);
  }
  return (
    <div className="flex flex-wrap gap-1.5 mt-2">
      {[...groups.values()].map((group) => {
        const first = group[0];
        const badge = group.map((s) => `S${s.n}`).join("·");
        const pages = [...new Set(group.map((s) => s.page).filter((p): p is number => p != null))];
        return (
          <button
            key={first.doc_id}
            type="button"
            title={first.snippet || ""}
            onClick={() => onSelect(group)}
            className="inline-flex items-center gap-1.5 rounded-full border border-amber/40 bg-amber/10 px-2.5 py-1 text-xs text-amber transition-colors hover:bg-amber/20"
          >
            <span className="font-semibold">{badge}</span>
            <span className="max-w-[14rem] truncate">{first.title || first.doc_id}</span>
            {/* `!= null`, not truthiness: JSX renders a falsy NUMBER as a text
                node, so `s.page && ...` would print a bare "0" for page 0. */}
            {pages.length > 0 && <span className="text-[0.65rem] opacity-70">p.{pages.join(", ")}</span>}
          </button>
        );
      })}
    </div>
  );
});

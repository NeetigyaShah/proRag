"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Copy, RotateCcw, ThumbsDown, ThumbsUp } from "lucide-react";
import { isCitationMarker, markerIndex, splitOnCitations } from "@/lib/citations";
import { parseBlocks, tokenizeInline, type Block } from "@/lib/markdown";
import { cn } from "@/lib/utils";
import type { ChatMessage, Source } from "@/lib/types";

// ---------------------------------------------------------------------------
// Lightweight markdown for streamed answers.
//
// The typewriter re-renders this component ~600 times per answer, so a full
// markdown engine (AST + plugins) is overkill. This parser covers exactly what
// the answer model emits: fenced code blocks, tables, lists, headings, and
// **bold** / *italic* / `inline code` inside prose (see lib/markdown.ts).
// Citation markers are handled separately (splitOnCitations) so they render as
// pills even inside paragraphs, list items and table cells.

// ---------------------------------------------------------------------------

const CODE_BLOCK_WRAP = "my-3 overflow-hidden rounded-lg border border-slate-800 bg-slate-900 last:mb-0";
const INLINE_CODE =
  "rounded-md bg-secondary px-1.5 py-0.5 font-mono text-[0.85em] text-foreground";
const CITATION_PILL =
  "inline-flex items-center justify-center rounded-full bg-amber-light px-1.5 py-0.5 text-[0.65rem] font-sans font-semibold text-amber-foreground transition-colors hover:bg-amber/15 disabled:opacity-50 disabled:hover:bg-amber-light focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring/70";

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
  const blocks = useMemo(() => parseBlocks(text), [text]);
  const byN = useMemo(() => new Map(sources.map((s) => [s.n, s])), [sources]);

  const renderContent = (part: string, keyPrefix: string) => {
    const pieces = splitOnCitations(part);
    return pieces.map((piece, i) => {
      if (isCitationMarker(piece)) {
        const n = markerIndex(piece);
        const src = byN.get(n);
        return (
          <sup key={`${keyPrefix}-c${i}`} className="mx-0.5">
            <button
              type="button"
              title={src?.snippet || "Open source"}
              aria-label={src ? `Open source ${n}` : `Source ${n} unavailable`}
              disabled={!src}
              onClick={() => src && onCite(src)}
              className={CITATION_PILL}
            >
              S{n}
            </button>
          </sup>
        );
      }
      return tokenizeInline(piece).map((tok, j) => {
        const key = `${keyPrefix}-${i}-${j}`;
        switch (tok.type) {
          case "code":
            return (
              <code key={key} className={INLINE_CODE}>
                {tok.content}
              </code>
            );
          case "bold":
            return <strong key={key}>{tok.content}</strong>;
          case "italic":
            return <em key={key}>{tok.content}</em>;
          default:
            return <span key={key}>{tok.content}</span>;
        }
      });
    });
  };

  return (
    <div className="answer-serif text-[1.05rem] leading-relaxed text-foreground">
      {blocks.map((block, i) => {
        switch (block.type) {
          case "heading":
            return (
              <h3
                key={i}
                className={cn(
                  "font-semibold text-foreground",
                  block.level === 1 ? "text-lg" : block.level === 2 ? "text-base" : "text-[1.05rem]",
                  i > 0 && "mt-4",
                )}
              >
                {renderContent(block.text, `h${i}`)}
              </h3>
            );
          case "paragraph":
            return (
              <p key={i} className={cn(i > 0 && "mt-3")}>
                {renderContent(block.lines.join(" "), `p${i}`)}
              </p>
            );
          case "list": {
            const items = block.items.map((it, j) => (
              <li key={j} className="pl-1">
                {renderContent(it, `li${i}-${j}`)}
              </li>
            ));
            return block.ordered ? (
              <ol key={i} className={cn("list-decimal pl-5", i > 0 && "mt-3", "space-y-1")}>
                {items}
              </ol>
            ) : (
              <ul key={i} className={cn("list-disc pl-5", i > 0 && "mt-3", "space-y-1")}>
                {items}
              </ul>
            );
          }
          case "code":
            return (
              <div key={i} className={CODE_BLOCK_WRAP}>
                <div className="flex items-center justify-between border-b border-slate-700/60 bg-slate-800/60 px-3 py-1.5">
                  <span className="font-mono text-[0.65rem] font-medium uppercase tracking-wider text-slate-400">
                    {block.lang}
                  </span>
                </div>
                <pre className="overflow-x-auto p-3 text-xs leading-relaxed text-slate-100">
                  <code className="font-mono">{block.code}</code>
                </pre>
              </div>
            );
          case "table":
            return (
              <div key={i} className={cn("overflow-x-auto rounded-lg border border-border", i > 0 && "mt-3")}>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border bg-secondary/60">
                      {block.header.map((h, j) => (
                        <th
                          key={j}
                          className="px-3 py-2 text-left font-semibold text-foreground"
                        >
                          {renderContent(h, `th${i}-${j}`)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, j) => (
                      <tr key={j} className="border-b border-border last:border-0">
                        {row.map((cell, k) => (
                          <td key={k} className="px-3 py-2 text-foreground/90">
                            {renderContent(cell, `td${i}-${j}-${k}`)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
        }
      })}
    </div>
  );
}

// ---- Hover action bar (copy / like / dislike / regenerate) -----------------

export function MessageActions({
  message,
  rating,
  streaming,
  onFeedback,
  onRegenerate,
}: {
  message: ChatMessage;
  rating: "up" | "down" | null;
  streaming: boolean;
  onFeedback: (messageKey: string, messageUuid: string, rating: "up" | "down") => void;
  onRegenerate: (messageId: string) => void;
}) {
  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    },
    [],
  );

  const copy = async () => {
    let ok = false;
    try {
      await navigator.clipboard.writeText(message.content);
      ok = true;
    } catch {
      // Fallback for contexts that deny the async clipboard API (headless
      // browsers, some enterprise policies): the execCommand path still works.
      try {
        const ta = document.createElement("textarea");
        ta.value = message.content;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        ok = document.execCommand("copy");
        document.body.removeChild(ta);
      } catch {
        ok = false;
      }
    }
    if (!ok) return; // leave the icon untouched rather than lying about success
    setCopied(true);
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    copyTimerRef.current = setTimeout(() => setCopied(false), 2000);
  };

  // The backend persists each assistant turn and hands back its UUID in the
  // meta SSE event; feedback is keyed on that, not on the local bubble id.
  const canFeedback = !!message.message_id;
  const feedbackTitle = canFeedback ? undefined : "Feedback unavailable";

  const baseBtn =
    "inline-flex size-7 items-center justify-center rounded-md transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring/70 disabled:pointer-events-none disabled:opacity-40";

  return (
    <div className="mt-2 flex items-center gap-0.5 opacity-100 md:opacity-0 md:transition-opacity md:duration-200 md:group-hover:opacity-100 md:focus-within:opacity-100">
      <button
        type="button"
        onClick={copy}
        aria-label={copied ? "Copied" : "Copy answer"}
        title="Copy answer"
        className={cn(baseBtn, "text-muted-foreground hover:bg-accent hover:text-foreground")}
      >
        {copied ? <Check className="size-3.5 text-emerald-600" /> : <Copy className="size-3.5" />}
      </button>
      <button
        type="button"
        disabled={!canFeedback}
        aria-pressed={rating === "up"}
        aria-label="Rate answer helpful"
        title={feedbackTitle ?? "Helpful"}
        onClick={() => canFeedback && onFeedback(message.id, message.message_id!, "up")}
        className={cn(
          baseBtn,
          rating === "up"
            ? "bg-amber-light text-amber-foreground hover:bg-amber-light"
            : "text-muted-foreground hover:bg-accent hover:text-foreground",
        )}
      >
        <ThumbsUp className="size-3.5" />
      </button>
      <button
        type="button"
        disabled={!canFeedback}
        aria-pressed={rating === "down"}
        aria-label="Rate answer unhelpful"
        title={feedbackTitle ?? "Not helpful"}
        onClick={() => canFeedback && onFeedback(message.id, message.message_id!, "down")}
        className={cn(
          baseBtn,
          rating === "down"
            ? "bg-amber-light text-amber-foreground hover:bg-amber-light"
            : "text-muted-foreground hover:bg-accent hover:text-foreground",
        )}
      >
        <ThumbsDown className="size-3.5" />
      </button>
      <button
        type="button"
        disabled={streaming}
        aria-label="Regenerate answer"
        title={streaming ? "Waiting for the current answer" : "Regenerate"}
        onClick={() => onRegenerate(message.id)}
        className={cn(baseBtn, "text-muted-foreground hover:bg-accent hover:text-foreground")}
      >
        <RotateCcw className="size-3.5" />
      </button>
    </div>
  );
}

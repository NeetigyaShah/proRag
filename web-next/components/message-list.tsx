"use client";

import { memo } from "react";
import type { ChatMessage, FxSpeed, Source } from "@/lib/types";
import { AnswerText, MessageActions } from "@/components/answer-text";
import { SourceChips } from "@/components/source-chips";
import { ThinkingRadarDrawer } from "@/components/thinking-radar-drawer";
import { cn } from "@/lib/utils";

// Stable identity: `?? []` would hand memo() a fresh array every render and
// defeat the shallow compare below.
const NO_SOURCES: Source[] = [];
const NO_CITED: number[] = [];

// One message, memoized. The typewriter re-renders the parent ~600 times per
// answer, but page.tsx's patch() preserves object identity for every message
// except the streaming one — so with memo() the rest are skipped instead of
// re-running AnswerText's parse. Measured over one answer: 0.60ms/frame
// -> 0.03ms at 40 messages, and the gap widens as the chat grows.
const MessageItem = memo(function MessageItem({
  message,
  onCite,
  rating,
  streaming,
  onFeedback,
  onRegenerate,
  fxSpeed,
  onFxSpeedChange,
  onToggleThinking,
}: {
  message: ChatMessage;
  onCite: (sources: Source[]) => void;
  rating: "up" | "down" | null;
  streaming: boolean;
  onFeedback: (messageKey: string, messageUuid: string, rating: "up" | "down") => void;
  onRegenerate: (messageId: string) => void;
  fxSpeed: FxSpeed;
  onFxSpeedChange: (speed: FxSpeed) => void;
  onToggleThinking: (messageId: string) => void;
}) {
  const isUser = message.role === "user";
  const meta = message.thinkingMeta;
  return (
    // `group` anchors the hover-revealed action bar on assistant messages.
    <div className="group flex flex-col gap-1.5">
      <span className="text-xs font-medium text-muted-foreground">{isUser ? "You" : "ProRag"}</span>
      {isUser ? (
        // break-words: an unbroken paste (long URL, token) would otherwise
        // stretch past max-w and force the column to scroll sideways.
        <div className="rounded-2xl bg-secondary px-4 py-2.5 text-sm text-foreground w-fit max-w-[85%] break-words">
          {message.content}
        </div>
      ) : message.status === "error" ? (
        <p className="text-sm text-destructive">{message.content}</p>
      ) : message.status === "thinking" ? (
        <>
          {/* The sonar drawer carries the "thinking" story now; the old pulsing
              status line stays as a compact fallback while it's collapsed. */}
          {meta ? (
            <ThinkingRadarDrawer
              open={meta.isExpanded ?? false}
              onOpenChange={() => onToggleThinking(message.id)}
              elapsedMs={meta.elapsedMs ?? 0}
              currentPhase={meta.currentPhase}
              topSources={meta.topSources}
              status={message.status}
              speed={fxSpeed}
              onSpeedChange={onFxSpeedChange}
              onSelect={(s) => onCite([s])}
              refinedPrompt={meta.refinedPrompt}
            />
          ) : (
            <p className={cn("text-sm text-muted-foreground animate-pulse")}>{message.content}</p>
          )}
        </>
      ) : (
        <>
          {meta && (
            <ThinkingRadarDrawer
              open={meta.isExpanded ?? false}
              onOpenChange={() => onToggleThinking(message.id)}
              elapsedMs={meta.elapsedMs ?? 0}
              currentPhase={meta.currentPhase}
              topSources={meta.topSources}
              // Status-less messages are finished ones; the drawer reads
              // `done` as "Thought about your request".
              status={message.status ?? "done"}
              speed={fxSpeed}
              onSpeedChange={onFxSpeedChange}
              onSelect={(s) => onCite([s])}
              refinedPrompt={meta.refinedPrompt}
            />
          )}
          <AnswerText
            text={message.content}
            sources={message.sources ?? NO_SOURCES}
            onCite={(s) => onCite([s])}
          />
          <SourceChips
            sources={message.sources ?? NO_SOURCES}
            citedNs={message.citedNs ?? NO_CITED}
            onSelect={onCite}
          />
          {message.status === "done" && (
            <MessageActions
              message={message}
              rating={rating}
              streaming={streaming}
              onFeedback={onFeedback}
              onRegenerate={onRegenerate}
            />
          )}
        </>
      )}
    </div>
  );
});

export function MessageList({
  messages,
  onCite,
  ratings,
  streaming,
  onFeedback,
  onRegenerate,
  fxSpeed,
  onFxSpeedChange,
  onToggleThinking,
}: {
  messages: ChatMessage[];
  onCite: (sources: Source[]) => void;
  ratings: Record<string, "up" | "down" | null>;
  streaming: boolean;
  onFeedback: (messageKey: string, messageUuid: string, rating: "up" | "down") => void;
  onRegenerate: (messageId: string) => void;
  fxSpeed: FxSpeed;
  onFxSpeedChange: (speed: FxSpeed) => void;
  onToggleThinking: (messageId: string) => void;
}) {
  return (
    // aria-live on the container, not per message: the assistant answer streams
    // in with no focus change, so nothing would be announced otherwise.
    <div
      className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-8"
      aria-live="polite"
      aria-busy={messages.at(-1)?.status === "streaming" || messages.at(-1)?.status === "thinking"}
    >
      {messages.map((m) => (
        <MessageItem
          key={m.id}
          message={m}
          onCite={onCite}
          rating={ratings[m.id] ?? null}
          streaming={streaming}
          onFeedback={onFeedback}
          onRegenerate={onRegenerate}
          fxSpeed={fxSpeed}
          onFxSpeedChange={onFxSpeedChange}
          onToggleThinking={onToggleThinking}
        />
      ))}
    </div>
  );
}

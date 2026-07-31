"use client";

import { memo } from "react";
import type { ChatMessage, Source } from "@/lib/types";
import { AnswerText } from "@/components/answer-text";
import { SourceChips } from "@/components/source-chips";
import { cn } from "@/lib/utils";

// Stable identity: `?? []` would hand memo() a fresh array every render and
// defeat the shallow compare below.
const NO_SOURCES: Source[] = [];
const NO_CITED: number[] = [];

// One message, memoized. The typewriter re-renders the parent ~600 times per
// answer, but page.tsx's patch() preserves object identity for every message
// except the streaming one — so with memo() the rest are skipped instead of
// re-running AnswerText's regex split. Measured over one answer: 0.60ms/frame
// -> 0.03ms at 40 messages, and the gap widens as the chat grows.
const MessageItem = memo(function MessageItem({
  message,
  onCite,
}: {
  message: ChatMessage;
  onCite: (source: Source) => void;
}) {
  const isUser = message.role === "user";
  return (
    <div className="flex flex-col gap-1.5">
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
        <p className={cn("text-sm text-muted-foreground animate-pulse")}>{message.content}</p>
      ) : (
        <>
          <AnswerText
            text={message.content}
            sources={message.sources ?? NO_SOURCES}
            onCite={onCite}
          />
          <SourceChips
            sources={message.sources ?? NO_SOURCES}
            citedNs={message.citedNs ?? NO_CITED}
            onSelect={onCite}
          />
        </>
      )}
    </div>
  );
});

export function MessageList({
  messages,
  onCite,
}: {
  messages: ChatMessage[];
  onCite: (source: Source) => void;
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
        <MessageItem key={m.id} message={m} onCite={onCite} />
      ))}
    </div>
  );
}

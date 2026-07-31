"use client";

import { motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/utils";
import { ChatInput, ChatInputTextArea, ChatInputSubmit } from "@/components/ui/chat-input";

// The composer never changes size between states — only its position animates.
// (The old version animated a flex container that morphed from tall+centered to
// a short docked bar, so the box resized and its children reflowed mid-flight:
// that was the "stretch".) Here the pill keeps one width and one intrinsic
// height; `top` glides from the vertical middle to just above the bottom.
export function ChatComposer({
  value,
  onChange,
  onSubmit,
  loading,
  onStop,
  docked,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  loading: boolean;
  onStop: () => void;
  docked: boolean;
}) {
  const reduceMotion = useReducedMotion();

  return (
    <div className={cn("relative w-full", docked ? "h-[104px] shrink-0" : "flex-1")}>
      <motion.div
        initial={false}
        animate={{ top: docked ? "auto" : "50%", bottom: docked ? 20 : "auto" }}
        transition={
          reduceMotion
            ? { duration: 0 }
            : { type: "spring", stiffness: 120, damping: 20, mass: 0.9 }
        }
        style={{ y: docked ? 0 : "-50%" }}
        className="absolute inset-x-0 mx-auto w-full max-w-2xl px-4"
      >
        {/* Hint fades on its own so it never collapses the box height. */}
        <motion.p
          initial={false}
          animate={{ opacity: docked ? 0 : 1, height: docked ? 0 : "auto" }}
          transition={{ duration: 0.2 }}
          className="mb-4 overflow-hidden text-center text-sm text-muted-foreground"
        >
          Ask anything about your documents. Answers cite exact pages.
        </motion.p>

        {/* focus-within (not JS state): the frost clears the moment any child
            gains focus and can't get stuck if React misses a focus event. */}
        <div
          onClick={(e) => {
            (e.currentTarget as HTMLElement).querySelector("textarea")?.focus();
          }}
          className={cn(
            "rounded-2xl bg-background/85 backdrop-blur-md transition-[filter,opacity] duration-300 ease-out",
            !docked &&
              "blur-[2px] opacity-80 hover:blur-[1px] hover:opacity-90 focus-within:blur-none focus-within:opacity-100",
          )}
        >
          <ChatInput
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onSubmit={onSubmit}
            loading={loading}
            onStop={onStop}
          >
            <ChatInputTextArea placeholder="Ask about your documents…" />
            <ChatInputSubmit />
          </ChatInput>
        </div>
      </motion.div>
    </div>
  );
}

# User-supplied shadcn components (paste verbatim, paths as given)

## components/ui/chat-input.tsx
```tsx
"use client";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { useTextareaResize } from "@/hooks/use-textarea-resize";
import { ArrowUpIcon } from "lucide-react";
import type React from "react";
import { createContext, useContext } from "react";

interface ChatInputContextValue {
    value?: string;
    onChange?: React.ChangeEventHandler<HTMLTextAreaElement>;
    onSubmit?: () => void;
    loading?: boolean;
    onStop?: () => void;
    variant?: "default" | "unstyled";
    rows?: number;
}

const ChatInputContext = createContext<ChatInputContextValue>({});

interface ChatInputProps extends Omit<ChatInputContextValue, "variant"> {
    children: React.ReactNode;
    className?: string;
    variant?: "default" | "unstyled";
    rows?: number;
}

function ChatInput({
    children,
    className,
    variant = "default",
    value,
    onChange,
    onSubmit,
    loading,
    onStop,
    rows = 1,
}: ChatInputProps) {
    const contextValue: ChatInputContextValue = {
        value,
        onChange,
        onSubmit,
        loading,
        onStop,
        variant,
        rows,
    };

    return (
        <ChatInputContext.Provider value={contextValue}>
            <div
                className={cn(
                    variant === "default" &&
                        "flex flex-col items-end w-full p-2 rounded-2xl border border-input bg-transparent focus-within:ring-1 focus-within:ring-ring focus-within:outline-none",
                    variant === "unstyled" && "flex items-start gap-2 w-full",
                    className,
                )}
            >
                {children}
            </div>
        </ChatInputContext.Provider>
    );
}

ChatInput.displayName = "ChatInput";

interface ChatInputTextAreaProps extends React.ComponentProps<typeof Textarea> {
    value?: string;
    onChange?: React.ChangeEventHandler<HTMLTextAreaElement>;
    onSubmit?: () => void;
    variant?: "default" | "unstyled";
}

function ChatInputTextArea({
    onSubmit: onSubmitProp,
    value: valueProp,
    onChange: onChangeProp,
    className,
    variant: variantProp,
    ...props
}: ChatInputTextAreaProps) {
    const context = useContext(ChatInputContext);
    const value = valueProp ?? context.value ?? "";
    const onChange = onChangeProp ?? context.onChange;
    const onSubmit = onSubmitProp ?? context.onSubmit;
    const rows = context.rows ?? 1;

    // Convert parent variant to textarea variant unless explicitly overridden
    const variant =
        variantProp ?? (context.variant === "default" ? "unstyled" : "default");

    const textareaRef = useTextareaResize(value, rows);
    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (!onSubmit) {
            return;
        }
        if (e.key === "Enter" && !e.shiftKey) {
            if (typeof value !== "string" || value.trim().length === 0) {
                return;
            }
            e.preventDefault();
            onSubmit();
        }
    };

    return (
        <Textarea
            ref={textareaRef}
            {...props}
            value={value}
            onChange={onChange}
            onKeyDown={handleKeyDown}
            className={cn(
                "max-h-[400px] min-h-0 resize-none overflow-x-hidden",
                variant === "unstyled" &&
                    "border-none focus-visible:ring-0 focus-visible:ring-offset-0 shadow-none",
                className,
            )}
            rows={rows}
        />
    );
}

ChatInputTextArea.displayName = "ChatInputTextArea";

interface ChatInputSubmitProps extends React.ComponentProps<typeof Button> {
    onSubmit?: () => void;
    loading?: boolean;
    onStop?: () => void;
}

function ChatInputSubmit({
    onSubmit: onSubmitProp,
    loading: loadingProp,
    onStop: onStopProp,
    className,
    ...props
}: ChatInputSubmitProps) {
    const context = useContext(ChatInputContext);
    const loading = loadingProp ?? context.loading;
    const onStop = onStopProp ?? context.onStop;
    const onSubmit = onSubmitProp ?? context.onSubmit;

    if (loading && onStop) {
        return (
            <Button
                onClick={onStop}
                className={cn(
                    "shrink-0 rounded-full p-1.5 h-fit border dark:border-zinc-600",
                    className,
                )}
                {...props}
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-label="Stop"
                >
                    <title>Stop</title>
                    <rect x="6" y="6" width="12" height="12" />
                </svg>
            </Button>
        );
    }

    const isDisabled =
        typeof context.value !== "string" || context.value.trim().length === 0;

    return (
        <Button
            className={cn(
                "shrink-0 rounded-full p-1.5 h-fit border dark:border-zinc-600",
                className,
            )}
            disabled={isDisabled}
            onClick={(event) => {
                event.preventDefault();
                if (!isDisabled) {
                    onSubmit?.();
                }
            }}
            {...props}
        >
            <ArrowUpIcon />
        </Button>
    );
}

ChatInputSubmit.displayName = "ChatInputSubmit";

export { ChatInput, ChatInputTextArea, ChatInputSubmit };
```

## components/ui/button.tsx (originui variant)
```tsx
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";

import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-lg text-sm font-medium transition-colors outline-offset-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring/70 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground shadow-sm shadow-black/5 hover:bg-primary/90",
        destructive:
          "bg-destructive text-destructive-foreground shadow-sm shadow-black/5 hover:bg-destructive/90",
        outline:
          "border border-input bg-background shadow-sm shadow-black/5 hover:bg-accent hover:text-accent-foreground",
        secondary:
          "bg-secondary text-secondary-foreground shadow-sm shadow-black/5 hover:bg-secondary/80",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-lg px-3 text-xs",
        lg: "h-10 rounded-lg px-8",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
    );
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
```

## components/ui/textarea.tsx (originui variant)
```tsx
import * as React from "react";

import { cn } from "@/lib/utils";

const Textarea = React.forwardRef<HTMLTextAreaElement, React.ComponentProps<"textarea">>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          "flex min-h-[80px] w-full rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground shadow-sm shadow-black/5 transition-shadow placeholder:text-muted-foreground/70 focus-visible:border-ring focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/20 disabled:cursor-not-allowed disabled:opacity-50",
          className,
        )}
        ref={ref}
        {...props}
      />
    );
  },
);
Textarea.displayName = "Textarea";

export { Textarea };
```

## hooks/use-textarea-resize.ts
```tsx
"use client";

import { useLayoutEffect, useRef } from "react";
import type { ComponentProps } from "react";

export function useTextareaResize(
    value: ComponentProps<"textarea">["value"],
    rows = 1,
) {
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    // biome-ignore lint/correctness/useExhaustiveDependencies: <explanation>
    useLayoutEffect(() => {
        const textArea = textareaRef.current;

        if (textArea) {
            // Get the line height to calculate minimum height based on rows
            const computedStyle = window.getComputedStyle(textArea);
            const lineHeight = Number.parseInt(computedStyle.lineHeight, 10) || 20;
            const padding =
                Number.parseInt(computedStyle.paddingTop, 10) +
                Number.parseInt(computedStyle.paddingBottom, 10);

            // Calculate minimum height based on rows
            const minHeight = lineHeight * rows + padding;

            // Reset height to auto first to get the correct scrollHeight
            textArea.style.height = "0px";
            const scrollHeight = Math.max(textArea.scrollHeight, minHeight);

            // Set the final height
            textArea.style.height = `${scrollHeight + 2}px`;
        }
    }, [textareaRef, value, rows]);

    return textareaRef;
}
```

NPM deps: lucide-react, @radix-ui/react-slot, class-variance-authority

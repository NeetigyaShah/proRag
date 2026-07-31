"use client";

import { Button } from "@/components/ui/button";
import { FileUp, FileText, Loader2 } from "lucide-react";
import { useRef } from "react";

interface TopBarProps {
  documents: number | null;
  ingesting: string | null;
  lastIngestResult: string | null;
  onUpload: (file: File) => void;
}

export function TopBar({ documents, ingesting, lastIngestResult, onUpload }: TopBarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <header className="flex items-center justify-between px-5 py-4 border-b border-border shrink-0">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold tracking-tight text-foreground">
          Pro<span className="text-amber">Rag</span>
        </span>
        <span className="text-xs text-muted-foreground hidden sm:inline">
          answers with receipts
        </span>
      </div>

      <div className="flex items-center gap-3 text-xs text-muted-foreground">
        {/* Upload progress/result lands here with no focus change, so a screen
            reader would otherwise never announce it. */}
        <span className="tabular-nums" aria-live="polite">
          {ingesting ? (
            <span className="inline-flex items-center gap-1.5">
              <Loader2 className="size-3.5 animate-spin" />
              Ingesting {ingesting}…
            </span>
          ) : lastIngestResult ? (
            lastIngestResult
          ) : documents === null ? (
            "…"
          ) : (
            <span className="inline-flex items-center gap-1.5">
              <FileText className="size-3.5" />
              {documents} {documents === 1 ? "document" : "documents"}
            </span>
          )}
        </span>
        <input
          ref={fileInputRef}
          type="file"
          // Mirrors ALLOWED_SUFFIXES in prorag/ingest/router.py — .txt/.md/.tsv
          // were accepted by the backend but hidden by the file picker.
          accept=".pdf,.txt,.md,.docx,.pptx,.csv,.xlsx,.tsv"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onUpload(file);
            e.target.value = "";
          }}
        />
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="gap-1.5"
          // ingest() is the real guard (drag-and-drop bypasses this button);
          // disabling here just stops the picker opening to no effect.
          disabled={!!ingesting}
          onClick={() => fileInputRef.current?.click()}
        >
          <FileUp className="size-3.5" />
          Upload
        </Button>
      </div>
    </header>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import { X, ChevronLeft, ChevronRight, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Source } from "@/lib/types";
import type { PDFDocumentProxy } from "pdfjs-dist";

// Parsed documents live past unmount: reopening a PDF you've already viewed is
// instant instead of re-downloading and re-parsing the whole file.
const pdfCache = new Map<string, Promise<PDFDocumentProxy>>();

// pdf.js viewer panel: renders the cited page and draws the amber bbox
// highlight. Y-axis math ported directly from gotoPage() in web/chat.js
// (PDF points grow up, canvas pixels grow down).
export function PdfViewer({ source, onClose }: { source: Source; onClose: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  const [page, setPage] = useState(source.page || 1);
  const [numPages, setNumPages] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(0); // bumps when a document finishes parsing

  // Clicking a different citation in the same document just jumps pages.
  useEffect(() => {
    if (source.page) setPage(source.page);
  }, [source.page, source.n]);

  // Backend URLs are relative to the FastAPI root — route through the /api proxy.
  const fileUrl = `/api/files/${source.doc_id}/original`;
  const isPdf = !source.title || source.title.toLowerCase().endsWith(".pdf");
  const [textContent, setTextContent] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    if (!isPdf) {
      // Non-PDF (txt/md/csv): show the raw text instead of a pdf.js error.
      fetch(fileUrl)
        .then((r) => r.text())
        .then((t) => {
          if (!cancelled) setTextContent(t.slice(0, 20000));
        })
        .catch(() => {
          if (!cancelled) setError("Could not load this document.");
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
      return () => {
        cancelled = true;
      };
    }

    setLoading(true);
    import("pdfjs-dist").then(async (pdfjsLib) => {
      pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
        "pdfjs-dist/build/pdf.worker.min.mjs",
        import.meta.url,
      ).toString();
      try {
        let pending = pdfCache.get(fileUrl);
        if (!pending) {
          // Fetch the whole file in one request: the Next dev proxy answers
          // range requests with a bodyless 204, so pdf.js range fetching can't
          // work through it. At 1.5MB served in ~70ms locally that's fine —
          // the win comes from pdfCache, not from partial fetches.
          pending = pdfjsLib.getDocument({ url: fileUrl }).promise;
          // Evict on failure so a transient error isn't cached forever.
          pending.catch(() => pdfCache.delete(fileUrl));
          pdfCache.set(fileUrl, pending);
        }
        const doc = await pending;
        if (cancelled) return;
        pdfRef.current = doc;
        setNumPages(doc.numPages);
        setReady((n) => n + 1); // tell the render effect the doc is available
      } catch {
        pdfCache.delete(fileUrl); // don't cache a failure
        if (!cancelled) setError("Could not load this PDF.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    });

    return () => {
      cancelled = true;
      // Document stays in pdfCache for instant reopen; only the local ref drops.
      pdfRef.current = null;
    };
  }, [fileUrl, isPdf]);

  useEffect(() => {
    const pdf = pdfRef.current;
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    if (!pdf || !canvas || !stage || page < 1 || page > pdf.numPages) return;

    let cancelled = false;
    (async () => {
      const pdfPage = await pdf.getPage(page);
      if (cancelled) return;
      const containerWidth = stage.parentElement?.clientWidth ?? 640;
      const scale = Math.min(1.6, (containerWidth - 48) / pdfPage.getViewport({ scale: 1 }).width);
      const viewport = pdfPage.getViewport({ scale });

      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      await pdfPage.render({ canvas, canvasContext: ctx, viewport }).promise;
      if (cancelled) return;

      stage.querySelectorAll("#pdf-highlight").forEach((el) => el.remove());
      const bbox = source.bbox && source.bbox.length === 4 ? source.bbox : null;
      if (bbox && page === (source.page ?? 1)) {
        const [x0, y0, x1, y1] = bbox;
        const rect = document.createElement("div");
        rect.id = "pdf-highlight";
        rect.style.left = `${x0 * viewport.scale}px`;
        rect.style.top = `${viewport.height - y1 * viewport.scale}px`;
        rect.style.width = `${(x1 - x0) * viewport.scale}px`;
        rect.style.height = `${(y1 - y0) * viewport.scale}px`;
        stage.appendChild(rect);
      }
    })();

    return () => {
      cancelled = true;
    };
    // `ready` re-runs this once the document finishes parsing (pdfRef is a ref,
    // so it can't trigger the effect on its own).
  }, [page, ready, source.bbox, source.page]);

  return (
    <div className="flex h-full w-full flex-col border-l border-border bg-card">
      <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-3 shrink-0">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-foreground">
            {source.title || source.doc_id}
          </p>
          {numPages > 0 && (
            <p className="text-xs text-muted-foreground tabular-nums">
              Page {page} / {numPages}
            </p>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <Button
            size="icon"
            variant="ghost"
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            aria-label="Previous page"
          >
            <ChevronLeft className="size-4" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            disabled={numPages > 0 && page >= numPages}
            onClick={() => setPage((p) => Math.min(numPages, p + 1))}
            aria-label="Next page"
          >
            <ChevronRight className="size-4" />
          </Button>
          <Button size="icon" variant="ghost" asChild>
            <a href={`${fileUrl}?download=1`} aria-label="Download original">
              <Download className="size-4" />
            </a>
          </Button>
          <Button size="icon" variant="ghost" onClick={onClose} aria-label="Close viewer">
            <X className="size-4" />
          </Button>
        </div>
      </div>

      <div className="relative flex-1 overflow-auto scrollbar-thin p-6">
        {loading && (
          <p className="text-sm text-muted-foreground">Loading page…</p>
        )}
        {error && <p className="text-sm text-destructive">{error}</p>}
        {textContent !== null ? (
          <pre className="mx-auto max-w-3xl whitespace-pre-wrap rounded-md bg-secondary/50 p-6 font-mono text-sm leading-relaxed text-foreground">
            {textContent}
          </pre>
        ) : (
          <div ref={stageRef} className="relative mx-auto w-fit">
            <canvas ref={canvasRef} className="rounded-md shadow-lg shadow-black/30" />
          </div>
        )}
      </div>
    </div>
  );
}

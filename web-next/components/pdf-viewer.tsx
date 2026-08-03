"use client";

import { useEffect, useRef, useState } from "react";
import { X, ChevronLeft, ChevronRight, Download, ZoomIn, ZoomOut, Maximize } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Source, ViewerTarget } from "@/lib/types";
import type { PDFDocumentProxy } from "pdfjs-dist";

// Zoom bounds and step: fit-page is 100%, in/out move in 20% steps.
const ZOOM_MIN = 0.2;
const ZOOM_MAX = 4;
const ZOOM_STEP = 0.2;
const clampZoom = (z: number) =>
  Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, Math.round(z * 100) / 100));

// Parsed documents live past unmount: reopening a PDF you've already viewed is
// instant instead of re-downloading and re-parsing the whole file.
const pdfCache = new Map<string, Promise<PDFDocumentProxy>>();

// pdf.js viewer panel: renders the cited page and draws the amber bbox
// highlight. Y-axis math ported directly from gotoPage() in web/chat.js
// (PDF points grow up, canvas pixels grow down).
export function PdfViewer({ target, onClose }: { target: ViewerTarget; onClose: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const pdfRef = useRef<PDFDocumentProxy | null>(null);
  // pdf.js module handle for the text layer (see load effect below). Stored in
  // a ref: the render effect needs TextLayer, and a second dynamic import would
  // create a duplicate pdf.js chunk instance in the dev server.
  const pdfjsRef = useRef<typeof import("pdfjs-dist") | null>(null);
  const [page, setPage] = useState(target.sources[0]?.page ?? 1);
  const [numPages, setNumPages] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(0); // bumps when a document finishes parsing
  // Multiplier on the fit-page scale; 1 = fit. Bbox positions below multiply
  // by viewport.scale, so the highlight tracks zoom automatically.
  const [zoom, setZoom] = useState(1);

  // A different selection (chunk or document) jumps to its page. Keyed on the
  // first source only — re-selecting the same group must not fight the user's
  // own page navigation.
  useEffect(() => {
    const p = target.sources[0]?.page;
    if (p) setPage(p);
  }, [target.sources[0]?.n, target.sources[0]?.page]);

  // Backend URLs are relative to the FastAPI root — route through the /api proxy.
  const title = target.title ?? target.sources[0]?.title;
  const fileUrl = `/api/files/${target.doc_id}/original`;
  const isPdf = !title || title.toLowerCase().endsWith(".pdf");
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
      pdfjsRef.current = pdfjsLib;
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
      const fitScale = Math.min(1.6, (containerWidth - 48) / pdfPage.getViewport({ scale: 1 }).width);
      const scale = fitScale * zoom;
      const viewport = pdfPage.getViewport({ scale });

      canvas.width = viewport.width;
      canvas.height = viewport.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      await pdfPage.render({ canvas, canvasContext: ctx, viewport }).promise;
      if (cancelled) return;

      stage.querySelectorAll(".textLayer").forEach((el) => el.remove());
      // Pen-highlighter pass: pdf.js's text layer places one absolutely
      // positioned span per text item, exactly over the rendered glyphs. Any
      // span whose rect falls inside a cited chunk's region gets a translucent
      // amber tint — the text itself is "marked", like a marker pen, instead
      // of a bordered box around it. Multiple cited chunks on the same page
      // each tint their own lines.
      const boxes = target.sources
        .filter((s) => (s.page ?? 1) === page && s.bbox && s.bbox.length === 4)
        .map((s) => {
          const [x0, y0, x1, y1] = s.bbox as [number, number, number, number];
          return {
            left: x0 * viewport.scale,
            top: viewport.height - y1 * viewport.scale,
            width: (x1 - x0) * viewport.scale,
            height: (y1 - y0) * viewport.scale,
          };
        });
      if (boxes.length === 0) return;
      const TextLayer = pdfjsRef.current?.TextLayer;
      if (!TextLayer) return;
      const textLayerDiv = document.createElement("div");
      textLayerDiv.className = "textLayer";
      textLayerDiv.style.width = `${viewport.width}px`;
      textLayerDiv.style.height = `${viewport.height}px`;
      stage.appendChild(textLayerDiv);
      const textLayer = new TextLayer({
        textContentSource: await pdfPage.getTextContent(),
        container: textLayerDiv,
        viewport,
      });
      await textLayer.render();
      if (cancelled) return;
      const stageRect = stage.getBoundingClientRect();
      for (const span of textLayerDiv.querySelectorAll("span")) {
        const r = span.getBoundingClientRect();
        const sx = r.left - stageRect.left;
        const sy = r.top - stageRect.top;
        const sw = r.width;
        const sh = r.height;
        if (sw === 0 || sh === 0) continue;
        for (const b of boxes) {
          const ix = Math.max(0, Math.min(sx + sw, b.left + b.width) - Math.max(sx, b.left));
          const iy = Math.max(0, Math.min(sy + sh, b.top + b.height) - Math.max(sy, b.top));
          // A line counts as highlighted when a quarter of it sits inside the
          // chunk region — boundary lines get fully tinted like a marker.
          if (iy > 0 && ix / sw > 0.25) {
            span.classList.add("pdf-hl");
            break;
          }
        }
      }
    })();

    return () => {
      cancelled = true;
    };
    // `ready` re-runs this once the document finishes parsing (pdfRef is a ref,
    // so it can't trigger the effect on its own). zoom re-runs it so the bbox
    // highlight and canvas are drawn at the new scale.
  }, [page, ready, zoom, target]);

  return (
    <div className="flex h-full w-full flex-col border-l border-border bg-card">
      <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-3 shrink-0">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-foreground">
            {title || target.doc_id}
          </p>
          {numPages > 0 && (
            <p className="text-xs text-muted-foreground tabular-nums">
              Page {page} / {numPages}
            </p>
          )}
        </div>
        <div className="flex items-center gap-0.5 shrink-0">
          <Button
            size="icon"
            variant="ghost"
            disabled={zoom <= ZOOM_MIN}
            onClick={() => setZoom((z) => clampZoom(z - ZOOM_STEP))}
            aria-label="Zoom out"
            title="Zoom out"
          >
            <ZoomOut className="size-4" />
          </Button>
          <button
            type="button"
            onClick={() => setZoom(1)}
            className="min-w-10 rounded-md px-1 text-center text-xs tabular-nums text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline focus-visible:outline-2 focus-visible:outline-ring/70"
            aria-label="Reset zoom to fit page"
            title="Fit page"
          >
            {Math.round(zoom * 100)}%
          </button>
          <Button
            size="icon"
            variant="ghost"
            disabled={zoom >= ZOOM_MAX}
            onClick={() => setZoom((z) => clampZoom(z + ZOOM_STEP))}
            aria-label="Zoom in"
            title="Zoom in"
          >
            <ZoomIn className="size-4" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            onClick={() => setZoom(1)}
            aria-label="Fit page"
            title="Fit page"
          >
            <Maximize className="size-4" />
          </Button>
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

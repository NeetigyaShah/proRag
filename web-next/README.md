# ProRag — web-next

Next.js + Tailwind + shadcn-style frontend for ProRag, a RAG chatbot with cited PDF sources.

## Run

```bash
npm run dev
```

Opens on http://localhost:3000. Requires the FastAPI backend running on
http://127.0.0.1:8000 — `next.config.ts` rewrites `/api/*` to it, so the
browser never talks to port 8000 directly (no CORS).

## Build

```bash
npm run build
```

## Notes

- `components/ui/*` and `hooks/use-textarea-resize.ts` are pasted verbatim
  from the supplied shadcn components; compose around them, don't restyle
  their internals.
- `lib/sse.ts` ports the SSE frame parser from the old `web/chat.js` client.
- `components/pdf-viewer.tsx` ports the pdf.js render + bbox-highlight math
  (PDF y-axis flip) from the same file.
- Single accent color (soft orange, `--amber` CSS var in `app/globals.css`)
  marks citations, source chips, and PDF highlights.

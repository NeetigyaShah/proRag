# CONTEXT — ProRag

## Glossary

- **Document** — one uploaded file (PDF, DOCX, PPTX, XLSX, CSV, TSV, TXT, MD, and — planned — scanned PDFs and images). The unit of citation: every answer references documents, never raw files.
- **Chunk** — a ~700-token piece of a document. The unit of retrieval, context, and citation (`[S1]` numbers are chunks). Produced either by `chunk_elements` (Docling path: never crosses a page or an H1/H2 boundary; carries a heading breadcrumb and a bbox) or `chunk_pages` (plain-text fallback: fixed windows that may span pages).
- **Page anchor** — the `page_start` of a chunk, used in citations ("p.289"). An approximation: a fallback-path chunk that starts on p.289 may also cover p.290–291.
- **Receipt** — the user-facing citation: an `[Sn]` marker in the answer, a source chip (one per cited document), and a highlight over the cited text in the PDF viewer.
- **Arm** — one retrieval search: vector (meaning), keyword/BM25 (words), structured (tables). Hybrid retrieval runs several arms per query.
- **Fusion (RRF)** — merging the arms' ranked lists into one list, rewarding chunks several arms agree on. The order of record when the reranker can't decide.
- **Rerank** — a paid cross-encoder (`cohere/rerank-v3.5` via OpenRouter) scoring the query against each fused chunk; the final ordering before the crop.
- **Flatness guard** — when the reranker's top-5 scores are all within `rerank_flat_spread`, its order is noise: the fused order is kept instead (scores still attached).
- **Crop** — selecting the chunks that enter the prompt: hard score floor, ≤3 chunks per document, ≤12 documents, ≤6000 tokens, ≥150 characters.
- **Ingestion** — the parse → chunk → embed → store pipeline for one file.
- **Setup mode** — single-box Docker Compose deployment (the supported install path).
- **Scale path** — production hardening of that same stack (backups, health, resource limits, managed-Postgres/S3 migration docs). Deliberately no new components.

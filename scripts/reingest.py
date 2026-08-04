"""Re-ingest documents whose chunks were cut by the fallback chunker.

Detects them structurally: fallback chunks can span pages and carry no
heading_path (docling chunks never cross a page and always carry one). Re-runs
each blob through ingest_bytes() so docling + chunk_elements() produce
section-aware chunks with exact page anchors.

Safe to re-run: documents are replaced, chunks cascade-deleted.

Usage:
    python scripts/reingest.py --doc-id <uuid> [--doc-id <uuid> ...]
    python scripts/reingest.py --all
    # default (no flags): only fallback-chunked documents
"""

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import func, select

from prorag.db import SessionLocal
from prorag.ingest.core import ingest_bytes
from prorag.models import Chunk, Document


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    async with SessionLocal() as s:
        stmt = select(Document)
        if args.doc_id:
            stmt = stmt.where(Document.id.in_(args.doc_id))
        elif not args.all:
            # Default: only fallback-chunked documents — any chunk whose page
            # span crosses a boundary or whose heading breadcrumb is empty.
            stmt = (
                stmt.join(Chunk)
                .group_by(Document.id)
                .having(
                    func.bool_or(Chunk.page_start != Chunk.page_end)
                    | func.bool_or(
                        func.coalesce(func.array_length(Chunk.heading_path, 1), 0) == 0
                    )
                )
            )
        docs = (await s.execute(stmt)).scalars().unique().all()
        print(f"re-ingesting {len(docs)} documents")
        for d in docs:
            data = Path(d.blob_path).read_bytes()
            print(f"  {d.filename} ({len(data)} bytes)")
            # ingest_bytes dedupes by sha256 — the identical blob would come
            # back as a duplicate no-op. Delete the row first (chunks cascade;
            # the blob file itself is reused).
            await s.delete(d)
            await s.commit()
            await ingest_bytes(s, data, d.filename, d.collection)
            await s.commit()


asyncio.run(main())

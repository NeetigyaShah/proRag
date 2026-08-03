"""Adaptive context crop (§4.4). Pure functions, no I/O — unit-testable.

crop_context(): sort by rerank score, dedupe revisions of the same document
(same title_norm, keep newest doc_date), drop chunks under 150 chars, clamp
to a dynamic score floor (max(top_score - gap, floor)), min 3 / max 12
docs, and stop at a hard token budget rather than a doc count.

normalize_title(): strips revision/version/date noise from a title so two
printings of the same manual ("Safety Manual Rev 3", "Safety Manual (May
2021)") collapse to the same key. Computed at ingest time in the real
pipeline (a stored column); exposed here as a pure function so both ingest
and this module (and its tests) share one implementation.
"""

import re

_REV_NOISE = re.compile(
    r"""
    \(?\brev(?:ision)?\.?\s*\d+[a-z]?\)?   |  # Rev 3 / Revision 2a / (Rev. 4)
    \(?\bv\.?\s*\d+(?:\.\d+)*\)?           |  # v2 / V.3.1
    \(?\d{4}-\d{2}-\d{2}\)?                |  # 2024-05-01
    \(?\d{1,2}/\d{1,2}/\d{2,4}\)?          |  # 05/01/2024
    \(?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)
        [a-z]*[-\s]?\d{2,4}\)?                # May-2021, Jan 2024
    """,
    re.IGNORECASE | re.VERBOSE,
)


def normalize_title(title: str | None) -> str:
    """When called: by crop_context() during retrieval when a hit carries no
    precomputed title_norm, and at ingest time in the real pipeline (it backs
    a stored column). What: strips revision/version/date noise and normalizes
    case and whitespace so printings of the same manual collapse to one key.
    Returns: the normalized lowercase title, "" for None/empty input."""
    if not title:
        return ""
    t = _REV_NOISE.sub("", title)
    t = re.sub(r"[()\[\]]", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def crop_context(
    hits: list[dict],
    *,
    min_docs: int = 3,
    max_docs: int = 12,
    max_chunks_per_doc: int = 3,
    score_gap: float = 0.15,
    score_floor: float = 0.0,
    token_budget: int = 6000,
    min_chars: int = 150,
) -> list[dict]:
    """hits: rerank-scored dicts carrying at least {text, score}; optionally
    title/title_norm, doc_id, doc_date, token_count. Returns the cropped,
    ordered (best-first) list to hand to build_context_block().

    Dedup is revision-aware, NOT one-chunk-per-document: printings of the
    same manual (same title_norm, different doc_id) collapse to the newest
    one, but genuinely different chunks of one document (different sections
    of a long PDF) all survive up to `max_chunks_per_doc` — otherwise a
    single-document answer could never cite more than one section."""
    if not hits:
        return []

    ranked = sorted(hits, key=lambda h: h["score"], reverse=True)

    # Per title_norm: pick the winning printing (newest doc_date, else best
    # rank), then keep up to max_chunks_per_doc of that printing's chunks.
    deduped: list[dict] = []
    winner_by_title: dict[str, dict] = {}
    taken_by_title: dict[str, int] = {}

    def _title(h: dict) -> str:
        return h.get("title_norm") or normalize_title(h.get("title"))

    for h in ranked:
        title_norm = _title(h)
        winner = winner_by_title.get(title_norm)
        if winner is None:
            winner_by_title[title_norm] = h
            taken_by_title[title_norm] = 1
            deduped.append(h)
            continue
        w_date, h_date = winner.get("doc_date"), h.get("doc_date")
        if h_date and w_date and h_date > w_date:
            # A strictly newer printing of the same document wins the slot:
            # drop the older printing's chunks entirely.
            deduped[:] = [x for x in deduped if _title(x) != title_norm]
            winner_by_title[title_norm] = h
            taken_by_title[title_norm] = 1
            deduped.append(h)
            continue
        # Same printing (same doc_id): more chunks may join up to the cap.
        # Different printing that isn't newer: drop it.
        if h.get("doc_id") and h.get("doc_id") == winner.get("doc_id") and taken_by_title[title_norm] < max_chunks_per_doc:
            taken_by_title[title_norm] += 1
            deduped.append(h)

    substantial = [h for h in deduped if len(h.get("text", "")) >= min_chars]
    pool = substantial if len(substantial) >= min_docs else deduped
    if not pool:
        return []

    top_score = pool[0]["score"]
    floor = max(top_score - score_gap, score_floor)
    kept = [h for h in pool if h["score"] >= floor]

    if len(kept) < min_docs:
        kept = pool[:min_docs]
    elif len(kept) > max_docs:
        kept = kept[:max_docs]

    budgeted: list[dict] = []
    total = 0
    for i, h in enumerate(kept):
        tc = h.get("token_count") or len(h.get("text", "").split())
        if i >= min_docs and total + tc > token_budget:
            break
        budgeted.append(h)
        total += tc

    return budgeted

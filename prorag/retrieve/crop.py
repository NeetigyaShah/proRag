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
    score_gap: float = 0.15,
    score_floor: float = 0.0,
    token_budget: int = 6000,
    min_chars: int = 150,
) -> list[dict]:
    """hits: rerank-scored dicts carrying at least {text, score}; optionally
    title/title_norm, doc_date, token_count. Returns the cropped, ordered
    (best-first) list to hand to build_context_block()."""
    if not hits:
        return []

    ranked = sorted(hits, key=lambda h: h["score"], reverse=True)

    # revision-aware dedup: one slot per title_norm, prefer newer doc_date.
    deduped: list[dict] = []
    best_by_title: dict[str, dict] = {}
    for h in ranked:
        title_norm = h.get("title_norm") or normalize_title(h.get("title"))
        existing = best_by_title.get(title_norm)
        if existing is None:
            best_by_title[title_norm] = h
            deduped.append(h)
            continue
        existing_date, new_date = existing.get("doc_date"), h.get("doc_date")
        if new_date and existing_date and new_date > existing_date:
            deduped[deduped.index(existing)] = h
            best_by_title[title_norm] = h
        # else existing wins — it was ranked higher (or dates don't decide it)

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

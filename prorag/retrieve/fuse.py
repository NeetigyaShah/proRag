"""RRF fusion (k=60), everywhere — §4.3. Pure function, no I/O.

Fuses N ranked lists of hit-dicts (each must carry "chunk_id") keyed by
chunk_id: score = sum(weight_i / (k + rank_i)) across every list a chunk
appears in, rank_i is 1-based within that list. Dedup keeps the first-seen
hit's fields (doc_id/text/page/title/...), just replaces "score" with the
fused RRF score.
"""

RRF_K = 60


def rrf_fuse(ranked_lists: list[list[dict]], weights: list[float] | None = None) -> list[dict]:
    """ranked_lists: each a list of hit-dicts already sorted best-first.
    weights: one per list, default 1.0 each (architecture's per-arm weights,
    e.g. vector 1.0 / fts 1.0 / structured 1.2, are passed in by the caller).
    Returns hits sorted by fused score, descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused: dict = {}
    for hits, weight in zip(ranked_lists, weights, strict=True):
        for rank, hit in enumerate(hits, start=1):
            cid = hit["chunk_id"]
            contribution = weight / (RRF_K + rank)
            if cid not in fused:
                fused[cid] = {**hit, "score": contribution}
            else:
                fused[cid]["score"] += contribution

    return sorted(fused.values(), key=lambda h: h["score"], reverse=True)

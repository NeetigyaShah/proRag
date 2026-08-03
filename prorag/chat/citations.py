"""Citation normalization + resolution. Pure functions, no I/O.

normalize_citations() maps common LLM deviations — (S1), [s1], [source 1], [1]
— to the canonical [S1] form. resolve_citations() then reads the normalized
text and, given the ordered list of chunks that were actually in context,
returns Source-shaping dicts for every index that was cited (in order of
first appearance), dropping out-of-range indices instead of raising.
"""

import re

_DEVIATION_PATTERNS = [
    re.compile(r"\(S(\d+)\)", re.IGNORECASE),
    re.compile(r"\[s(\d+)\]", re.IGNORECASE),
    re.compile(r"\[source\s+(\d+)\]", re.IGNORECASE),
    re.compile(r"\[(\d+)\]"),
]

_CANONICAL = re.compile(r"\[S(\d+)\]")


def normalize_citations(text: str) -> str:
    """When called: on the raw LLM answer — chat/router.py before persisting
    and resolving sources, and eval/runner.py when scoring answers. What:
    maps common LLM citation deviations ((S1), [s1], [source 1], [1]) to the
    canonical [S1] form. Returns: the normalized text."""
    normalized = text
    for pattern in _DEVIATION_PATTERNS:
        normalized = pattern.sub(lambda m: f"[S{m.group(1)}]", normalized)
    return normalized


def extract_cited_indices(normalized_text: str) -> list[int]:
    """Indices in order of first appearance, deduped."""
    seen: dict[int, None] = {}
    for match in _CANONICAL.finditer(normalized_text):
        seen.setdefault(int(match.group(1)), None)
    return list(seen.keys())


def resolve_citations(normalized_text: str, chunks: list[dict]) -> list[dict]:
    """chunks[i] corresponds to [S{i+1}] (1-indexed, matching the context block
    numbering handed to the LLM). Returns sources only for indices that were
    actually cited and that exist in `chunks`; out-of-range indices are dropped.
    """
    cited = extract_cited_indices(normalized_text)
    sources = []
    for n in cited:
        idx = n - 1
        if 0 <= idx < len(chunks):
            sources.append({"n": n, **chunks[idx]})
    return sources


def build_context_block(chunks: list[dict]) -> str:
    """chunks: list of {doc_id, title, page, text} in retrieval order.
    Produces the [Sn]-numbered context block the answerer is prompted with.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        page = c.get("page")
        page_str = f" — p.{page}" if page is not None else ""
        blocks.append(f"[S{i}] {c.get('title', 'Untitled')}{page_str}\n{c['text']}")
    return "\n\n".join(blocks)

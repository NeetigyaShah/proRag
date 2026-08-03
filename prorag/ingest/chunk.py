"""Chunking. Pure functions — no I/O, no DB.

chunk_pages(): Phase 1's fixed ~700-token chunker over per-page plain text —
kept as the fallback path (plain .txt/.md, or PyMuPDF text with no structure).

chunk_elements(): Phase 3's heading- and page-boundary aware chunker (§3.2)
over a Docling-style element stream. Never merges across a page boundary or
an H1/H2 heading boundary; prepends a heading breadcrumb to embed_text.

# ponytail: token count is approximated by whitespace word count (~1 word ~=
# 1 token, close enough for chunk sizing at this scale). Swap for a real BPE
# tokenizer (tiktoken) if chunk-size drift starts mattering.
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One fixed-window chunk from chunk_pages(): the plain text plus the page
    span it was cut from (used for citation page anchors)."""

    ord: int
    text: str
    page_start: int
    page_end: int
    token_count: int


def _tokens(text: str) -> list[str]:
    return text.split()


def chunk_pages(
    pages: list[str],
    target_tokens: int = 700,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Chunk a document given as a list of per-page plain text (1 entry per page,
    index 0 = page 1). Packs pages into ~target_tokens windows with overlap,
    never losing which page(s) a chunk came from.
    """
    # Flatten to a single (word, page_no) stream so overlap can cross page
    # boundaries while still tracking page provenance per chunk.
    stream: list[tuple[str, int]] = []
    for page_no, page_text in enumerate(pages, start=1):
        for word in _tokens(page_text):
            stream.append((word, page_no))

    if not stream:
        return []

    chunks: list[Chunk] = []
    i = 0
    ord_ = 0
    n = len(stream)
    step = max(target_tokens - overlap_tokens, 1)
    while i < n:
        window = stream[i : i + target_tokens]
        if not window:
            break
        words = [w for w, _ in window]
        pages_in_window = [p for _, p in window]
        chunks.append(
            Chunk(
                ord=ord_,
                text=" ".join(words),
                page_start=pages_in_window[0],
                page_end=pages_in_window[-1],
                token_count=len(words),
            )
        )
        ord_ += 1
        if i + target_tokens >= n:
            break
        i += step
    return chunks


# ---------------------------------------------------------------------------
# Phase 3: heading- and page-boundary aware chunking (§3.2)
# ---------------------------------------------------------------------------


@dataclass
class Element:
    """One prose element from the Docling stream (a paragraph/list-item/etc)."""

    text: str
    page: int
    heading_path: list[str] = field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class StructuredChunk:
    """One chunk from chunk_elements(): raw text, embed_text with the heading
    breadcrumb prepended, the page span, and the union bbox used for citation
    highlighting."""

    ord: int
    text: str
    embed_text: str
    heading_path: list[str]
    page_start: int
    page_end: int
    bbox: list[float] | None
    token_count: int


def _heading_key(heading_path: list[str]) -> tuple:
    return tuple(heading_path[:2])  # H1/H2 only — deeper headings may pack together


def _union_bbox(boxes: list[tuple[float, float, float, float]]) -> list[float] | None:
    """When called: chunk_elements(), once per chunk window. What: tight
    bounding box around every element box in the window. Returns: [x0, y0,
    x1, y1], or None when no element carries a bbox."""
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return [x0, y0, x1, y1]


def breadcrumb(heading_path: list[str]) -> str:
    """When called: chunk_elements(), once per chunk, to build embed_text.
    What: renders the heading path as a '#'-prefixed breadcrumb, '' when there
    is no heading. Returns: the breadcrumb string."""
    if not heading_path:
        return ""
    return "# " + " > ".join(heading_path)


def _group_by_page_and_heading(elements: list[Element]) -> list[list[Element]]:
    """Never merging across a page boundary *and* never merging across an
    H1/H2 (§3.2) means a chunk-packing group can't cross either — split the
    reading-order stream wherever page or heading[:2] changes."""
    groups: list[list[Element]] = []
    current: list[Element] = []
    current_key = None
    for el in elements:
        key = (el.page, _heading_key(el.heading_path))
        if current and key != current_key:
            groups.append(current)
            current = []
        current.append(el)
        current_key = key
    if current:
        groups.append(current)
    return groups


def chunk_elements(
    elements: list[Element],
    target_tokens: int = 700,
    overlap_tokens: int = 100,
) -> list[StructuredChunk]:
    """Pack a Docling-style element stream into ~target_tokens chunks, never
    crossing a page boundary or an H1/H2 heading boundary, with a heading
    breadcrumb prepended to embed_text (§3.2)."""
    if not elements:
        return []

    chunks: list[StructuredChunk] = []
    ord_ = 0
    step = max(target_tokens - overlap_tokens, 1)

    for group in _group_by_page_and_heading(elements):
        stream: list[tuple[str, Element]] = [(w, el) for el in group for w in el.text.split()]
        n = len(stream)
        if n == 0:
            continue
        i = 0
        while i < n:
            window = stream[i : i + target_tokens]
            words = [w for w, _ in window]
            els_in_window = [e for _, e in window]
            heading_path = els_in_window[0].heading_path
            page = els_in_window[0].page
            bbox = _union_bbox([e.bbox for e in els_in_window if e.bbox])
            text = " ".join(words)
            crumb = breadcrumb(heading_path)
            embed_text = f"{crumb}\n{text}" if crumb else text
            chunks.append(
                StructuredChunk(
                    ord=ord_,
                    text=text,
                    embed_text=embed_text,
                    heading_path=heading_path,
                    page_start=page,
                    page_end=page,
                    bbox=bbox,
                    token_count=len(words),
                )
            )
            ord_ += 1
            if i + target_tokens >= n:
                break
            i += step

    return chunks

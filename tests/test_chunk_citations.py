"""Pure-function smoke test: chunking + citation resolution. No DB, no LLM."""

from prorag.chat.citations import (
    build_context_block,
    extract_cited_indices,
    normalize_citations,
    resolve_citations,
)
from prorag.ingest.chunk import Element, breadcrumb, chunk_elements, chunk_pages


def test_chunk_pages_basic():
    pages = ["word " * 500, "word " * 500]  # 500 words/page, 2 pages = 1000 words
    chunks = chunk_pages(pages, target_tokens=700, overlap_tokens=100)

    assert len(chunks) >= 2
    assert chunks[0].page_start == 1
    # first chunk (700 words) spans into page 2 given 500 words on page 1
    assert chunks[0].page_end == 2
    assert chunks[0].token_count == 700
    # ord is sequential
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_chunk_pages_empty():
    assert chunk_pages([], 700, 100) == []
    assert chunk_pages([""], 700, 100) == []


def test_chunk_pages_single_small_page():
    chunks = chunk_pages(["hello world"], 700, 100)
    assert len(chunks) == 1
    assert chunks[0].page_start == chunks[0].page_end == 1
    assert chunks[0].token_count == 2


def test_breadcrumb_empty_for_no_heading():
    assert breadcrumb([]) == ""


def test_breadcrumb_joins_path():
    assert breadcrumb(["Safety Manual", "4. Drills", "4.2 Fire"]) == "# Safety Manual > 4. Drills > 4.2 Fire"


def test_chunk_elements_never_crosses_page_boundary():
    elements = [
        Element(text="word " * 400, page=1, heading_path=["Manual", "Intro"]),
        Element(text="word " * 400, page=2, heading_path=["Manual", "Intro"]),
    ]
    chunks = chunk_elements(elements, target_tokens=700, overlap_tokens=100)
    for c in chunks:
        assert c.page_start == c.page_end  # never spans two pages


def test_chunk_elements_never_crosses_heading_boundary():
    elements = [
        Element(text="alpha " * 300, page=1, heading_path=["Manual", "4. Drills"]),
        Element(text="beta " * 300, page=1, heading_path=["Manual", "5. Appendix"]),
    ]
    chunks = chunk_elements(elements, target_tokens=700, overlap_tokens=100)
    # two distinct heading sections on the same page never merge into one chunk
    assert not any("alpha" in c.text and "beta" in c.text for c in chunks)


def test_chunk_elements_embed_text_has_breadcrumb_prefix():
    elements = [Element(text="fire drills monthly", page=1, heading_path=["Safety Manual", "4. Drills"])]
    chunks = chunk_elements(elements)
    assert chunks[0].embed_text.startswith("# Safety Manual > 4. Drills\n")
    assert chunks[0].text == "fire drills monthly"  # text itself has no breadcrumb


def test_chunk_elements_no_heading_no_breadcrumb():
    elements = [Element(text="plain text", page=1, heading_path=[])]
    chunks = chunk_elements(elements)
    assert chunks[0].embed_text == "plain text"


def test_chunk_elements_bbox_union():
    elements = [
        Element(text="a b c", page=1, heading_path=["H"], bbox=(10, 10, 50, 50)),
        Element(text="d e f", page=1, heading_path=["H"], bbox=(20, 5, 60, 40)),
    ]
    chunks = chunk_elements(elements, target_tokens=700, overlap_tokens=100)
    assert chunks[0].bbox == [10, 5, 60, 50]


def test_chunk_elements_empty():
    assert chunk_elements([]) == []


def test_normalize_citations_deviations():
    text = "Fire drills are monthly (S1). See also [s2] and [source 3] plus [4]."
    normalized = normalize_citations(text)
    assert "[S1]" in normalized
    assert "[S2]" in normalized
    assert "[S3]" in normalized
    assert "[S4]" in normalized
    assert "(S1)" not in normalized


def test_extract_cited_indices_order_and_dedup():
    normalized = "See [S3] and [S1], also [S3] again."
    assert extract_cited_indices(normalized) == [3, 1]


def test_resolve_citations_drops_out_of_range():
    chunks = [
        {"doc_id": "d1", "text": "chunk one", "page": 1, "title": "Doc"},
        {"doc_id": "d1", "text": "chunk two", "page": 2, "title": "Doc"},
    ]
    normalized = "Claim one [S1]. Claim two [S2]. Bogus claim [S99]."
    sources = resolve_citations(normalized, chunks)

    assert len(sources) == 2  # S99 silently dropped, never a 500
    assert sources[0]["n"] == 1
    assert sources[0]["text"] == "chunk one"
    assert sources[1]["n"] == 2


def test_build_context_block_numbers_in_order():
    chunks = [
        {"title": "Manual A", "page": 5, "text": "alpha"},
        {"title": "Manual B", "page": None, "text": "beta"},
    ]
    block = build_context_block(chunks)
    assert "[S1] Manual A — p.5\nalpha" in block
    assert "[S2] Manual B\nbeta" in block


def test_end_to_end_chunk_then_cite():
    """Chunk a fake 2-page doc, pretend the LLM cited chunk 1, resolve it back
    to a source with the right page number — the whole citation contract in
    one pass, no DB or LLM involved."""
    pages = ["Fire drills occur monthly per SOLAS. " * 20, "Unrelated appendix text. " * 20]
    chunks = chunk_pages(pages, target_tokens=700, overlap_tokens=100)
    context_chunks = [
        {"doc_id": "doc-1", "page": c.page_start, "title": "Safety Manual", "text": c.text} for c in chunks
    ]

    raw_answer = "Fire drills must happen monthly [S1]."
    normalized = normalize_citations(raw_answer)
    sources = resolve_citations(normalized, context_chunks)

    assert len(sources) == 1
    assert sources[0]["page"] == 1
    assert sources[0]["doc_id"] == "doc-1"

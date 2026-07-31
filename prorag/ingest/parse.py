"""MIME routing → Docling (PDF/DOCX/PPTX, optional) → PyMuPDF fallback →
pandas (CSV/XLSX/TSV) → plain text (§3.1).

Docling is an optional import: this module stays import-safe with docling
absent, and callers automatically fall back to the PyMuPDF path on import
failure or parse failure.
"""

from dataclasses import dataclass, field

import fitz  # PyMuPDF

from prorag.ingest.chunk import Element

try:
    from docling.document_converter import DocumentConverter  # type: ignore

    DOCLING_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where docling is absent
    DocumentConverter = None  # type: ignore
    DOCLING_AVAILABLE = False


# Magic-byte signatures. Extension-only routing trusts the uploader; a file
# renamed to .pdf would otherwise be handed straight to a PDF parser.
# ponytail: stdlib signature check, not python-magic — these four families are
# the only binary formats we accept. Text formats (txt/md/csv/tsv) have no
# signature, so they return None and fall through to extension routing.
_SIGNATURES: list[tuple[bytes, str]] = [
    (b"%PDF-", "application/pdf"),
    (b"PK\x03\x04", "zip"),  # docx/pptx/xlsx are all zip containers
]

_OOXML_MARKERS: list[tuple[bytes, str]] = [
    (b"word/", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    (b"ppt/", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    (b"xl/", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
]


def sniff_mime(data: bytes) -> str | None:
    """Detect MIME from content. Returns None when the format has no reliable
    signature (plain text), meaning the caller should trust the extension."""
    for sig, mime in _SIGNATURES:
        if data.startswith(sig):
            if mime != "zip":
                return mime
            # OOXML: peek at the zip's local file headers to tell the three apart.
            head = data[:4096]
            for marker, ooxml_mime in _OOXML_MARKERS:
                if marker in head:
                    return ooxml_mime
            return None  # some other zip; let the extension decide
    return None


@dataclass
class ParsedTable:
    caption: str | None
    columns: list[str]
    rows: list[dict]
    page_no: int | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class ParsedDocument:
    """Docling's split element stream: prose elements + Table elements (§3.1)."""

    prose: list[Element] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)
    page_count: int | None = None


def parse_to_pages(data: bytes, mime: str) -> list[str]:
    """Return a list of per-page plain text. Non-PDF text is treated as one page.
    Phase 1 fallback path — no layout, no tables."""
    if mime == "application/pdf":
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            return [page.get_text() for page in doc]
        finally:
            doc.close()
    # .txt / .md: no real pagination, whole file is "page 1"
    return [data.decode("utf-8", errors="replace")]


DOCLING_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
}

STRUCTURED_MIMES = {
    "text/csv",
    "text/tab-separated-values",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
}


def try_docling_parse(data: bytes, mime: str, filename: str) -> ParsedDocument | None:
    """Parse via Docling, splitting the element stream into prose vs Table
    elements (§3.1/§3.2). Returns None if Docling isn't installed or parsing
    fails — the caller falls back to PyMuPDF."""
    if not DOCLING_AVAILABLE or mime not in DOCLING_MIMES:
        return None
    try:
        import tempfile
        from pathlib import Path

        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            tmp_path = f.name

        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        dl_doc = result.document

        prose: list[Element] = []
        tables: list[ParsedTable] = []
        heading_stack: list[str] = []

        for item, _level in dl_doc.iterate_items():
            label = getattr(item, "label", "") or type(item).__name__
            prov = item.prov[0] if getattr(item, "prov", None) else None
            page_no = prov.page_no if prov else None
            bbox = None
            if prov is not None and getattr(prov, "bbox", None) is not None:
                b = prov.bbox
                bbox = (b.l, b.t, b.r, b.b)

            if str(label).lower() in ("section_header", "title", "heading"):
                text = getattr(item, "text", "") or ""
                heading_stack = [text]  # ponytail: flat 1-level breadcrumb; nest by level if needed
                continue

            if str(label).lower() == "table" or type(item).__name__ == "TableItem":
                table_df = item.export_to_dataframe() if hasattr(item, "export_to_dataframe") else None
                if table_df is not None:
                    columns = [str(c) for c in table_df.columns]
                    rows = table_df.to_dict(orient="records")
                    tables.append(
                        ParsedTable(
                            caption=getattr(item, "caption_text", None) or None,
                            columns=columns,
                            rows=rows,
                            page_no=page_no,
                            bbox=bbox,
                        )
                    )
                continue

            text = getattr(item, "text", "") or ""
            if text.strip():
                prose.append(Element(text=text, page=page_no or 1, heading_path=list(heading_stack), bbox=bbox))

        return ParsedDocument(prose=prose, tables=tables, page_count=getattr(dl_doc, "num_pages", None))
    except Exception:
        return None


def parse_structured(data: bytes, mime: str) -> list[ParsedTable]:
    """CSV/XLSX/TSV via pandas — each sheet becomes a table (§3.3). page_no is
    always None; citation degrades to a row-range anchor."""
    import io

    import pandas as pd

    if mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        return [
            ParsedTable(caption=name, columns=[str(c) for c in df.columns], rows=df.to_dict(orient="records"))
            for name, df in sheets.items()
        ]

    sep = "\t" if mime == "text/tab-separated-values" else ","
    df = pd.read_csv(io.BytesIO(data), sep=sep)
    return [ParsedTable(caption=None, columns=[str(c) for c in df.columns], rows=df.to_dict(orient="records"))]


def guess_mime(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if lower.endswith(".pptx"):
        return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    if lower.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if lower.endswith(".tsv"):
        return "text/tab-separated-values"
    if lower.endswith(".csv"):
        return "text/csv"
    if lower.endswith(".md"):
        return "text/markdown"
    return "text/plain"

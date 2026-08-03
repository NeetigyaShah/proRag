"""MIME routing → Docling (PDF/DOCX/PPTX, optional) → PyMuPDF fallback →
pandas (CSV/XLSX/TSV) → plain text (§3.1).

Docling is an optional import: this module stays import-safe with docling
absent, and callers automatically fall back to the PyMuPDF path on import
failure or parse failure.
"""

import importlib.util
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from prorag.ingest.chunk import Element

# Probe WITHOUT importing: `import docling` pulls torch, which costs ~30-75s
# of process start on this machine. find_spec only checks the package exists;
# the real import happens once, lazily, inside _get_docling_converter() on the
# first docling parse — same idiom as llm.py's local embedder.
DOCLING_AVAILABLE = importlib.util.find_spec("docling") is not None


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
    """One detected table from any parse path (Docling or pandas): header
    columns plus record rows, ready for table-artifact generation."""

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


def _make_docling_converter():
    """Build the memory-tuned DocumentConverter (docling >= 2.117 API).

    Measured on this machine (i5-1135G7 laptop, 16 GB): the default pipeline
    OOM'd (std::bad_alloc) at page ~99 of a 474-page PDF. With these options
    the same PDF converts fully (~218s) with 19 tables extracted and only the
    last few pages lost to memory pressure — docling reports per-page
    failures and continues, so the loss is graceful either way.

    Settings that matter for memory (per docling docs):
    - batch sizes pinned to 1 (defaults process several pages at once)
    - do_ocr=False — digital PDFs have a text layer; OCR is pure memory cost
    - images_scale=1.0 — the default 2.0 quadruples pixel memory per page
    - picture/table image generation off (nothing here consumes those)
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorOptions, ThreadedPdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline

    opts = ThreadedPdfPipelineOptions(
        ocr_batch_size=1,
        layout_batch_size=1,
        table_batch_size=1,
        accelerator_options=AcceleratorOptions(num_threads=1, device="cpu"),
    )
    opts.do_ocr = False
    opts.images_scale = 1.0
    opts.generate_page_images = False
    opts.generate_picture_images = False
    opts.generate_table_images = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=opts,
            )
        }
    )


_docling_converter = None


def _get_docling_converter():
    """Lazy singleton, same idiom as llm.py's local embedder: the pipeline
    holds ~100MB of layout-model weights; rebuilding it per ingest request
    would reload them every time. Imports docling here (first call only) so
    process startup and the test suite never pay torch's import cost."""
    global _docling_converter
    if _docling_converter is None:
        if not DOCLING_AVAILABLE:
            raise ImportError("docling not installed; PyMuPDF fallback is active")
        _docling_converter = _make_docling_converter()
    return _docling_converter


def try_docling_parse(data: bytes, mime: str, filename: str) -> ParsedDocument | None:
    """Parse via Docling, splitting the element stream into prose vs Table
    elements (§3.1/§3.2). Returns None if Docling isn't installed or parsing
    fails — the caller falls back to PyMuPDF."""
    if not DOCLING_AVAILABLE or mime not in DOCLING_MIMES:
        return None
    tmp_path: str | None = None
    try:
        suffix = Path(filename).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            tmp_path = f.name
        converter = _get_docling_converter()
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
                # doc=dl_doc: docling >= 2.117 deprecates the no-arg call (and
                # the table's text extraction needs the parent document for
                # header-row resolution in newer versions).
                table_df = item.export_to_dataframe(doc=dl_doc) if hasattr(item, "export_to_dataframe") else None
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

        # docling's num_pages flipped from a property to a method in 2.117 —
        # call it when it's callable so the value survives both APIs. A bound
        # method object here would crash the Document INSERT (asyncpg DataError)
        # after a successful parse, 500-ing the ingest and orphaning the blob.
        num_pages = getattr(dl_doc, "num_pages", None)
        page_count = num_pages() if callable(num_pages) else num_pages

        return ParsedDocument(prose=prose, tables=tables, page_count=page_count)
    except Exception:
        return None
    finally:
        # The converter reads the file at convert() time; the parsed document
        # lives in memory after that, so the temp copy is always disposable.
        # Without this, every docling parse leaked one file into %TEMP%.
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)


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
    """When called: ingest_bytes() on every upload, before parsing, to route
    the file to a parser. What: maps the filename extension to a MIME type;
    unknown suffixes fall back to text/plain. Returns: the MIME string."""
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

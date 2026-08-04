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
from prorag.settings import settings

# Probe WITHOUT importing: `import docling` pulls torch, which costs ~30-75s
# of process start on this machine. find_spec only checks the package exists;
# the real import happens once, lazily, inside _get_docling_converter() on the
# first docling parse — same lazy-singleton idiom as llm.py's embedder paths.
DOCLING_AVAILABLE = importlib.util.find_spec("docling") is not None


# Magic-byte signatures. Extension-only routing trusts the uploader; a file
# renamed to .pdf would otherwise be handed straight to a PDF parser.
# ponytail: stdlib signature check, not python-magic — these four families are
# the only binary formats we accept. Text formats (txt/md/csv/tsv) have no
# signature, so they return None and fall through to extension routing.
_SIGNATURES: list[tuple[bytes, str]] = [
    (b"%PDF-", "application/pdf"),
    (b"PK\x03\x04", "zip"),  # docx/pptx/xlsx are all zip containers
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
    (b"RIFF", "image/webp"),  # RIFF....WEBP — the WEBP tag sits at offset 8
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

    Scanned PDFs and raster uploads never reach this converter: they are
    OCR'd via the OpenRouter vision API (ocr_pages) and take the plain-text
    chunk path instead (core.py).
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
    """Lazy singleton: the pipeline holds ~100MB of layout-model weights;
    rebuilding it per ingest request would reload them every time. Imports
    docling here (first call only) so process startup and the test suite
    never pay torch's import cost."""
    global _docling_converter
    if _docling_converter is None:
        if not DOCLING_AVAILABLE:
            raise ImportError("docling not installed; PyMuPDF fallback is active")
        _docling_converter = _make_docling_converter()
    return _docling_converter


def _pdf_page_count(data: bytes, mime: str) -> int:
    """Page count for OCR routing: 1 for raster uploads, the PDF's real page
    count otherwise (a scan may be many pages)."""
    if mime != "application/pdf":
        return 1
    try:
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        try:
            return doc.page_count
        finally:
            doc.close()
    except Exception:
        return 1


def _pdf_has_text_layer(data: bytes) -> bool:
    """Cheap text-layer probe for PDFs: PyMuPDF's get_text per page. A scan
    renders glyphs but has no text layer — those go through the OpenRouter
    OCR path, while digital PDFs keep docling's no-OCR pipeline."""
    try:
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        try:
            return any(len(page.get_text().strip()) > 0 for page in doc)
        finally:
            doc.close()
    except Exception:
        return True  # probe failed — assume digital, let the parse decide


def _render_png(data: bytes, mime: str, page_no: int) -> bytes:
    """Render one page as PNG bytes for the vision OCR call: page `page_no`
    (1-based) of a PDF via PyMuPDF, or the image itself via Pillow."""
    if mime == "application/pdf":
        import fitz

        doc = fitz.open(stream=data, filetype="pdf")
        try:
            page = doc[page_no - 1]
            pix = page.get_pixmap(dpi=200)
            return pix.tobytes("png")
        finally:
            doc.close()
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(data))
    img.load()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def ocr_pages(data: bytes, mime: str, page_count: int) -> list[str]:
    """OCR a scanned PDF or raster upload via a paid OpenRouter vision model
    (settings.ocr_model). One API call per page: render the page to PNG,
    ask the model to transcribe all text exactly. No local OCR engine — the
    laptop stays cold. Returns per-page plain text; raises on any failure
    (the caller rejects the upload rather than ingesting binary garbage)."""
    import base64

    import httpx

    api_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — OCR needs it (set it in .env)")

    prompt = "Transcribe ALL text in this image exactly as written. Preserve line breaks and section order. Output only the transcribed text, no commentary."
    pages: list[str] = []
    async with httpx.AsyncClient(timeout=settings.ocr_timeout) as client:
        for page_no in range(1, page_count + 1):
            png = _render_png(data, mime, page_no)
            b64 = base64.b64encode(png).decode()
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": settings.ocr_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            ],
                        }
                    ],
                },
            )
            resp.raise_for_status()
            pages.append(resp.json()["choices"][0]["message"]["content"] or "")
    return pages


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
    if lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp")):
        return "image/png" if lower.endswith(".png") else (
            "image/jpeg" if lower.endswith((".jpg", ".jpeg")) else (
                "image/tiff" if lower.endswith((".tiff", ".tif")) else "image/webp"))
    return "text/plain"

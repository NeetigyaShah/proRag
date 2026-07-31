"""Table artifact generation (§3.3). Pure functions — no I/O, no DB.

Three artifacts per table:
  1. table_rows (JSONB) — one row per record, done by the caller via build_table_rows().
  2. a summary chunk — synthesized text, no LLM.
  3. row-window chunks — markdown, ~25 rows per window, header repeated.

Wide tables (>12 columns) additionally get row-per-chunk key-value serialization
instead of (or alongside) the row windows.
"""

from dataclasses import dataclass

WIDE_TABLE_COLUMNS = 12
DEFAULT_WINDOW_SIZE = 25


@dataclass
class TableArtifact:
    kind: str  # table_summary | table_window | row
    text: str
    row_start: int | None = None  # 0-based row index range this artifact covers
    row_end: int | None = None


def is_wide_table(columns: list[str]) -> bool:
    return len(columns) > WIDE_TABLE_COLUMNS


def build_table_rows(columns: list[str], rows: list[list]) -> list[dict]:
    """Zip header + row values into the JSONB `data` dict for table_rows."""
    return [dict(zip(columns, row, strict=False)) for row in rows]


def _row_to_markdown(row: dict, columns: list[str]) -> str:
    return "| " + " | ".join(str(row.get(c, "")) for c in columns) + " |"


def build_table_summary(
    caption: str | None,
    columns: list[str],
    rows: list[dict],
    sample_size: int = 3,
) -> str:
    """ "Table: <caption>. Columns: a, b, c. 240 rows. Sample: <first 3 rows as markdown>" (§3.3)."""
    title = caption or "Untitled table"
    col_list = ", ".join(columns)
    sample_rows = rows[:sample_size]
    sample_md = "\n".join(_row_to_markdown(r, columns) for r in sample_rows)
    return f"Table: {title}. Columns: {col_list}. {len(rows)} rows.\nSample:\n| {' | '.join(columns)} |\n{sample_md}"


def build_row_windows(
    columns: list[str],
    rows: list[dict],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> list[TableArtifact]:
    """Serialize rows to markdown in windows of ~window_size rows, header repeated
    in every window (§3.3, artifact 3)."""
    if not rows:
        return []
    header = f"| {' | '.join(columns)} |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    artifacts = []
    for start in range(0, len(rows), window_size):
        window = rows[start : start + window_size]
        body = "\n".join(_row_to_markdown(r, columns) for r in window)
        text = f"{header}\n{sep}\n{body}"
        artifacts.append(
            TableArtifact(kind="table_window", text=text, row_start=start, row_end=start + len(window) - 1)
        )
    return artifacts


def build_row_kv_chunks(columns: list[str], rows: list[dict]) -> list[TableArtifact]:
    """Wide tables (>12 cols) and forms: one chunk per row, serialized as
    key-value pairs (`vessel: X | inspection_date: 2026-03-01 | result: pass`)."""
    artifacts = []
    for i, row in enumerate(rows):
        text = " | ".join(f"{c}: {row.get(c, '')}" for c in columns)
        artifacts.append(TableArtifact(kind="row", text=text, row_start=i, row_end=i))
    return artifacts


def build_table_artifacts(
    caption: str | None,
    columns: list[str],
    rows: list[dict],
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> list[TableArtifact]:
    """Full artifact set for one table per §3.3: summary + windows, plus
    row-per-chunk key-value for wide tables."""
    artifacts = [TableArtifact(kind="table_summary", text=build_table_summary(caption, columns, rows))]
    if is_wide_table(columns):
        artifacts.extend(build_row_kv_chunks(columns, rows))
    else:
        artifacts.extend(build_row_windows(columns, rows, window_size))
    return artifacts

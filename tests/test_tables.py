"""Pure-function tests for table artifact generation (§3.3). No DB, no I/O."""

from prorag.ingest.tables import (
    build_row_kv_chunks,
    build_row_windows,
    build_table_artifacts,
    build_table_rows,
    build_table_summary,
    is_wide_table,
)

COLUMNS = ["vessel", "inspection_date", "result"]
ROWS = [
    {"vessel": "Vessel A", "inspection_date": "2026-01-01", "result": "pass"},
    {"vessel": "Vessel B", "inspection_date": "2026-02-01", "result": "fail"},
    {"vessel": "Vessel C", "inspection_date": "2026-03-01", "result": "pass"},
]


def test_build_table_rows_zips_header_and_values():
    rows = build_table_rows(COLUMNS, [["A", "2026-01-01", "pass"]])
    assert rows == [{"vessel": "A", "inspection_date": "2026-01-01", "result": "pass"}]


def test_is_wide_table():
    assert not is_wide_table(COLUMNS)
    assert is_wide_table([f"col{i}" for i in range(13)])


def test_build_table_summary_contains_caption_columns_count_and_sample():
    summary = build_table_summary("Inspections", COLUMNS, ROWS)
    assert "Table: Inspections." in summary
    assert "Columns: vessel, inspection_date, result." in summary
    assert "3 rows." in summary
    assert "Vessel A" in summary
    # only sample_size rows by default
    assert summary.count("Vessel") == 3  # header col name doesn't contain "Vessel"


def test_build_table_summary_no_caption_falls_back():
    summary = build_table_summary(None, COLUMNS, ROWS)
    assert "Untitled table" in summary


def test_build_row_windows_repeats_header_per_window():
    many_rows = [{"vessel": f"V{i}", "inspection_date": "2026-01-01", "result": "pass"} for i in range(60)]
    windows = build_row_windows(COLUMNS, many_rows, window_size=25)
    assert len(windows) == 3  # 25 + 25 + 10
    assert windows[0].row_start == 0 and windows[0].row_end == 24
    assert windows[-1].row_start == 50 and windows[-1].row_end == 59
    for w in windows:
        assert w.kind == "table_window"
        assert "vessel" in w.text.splitlines()[0]  # header repeated


def test_build_row_windows_empty():
    assert build_row_windows(COLUMNS, []) == []


def test_build_row_kv_chunks_one_per_row():
    kv_chunks = build_row_kv_chunks(COLUMNS, ROWS)
    assert len(kv_chunks) == 3
    assert kv_chunks[0].text == "vessel: Vessel A | inspection_date: 2026-01-01 | result: pass"
    assert all(c.kind == "row" for c in kv_chunks)


def test_build_table_artifacts_narrow_table_uses_windows():
    artifacts = build_table_artifacts("Inspections", COLUMNS, ROWS)
    assert artifacts[0].kind == "table_summary"
    assert all(a.kind == "table_window" for a in artifacts[1:])


def test_build_table_artifacts_wide_table_uses_row_kv():
    wide_columns = [f"col{i}" for i in range(13)]
    wide_rows = [{c: "v" for c in wide_columns}]
    artifacts = build_table_artifacts("Wide", wide_columns, wide_rows)
    assert artifacts[0].kind == "table_summary"
    assert artifacts[1].kind == "row"

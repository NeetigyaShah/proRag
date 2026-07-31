"""Pure-function tests for Phase 6 eval: golden-set parsing and deterministic
metric computation (hit-rate, keyword coverage, citation validity). No DB, no
LLM, no ragas — run_eval() itself needs a live session and isn't exercised
here (§8 Phase 6)."""

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import prorag.eval.router as router_mod
from prorag.eval.runner import (
    DEFAULT_GOLDEN_PATH,
    aggregate_metrics,
    citation_validity,
    keyword_coverage,
    load_golden,
    ragas_available,
    retrieval_hit_rate,
)
from prorag.settings import settings

# ---- golden-set loading -----------------------------------------------------


def test_load_golden_checked_in_template_parses():
    entries = load_golden(DEFAULT_GOLDEN_PATH)
    assert len(entries) == 8
    for e in entries:
        assert "question" in e
        assert "expected_answer_contains" in e


def test_load_golden_custom_path(tmp_path):
    path = tmp_path / "mini.jsonl"
    path.write_text(
        '{"question": "q1", "expected_answer_contains": ["a"], "expected_doc_ids": ["d1"], "notes": ""}\n'
        "\n"  # blank lines are skipped
        '{"question": "q2", "expected_answer_contains": [], "expected_source_titles": ["Doc"], "notes": ""}\n',
        encoding="utf-8",
    )
    entries = load_golden(path)
    assert len(entries) == 2
    assert entries[0]["question"] == "q1"
    assert entries[1]["expected_source_titles"] == ["Doc"]


# ---- retrieval hit-rate -----------------------------------------------------


def test_hit_rate_matches_by_doc_id():
    entry = {"expected_doc_ids": ["abc"]}
    hits = [{"doc_id": "xyz"}, {"doc_id": "abc"}]
    assert retrieval_hit_rate(entry, hits) == 1.0


def test_hit_rate_matches_by_title_substring():
    entry = {"expected_source_titles": ["Safety Manual"]}
    hits = [{"doc_id": "d1", "title": "Safety Manual (Rev 3)"}]
    assert retrieval_hit_rate(entry, hits) == 1.0


def test_hit_rate_miss():
    entry = {"expected_doc_ids": ["abc"]}
    hits = [{"doc_id": "xyz"}]
    assert retrieval_hit_rate(entry, hits) == 0.0


def test_hit_rate_vacuous_when_no_expectation_set():
    entry = {"expected_doc_ids": [], "expected_source_titles": []}
    assert retrieval_hit_rate(entry, []) == 1.0


# ---- keyword coverage -------------------------------------------------------


def test_keyword_coverage_full_match():
    assert keyword_coverage("Fire drills are monthly per SOLAS", ["monthly", "SOLAS"]) == 1.0


def test_keyword_coverage_partial_match():
    assert keyword_coverage("Fire drills are monthly", ["monthly", "SOLAS"]) == 0.5


def test_keyword_coverage_case_insensitive():
    assert keyword_coverage("MONTHLY drills", ["monthly"]) == 1.0


def test_keyword_coverage_empty_expectation_is_full_score():
    assert keyword_coverage("anything", []) == 1.0


def test_keyword_coverage_no_match():
    assert keyword_coverage("irrelevant text", ["monthly"]) == 0.0


# ---- citation validity -------------------------------------------------------


def test_citation_validity_all_in_range():
    assert citation_validity("The rule applies [S1] and also [S2].", num_sources=2) == 1.0


def test_citation_validity_out_of_range_lowers_score():
    assert citation_validity("See [S1] and [S5].", num_sources=1) == 0.5


def test_citation_validity_no_citations_is_full_score():
    assert citation_validity("No citations here.", num_sources=3) == 1.0


def test_citation_validity_normalizes_deviations_first():
    # (S1) is a known LLM deviation normalize_citations() repairs to [S1]
    assert citation_validity("See (S1) for details.", num_sources=1) == 1.0


# ---- aggregation -------------------------------------------------------------


def test_aggregate_metrics_averages():
    per_question = [
        {"hit_rate": 1.0, "keyword_coverage": 1.0, "citation_validity": 1.0},
        {"hit_rate": 0.0, "keyword_coverage": 0.5, "citation_validity": 1.0},
    ]
    agg = aggregate_metrics(per_question)
    assert agg["hit_rate"] == 0.5
    assert agg["keyword_coverage"] == 0.75
    assert agg["citation_validity"] == 1.0
    assert agg["n_questions"] == 2


def test_aggregate_metrics_empty():
    agg = aggregate_metrics([])
    assert agg["hit_rate"] == 0.0
    assert agg["n_questions"] == 0


def test_ragas_available_is_a_bool():
    # Environment doesn't install ragas (heavy, optional per §8 Phase 6) —
    # just assert the probe doesn't raise and returns a bool either way.
    assert isinstance(ragas_available(), bool)


# ---- router guards ------------------------------------------------------------
# POST /eval/run spends one answer call per golden entry, so it has to honour the
# same daily cap /chat does and cannot run unbounded. No DB: get_session is
# overridden and run_eval is stubbed.


@pytest.fixture
def client(monkeypatch):
    from prorag.db import get_session
    from prorag.main import app

    app.dependency_overrides[get_session] = lambda: None
    monkeypatch.setattr(router_mod, "check_daily_cap", _noop_cap)
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


async def _noop_cap(session):
    return None


def test_eval_run_honours_daily_cost_cap(client, monkeypatch):
    async def over_cap(session):
        raise HTTPException(429, "daily cost cap reached")

    monkeypatch.setattr(router_mod, "check_daily_cap", over_cap)
    monkeypatch.setattr(router_mod, "run_eval", _should_not_run)

    assert client.post("/eval/run").status_code == 429


async def _should_not_run(session):
    raise AssertionError("run_eval must not be called once the cap is hit")


def test_eval_run_times_out_instead_of_hanging(client, monkeypatch):
    async def hangs(session):
        await asyncio.sleep(10)

    monkeypatch.setattr(router_mod, "run_eval", hangs)
    monkeypatch.setattr(settings, "eval_timeout_seconds", 0.05)

    assert client.post("/eval/run").status_code == 504

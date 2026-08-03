"""Pure-function tests for the prefill agent (query refinement): JSON parsing,
fallback safety, and the changed:false no-op contract. No DB, no LLM, no
network — every completion is monkeypatched."""

import pytest

from prorag.retrieve.prefill import _parse, prefill


def _async(value):
    async def _coro(*_args, **_kwargs):
        return value

    return _coro


# ---- JSON parsing ----------------------------------------------------------


def test_parse_plain():
    data = _parse('{"cleaned": "What is the deadline?", "changed": true}')
    assert data["cleaned"] == "What is the deadline?"
    assert data["changed"] is True


def test_parse_strips_code_fence():
    # The free gemma model wraps its JSON in ```json fences — must not break.
    raw = '```json\n{"cleaned": "fire drill frequency requirement", "changed": true}\n```'
    data = _parse(raw)
    assert data["cleaned"] == "fire drill frequency requirement"


def test_parse_garbage_raises():
    with pytest.raises(Exception):
        _parse("sure, here you go!")


# ---- fallback safety (never worsens the prompt) -----------------------------


async def test_prefill_falls_back_on_llm_error(monkeypatch):
    async def boom(*_args, **_kwargs):
        raise RuntimeError("free tier rate limited")

    monkeypatch.setattr("prorag.retrieve.prefill.prefill_completion", boom)
    assert await prefill("what is the deadline") == "what is the deadline"


async def test_prefill_falls_back_on_unparseable_output(monkeypatch):
    monkeypatch.setattr("prorag.retrieve.prefill.prefill_completion", _async("just fixing typos for you!"))
    assert await prefill("wat is the deadlne") == "wat is the deadlne"


async def test_prefill_returns_raw_when_model_says_unchanged(monkeypatch):
    # changed:false = model judged the prompt already good — keep it verbatim.
    monkeypatch.setattr(
        "prorag.retrieve.prefill.prefill_completion",
        _async('{"cleaned": "summarize", "changed": false}'),
    )
    assert await prefill("summarize") == "summarize"


async def test_prefill_returns_cleaned_on_happy_path(monkeypatch):
    monkeypatch.setattr(
        "prorag.retrieve.prefill.prefill_completion",
        _async('```json\n{"cleaned": "What is the deadline for the quarterly report submission?", "changed": true}\n```'),
    )
    assert await prefill("wat is the deadlne for the quarterly report submisson?") == (
        "What is the deadline for the quarterly report submission?"
    )


async def test_prefill_ignores_missing_cleaned_field(monkeypatch):
    monkeypatch.setattr(
        "prorag.retrieve.prefill.prefill_completion",
        _async('{"changed": true}'),
    )
    assert await prefill("some query") == "some query"


async def test_prefill_blank_query_returns_verbatim():
    # Never raises even with an empty input (stub completion would break).
    assert await prefill("   ") == "   "

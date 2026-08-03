"""Pure-function tests for Phase 2 retrieval: RRF fusion, adaptive crop,
planner JSON parsing fallback. No DB, no LLM, no network — except the one
EXPLAIN regression test at the bottom, which skips if the DB is unreachable."""

import asyncio
import random

import pytest

import prorag.retrieve.plan as plan_module
from prorag.retrieve.arms import _distance_expr, build_structured_query
from prorag.retrieve.crop import crop_context, normalize_title
from prorag.retrieve.fuse import rrf_fuse
from prorag.retrieve.plan import _extract_json, _fallback, plan
from prorag.retrieve.rerank import _rerank_api


def _hit(chunk_id, **kw):
    return {"chunk_id": chunk_id, "text": "x" * 200, "score": 0.0, **kw}


# ---- RRF fusion ----------------------------------------------------------


def test_rrf_fuse_single_list_preserves_order():
    hits = [_hit(1), _hit(2), _hit(3)]
    fused = rrf_fuse([hits])
    assert [h["chunk_id"] for h in fused] == [1, 2, 3]


def test_rrf_fuse_boosts_chunk_appearing_in_multiple_lists():
    list_a = [_hit(1), _hit(2), _hit(3)]
    list_b = [_hit(2), _hit(1), _hit(4)]
    fused = rrf_fuse([list_a, list_b])
    # chunk 1 and 2 both appear near the top of both lists -> should outrank
    # chunk 3/4 which each appear in only one list
    top_two = {fused[0]["chunk_id"], fused[1]["chunk_id"]}
    assert top_two == {1, 2}


def test_rrf_fuse_weights_apply_per_list():
    list_a = [_hit(1)]
    list_b = [_hit(2)]
    fused = rrf_fuse([list_a, list_b], weights=[1.0, 100.0])
    assert fused[0]["chunk_id"] == 2  # heavily weighted list wins


def test_rrf_fuse_empty():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


# ---- adaptive crop ---------------------------------------------------------


def test_crop_floor_drops_low_scorers():
    hits = [
        _hit(1, score=0.9, title="A"),
        _hit(2, score=0.85, title="B"),
        _hit(3, score=0.2, title="C"),  # below the 0.3 absolute floor
    ]
    cropped = crop_context(hits, min_docs=1, max_docs=12, score_floor=0.3)
    ids = [h["chunk_id"] for h in cropped]
    assert 3 not in ids
    assert ids == [1, 2]


def test_crop_enforces_min_docs_even_below_floor():
    hits = [_hit(i, score=1.0 - i * 0.3, title=f"T{i}") for i in range(5)]
    cropped = crop_context(hits, min_docs=3, max_docs=12, score_floor=0.5)
    assert len(cropped) >= 3


def test_crop_enforces_max_docs():
    hits = [_hit(i, score=0.9, title=f"T{i}") for i in range(20)]
    cropped = crop_context(hits, min_docs=3, max_docs=12, score_floor=0.02)
    assert len(cropped) <= 12


def test_crop_revision_aware_dedup_prefers_newer_doc_date():
    hits = [
        _hit(1, score=0.9, title="Safety Manual Rev 2", doc_date="2020-01-01"),
        _hit(2, score=0.85, title="Safety Manual Rev 3", doc_date="2023-01-01"),
        _hit(3, score=0.5, title="Unrelated Doc"),
    ]
    cropped = crop_context(hits, min_docs=1, max_docs=12, score_floor=0.4)
    ids = [h["chunk_id"] for h in cropped]
    assert 1 not in ids  # superseded by the newer revision
    assert 2 in ids


def test_crop_keeps_multiple_chunks_of_one_document():
    """Sections of one PDF are all answerable: the dedup collapses *revisions*
    (same title, different doc), not chunks of the same document."""
    hits = [
        _hit(1, score=0.9, title="Resume.pdf", doc_id="doc-a"),
        _hit(2, score=0.85, title="Resume.pdf", doc_id="doc-a"),
        _hit(3, score=0.8, title="Resume.pdf", doc_id="doc-a"),
        _hit(4, score=0.7, title="Resume.pdf", doc_id="doc-a"),
        _hit(5, score=0.6, title="Other.pdf", doc_id="doc-b"),
    ]
    cropped = crop_context(hits, min_docs=1, max_docs=12, score_floor=0.4, max_chunks_per_doc=3)
    ids = [h["chunk_id"] for h in cropped]
    assert ids[:3] == [1, 2, 3]  # top-3 chunks of the winning document
    assert 4 not in ids  # beyond the per-document cap
    assert 5 in ids


def test_normalize_title_strips_revision_noise():
    assert normalize_title("Safety Manual Rev 3") == normalize_title("Safety Manual Rev 2")
    assert normalize_title("Safety Manual (May 2021)") == normalize_title("Safety Manual (Jan 2024)")
    assert normalize_title(None) == ""


def test_crop_empty_input():
    assert crop_context([]) == []


def test_crop_token_budget_stops_after_min_docs():
    big_text = "word " * 5000
    hits = [_hit(i, score=0.9 - i * 0.01, title=f"T{i}", text=big_text) for i in range(6)]
    cropped = crop_context(hits, min_docs=2, max_docs=12, score_floor=0.02, token_budget=6000)
    assert len(cropped) == 2  # min_docs always kept, budget stops further growth


# ---- structured arm query building -----------------------------------------


def test_build_structured_query_filters_on_row_kinds():
    stmt = build_structured_query("fire drill", k=10)
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "table_rows" in compiled
    assert "like" in compiled.lower()
    assert "fire drill" in compiled.lower()
    assert "row" in compiled and "table_window" in compiled and "table_summary" in compiled


def test_build_structured_query_applies_field_filters():
    stmt = build_structured_query("x", k=5, field_filters={"vessel": "Vessel A"})
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "Vessel A" in compiled


# ---- session usage in retrieve() --------------------------------------------
# An AsyncSession is one connection and is not concurrency-safe: overlapping
# execute() calls raise InvalidRequestError("session is provisioning a new
# connection"). retrieve() used to asyncio.gather() its arms over the shared
# session, which hit that whenever the session arrived cold (GET /search).
# StrictSession below enforces the same one-at-a-time rule without a DB.


class _StrictSession:
    """Mimics AsyncSession's non-reentrancy: two overlapping executes raise."""

    def __init__(self):
        self.busy = False
        self.commits = 0

    async def execute(self, *_a, **_kw):
        if self.busy:
            raise AssertionError("concurrent execute() on one session — AsyncSession forbids this")
        self.busy = True
        try:
            await asyncio.sleep(0)  # yields, so a gathered sibling can interleave
            return None
        finally:
            self.busy = False

    async def commit(self):
        self.commits += 1

    def add(self, _obj):
        pass


@pytest.fixture
def patched_retrieve(monkeypatch):
    """retrieve() with the network stubbed; each arm goes through session.execute.
    Patches names on the operations module (operations/retrieval.py) — retrieve()
    looks them up in its own module globals at call time."""
    import prorag.operations.retrieval as cr

    async def arm(session, _q, _k, **_kw):
        await session.execute()
        return []

    monkeypatch.setattr(cr, "plan", lambda q, session=None, **_kw: _async({"queries": [q, q], "mode": "table"}))
    monkeypatch.setattr(cr, "embed_texts", lambda t, session=None, **_kw: _async([[0.0], [0.0]]))
    monkeypatch.setattr(cr, "rerank", lambda q, hits: _async(hits))
    monkeypatch.setattr(cr, "vector_search", arm)
    monkeypatch.setattr(cr, "keyword_search", arm)
    monkeypatch.setattr(cr, "structured_search", arm)
    return cr


def _async(value):
    async def _coro():
        return value

    return _coro()


def test_retrieve_never_overlaps_queries_on_one_session(patched_retrieve):
    session = _StrictSession()
    asyncio.run(patched_retrieve.retrieve(session, "fire drill interval"))


def test_retrieve_releases_the_connection_before_the_llm_call(patched_retrieve):
    """Retrieval opens a read transaction; the caller then spends seconds on an
    LLM call. Without a commit the connection sits `idle in transaction` for the
    whole answer, so pool_size chats saturate the pool."""
    session = _StrictSession()
    asyncio.run(patched_retrieve.retrieve(session, "fire drill interval"))
    assert session.commits == 1


def test_build_structured_query_respects_limit():
    stmt = build_structured_query("x", k=7)
    assert stmt._limit == 7


# ---- planner JSON parsing --------------------------------------------------


def test_extract_json_plain():
    data = _extract_json('{"search_needed": true, "queries": ["a", "b"], "mode": "default"}')
    assert data["queries"] == ["a", "b"]


def test_extract_json_strips_code_fence():
    raw = '```json\n{"search_needed": false, "queries": ["a", "b"], "mode": "default"}\n```'
    data = _extract_json(raw)
    assert data["search_needed"] is False


def test_extract_json_garbage_raises():
    with pytest.raises(Exception):
        _extract_json("not json at all")


def test_fallback_uses_raw_query_both_slots():
    fb = _fallback("what is the fire drill interval")
    assert fb["search_needed"] is True
    assert fb["queries"] == ["what is the fire drill interval"] * 2


async def test_plan_falls_back_on_llm_error(monkeypatch):
    async def boom(system, user):
        raise RuntimeError("provider down")

    monkeypatch.setattr(plan_module, "plan_completion", boom)
    result = await plan("what is the deadline")
    assert result["queries"] == ["what is the deadline"] * 2


async def test_plan_falls_back_on_unparseable_json(monkeypatch):
    async def garbage(system, user):
        return "I'm sorry, I cannot produce JSON right now."

    monkeypatch.setattr(plan_module, "plan_completion", garbage)
    result = await plan("what is the deadline")
    assert result["queries"] == ["what is the deadline"] * 2


async def test_plan_happy_path(monkeypatch):
    async def good(system, user):
        return '{"search_needed": true, "queries": ["primary", "alt"], "mode": "table"}'

    monkeypatch.setattr(plan_module, "plan_completion", good)
    result = await plan("anything")
    assert result == {"search_needed": True, "queries": ["primary", "alt"], "mode": "table"}


# ---- Rerank API (OpenRouter hosted cross-encoder backend) ------------------


class _FakeRerankResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeRerankClient:
    """Stands in for httpx.AsyncClient inside _rerank_api: records the POST
    payload so tests can assert what the endpoint was asked, and returns a
    canned rerank response."""

    def __init__(self, payload, **kwargs):
        self._payload = payload
        self.sent = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, **kwargs):
        self.sent = kwargs.get("json")
        return _FakeRerankResponse(self._payload)


async def test_rerank_api_happy_path(monkeypatch):
    payload = {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
            {"index": 2, "relevance_score": 0.55},
        ]
    }
    fake = _FakeRerankClient(payload)
    monkeypatch.setattr("prorag.retrieve.rerank.httpx.AsyncClient", lambda **kw: fake)
    hits = [_hit(1), _hit(2), _hit(3)]
    out = await _rerank_api("q", hits)
    # re-sorted by relevance desc; scores are the API's 0..1 as-is
    assert [h["chunk_id"] for h in out] == [2, 3, 1]
    assert abs(out[0]["score"] - 0.9) < 1e-9
    assert abs(out[2]["score"] - 0.2) < 1e-9
    # the endpoint got every chunk text plus the query
    assert fake.sent["query"] == "q"
    assert fake.sent["documents"] == ["x" * 200] * 3


async def test_rerank_api_skips_out_of_range_indices(monkeypatch):
    fake = _FakeRerankClient(
        {"results": [{"index": 7, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]}
    )
    monkeypatch.setattr("prorag.retrieve.rerank.httpx.AsyncClient", lambda **kw: fake)
    hits = [_hit(1), _hit(2)]
    out = await _rerank_api("q", hits)
    assert [h["chunk_id"] for h in out] == [1]
    assert abs(out[0]["score"] - 0.2) < 1e-9


async def test_rerank_api_falls_back_on_error(monkeypatch):
    class Boom:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kwargs):
            raise RuntimeError("provider outage")

    monkeypatch.setattr("prorag.retrieve.rerank.httpx.AsyncClient", Boom)
    hits = [_hit(1, score=0.1), _hit(2, score=0.2)]
    assert await _rerank_api("q", hits) is hits  # input order, same objects


# ---- vector_search ORDER BY must match the hnsw expression index (#16) ------


async def test_vector_search_orders_by_the_expression_the_hnsw_index_was_built_on():
    """0001_initial.py builds ix_chunks_embedding_hnsw on embedding::halfvec(dim)
    when embed_dim > 2000. If the query orders by the raw column instead, the
    expression never matches and the planner can't use the index at any corpus
    size (confirmed below by forcing enable_seqscan=off, which the un-cast query
    can't route around). Skips if the DB isn't reachable."""
    from sqlalchemy import select, text

    from prorag.db import SessionLocal, engine
    from prorag.models import Chunk
    from prorag.settings import settings

    if settings.embed_dim <= 2000:
        pytest.skip("embed_dim <= 2000 indexes the raw vector column, not halfvec")

    try:
        session = SessionLocal()
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")

    async with session:
        query_embedding = [random.random() for _ in range(settings.embed_dim)]
        stmt = select(Chunk.id, _distance_expr(query_embedding).label("distance")).order_by("distance").limit(5)
        compiled = stmt.compile(dialect=engine.sync_engine.dialect, compile_kwargs={"literal_binds": True})

        await session.execute(text("SET LOCAL enable_seqscan = off"))
        rows = (await session.execute(text("EXPLAIN " + str(compiled)))).all()
    plan_text = "\n".join(r[0] for r in rows)

    assert "Index Scan using ix_chunks_embedding_hnsw" in plan_text, plan_text

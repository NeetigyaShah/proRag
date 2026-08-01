"""Endpoint-level tests for the file/table visibility guard (#18): a doc or
table invisible to the caller must 404 exactly like a missing one — never
403, which would confirm the id is real (#3's existence-must-not-leak rule).

Sessions are scripted fakes, not a real DB — this exercises only the
endpoints' control flow (guard-then-fetch, same 404 body either way); the
actual ACL predicate is covered end-to-end against Postgres in
tests/test_visibility.py.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from prorag.auth import current_user
from prorag.db import get_session
from prorag.main import app
from prorag.models import User


class _FakeResult:
    """Stands in for whatever `await session.execute(...)` returns — the
    endpoints under test only ever call .scalar_one_or_none() or
    .scalars().all() on it."""

    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return self

    def all(self):
        return self._value


class _ScriptedSession:
    """Returns queued _FakeResults from .execute(), one per call, in order —
    enough to drive an endpoint's control flow without a real DB."""

    def __init__(self, results):
        self._results = list(results)

    async def execute(self, *_a, **_kw):
        return self._results.pop(0)


class _FakeTable:
    def __init__(self, doc_id):
        self.doc_id = doc_id


class _FakeRow:
    def __init__(self, row_no, data):
        self.row_no = row_no
        self.data = data


@pytest.fixture
def client():
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def _as_user(u):
    app.dependency_overrides[current_user] = lambda: u


def _with_session(results):
    app.dependency_overrides[get_session] = lambda: _ScriptedSession(results)


def test_get_original_404s_not_403_when_doc_invisible(client):
    """visible_doc_guard's own query comes back empty -> the guard fails
    before the real Document row is ever fetched."""
    _as_user(User(id=uuid.uuid4(), email="u@example.com", is_admin=False))
    _with_session([_FakeResult(None)])

    resp = client.get(f"/files/{uuid.uuid4()}/original")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "document not found"}


def test_get_original_404s_same_as_visible_but_since_deleted(client):
    """Same status/body whether the id never existed or is merely hidden —
    a differing response would itself leak which case it is."""
    _as_user(None)  # super-principal: visible_doc_guard passes any real id
    _with_session([_FakeResult(uuid.uuid4()), _FakeResult(None)])

    resp = client.get(f"/files/{uuid.uuid4()}/original")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "document not found"}


def test_get_table_rows_404s_not_403_when_owning_doc_invisible(client):
    doc_id = uuid.uuid4()
    _as_user(User(id=uuid.uuid4(), email="u2@example.com", is_admin=False))
    _with_session([_FakeResult(_FakeTable(doc_id)), _FakeResult(None)])

    resp = client.get(f"/tables/{uuid.uuid4()}/rows")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "table not found"}


def test_get_table_rows_404s_when_table_id_unknown(client):
    _as_user(User(id=uuid.uuid4(), email="u3@example.com", is_admin=False))
    _with_session([_FakeResult(None)])

    resp = client.get(f"/tables/{uuid.uuid4()}/rows")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "table not found"}


def test_get_table_rows_returns_rows_when_visible(client):
    doc_id = uuid.uuid4()
    _as_user(User(id=uuid.uuid4(), email="u4@example.com", is_admin=False))
    rows = [_FakeRow(1, {"a": 1}), _FakeRow(2, {"a": 2})]
    _with_session([_FakeResult(_FakeTable(doc_id)), _FakeResult(doc_id), _FakeResult(rows)])

    resp = client.get(f"/tables/{uuid.uuid4()}/rows")
    assert resp.status_code == 200
    assert resp.json() == {"rows": [{"row_no": 1, "data": {"a": 1}}, {"row_no": 2, "data": {"a": 2}}]}

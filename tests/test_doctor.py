"""Unit tests for `prorag doctor`'s check functions (#20) — every check takes
an injectable stub (`probe=`/`ping=`/`embed=`) so these exercise
every OK/WARN/FAIL branch with no network and no DB, mirroring the fake-session
pattern already used for /readyz in tests/test_ops.py.

A separate integration test (test_doctor_run_all_against_live_stack) calls
run_all() against the real stack and skips cleanly if the DB is unreachable.
"""

import types

import pytest

from prorag import doctor
from prorag.settings import Settings, settings

# ---- check_settings -----------------------------------------------------------


def test_check_settings_ok_with_real_settings():
    name, ok, detail = doctor.check_settings()
    assert name == "settings"
    assert ok is True
    assert "embed_dim" in detail


def test_check_settings_fails_on_a_stub_missing_fields():
    name, ok, detail = doctor.check_settings(cfg=object())
    assert name == "settings"
    assert ok is False
    assert "missing expected field" in detail


# ---- check_db -------------------------------------------------------------


async def test_check_db_ok_when_probe_succeeds():
    async def probe():
        return None

    name, ok, detail = await doctor.check_db(probe=probe)
    assert (name, ok) == ("db", True)
    assert detail == "reachable"


async def test_check_db_fails_when_probe_raises():
    async def probe():
        raise OSError("connection refused")

    name, ok, detail = await doctor.check_db(probe=probe)
    assert (name, ok) == ("db", False)
    assert "connection refused" in detail


# ---- check_migrations -------------------------------------------------------


async def test_check_migrations_ok_when_db_matches_head():
    async def probe():
        return "0009"

    name, ok, detail = await doctor.check_migrations(probe=probe, head="0009")
    assert (name, ok) == ("migrations", True)
    assert "0009" in detail


async def test_check_migrations_fails_on_mismatch():
    async def probe():
        return "0007"

    name, ok, detail = await doctor.check_migrations(probe=probe, head="0009")
    assert (name, ok) == ("migrations", False)
    assert "0007" in detail and "0009" in detail


async def test_check_migrations_fails_when_probe_raises():
    async def probe():
        raise OSError("no db")

    name, ok, detail = await doctor.check_migrations(probe=probe, head="0009")
    assert (name, ok) == ("migrations", False)


# ---- check_blob_dir ----------------------------------------------------------


def test_check_blob_dir_ok_when_writable(tmp_path):
    name, ok, detail = doctor.check_blob_dir(str(tmp_path))
    assert (name, ok) == ("blob_dir", True)
    assert "writable" in detail


def test_check_blob_dir_fails_when_path_collides_with_a_file(tmp_path):
    # mkdir(parents=True, exist_ok=True) raises FileExistsError (an OSError
    # subclass) when a path component is an existing plain file.
    blocker = tmp_path / "blobs"
    blocker.write_text("not a directory")
    target = blocker / "nested"

    name, ok, detail = doctor.check_blob_dir(str(target))
    assert (name, ok) == ("blob_dir", False)
    assert "not writable" in detail


# ---- check_llm --------------------------------------------------------------


async def test_check_llm_warns_when_no_key():
    name, ok, detail = await doctor.check_llm(has_key=False)
    assert (name, ok) == ("llm", True)
    assert detail.startswith("WARN:")


async def test_check_llm_ok_when_ping_succeeds():
    async def ping():
        return None

    name, ok, detail = await doctor.check_llm(ping=ping, has_key=True)
    assert (name, ok) == ("llm", True)
    assert not detail.startswith("WARN:")


async def test_check_llm_fails_when_ping_raises():
    async def ping():
        raise RuntimeError("401 unauthorized")

    name, ok, detail = await doctor.check_llm(ping=ping, has_key=True)
    assert (name, ok) == ("llm", False)
    assert "401" in detail


# ---- check_embed --------------------------------------------------------------


async def test_check_embed_warns_when_no_key():
    name, ok, detail = await doctor.check_embed(has_key=False)
    assert (name, ok) == ("embed", True)
    assert detail.startswith("WARN:")


async def test_check_embed_ok_when_dim_matches(monkeypatch):
    monkeypatch.setattr(settings, "embed_dim", 3)

    async def embed():
        return [[0.1, 0.2, 0.3]]

    name, ok, detail = await doctor.check_embed(embed=embed, has_key=True)
    assert (name, ok) == ("embed", True)


async def test_check_embed_fails_when_dim_mismatches(monkeypatch):
    monkeypatch.setattr(settings, "embed_dim", 1536)

    async def embed():
        return [[0.1, 0.2, 0.3]]

    name, ok, detail = await doctor.check_embed(embed=embed, has_key=True)
    assert (name, ok) == ("embed", False)
    assert "1536" in detail


# ---- check_rerank -------------------------------------------------------------


async def test_check_rerank_ok_disabled(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", False)
    name, ok, detail = await doctor.check_rerank()
    assert (name, ok, detail) == ("rerank", True, "disabled")


async def test_check_rerank_warns_without_key(monkeypatch):
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    name, ok, detail = await doctor.check_rerank()
    assert (name, ok) == ("rerank", True)
    assert detail.startswith("WARN:")


# ---- check_bm25 ---------------------------------------------------------------


async def test_check_bm25_ok_when_index_present():
    async def probe():
        return True

    name, ok, detail = await doctor.check_bm25(probe=probe)
    assert (name, ok) == ("bm25", True)
    assert "present" in detail


async def test_check_bm25_warns_when_index_absent():
    async def probe():
        return False

    name, ok, detail = await doctor.check_bm25(probe=probe)
    assert (name, ok) == ("bm25", True)
    assert detail.startswith("WARN:")


async def test_check_bm25_fails_when_probe_raises():
    async def probe():
        raise OSError("no db")

    name, ok, detail = await doctor.check_bm25(probe=probe)
    assert (name, ok) == ("bm25", False)


# ---- _label --------------------------------------------------------------------


@pytest.mark.parametrize(
    "ok,detail,expected",
    [
        (False, "unreachable", "FAIL"),
        (True, "WARN: skipped", "WARN"),
        (True, "reachable", "OK"),
    ],
)
def test_label(ok, detail, expected):
    assert doctor._label(ok, detail) == expected


# ---- settings validators (#20 point 5) -----------------------------------------


def test_settings_rejects_non_positive_daily_cost_cap():
    with pytest.raises(Exception, match="daily_cost_cap_usd"):
        Settings(daily_cost_cap_usd=0)


def test_settings_rejects_non_positive_session_ttl():
    with pytest.raises(Exception, match="session_ttl_days"):
        Settings(session_ttl_days=-1)


def test_settings_rejects_empty_blob_dir():
    with pytest.raises(Exception, match="blob_dir"):
        Settings(blob_dir="   ")


# ---- integration ---------------------------------------------------------------


async def test_doctor_run_all_against_live_stack():
    """Same skip-cleanly-without-DB pattern as tests/test_identity_schema.py:
    no live stack in CI means "skipped", not "failed"."""
    from sqlalchemy import text

    from prorag.db import SessionLocal, engine

    try:
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    finally:
        await engine.dispose()

    # Force the no-network WARN branch for llm/embed regardless of what's in
    # the environment (a provider key set in .env for other tests would
    # otherwise make this specific check flake on a real network call);
    # db/migrations/blob_dir/bm25 below still run against the live stack.
    results = await doctor.run_all(llm_has_key=False, embed_has_key=False)
    names = {name for name, _, _ in results}
    assert names == {"settings", "db", "migrations", "blob_dir", "llm", "embed", "rerank", "bm25"}
    fails = [(n, d) for n, ok, d in results if not ok]
    assert not fails, f"doctor reported FAIL against the live stack: {fails}"

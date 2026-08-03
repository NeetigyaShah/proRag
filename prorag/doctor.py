"""`python -m prorag.doctor` (#20): the day-one smoke check per #8's
resolution — one aligned line per check, OK/WARN/FAIL, exit nonzero only on
FAIL. WARN covers things that degrade gracefully at runtime (no LLM key set,
reranker stuck on CPU) rather than genuinely broken deployments.

Each check_x() returns (name, ok, detail) and is independently importable —
network-touching checks take an injectable stub (`probe=`/`ping=`/`get_model=`)
so tests can exercise the pass/fail branches with no network, per the pattern
already used for /readyz's fake sessions in tests/test_ops.py. WARN is spelled
as an `ok=True` result whose detail starts with "WARN: "; the tuple has no
separate status field, so that prefix is the whole convention.
"""

import asyncio
import os
from pathlib import Path

from sqlalchemy import text

from prorag import db
from prorag.settings import settings

ROOT = Path(__file__).resolve().parent.parent

_PROVIDER_KEY_ENVVARS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "AZURE_API_KEY")


def _has_provider_key() -> bool:
    # os.environ first (real env vars), then the settings field — a key that
    # lives only in .env is visible to Settings but NOT to os.environ, and the
    # app must not report "no provider key" for a configured provider.
    return any(os.environ.get(k) for k in _PROVIDER_KEY_ENVVARS) or bool(settings.openrouter_api_key)


def _label(ok: bool, detail: str) -> str:
    if not ok:
        return "FAIL"
    if detail.startswith("WARN:"):
        return "WARN"
    return "OK"


# ---- checks -----------------------------------------------------------------


def check_settings(cfg=None) -> tuple[str, bool, str]:
    """Mostly a formality — a bad .env already fails at import via the
    field_validators in settings.py — but kept as an independently callable
    check like the others, and testable with a stub missing an attribute."""
    cfg = cfg if cfg is not None else settings
    try:
        detail = f"loaded (embed_dim={cfg.embed_dim}, auth_enabled={cfg.auth_enabled}, blob_dir={cfg.blob_dir})"
        return ("settings", True, detail)
    except AttributeError as exc:
        return ("settings", False, f"settings object missing expected field: {exc}")


async def check_db(probe=None, timeout: float = 5.0) -> tuple[str, bool, str]:
    """When called: by run_all() (and directly by tests with an injected
    probe). What: runs a trivial SELECT 1 against the app's DB engine.
    Returns: ("db", ok, detail) — FAIL when the connection errors."""
    async def _default_probe() -> None:
        async with db.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    probe = probe or _default_probe
    try:
        async with asyncio.timeout(timeout):
            await probe()
        return ("db", True, "reachable")
    except Exception as exc:
        return ("db", False, f"unreachable: {exc}")


def _script_head() -> str:
    """When called: by check_migrations. What: reads the alembic revision the
    code expects — the head of the local alembic scripts directory. Returns:
    the revision id string."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    return ScriptDirectory.from_config(cfg).get_current_head()


async def check_migrations(probe=None, head=None, timeout: float = 5.0) -> tuple[str, bool, str]:
    """When called: by run_all() (and directly by tests with an injected
    probe). What: compares the DB's alembic_version to the expected head.
    Returns: ("migrations", ok, detail) — FAIL on drift or probe error."""
    async def _default_probe() -> str | None:
        async with db.engine.connect() as conn:
            result = await conn.execute(text("SELECT version_num FROM alembic_version"))
            return result.scalar_one_or_none()

    probe = probe or _default_probe
    try:
        expected_head = head if head is not None else _script_head()
        async with asyncio.timeout(timeout):
            current = await probe()
    except Exception as exc:
        return ("migrations", False, f"could not compare revisions: {exc}")
    if current != expected_head:
        return ("migrations", False, f"DB at {current!r}, code expects head {expected_head!r}")
    return ("migrations", True, f"at head ({expected_head})")


def check_blob_dir(blob_dir: str | None = None) -> tuple[str, bool, str]:
    """When called: by run_all() (and directly by tests). What: ensures the
    blob directory exists and is writable via a write+delete probe. Returns:
    ("blob_dir", ok, detail) — FAIL on any OSError."""
    path = Path(blob_dir if blob_dir is not None else settings.blob_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_file = path / ".doctor_write_test"
        probe_file.write_text("ok")
        probe_file.unlink()
        return ("blob_dir", True, f"writable ({path})")
    except OSError as exc:
        return ("blob_dir", False, f"not writable ({path}): {exc}")


async def check_llm(ping=None, has_key: bool | None = None, timeout: float = 20.0) -> tuple[str, bool, str]:
    """When called: by run_all() (and directly by tests with an injected
    ping). What: sends a 1-token completion to settings.answer_model to prove
    the provider is reachable. Returns: ("llm", ok, detail) — WARN (ok) when
    no provider key is set, FAIL when the ping errors. Default timeout is 20s
    deliberately: litellm's FIRST call in a cold process does provider
    init (cost table, model info) that a 5s cap reliably trips — measured
    ~6-10s cold, ~1s warm."""
    has_key = _has_provider_key() if has_key is None else has_key
    if not has_key:
        return ("llm", True, "WARN: skipped — no provider API key set")

    async def _default_ping() -> None:
        import litellm

        await litellm.acompletion(
            model=settings.answer_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )

    ping = ping or _default_ping
    try:
        async with asyncio.timeout(timeout):
            await ping()
        return ("llm", True, f"reachable ({settings.answer_model})")
    except Exception as exc:
        return ("llm", False, f"unreachable ({settings.answer_model}): {exc}")


async def check_embed(embed=None, has_key: bool | None = None, timeout: float = 5.0) -> tuple[str, bool, str]:
    """When called: by run_all() (and directly by tests with an injected
    embed). What: embeds a single "ping" and verifies the returned dimension
    matches EMBED_DIM. Returns: ("embed", ok, detail) — WARN (ok) when no key
    and no local model, FAIL on error or dimension mismatch."""
    has_key = (_has_provider_key() or settings.embed_model.startswith("local/")) if has_key is None else has_key
    if not has_key:
        return ("embed", True, "WARN: skipped — no provider API key set")

    async def _default_embed() -> list[list[float]]:
        from prorag.llm import embed_texts

        return await embed_texts(["ping"])

    embed = embed or _default_embed
    try:
        async with asyncio.timeout(timeout):
            vectors = await embed()
    except Exception as exc:
        return ("embed", False, f"unreachable ({settings.embed_model}): {exc}")
    dim = len(vectors[0]) if vectors else 0
    if dim != settings.embed_dim:
        return ("embed", False, f"returned dim {dim} != EMBED_DIM {settings.embed_dim}")
    return ("embed", True, f"reachable, dim {dim} matches EMBED_DIM ({settings.embed_model})")


async def check_rerank(get_model=None, timeout: float = 30.0) -> tuple[str, bool, str]:
    """When called: by run_all(). What: verifies the CONFIGURED rerank
    backend — a live 2-doc probe of OpenRouter's rerank endpoint for
    backend="api", else loads the local model and reports where it landed.
    Returns: ("rerank", ok, detail) — always ok: "disabled" when reranking
    is off, WARN for CPU or unloadable (reranking degrades gracefully, #11)."""
    if not settings.rerank_enabled:
        return ("rerank", True, "disabled")

    if settings.rerank_backend == "api":
        api_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return ("rerank", True, "WARN: OPENROUTER_API_KEY unset — rerank API unavailable")
        try:
            async with asyncio.timeout(timeout):
                import httpx

                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/rerank",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": settings.rerank_api_model,
                            "query": "doctor probe",
                            "documents": ["doctor probe chunk"],
                        },
                    )
                    resp.raise_for_status()
            return (
                "rerank",
                True,
                f"api ok ({settings.rerank_api_model}, {resp.json().get('model', '?')})",
            )
        except Exception as exc:
            return ("rerank", True, f"WARN: rerank API probe failed ({exc})")

    if get_model is None:
        from prorag.retrieve.rerank import get_model as _get_model
    else:
        _get_model = get_model

    try:
        async with asyncio.timeout(timeout):
            model = await asyncio.to_thread(_get_model)
    except Exception as exc:
        return ("rerank", True, f"WARN: could not load {settings.rerank_model} — reranking will no-op ({exc})")
    if model is None:
        return ("rerank", True, f"WARN: could not load {settings.rerank_model} — reranking will no-op")
    device = str(getattr(model, "device", "cpu"))
    if "cpu" in device.lower():
        return ("rerank", True, f"WARN: loaded on CPU ({settings.rerank_model}) — reranking will be slow (#11)")
    return ("rerank", True, f"loaded on {device} ({settings.rerank_model})")


async def check_bm25(probe=None, timeout: float = 5.0) -> tuple[str, bool, str]:
    """When called: by run_all() (and directly by tests with an injected
    probe). What: checks whether the pg_search BM25 index exists. Returns:
    ("bm25", ok, detail) — WARN (ok) when absent, since the tsvector fallback
    keeps search working."""
    async def _default_probe() -> bool:
        async with db.engine.connect() as conn:
            result = await conn.execute(text("SELECT to_regclass('ix_chunks_bm25') IS NOT NULL"))
            return bool(result.scalar_one())

    probe = probe or _default_probe
    try:
        async with asyncio.timeout(timeout):
            present = await probe()
    except Exception as exc:
        return ("bm25", False, f"could not check: {exc}")
    if present:
        return ("bm25", True, "BM25 index present (pg_search)")
    return ("bm25", True, "WARN: BM25 index absent — tsvector fallback active")


# ---- orchestration ------------------------------------------------------------


async def run_all(*, llm_has_key: bool | None = None, embed_has_key: bool | None = None) -> list[tuple[str, bool, str]]:
    """`llm_has_key`/`embed_has_key` let a caller force the no-network WARN
    branch of check_llm/check_embed (tests/test_doctor.py's live-stack
    integration test does this so a provider key set in .env for other
    tests can't make it flake on a real network call) — default None keeps
    the normal environment-autodetected behaviour for the `doctor` CLI."""
    return [
        check_settings(),
        await check_db(),
        await check_migrations(),
        check_blob_dir(),
        await check_llm(has_key=llm_has_key),
        await check_embed(has_key=embed_has_key),
        await check_rerank(),
        await check_bm25(),
    ]


async def _main() -> int:
    """When called: `python -m prorag.doctor`. What: runs every check, prints
    one aligned OK/WARN/FAIL line per check, and disposes the DB engine.
    Returns: exit code 1 if any check FAILed, else 0."""
    try:
        results = await run_all()
    finally:
        await db.engine.dispose()

    width = max(len(name) for name, _, _ in results)
    any_fail = False
    for name, ok, detail in results:
        label = _label(ok, detail)
        any_fail = any_fail or label == "FAIL"
        print(f"{name.ljust(width)}  {label:<4}  {detail.removeprefix('WARN: ')}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(_main()))

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from prorag.auth import require_api_key
from prorag.chat.router import router as chat_router
from prorag.db import engine, get_session
from prorag.eval.router import router as eval_router
from prorag.files.router import router as files_router
from prorag.ingest.router import router as ingest_router
from prorag.middleware import GZipSkipSSEMiddleware, RequestTimingMiddleware
from prorag.retrieve.router import router as search_router
from prorag.settings import settings

# ponytail: stdlib logging, not structlog — one line per request is all this
# app needs; reach for structlog when structured/JSON logs are actually consumed
# by something (Loki, Datadog...).
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # Without this, a reload/SIGTERM leaves pooled connections for the DB to time
    # out on its own; dispose() closes them and lets the worker drain cleanly.
    await engine.dispose()


app = FastAPI(title="ProRag", version="0.1.0", lifespan=lifespan)

app.add_middleware(RequestTimingMiddleware)
app.add_middleware(GZipSkipSSEMiddleware)

# Bearer auth on every route except /healthz, /readyz, and /web static (§6, §8
# Phase 5). No-ops when settings.auth_enabled is False (local dev default).
_auth = [Depends(require_api_key)]
app.include_router(ingest_router, dependencies=_auth)
app.include_router(chat_router, dependencies=_auth)
app.include_router(files_router, dependencies=_auth)
app.include_router(search_router, dependencies=_auth)
app.include_router(eval_router, dependencies=_auth)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.is_dir():
    # Phase 4 viewer: one static HTML/JS page, no build step (§8 Phase 4 deviation).
    app.mount("/web", StaticFiles(directory=WEB_DIR, html=True), name="web")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(session: AsyncSession = Depends(get_session)):
    """Liveness isn't enough to load-balance on — readyz actually checks the
    DB is reachable (§6, §8 Phase 5).

    The timeout is the point: a saturated pool makes the checkout itself block
    for db_pool_timeout (30s) before raising, so an un-timed probe *hangs*
    rather than answering. Most probes give up at 5-10s, so the orchestrator
    sees a dead endpoint instead of a clean 503 — measured against a
    deliberately saturated pool. Failing fast reports "not ready", which is
    exactly what a pool-exhausted worker is.
    """
    try:
        async with asyncio.timeout(settings.readyz_timeout_seconds):
            await session.execute(text("SELECT 1"))
    except TimeoutError:
        # Expected under load, and probes are frequent — a full traceback per
        # probe would bury the actual incident in its own noise.
        logger.warning("readiness check timed out after %ss", settings.readyz_timeout_seconds)
        return JSONResponse({"status": "not ready"}, status_code=503)
    except Exception:
        # An unauthenticated probe must not echo the DB error: the connection
        # string (host, user, sometimes password) shows up in asyncpg messages.
        logger.exception("readiness check failed")
        return JSONResponse({"status": "not ready"}, status_code=503)
    return {"status": "ready"}

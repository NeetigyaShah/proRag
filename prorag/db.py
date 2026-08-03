"""Persistence infrastructure: builds the async SQLAlchemy engine and session
factory, and exposes `get_session`, FastAPI's per-request DB-session
dependency. Imported at startup and used by every handler that touches the DB.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from prorag.settings import settings

# pool_pre_ping survives idle-connection drops. The explicit pool bounds matter
# because ingestion holds a session across a slow embed phase: with the default
# pool (5 + 10 overflow) a handful of concurrent uploads can starve chat
# requests, and pool_timeout turns that starvation into a fast error instead of
# an indefinite hang. pool_recycle keeps connections under any DB idle timeout.
engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_recycle=1800,
    # Without this, an unreachable DB makes the first connection attempt hang
    # indefinitely instead of failing fast at startup.
    connect_args={"timeout": settings.db_connect_timeout},
    # LIFO reuses hot connections, so idle ones age out and get recycled rather
    # than every connection being kept marginally alive.
    pool_use_lifo=True,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency (Depends(get_session)): yields one AsyncSession per
    request; handlers commit/rollback, and the context manager closes it when
    the request ends."""

    async with SessionLocal() as session:
        yield session

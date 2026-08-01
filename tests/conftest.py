"""Centralizes the `await engine.dispose()` that used to live in a
try/finally at the bottom of every DB-touching test (#23) — this bug (a
missing dispose leaving pooled connections bound to a closed event loop,
breaking the *next* test's checkout) recurred across three separate worker
sessions before this fixture existed.

pytest-asyncio (asyncio_mode = "auto", function-scoped event loop by
default per pyproject.toml) gives each test its own loop, so this has to run
after every single test, not once for the whole session — hence
function-scoped autouse rather than session-scoped. Cheap to do
unconditionally: a test that never touched the DB just disposes an
already-empty pool.
"""

import pytest_asyncio

from prorag.db import engine


@pytest_asyncio.fixture(autouse=True)
async def _dispose_engine_after_each_test():
    yield
    await engine.dispose()

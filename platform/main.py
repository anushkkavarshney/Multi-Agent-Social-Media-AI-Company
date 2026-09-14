"""FastAPI app for the mock social media platform.

Run from the repo root:
    python -m platform.main          # http://127.0.0.1:8010  (docs at /docs)

Port note: the default is 8010 because 8000 is occupied by an unrelated dev
server on the build machine; override with PLATFORM_PORT in .env if needed.

Startup: creates tables (idempotent) and seeds the 3 channels from
platform/db.py::_seed_channels when the DB is empty.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.settings import get_settings
from platform.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables + seed channels on boot (guarded, restart-safe)."""
    settings = get_settings()
    if settings.create_tables_on_startup:
        await init_db()
    yield
    # No teardown needed: SQLite file persists; the engine closes with the process.


app = FastAPI(
    title="Mock Social Media Platform",
    description=(
        "Self-contained social platform with a statistical engagement simulator. "
        "Publish posts, advance simulated time with /simulate/tick, read metrics "
        "and comments. Part of the Prodigal AI Task 1 multi-agent system."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Routers (paths are defined relative to root; the section-5 table maps 1:1)
from platform.routes.comments import router as comments_router  # noqa: E402
from platform.routes.metrics import router as metrics_router  # noqa: E402
from platform.routes.posts import router as posts_router  # noqa: E402
from platform.routes.simulation import router as simulation_router  # noqa: E402

app.include_router(posts_router)
app.include_router(metrics_router)
app.include_router(comments_router)
app.include_router(simulation_router)


@app.get("/health")
async def health() -> dict:
    """Liveness probe (also verifies the DB is reachable)."""
    from sqlalchemy import select

    from platform.db import get_session_factory
    from platform.schema import Channel

    factory = get_session_factory()
    async with factory() as session:
        channels = (await session.execute(select(Channel.id))).scalars().all()
    return {"status": "ok", "channels": list(channels)}


if __name__ == "__main__":
    import os

    import uvicorn

    # 127.0.0.1 on purpose: the mock platform is local-only infrastructure.
    port = int(os.environ.get("PLATFORM_PORT", "8010"))
    uvicorn.run("platform.main:app", host="127.0.0.1", port=port, reload=False)

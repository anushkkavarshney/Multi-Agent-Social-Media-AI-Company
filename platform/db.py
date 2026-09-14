"""Async SQLAlchemy engine/session for the mock platform.

Design notes:
- One engine per process, created lazily (not at import time) so tests can
  point it at a throwaway DB file and the app can read DB_PATH from settings.
- aiosqlite driver => fully async FastAPI handlers on a file-based SQLite DB
  (architecture doc section 1: "SQLite via SQLAlchemy ORM (async, aiosqlite)").
- `check_same_thread=False` is required because aiosqlite runs SQLite on its
  own worker thread; SQLAlchemy needs permission for that cross-thread handoff.
- SQLite serializes writes on the single file, so short-lived sessions per
  request are enough; no pool tuning needed at this scale.
"""

from __future__ import annotations

import json

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config.settings import get_settings


class Base(DeclarativeBase):
    """Declarative base for all platform ORM models."""


_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    """Return the process-wide async engine, creating it on first use.

    Lazy so that importing this module never touches disk or settings —
    keeps unit tests import-safe.
    """
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        # aiosqlite is file-backed: `sqlite+aiosqlite:///relative/path.db`
        _engine = create_async_engine(
            f"sqlite+aiosqlite:///{settings.db_path}",
            connect_args={"check_same_thread": False},
        )
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Session factory for `async with get_session_factory()() as session:` use."""
    if _session_factory is None:
        get_engine()  # triggers engine + factory creation
    return _session_factory


def reset_engine_for_tests() -> None:
    """Drop the process-wide engine so a test can re-point DB_PATH.

    Only for tests: after changing settings, the next get_engine() call
    creates a fresh engine/session factory against the new path.
    """
    global _engine, _session_factory
    _engine = None
    _session_factory = None


async def init_db() -> None:
    """Create all tables if missing, then seed the 3 channels (doc section 5).

    Called by the FastAPI startup hook. Restart-safe: seeding is skipped when
    channels already exist, and create_all is a no-op on existing tables.
    """
    from platform.schema import Channel  # local import avoids circulars

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = get_session_factory()
    async with factory() as session:
        count = (await session.execute(select(func.count(Channel.id)))).scalar_one()
        if count == 0:
            session.add_all(_seed_channels())
            await session.commit()


def _profile(**kwargs) -> str:
    """Serialize a channel character profile to the JSON blob the engine reads."""
    return json.dumps(kwargs)


def _seed_channels() -> list[Channel]:
    """The 3 channel archetypes from architecture doc section 5.

    These profiles are REAL simulation inputs — platform/simulation reads them:
      - penalize_copy_over_words / penalty_strength drive the channel-specific
        length-penalty rule (doc section 6, rule 4).
      - peak_hours drives each channel's time-window boost window.
      - hashtag_sweet_spot drives the non-linear hashtag rule's per-channel tuning.
      - tone/audience make the three channels measurably different rather than
        decorative (the same post MUST perform differently across them).

    Deliberate asymmetries (so Analytics has real cross-channel structure):
      QuickBites: strongest evening peak, harsh length penalty (short copy wins)
      DebateHall: mild morning peak, questions thrive, tolerant of long copy
      LeadDesk:   morning professional peak, punishes >150-word essays 30%
    """
    from platform.schema import Channel

    return [
        Channel(
            id="ch_shortform",
            name="QuickBites",
            type="shortform_video",
            character_profile_json=_profile(
                audience="gen-z snackers, 18-25",
                tone="punchy, meme-fluent, hook in the first 5 words",
                # rule 4 (length penalty): >60 words loses 60% of reach here
                # (doc says ~40%; we use 0.60 to make the channel visibly distinct)
                penalize_copy_over_words=60,
                penalty_strength=0.60,
                # rule 1 (time-window boost): evening prime time for short video
                peak_hours="18:00-21:00",
                peak_multiplier=1.8,
                # rule 3 (hashtags): reach peaks at 3-5 tags
                hashtag_sweet_spot=(3, 5),
            ),
        ),
        Channel(
            id="ch_discussion",
            name="DebateHall",
            type="discussion",
            character_profile_json=_profile(
                audience="curious generalists who reply to take a position",
                tone="conversational, debate-starter, ends with a question",
                # discussion threads tolerate longer copy: no length penalty
                penalize_copy_over_words=10_000,  # effectively disabled
                penalty_strength=0.0,
                # rule 1: mild morning scroll peak (commute hours)
                peak_hours="07:00-10:00",
                peak_multiplier=1.4,
                hashtag_sweet_spot=(3, 5),
            ),
        ),
        Channel(
            id="ch_professional",
            name="LeadDesk",
            type="professional",
            character_profile_json=_profile(
                audience="operators and founders, 25-40, B2B-minded",
                tone="crisp, specific, credibility over hype",
                # rule 4, straight from the doc: professional penalizes copy
                # >150 words by ~30% reach
                penalize_copy_over_words=150,
                penalty_strength=0.30,
                # rule 1: professional reading happens in the workday morning
                peak_hours="08:00-11:00",
                peak_multiplier=1.5,
                hashtag_sweet_spot=(3, 5),
            ),
        ),
    ]

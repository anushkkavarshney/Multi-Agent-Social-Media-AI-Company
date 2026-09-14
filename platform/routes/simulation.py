"""Simulation routes: advance the simulated clock and accrue engagement.

Tick semantics (this is the platform's time machine):
    POST /simulate/tick {"hours": 24, "campaign_id": "..."}
      -> sim_time advances by `hours`
      -> every post with published_at < window_start and NO metric yet is
         accrued at its maturity moment (full exposure)
      -> posts published DURING the window get one accrual scaled by their
         exposure_fraction (share of the window they were live for)

Windows never overlap because the state clock advances to window_end on every
tick — so "first tick covering a post" is unambiguous, and no post is ever
double-counted.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from platform.routes.posts import (
    _create_comment_rows,
    _load_format_history,
    _record_history,
    _store_multipliers,
    channel_profile,
    get_db,
    sim_post_dict,
)
from platform.schema import Channel, Metric, Post, SimulationState, as_utc
from platform.simulation import engine as sim_engine

router = APIRouter()


def _rng(seed: int | None = None) -> np.random.Generator:
    base = seed if seed is not None else get_settings().sim_seed
    return np.random.default_rng(base)


async def _get_or_create_state(session: AsyncSession, campaign_id: str) -> SimulationState:
    state = (
        await session.execute(
            select(SimulationState).where(SimulationState.campaign_id == campaign_id)
        )
    ).scalar_one_or_none()
    if state is None:
        state = SimulationState(
            campaign_id=campaign_id, current_sim_time=datetime.now(timezone.utc)
        )
        session.add(state)
        await session.flush()
    return state


@router.get("/channels")
async def list_channels(session: AsyncSession = Depends(get_db)) -> dict:
    """List channels with their parsed character profiles (engine inputs)."""
    channels = (await session.execute(select(Channel))).scalars().all()
    return {
        "channels": [
            {
                "id": c.id,
                "name": c.name,
                "type": c.type,
                "character_profile": channel_profile(c),
            }
            for c in channels
        ]
    }


@router.post("/simulate/tick")
async def simulate_tick(payload: dict, session: AsyncSession = Depends(get_db)) -> dict:
    """Advance simulated time and trigger the engagement engine.

    Body: {"hours": number in (0, 720] (default 24),
           "campaign_id": str (optional; namespace for the sim clock),
           "seed": int (optional; overrides settings.sim_seed for this tick)}.
    """
    hours = payload.get("hours", 24)
    if not isinstance(hours, (int, float)) or isinstance(hours, bool) or hours <= 0 or hours > 720:
        raise HTTPException(status_code=400, detail="hours must be a number in (0, 720]")
    campaign_id = payload.get("campaign_id")
    if campaign_id is not None and (
        not isinstance(campaign_id, str) or not campaign_id.strip()
    ):
        raise HTTPException(status_code=400, detail="campaign_id must be a non-empty string")
    seed = payload.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise HTTPException(status_code=400, detail="seed must be an int")
    rng = _rng(seed)

    state = await _get_or_create_state(session, campaign_id or "default")
    window_start = as_utc(state.current_sim_time)
    window_end = window_start + timedelta(hours=hours)

    # --- Candidate set 1: posts matured before this window, no metrics yet ---
    matured = (
        await session.execute(
            select(Post).where(Post.published_at < window_start).order_by(Post.published_at)
        )
    ).scalars().all()
    if campaign_id is not None:
        matured = [p for p in matured if p.campaign_id == campaign_id]

    accrued_ids: set[str] = set()
    if matured:
        have_metrics = set(
            (
                await session.execute(
                    select(Metric.post_id).where(Metric.post_id.in_([p.id for p in matured]))
                )
            ).scalars()
        )
        matured = [p for p in matured if p.id not in have_metrics]

    # --- Candidate set 2: posts published inside this window (partial exposure) ---
    published_in_window = (
        await session.execute(
            select(Post).where(Post.published_at >= window_start, Post.published_at < window_end)
        )
    ).scalars().all()
    if campaign_id is not None:
        published_in_window = [p for p in published_in_window if p.campaign_id == campaign_id]

    channels = {c.id: c for c in (await session.execute(select(Channel))).scalars()}

    results = []

    async def accrue(post: Post, exposure: float, at_time: datetime) -> dict | None:
        channel = channels.get(post.channel_id)
        if channel is None:
            return {"post_id": post.id, "status": "skipped_unknown_channel"}
        await _accrue_with(session, post, channel, exposure, at_time, rng)
        accrued_ids.add(post.id)
        return {"post_id": post.id, "status": "accrued", "exposure_fraction": round(exposure, 3)}

    for post in matured:
        results.append(await accrue(post, 1.0, as_utc(post.published_at)))

    for post in published_in_window:
        if post.id in accrued_ids:
            continue
        live = (window_end - as_utc(post.published_at)).total_seconds() / 3600.0
        exposure = min(max(live / hours, 0.0), 1.0)
        results.append(await accrue(post, exposure, window_end))

    # Advance the clock LAST (after all accruals used the window bounds).
    state.current_sim_time = window_end
    await session.commit()

    return {
        "campaign_id": campaign_id or "default",
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "hours": hours,
        "posts_processed": [r for r in results if r is not None],
        "accrued_count": len(accrued_ids),
    }


async def _accrue_with(
    session: AsyncSession,
    post: Post,
    channel: Channel,
    exposure: float,
    at_time: datetime,
    rng,
) -> None:
    """Shared accrual used by both candidate paths (keeps metric creation in one place)."""
    sim_post = sim_post_dict(post, exposure)
    recent_history = await _load_format_history(session, post.campaign_id)
    result = sim_engine.simulate_post_engagement(
        sim_post, channel_profile(channel), recent_history, rng
    )
    metric = Metric(
        post_id=post.id,
        impressions=result["impressions"],
        likes=result["likes"],
        comments=result["comments"],
        shares=result["shares"],
        saves=result["saves"],
        clicks=result["clicks"],
        follower_delta=result["follower_delta"],
        recorded_at=at_time,
    )
    session.add(metric)
    # Simulated comment TEXT for this accrual (Community Manager input).
    await _create_comment_rows(session, post, metric.comments, rng)
    await _store_multipliers(session, post.campaign_id, post.id, result["multipliers"])
    await _record_history(session, post, at_time)
    await session.flush()

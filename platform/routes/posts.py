"""Post routes: publish + feed (the two **required** post endpoints).

Publish contract: the Scheduler Agent (Day 2) calls POST /posts once per
approved post. The platform stamps its own id and returns it — that returned
id is what lands in `Post.published_post_id` on the agent side.

On publish, if `published_at` is not in the future, one immediate accrual
runs so /metrics and /comments return real data right away (nice for curl
testing); further engagement accrues on later /simulate/tick calls.
"""

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from platform.db import get_session_factory
from platform.schema import Channel, Metric, Post, as_utc
from platform.simulation import engine as sim_engine

router = APIRouter()


async def get_db() -> AsyncSession:
    """FastAPI dependency: one short-lived session per request."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


def post_to_dict(post: Post, channel_name: str | None = None) -> dict:
    """Serialize a Post ORM row to the API-facing dict (tz-aware timestamps)."""
    return {
        "id": post.id,
        "campaign_id": post.campaign_id,
        "channel_id": post.channel_id,
        "channel_name": channel_name,
        "copy": post.copy,
        "hashtags": list(post.hashtags_json or []),
        "cta": post.cta,
        "format": post.format,
        "published_at": as_utc(post.published_at).isoformat(),
        "created_at": as_utc(post.created_at).isoformat(),
    }


async def load_channel(session: AsyncSession, channel_id: str) -> Channel:
    channel = (
        await session.execute(select(Channel).where(Channel.id == channel_id))
    ).scalar_one_or_none()
    if channel is None:
        raise HTTPException(status_code=404, detail=f"unknown channel: {channel_id}")
    return channel


def channel_profile(channel: Channel) -> dict:
    """Parse a channel's character_profile_json blob."""
    return json.loads(channel.character_profile_json)


def sim_post_dict(post: Post, exposure_fraction: float = 1.0) -> dict:
    """Shape a Post row into the dict the simulation engine expects."""
    return {
        "copy": post.copy,
        "hashtags": list(post.hashtags_json or []),
        "cta": post.cta,
        "format": post.format,
        "published_at": as_utc(post.published_at),
        "channel_id": post.channel_id,
        "exposure_fraction": exposure_fraction,
    }


from platform.routes._rng import get_rng as _rng


async def _record_history(session: AsyncSession, post: Post, sim_time: datetime) -> None:
    """Append {date, format} to the campaign's format history (novelty rule input)."""
    from platform.schema import SimulationState

    state = (
        await session.execute(
            select(SimulationState).where(SimulationState.campaign_id == post.campaign_id)
        )
    ).scalar_one_or_none()
    if state is None:
        state = SimulationState(campaign_id=post.campaign_id)
        session.add(state)
    history = dict(state.format_history_json or {})
    entries = list(history.get(post.channel_id, []))
    entries.append({"date": as_utc(post.published_at).date().isoformat(), "format": post.format})
    history[post.channel_id] = entries
    state.format_history_json = history
    state.current_sim_time = sim_time


async def _create_comment_rows(session: AsyncSession, post: Post, count: int, rng) -> None:
    """Generate + persist `count` simulated Comment rows for a post.

    The Community Manager Agent (Day 2) reads these via GET /posts/{id}/comments,
    so the platform must produce real TEXT, not just a comment count.
    """
    if count <= 0:
        return
    from platform.schema import Comment as CommentRow
    from platform.simulation.engine import generate_comment_texts

    rows = generate_comment_texts(sim_post_dict(post), count, rng)
    for r in rows:
        session.add(CommentRow(id=f"cmt_{uuid.uuid4().hex[:12]}", post_id=post.id, **r))


async def _accrue(
    session: AsyncSession,
    post: Post,
    channel: Channel,
    exposure_fraction: float,
    sim_time: datetime,
) -> Metric:
    """Run the simulation engine once for a post and append a Metric row.

    Multipliers + per-rule contributions are stored on the SimulationState's
    observability blob so we can debug why a specific post performed the way
    it did (and so the write-up can quote real multiplier chains).
    """
    profile = channel_profile(channel)
    rng = _rng()
    sim_post = sim_post_dict(post, exposure_fraction)
    recent_history = await _load_format_history(session, post.campaign_id)
    result = sim_engine.simulate_post_engagement(sim_post, profile, recent_history, rng)

    metric = Metric(
        post_id=post.id,
        impressions=result["impressions"],
        likes=result["likes"],
        comments=result["comments"],
        shares=result["shares"],
        saves=result["saves"],
        clicks=result["clicks"],
        follower_delta=result["follower_delta"],
        recorded_at=sim_time,
    )
    session.add(metric)

    # Simulated comment TEXT for this accrual (Community Manager input).
    await _create_comment_rows(session, post, metric.comments, rng)

    # Store the multiplier trace (observability only; not part of the API).
    await _store_multipliers(session, post.campaign_id, post.id, result["multipliers"])
    await _record_history(session, post, sim_time)
    await session.commit()
    return metric


async def _store_multipliers(
    session: AsyncSession, campaign_id: str, post_id: str, multipliers: dict
) -> None:
    from platform.schema import SimulationState

    state = (
        await session.execute(
            select(SimulationState).where(SimulationState.campaign_id == campaign_id)
        )
    ).scalar_one_or_none()
    if state is None:
        state = SimulationState(campaign_id=campaign_id)
        session.add(state)
    blob = dict(state.last_multipliers_json or {})
    blob[post_id] = multipliers
    state.last_multipliers_json = blob


async def _load_format_history(session: AsyncSession, campaign_id: str) -> dict:
    from platform.schema import SimulationState

    state = (
        await session.execute(
            select(SimulationState).where(SimulationState.campaign_id == campaign_id)
        )
    ).scalar_one_or_none()
    if state is None:
        return {}
    return dict(state.format_history_json or {})


@router.post("/posts", status_code=201)
async def publish_post(
    payload: dict,
    session: AsyncSession = Depends(get_db),
) -> dict:
    """**Publish a post** (required endpoint).

    Body: {"campaign_id": str, "channel_id": str, "copy": str,
           "hashtags": [str], "cta": str|null, "format": str,
           "published_at": ISO-datetime (optional, defaults to now)}
    Returns the stored post including the platform-assigned `id`.
    """
    # Explicit field checks (clear 400s) instead of a Pydantic body model —
    # the platform is the *receiver* of agent output and must fail loudly,
    # not coerce, when the Scheduler sends malformed payloads.
    for field in ("campaign_id", "channel_id", "copy"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise HTTPException(status_code=400, detail=f"missing or empty field: {field}")
    channel = await load_channel(session, payload["channel_id"])

    hashtags = payload.get("hashtags") or []
    if not isinstance(hashtags, list) or any(not isinstance(h, str) or not h.strip() for h in hashtags):
        raise HTTPException(status_code=400, detail="hashtags must be a list of non-empty strings")

    published_at_raw = payload.get("published_at")
    if published_at_raw:
        try:
            published_at = datetime.fromisoformat(published_at_raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"bad published_at: {exc}") from exc
    else:
        published_at = datetime.now(timezone.utc)
    if published_at.tzinfo is None:
        # The platform simulates one timezone; naive inputs are treated as UTC.
        published_at = published_at.replace(tzinfo=timezone.utc)

    post = Post(
        id=f"post_{uuid.uuid4().hex[:12]}",
        campaign_id=payload["campaign_id"],
        channel_id=payload["channel_id"],
        copy=payload["copy"],
        # Canonical "#tag" form so the engine and Analytics never see
        # "espresso" and "#espresso" as different tags.
        hashtags_json=[f"#{h.strip().lstrip('#')}" for h in hashtags],
        cta=payload.get("cta"),
        format=payload.get("format", "text_only"),
        published_at=published_at,
    )
    session.add(post)
    await session.commit()

    # Immediate accrual for already-live posts so /metrics returns data now.
    sim_time = max(published_at, datetime.now(timezone.utc))
    if published_at <= datetime.now(timezone.utc):
        await _accrue(session, post, channel, exposure_fraction=1.0, sim_time=sim_time)

    return {"status": "published", "post": post_to_dict(post)}


@router.get("/feed")
async def get_feed(
    channel: str | None = Query(None, description="filter by channel id"),
    campaign_id: str | None = Query(None, description="filter by campaign"),
    session: AsyncSession = Depends(get_db),
) -> dict:
    """**Fetch a channel feed** (required endpoint); newest first."""
    q = select(Post).order_by(Post.published_at.desc()).limit(200)
    if channel:
        q = q.where(Post.channel_id == channel)
    if campaign_id:
        q = q.where(Post.campaign_id == campaign_id)
    posts = (await session.execute(q)).scalars().all()

    names = {c.id: c.name for c in (await session.execute(select(Channel))).scalars()}
    return {"posts": [post_to_dict(p, names.get(p.channel_id)) for p in posts]}

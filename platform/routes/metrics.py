"""Metrics routes: per-post engagement + the weekly analytics aggregation.

The weekly endpoint's job is to hand the Analytics Agent *raw computed numbers*
(architecture doc section 7: don't let the LLM eyeball raw rows — compute the
stats in Python, then let the model narrate them). All grouping/correlation
outputs here are plain dicts of real numbers from the DB.
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from platform.routes.posts import get_db, post_to_dict
from platform.schema import Comment, Metric, Post, as_utc

router = APIRouter()


def _window_label(pub: datetime) -> str:
    """Bucket publish time into the ground-truth windows the engine uses.

    Mirrors hidden_rules.time_window_boost so the Analytics Agent's group-bys
    line up 1:1 with the planted rules (evening peak / dead zone / normal)."""
    if 18 <= pub.hour < 21:
        return "evening_peak(18-21)"
    if 2 <= pub.hour < 6:
        return "dead_zone(02-06)"
    return "normal"


async def _get_post(session: AsyncSession, post_id: str) -> Post:
    post = (
        await session.execute(select(Post).where(Post.id == post_id))
    ).scalar_one_or_none()
    if post is None:
        raise HTTPException(status_code=404, detail=f"unknown post: {post_id}")
    return post


async def _latest_metric(session: AsyncSession, post_id: str) -> Metric | None:
    rows = (
        await session.execute(
            select(Metric).where(Metric.post_id == post_id).order_by(Metric.recorded_at.desc())
        )
    ).scalars().first()
    return rows


def _metric_to_dict(m: Metric) -> dict:
    return {
        "post_id": m.post_id,
        "impressions": m.impressions,
        "likes": m.likes,
        "comments": m.comments,
        "shares": m.shares,
        "saves": m.saves,
        "clicks": m.clicks,
        "follower_delta": m.follower_delta,
        "recorded_at": as_utc(m.recorded_at).isoformat(),
    }


@router.get("/posts/{post_id}/metrics")
async def get_metrics(
    post_id: str,
    session: AsyncSession = Depends(get_db),
) -> dict:
    """**Fetch simulated engagement for a post** (required endpoint).

    Returns the latest cumulative snapshot plus derived rates the Analytics
    Agent would otherwise have to compute itself.
    """
    post = await _get_post(session, post_id)
    metric = await _latest_metric(session, post_id)
    if metric is None:
        return {
            "post": post_to_dict(post),
            "metric": None,
            "note": "no engagement yet — post is scheduled for the future or no tick has run",
        }
    impressions = metric.impressions
    rates = {
        "like_rate": metric.likes / impressions if impressions else 0.0,
        "save_rate": metric.saves / impressions if impressions else 0.0,
        "click_rate": metric.clicks / impressions if impressions else 0.0,
    }
    return {"post": post_to_dict(post), "metric": _metric_to_dict(metric), "rates": rates}


# --- Weekly analytics (data for the Analytics Agent) -------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson r, hand-rolled (no scipy dependency in the API layer).
    Returns 0.0 when either series is constant (correlation undefined)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / sqrt(vx * vy)


@router.get("/analytics/week/{campaign_id}/{week}")
async def get_weekly_analytics(
    campaign_id: str,
    week: int,
    session: AsyncSession = Depends(get_db),
) -> dict:
    """Raw aggregated week data for the Analytics Agent (doc section 5).

    Week 1 = days 1-7 from the campaign's first publish, week 2 = days 8-14, etc.
    Computes per-post rows with candidate explanatory features (hour, length,
    hashtags, format, question-ending) alongside outcomes, then cross-cutting
    group means and Pearson correlations — the exact tables the Analytics Agent
    narrates into a WeeklyReport.
    """
    posts = (
        await session.execute(
            select(Post).where(Post.campaign_id == campaign_id).order_by(Post.published_at)
        )
    ).scalars().all()
    if not posts:
        raise HTTPException(status_code=404, detail=f"no posts for campaign: {campaign_id}")

    first_pub = min(as_utc(p.published_at) for p in posts)
    week_start = first_pub + timedelta(days=7 * (week - 1))
    week_end = week_start + timedelta(days=7)
    week_posts = [p for p in posts if week_start <= as_utc(p.published_at) < week_end]
    if not week_posts:
        raise HTTPException(status_code=404, detail=f"no posts found in week {week}")

    # Fetch latest metric per post in one query set.
    metrics: dict[str, Metric] = {}
    for p in week_posts:
        m = await _latest_metric(session, p.id)
        if m is not None:
            metrics[p.id] = m

    post_rows = []
    for p in week_posts:
        m = metrics.get(p.id)
        if m is None:
            continue
        pub = as_utc(p.published_at)
        post_rows.append(
            {
                "post_id": p.id,
                "channel_id": p.channel_id,
                "format": p.format,
                "hour": pub.hour,
                "time_window": _window_label(pub),
                "word_count": len(p.copy.split()),
                "hashtag_count": len(p.hashtags_json or []),
                "ends_in_question": p.copy.rstrip().endswith("?"),
                "has_cta": p.cta is not None,
                "impressions": m.impressions,
                "likes": m.likes,
                "comments": m.comments,
                "shares": m.shares,
                "saves": m.saves,
                "clicks": m.clicks,
                "follower_delta": m.follower_delta,
                "save_rate": m.saves / m.impressions if m.impressions else 0.0,
            }
        )

    def group_mean(key_fn, value_fn) -> dict:
        groups: dict[str, list[float]] = {}
        for r in post_rows:
            groups.setdefault(key_fn(r), []).append(value_fn(r))
        return {k: round(_mean(v), 2) for k, v in sorted(groups.items())}

    impressions_of = lambda r: float(r["impressions"])
    comments_of = lambda r: float(r["comments"])

    patterns = {
        "mean_impressions_by_time_window": group_mean(lambda r: r["time_window"], impressions_of),
        "mean_impressions_by_hour_bucket": group_mean(lambda r: f"{r['hour']:02d}", impressions_of),
        "mean_impressions_by_channel": group_mean(lambda r: r["channel_id"], impressions_of),
        "mean_impressions_by_format": group_mean(lambda r: r["format"], impressions_of),
        "mean_impressions_by_hashtag_bucket": group_mean(
            lambda r: "0" if r["hashtag_count"] == 0 else "1-2" if r["hashtag_count"] <= 2 else "3-5" if r["hashtag_count"] <= 5 else "6-7" if r["hashtag_count"] <= 7 else "8+",
            impressions_of,
        ),
        "mean_impressions_long_vs_short_copy": group_mean(
            lambda r: "long" if r["word_count"] > 150 else "short", impressions_of
        ),
        "mean_comments_question_vs_statement": group_mean(
            lambda r: "question" if r["ends_in_question"] else "statement", comments_of
        ),
        "correlations": {
            "word_count_vs_impressions": round(_pearson([r["word_count"] for r in post_rows], [impressions_of(r) for r in post_rows]), 3),
            "hashtag_count_vs_impressions": round(_pearson([r["hashtag_count"] for r in post_rows], [impressions_of(r) for r in post_rows]), 3),
            "save_rate_vs_follower_delta": round(_pearson([r["save_rate"] for r in post_rows], [float(r["follower_delta"]) for r in post_rows]), 3),
        },
    }

    totals = {
        "posts": len(post_rows),
        "impressions": sum(r["impressions"] for r in post_rows),
        "likes": sum(r["likes"] for r in post_rows),
        "comments": sum(r["comments"] for r in post_rows),
        "shares": sum(r["shares"] for r in post_rows),
        "saves": sum(r["saves"] for r in post_rows),
        "clicks": sum(r["clicks"] for r in post_rows),
        "follower_delta": sum(r["follower_delta"] for r in post_rows),
    }

    return {
        "campaign_id": campaign_id,
        "week": week,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "totals": totals,
        "post_rows": post_rows,
        "patterns": patterns,
    }

"""Analytics Agent — two-stage design (doc section 7, "don't let the LLM
eyeball raw numbers and hallucinate patterns; compute the patterns, then
have it explain them").

Stage 1 — PLAIN PYTHON (`compute_stats`): group-bys, rates, correlations,
top/bottom ranking over raw EngagementSnapshot + Comment data. Deterministic,
unit-testable, no model in sight.

Stage 2 — MODEL NARRATION (`run`): the computed stats dict goes into
llm/prompts/analytics.jinja with the instruction that every number in the
narrative must appear verbatim in the stats. The model produces the
WeeklyReport's narrative fields ONLY — kpi_performance is filled from the
computed stats, never from model output (the model literally cannot put a
wrong number in the report's KPI table).

Day-2 scope: runs against FIXTURE data (the same synthetic snapshots the
simulation tests use); Day 3 points it at GET /analytics/week/{campaign_id}
from the live platform. The stats function takes plain dicts on purpose —
they are the JSON shape the platform endpoint returns, so the Day-3 wiring
is "fetch, then call the same code".
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from agents.base_agent import BaseAgent
from llm.trace import append_event
from models.report import Recommendation, WeeklyReport


def compute_stats(
    snapshots: list[dict],
    comments: list[dict],
    kpis: list[dict],
    channel_of_post: dict[str, str] | None = None,
    window_hours_of_post: dict[str, str] | None = None,
) -> dict:
    """Stage 1: every number the narrative is allowed to cite.

    Args:
        snapshots: [{post_id, recorded_at, impressions, likes, comments,
                    shares, saves, clicks, follower_delta}] — latest snapshot
                   per post (cumulative counters).
        comments: [{post_id, text, sentiment}] — the week's comments.
        kpis: [{metric, target, channel}] — the campaign's commitments.
        channel_of_post: post_id -> channel id (enables per-channel groups).
        window_hours_of_post: post_id -> "peak" | "offpeak" label for the
                   publish hour (enables the timing group comparison; the
                   platform provides peak windows per channel profile).

    Returns a JSON-serializable dict. Kept flat and named — this dict IS
    the interface between Python and the model; the template tells the
    model it may only cite numbers that appear here.
    """
    channel_of_post = channel_of_post or {}
    window_hours_of_post = window_hours_of_post or {}

    # --- per-post rollup (latest cumulative snapshot per post) -------------
    per_post: dict[str, dict] = {}
    for s in snapshots:
        cur = per_post.get(s["post_id"])
        if cur is None or s["recorded_at"] > cur["recorded_at"]:
            per_post[s["post_id"]] = s

    total_impressions = sum(p["impressions"] for p in per_post.values())
    total_engagements = sum(
        p["likes"] + p["comments"] + p["shares"] + p["saves"] + p["clicks"]
        for p in per_post.values()
    )
    total_comments = sum(p["comments"] for p in per_post.values())
    total_saves = sum(p["saves"] for p in per_post.values())
    total_followers = sum(p["follower_delta"] for p in per_post.values())

    stats: dict = {
        "posts_analyzed": len(per_post),
        "total_impressions": total_impressions,
        "total_engagements": total_engagements,
        "mean_impressions_per_post": round(total_impressions / len(per_post), 1) if per_post else 0.0,
        "engagement_rate_total": round(
            total_engagements / total_impressions, 4
        ) if total_impressions else 0.0,
        "comment_rate_total": round(total_comments / total_impressions, 4) if total_impressions else 0.0,
        "save_rate_total": round(total_saves / total_impressions, 4) if total_impressions else 0.0,
        "total_follower_delta": total_followers,
    }

    # --- timing group comparison (the time-window hidden rule, visible) ----
    peak, offpeak = [], []
    for pid, p in per_post.items():
        label = window_hours_of_post.get(pid)
        if label == "peak":
            peak.append(p["impressions"])
        elif label == "offpeak":
            offpeak.append(p["impressions"])
    stats["timing"] = {
        "peak_posts": len(peak),
        "peak_mean_impressions": round(sum(peak) / len(peak), 1) if peak else None,
        "offpeak_posts": len(offpeak),
        "offpeak_mean_impressions": round(sum(offpeak) / len(offpeak), 1) if offpeak else None,
    }

    # --- question-CTA vs statement (the comment-lift rule, visible) --------
    q_comments, s_comments = [], []
    for pid, p in per_post.items():
        # Platform emits ``ends_in_question``; Day-2 fixture data sets
        # ``copy_is_question``.  Accept either key so both paths work.
        is_question = p.get("ends_in_question", p.get("copy_is_question", False))
        bucket = q_comments if is_question else s_comments
        bucket.append(p["comments"])
    stats["question_cta"] = {
        "question_posts": len(q_comments),
        "question_mean_comments": round(sum(q_comments) / len(q_comments), 2) if q_comments else None,
        "statement_mean_comments": round(sum(s_comments) / len(s_comments), 2) if s_comments else None,
    }

    # --- per-channel group (channel asymmetry, visible) ---------------------
    by_channel: dict[str, list[int]] = defaultdict(list)
    for pid, p in per_post.items():
        ch = channel_of_post.get(pid, "unknown")
        by_channel[ch].append(p["impressions"])
    stats["by_channel"] = {
        ch: {
            "posts": len(imps),
            "mean_impressions": round(sum(imps) / len(imps), 1),
            "total_follower_delta": sum(
                p["follower_delta"] for pid, p in per_post.items()
                if channel_of_post.get(pid) == ch
            ),
        }
        for ch, imps in sorted(by_channel.items())
    }

    # --- top/bottom posts per channel (drives post_insights) ----------------
    ranked = sorted(per_post.values(), key=lambda p: p["impressions"], reverse=True)
    stats["top_post"] = _brief_post(ranked[0], channel_of_post) if ranked else None
    stats["bottom_post"] = _brief_post(ranked[-1], channel_of_post) if ranked else None

    # --- KPI measurement (rate metrics computed from the same totals) ------
    stats["kpi_performance"] = {
        "impressions": float(total_impressions),
        "engagement_rate": stats["engagement_rate_total"],
        "comment_rate": stats["comment_rate_total"],
        "save_rate": stats["save_rate_total"],
        "follower_growth": float(total_followers),
        "likes": sum(p["likes"] for p in per_post.values()),
        "clicks": sum(p["clicks"] for p in per_post.values()),
        "shares": sum(p["shares"] for p in per_post.values()),
    }

    # KPI vs target (the model can only reference these verified numbers)
    available_metrics = {
        "impressions": float(total_impressions),
        "engagement_rate": stats["engagement_rate_total"],
        "comment_rate": stats["comment_rate_total"],
        "save_rate": stats["save_rate_total"],
        "follower_growth": float(total_followers),
        "click_through_rate": stats["kpi_performance"]["clicks"] / total_impressions if total_impressions else 0.0,
        "share_rate": stats["kpi_performance"]["shares"] / total_impressions if total_impressions else 0.0,
    }
    stats["kpi_vs_target"] = [
        {
            "metric": k.get("metric"),
            "target": k.get("target"),
            "measured": round(available_metrics.get(k.get("metric"), 0.0), 4),
            "channel": k.get("channel"),
        }
        for k in kpis
    ]

    # --- comment sentiment counts -------------------------------------------
    sentiment_counts: dict[str, int] = defaultdict(int)
    for c in comments:
        sentiment_counts[c.get("sentiment", "neutral")] += 1
    stats["comment_sentiment_counts"] = dict(sentiment_counts)

    return stats


def _brief_post(p: dict, channel_of_post: dict[str, str]) -> dict:
    """Compact per-post summary for the narrative layer."""
    return {
        "post_id": p["post_id"],
        "channel": channel_of_post.get(p["post_id"], "unknown"),
        "impressions": p["impressions"],
        "comments": p["comments"],
        "saves": p["saves"],
        "follower_delta": p["follower_delta"],
    }


class _PostInsightOut(BaseModel):
    post_id: str
    performance_rank: str  # constrained by the validator below (top|bottom|mid)
    hypothesis: str = Field(..., min_length=1)


class _RecommendationOut(BaseModel):
    change: str
    evidence: str
    expected_effect: str


class _Narrative(BaseModel):
    """The narrative fields stage 2 produces. Numeric report fields are NOT
    here on purpose — the model cannot touch kpi_performance."""

    patterns_found: list[str] = Field(..., min_length=1)
    post_insights: list[_PostInsightOut] = Field(..., min_length=1)
    comment_sentiment_summary: str = Field(..., min_length=1)
    recommendations: list[_RecommendationOut] = Field(..., min_length=1)

    @field_validator("post_insights")
    @classmethod
    def rank_must_be_known(cls, v: list[_PostInsightOut]) -> list[_PostInsightOut]:
        # Constrain here (retryable at the model boundary) instead of letting
        # the WeeklyReport Literal reject after the loop has already closed.
        for i in v:
            if i.performance_rank not in ("top", "bottom", "mid"):
                raise ValueError(
                    f"performance_rank must be one of top|bottom|mid, got '{i.performance_rank}'"
                )
        return v


class AnalyticsAgent(BaseAgent):
    name = "analytics"
    uses_router_model = False

    async def run(
        self,
        campaign_id: str,
        week_number: int,
        snapshots: list[dict],
        comments: list[dict],
        kpis: list[dict],
        *,
        channel_of_post: dict[str, str] | None = None,
        window_hours_of_post: dict[str, str] | None = None,
        precomputed_stats: dict | None = None,
        transport=None,
    ) -> WeeklyReport:
        """Compute stats (stage 1), narrate (stage 2), assemble the report.

        The returned WeeklyReport's kpi_performance comes from compute_stats,
        not from the model. Day 3 swaps the fixture snapshots/comments for
        the platform's /analytics/week payload; nothing else changes.

        `precomputed_stats` is the Day-3 wiring's one-time stage-1 result
        (already logged as its own step), passed in so stats are computed and
        shown exactly once in a run's trace — not twice under slightly
        different noise.
        """
        self.log_io(
            "input",
            campaign_id=campaign_id,
            week_number=week_number,
            snapshots=len(snapshots),
            comments=len(comments),
        )

        stats = precomputed_stats or compute_stats(
            snapshots, comments, kpis,
            channel_of_post=channel_of_post,
            window_hours_of_post=window_hours_of_post,
        )

        prompt = self.render_prompt(
            "analytics.jinja",
            stats=stats,  # Jinja renders the dict readably; template forbids inventing numbers
            kpis=kpis,
        )
        narrative: _Narrative = await self.call_model(
            instruction=prompt,
            schema_cls=_Narrative,
            transport=transport,
            campaign_id=campaign_id,
        )

        # Assemble the contract: narrative fields from the model, numeric
        # fields from Python. PostInsight/Recommendation contracts are
        # validated on construction (blank strings rejected here).
        report = WeeklyReport(
            campaign_id=campaign_id,
            week_number=week_number,
            kpi_performance=stats["kpi_performance"],
            post_insights=[
                {"post_id": i.post_id, "performance_rank": i.performance_rank, "hypothesis": i.hypothesis}
                for i in narrative.post_insights
            ],
            patterns_found=narrative.patterns_found,
            comment_sentiment_summary=narrative.comment_sentiment_summary,
            recommendations=[
                {"change": r.change, "evidence": r.evidence, "expected_effect": r.expected_effect}
                for r in narrative.recommendations
            ],
        )

        self.log_io(
            "output",
            campaign_id=campaign_id,
            week_number=week_number,
            patterns=len(report.patterns_found),
            recommendations=len(report.recommendations),
            kpi_performance=report.kpi_performance,
        )
        await self.emit(
            to_agent="orchestrator",
            campaign_id=campaign_id,
            message_type="weekly_report_ready",
            payload=report.model_dump(mode="json"),
        )
        return report

    # ------------------------------------------------------------------
    # Day 3 wiring: fetch real week data from the platform, then narrate
    # ------------------------------------------------------------------

    async def run_week(
        self,
        platform,
        campaign_id: str,
        week_number: int,
        campaign,
        channels: list[dict],
        *,
        posture: bool = True,
    ) -> WeeklyReport:
        """Full two-stage analysis over the LIVE platform week (no fixtures).

        Stage 0 — fetch: GET /analytics/week/{campaign_id}/{week} returns the
        platform's own computed aggregation (post_rows with per-post features +
        outcomes, totals, and the platform's built-in patterns dict).  The
        aggregate post_rows are reshaped to compute_stats' snapshot contract.

        Stage 1 — compute: compute_stats() runs in plain Python and its output
        is logged as its OWN trace/console step (``analytics_computed_stats``)
        BEFORE the model is consulted. This is the write-up's proof that the
        numbers the narrative cites were computed, not eyeballed.

        Stage 2 — narrate: run() summons the model over the precomputed stats.

        Also fetches the week's comment texts (GET /posts/{id}/comments) so
        sentiment counts come from real comment data, not comment totals.
        """
        payload = await platform.get_weekly(campaign_id, week_number)
        post_rows = payload["post_rows"]

        channel_of_post = {r["post_id"]: r["channel_id"] for r in post_rows}
        window_hours_of_post = {
            r["post_id"]: ("peak" if "peak" in r["time_window"] else "offpeak")
            for r in post_rows
        }

        # Reshape post_rows (aggregate rows without recorded_at) to the
        # per-post snapshot contract compute_stats expects. post_rows are ALREADY
        # the latest snapshot per post, so one recorded_at is enough to disambiguate.
        snapshots = [
            {
                **{k: r[k] for k in (
                    "post_id", "impressions", "likes", "comments", "shares",
                    "saves", "clicks", "follower_delta", "ends_in_question",
                )},
                "recorded_at": payload["week_end"],
            }
            for r in post_rows
        ]

        # Real comment text per post (feed get_comments).
        comments: list[dict] = []
        for post_id in channel_of_post:
            for c in await platform.get_comments(post_id):
                comments.append(
                    {"post_id": post_id, "text": c.get("text", ""), "sentiment": c.get("sentiment", "neutral")}
                )

        kpis = [k.model_dump(mode="json") for k in campaign.kpis]

        # ---- STAGE 1: compute + log as a DISTINCT step (before any model) ----
        stats = compute_stats(
            snapshots, comments, kpis,
            channel_of_post=channel_of_post,
            window_hours_of_post=window_hours_of_post,
        )
        append_event(
            {
                "event": "analytics_computed_stats",
                "campaign_id": campaign_id,
                "week_number": week_number,
                "stage": 1,
                "channel_of_post": channel_of_post,
                "window_hours_of_post": window_hours_of_post,
                "stats": stats,
            }
        )
        self.log_io("computed_stats", campaign_id=campaign_id, week_number=week_number, stats=stats)

        # ---- STAGE 2: narrate over the precomputed stats (never re-drawn) ----
        report = await self.run(
            campaign_id, week_number,
            snapshots, comments, kpis,
            channel_of_post=channel_of_post,
            window_hours_of_post=window_hours_of_post,
            precomputed_stats=stats,
        )
        return report

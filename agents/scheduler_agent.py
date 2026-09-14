"""Scheduler Agent — deterministic slot assignment, NO model call.

Doc section 7 is explicit: "No LLM needed — deterministic logic; document
this choice (not every 'agent' needs to be a model call)." This module is
that documentation: the scheduler distributes approved posts across the
campaign's days, inside each channel's engagement peak window where
possible (it reads the same channel profiles the simulation engine does —
a first signal, Day 3's Analytics discovers the rest from data).

Day-2 scope: produce ScheduleSlots only. Calling POST /posts on the mock
platform is Day 3 (publish step); this agent deliberately has no HTTP code.

Why the peak-window heuristic is legitimate here (and not the Analytics
Agent "discovering" it): the scheduler must place posts BEFORE any
engagement data exists. Using the platform's declared audience profile is
what a real social team does (channel best practices); Analytics then
verifies/corrects from measured data in week 2 — which is exactly the
closed loop the project wants to demonstrate.
"""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone

from agents.base_agent import BaseAgent
from models.post import ScheduleSlot

# Local publish times per channel type (24h windows in sim-local time).
# The profiles carry each channel's PEAK window; publishing at/inside the
# window's start is the deterministic "best available slot" rule.
_PREFERRED_LOCAL_TIME: dict[str, time] = {
    "shortform_video": time(18, 0),   # QuickBites peak 18:00-21:00
    "discussion": time(7, 30),        # DebateHall peak 07:00-10:00
    "professional": time(8, 0),       # LeadDesk peak 08:00-11:00
}
_DEFAULT_TIME = time(12, 0)  # neutral fallback for unknown channel types


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


class SchedulerAgent(BaseAgent):
    name = "scheduler"
    # This agent never calls a model; the flag documents the choice and
    # makes it testable that no accidental model call was introduced.
    uses_model = False

    async def build_schedule(
        self,
        campaign_id: str,
        posts: list,  # list[models.post.Post] — approved posts
        channel_profiles: list[dict],
        *,
        start_day: datetime | None = None,
        posts_per_day_per_channel: int = 1,
    ) -> list[ScheduleSlot]:
        """Assign each post a publish time; returns ScheduleSlots.

        Algorithm (deterministic, reviewed line-by-line):
        1. Group posts by channel, preserving the input order.
        2. Distribute each channel's posts evenly across the campaign's
           days (one per day by default — cadence, not flooding).
        3. Publish each at that channel's preferred local time, which for
           the seeded channels sits at the START of its peak window.
        4. Slots are tz-aware UTC (the platform stores/compares UTC).

        Emits schedule_ready with the full slot list.
        """
        self.log_io("input", campaign_id=campaign_id, posts=len(posts))
        profiles = {p["id"]: p for p in channel_profiles}
        start = start_day or datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        slots: list[ScheduleSlot] = []
        by_channel: dict[str, list] = {}
        for post in posts:
            by_channel.setdefault(post.channel, []).append(post)

        for channel_id, channel_posts in by_channel.items():
            profile = profiles.get(channel_id, {})
            ctype = profile.get("type", "")
            preferred = _PREFERRED_LOCAL_TIME.get(ctype, _DEFAULT_TIME)

            # Even spread: post k of N goes on day k * days_available // N,
            # guaranteeing monotonic days without clustering at the start.
            n = len(channel_posts)
            days_available = max(1, min(
                _campaign_days(profile, start),
                max(1, n),  # at least enough days for the posts we have
            ))
            for k, post in enumerate(channel_posts):
                day_offset = (k * days_available) // n if n > 1 else 0
                local_dt = datetime.combine(start + timedelta(days=day_offset), preferred, tzinfo=timezone.utc)
                slots.append(
                    ScheduleSlot(
                        post_id=post.id,
                        channel=channel_id,
                        scheduled_at=local_dt,
                    )
                )

        self.log_io(
            "output",
            campaign_id=campaign_id,
            slots=[
                {"post_id": s.post_id, "channel": s.channel, "scheduled_at": s.scheduled_at.isoformat()}
                for s in slots
            ],
        )
        await self.emit(
            to_agent="orchestrator",
            campaign_id=campaign_id,
            message_type="schedule_ready",
            payload={
                "slots": [
                    {"post_id": s.post_id, "channel": s.channel, "scheduled_at": s.scheduled_at.isoformat()}
                    for s in slots
                ]
            },
        )
        return slots


def _campaign_days(profile: dict, start: datetime) -> int:
    """Days the campaign spans (from the campaign, not the profile).

    Kept separate for testability; the profile doesn't carry duration —
    the runner passes duration through the posts' campaign context.
    """
    # Day-2 simplification: slots spread over 7 days max. Day 3's publish
    # loop knows the real campaign duration and can re-run this cheaply.
    return 7

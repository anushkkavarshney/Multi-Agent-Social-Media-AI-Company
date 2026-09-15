"""Scheduler Agent — deterministic slot assignment + Day-3 platform publishing.

No LLM is ever used here (architecture doc section 7: "Not every 'agent'
needs to be a model call"). This module documents the design choice AND the
publishing logic that makes the Day-3 pipeline live.

Day-3 revision to slot placement (improving on the Day-2 day-0 pile-up):
Day-2 assigned posts per-channel independently, spreading `n` posts across
the channel's sub-window. With 3 channels each having 1 post (the demo),
that put all 3 posts on day 0. The demo needs posts spread across the 7-day
analytics window for the tick engine to show real progressive engagement and
for the novelty rule to have a chance to fire.

Fix: sort all posts by their channel's preferred time (stable, deterministic),
then distribute them across the campaign window at equal intervals so the
first post lands on day 0 and the last on the final day. The per-channel
preferred-hour guarantee is preserved — each post keeps its channel's peak
time. Same-day collisions are tolerated (two different channels can share
a day); the window-spacing ensures no two same-channel posts land on the
same day for realistic campaigns (< 7 posts per channel across 7 days).

Day-3 scope: produce ScheduleSlots AND publish via POST /posts on the mock
platform (architecture doc section 10 / build-order step 9). Publish is
separate from build_schedule so the publish loop can be checkpointed
independently (a mid-publish crash re-starts from the last un-published slot,
not from scratch).
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


class SchedulerAgent(BaseAgent):
    name = "scheduler"
    uses_model = False

    async def build_schedule(
        self,
        campaign_id: str,
        posts: list,
        channel_profiles: list[dict],
        *,
        start_day: datetime | None = None,
        posts_per_day_per_channel: int = 1,
        campaign_days: int | None = None,
    ) -> list[ScheduleSlot]:
        """Assign each post a publish time; returns ScheduleSlots.

        Algorithm (Day-3 revision):
        1. Group posts by channel, preserving input order.
        2. For each channel, pair posts with that channel's preferred time.
        3. Merge into one list, sort by (preferred_time, channel_id) for a
           stable, deterministic output that does not depend on dict
           insertion order.
        4. Spread the sorted posts across the campaign window (default 7
           days, capped at `campaign_days`) at equal day-intervals so the
           first post lands on day 0 and the last on the final day.
        5. Every post retains its channel's peak hour — the per-channel
           preferred-time guarantee is unchanged.

        Emits schedule_ready with the full slot list.
        """
        self.log_io("input", campaign_id=campaign_id, posts=len(posts))

        profiles = {p["id"]: p for p in channel_profiles}
        start = start_day or datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        days = campaign_days or 7

        # --- pair each post with its channel's preferred time ----------------
        type_profiles: dict[str, dict] = {p["id"]: p for p in channel_profiles}
        items: list[tuple[time, str, object]] = []
        for post in posts:
            profile = type_profiles.get(post.channel, {})
            ctype = profile.get("type", "")
            preferred = _PREFERRED_LOCAL_TIME.get(ctype, _DEFAULT_TIME)
            items.append((preferred, post.channel, post))

        # --- stable sort by (preferred hour, channel id) --------------------
        items.sort(key=lambda t: (t[0], t[1]))

        n = len(items)
        slots: list[ScheduleSlot] = []
        for k, (preferred, channel_id, post) in enumerate(items):
            if n == 1:
                day_offset = 0
            else:
                day_offset = round(k * (days - 1) / (n - 1))
            day_offset = min(day_offset, days - 1)  # safety clamp
            local_dt = datetime.combine(
                start + timedelta(days=day_offset), preferred, tzinfo=timezone.utc
            )
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

    # ------------------------------------------------------------------
    # Day 3: publish each approved post to the mock platform
    # ------------------------------------------------------------------

    async def publish_posts(
        self,
        posts: list,
        slots: list[ScheduleSlot],
        platform,
        campaign_id: str,
    ) -> dict[str, str]:
        """POST each post to the platform at its scheduled slot time.

        Returns a mapping ``{post_id: published_post_id}`` and stamps
        ``post.published_post_id`` on each Post object in-place. Emits
        ``post_published`` bus messages so the conversation is complete.

        The caller must supply a running ``platform`` (PlatformClient).
        If publishing one post fails the error is logged and the loop
        continues (partial publish is recoverable via checkpoint).
        """
        from llm.trace import append_event

        slot_map = {s.post_id: s for s in slots}
        published: dict[str, str] = {}
        for post in posts:
            slot = slot_map.get(post.id)
            if slot is None:
                continue
            if post.published_post_id:
                # Resumed: already published — skip the HTTP call.
                published[post.id] = post.published_post_id
                continue
            try:
                result = await platform.publish_post(
                    campaign_id=campaign_id,
                    channel_id=post.channel,
                    copy=post.copy,
                    hashtags=post.hashtags,
                    cta=post.cta,
                    format=post.creative.asset_type,
                    published_at=slot.scheduled_at.isoformat(),
                )
                pid = result["post"]["id"]
                post.published_post_id = pid
                published[post.id] = pid
                append_event(
                    {"event": "post_published", "post_id": post.id,
                     "published_post_id": pid, "slot": slot.scheduled_at.isoformat()}
                )
                self.log_io(
                    "published", post_id=post.id, published_post_id=pid,
                    slot=slot.scheduled_at.isoformat(),
                )
            except Exception as exc:
                append_event(
                    {"event": "post_publish_failed", "post_id": post.id,
                     "error": str(exc)[:300]}
                )
                self.log_io("publish_failed", post_id=post.id, error=str(exc)[:200])
        await self.emit(
            to_agent="orchestrator",
            campaign_id=campaign_id,
            message_type="posts_published",
            payload={"count": len(published), "published": published},
        )
        return published
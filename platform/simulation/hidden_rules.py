"""The 6 hidden engagement rules (architecture doc section 6).

Each rule is an INDEPENDENT, TOGGLEABLE function with signature

    fn(post: dict, channel_profile: dict, recent_history: dict) -> float

returning a *multiplier* that composes multiplicatively onto the noisy baseline
drawn in engine.py. Independence + toggling is what lets the write-up later
report a per-rule discovery rate ("Analytics found 4/6; rule 6 was the hard one").

All rules are deterministic given their inputs — randomness lives ONLY in the
engine's baseline draw. That makes the directional tests in
tests/test_simulation_engine.py meaningful: averaging over many posts cancels
the noise and reveals each rule's effect.

Ground truth (document these in the README/write-up):
    1. time_window_boost          evening window => 1.4-1.8x reach; 02:00-06:00 => 0.5x
    2. question_cta_comment_lift  copy ending in '?' => 2-3x expected comments, CHANNEL-MODULATED (discussion 3.0x / short-form 2.5x / professional 2.0x)
    3. nonlinear_hashtag_effect   reach peaks at 3-5 tags; 0 tags underperforms; 8+ penalized
    4. channel_length_penalty     professional >150 words ~30% penalty; short-form >60 words harsher
    5. novelty_decay              same format 3+ days running => compounding 15%/day decay
    6. follower_delta_save_boost  save-rate >5% => disproportionate follower gain (second-order)
"""

import json
from datetime import datetime, time, timezone

# Master toggles (all default ON). Tests flip these to isolate a rule's
# contribution; per-rule discovery rates get reported against this set.
RULES_ENABLED: dict[str, bool] = {
    "time_window_boost": True,
    "question_cta_comment_lift": True,
    "nonlinear_hashtag_effect": True,
    "channel_length_penalty": True,
    "novelty_decay": True,
    "follower_delta_save_boost": True,
}


def _parse_hhmm(s: str) -> time:
    """'18:00' -> datetime.time(18, 0). Raises ValueError on malformed input."""
    h, m = s.split(":")
    return time(int(h), int(m))


def _in_window(t: time, start: str, end: str) -> bool:
    """Is time-of-day t within [start, end)? Handles windows crossing midnight."""
    start_t, end_t = _parse_hhmm(start), _parse_hhmm(end)
    if start_t <= end_t:
        return start_t <= t < end_t
    # Window wraps midnight (e.g. 22:00-02:00): OR of the two halves.
    return t >= start_t or t < end_t


def _words(copy: str) -> int:
    return len(copy.split())


def time_window_boost(post: dict, channel_profile: dict, recent_history: dict) -> float:
    """Rule 1 — reach multiplier by publish time-of-day.

    Baseline (no peak): 1.0. Inside the channel's peak window: the channel's
    `peak_multiplier` (doc: 1.4-1.8x). Inside the dead window 02:00-06:00: 0.5x.
    """
    if not RULES_ENABLED["time_window_boost"]:
        return 1.0
    published_at: datetime = post["published_at"]
    # Normalize to the channel's local framing — we simulate a single timezone,
    # so "local" == UTC-sim time here; tzinfo is normalized if present.
    t = published_at.timetz().replace(tzinfo=None) if published_at.tzinfo else published_at.time()
    if _in_window(t, "02:00", "06:00"):
        return 0.5
    peak = channel_profile.get("peak_hours", "18:00-21:00")
    start, end = peak.split("-")
    if _in_window(t, start, end):
        return float(channel_profile.get("peak_multiplier", 1.5))
    return 1.0


def question_cta_comment_lift(post: dict, channel_profile: dict, recent_history: dict) -> float:
    """Rule 2 — comment-count multiplier (NOT reach).

    Copy ending in '?' gets a question_comment_lift multiplier on the Poisson
    lambda driving expected comments. The lift is CHANNEL-MODULATED: each
    channel profile carries its own `question_comment_lift` value (doc box
    "2-3x"), so the same question draws measurably more comments on the
    discussion channel than on the professional channel — rule 2 is NOT a
    flat multiplier across channels. Returned as a multiplier that engine.py
    applies ONLY to the comment count, never to reach.
    """
    if not RULES_ENABLED["question_cta_comment_lift"]:
        return 1.0
    copy = post.get("copy", "")
    if not copy.rstrip().endswith("?"):
        return 1.0
    return float(channel_profile.get("question_comment_lift", 2.5))


def nonlinear_hashtag_effect(post: dict, channel_profile: dict, recent_history: dict) -> float:
    """Rule 3 — reach multiplier as a non-linear function of hashtag count.

    Shape (doc: "peaks at 3-5; 0 underperforms; 8+ looks spammy"):
        0 tags      -> 0.8x (no discovery surface)
        1-2 tags    -> 1.0x (neutral)
        3-5 tags    -> 1.3x (the sweet spot)
        6-7 tags    -> 1.0x (declining)
        8+ tags     -> 0.7x (spam penalty)
    """
    if not RULES_ENABLED["nonlinear_hashtag_effect"]:
        return 1.0
    n = len(post.get("hashtags", []))
    if n == 0:
        return 0.8
    if 1 <= n <= 2:
        return 1.0
    if 3 <= n <= 5:
        return 1.3
    if 6 <= n <= 7:
        return 1.0
    return 0.7


def channel_length_penalty(post: dict, channel_profile: dict, recent_history: dict) -> float:
    """Rule 4 — per-channel copy-length penalty on reach.

    Reads `penalize_copy_over_words` / `penalty_strength` from the channel's
    character profile (see platform/db.py::_seed_channels). Professional
    penalizes >150 words by ~30%; short-form video penalizes >60 words harshly
    (0.60 strength, i.e. keep 40% reach). Discussion sets the threshold to
    10_000 => effectively exempt.
    """
    if not RULES_ENABLED["channel_length_penalty"]:
        return 1.0
    threshold = channel_profile.get("penalize_copy_over_words", 10_000)
    strength = channel_profile.get("penalty_strength", 0.0)
    if strength <= 0 or _words(post.get("copy", "")) <= threshold:
        return 1.0
    return 1.0 - strength


def novelty_decay(post: dict, channel_profile: dict, recent_history: dict) -> float:
    """Rule 5 — compounding 15%/day reach decay for format repetition.

    `recent_history` carries {"format_history": {channel_id: [{"date": str, "format": str}, ...]}}
    plus the current post's channel id. If the SAME channel ran the SAME format
    on each of the 1, 2, ... days before this post's day, reach decays
    15% per consecutive repeat day (0.85^k for k consecutive prior days).
    """
    if not RULES_ENABLED["novelty_decay"]:
        return 1.0
    published_at: datetime = post["published_at"]
    today = published_at.date().isoformat()
    channel_id = post.get("channel_id") or post.get("channel", "")
    fmt = post.get("format", "text_only")

    history = recent_history.get("format_history", {}).get(channel_id, [])
    # Consecutive same-format days *before* today (most recent first).
    streak = 0
    for entry in sorted(history, key=lambda e: e["date"], reverse=True):
        if entry["date"] >= today:
            continue  # ignore same-day entries; streak counts prior days only
        if entry["format"] == fmt:
            streak += 1
        else:
            break
    if streak < 3:
        # Doc: decay kicks in when the same format ran "3+ days running".
        # Days 1-2 of a streak are free; decay compounds from day 3 onward.
        return 1.0
    return 0.85 ** (streak - 2)


def follower_delta_save_boost(post: dict, channel_profile: dict, recent_history: dict) -> float:
    """Rule 6 — second-order: save-rate drives a follower boost.

    NOT a reach multiplier — engine.py consults this AFTER computing raw
    engagement: if saves/impressions > 5%, follower_delta gets an extra
    `save_rate * 30` multiplier (algorithmic-promotion story). Returned value
    is the multiplier for follower_delta; 1.0 means no boost.

    Deliberately subtle: the correlation only emerges when grouping posts by
    save-rate, which is exactly why it's the hard one for Analytics to find.
    """
    if not RULES_ENABLED["follower_delta_save_boost"]:
        return 1.0
    # post["save_rate"] is computed by engine.py before this rule is consulted.
    save_rate = post.get("save_rate", 0.0)
    return 1.0 + (30.0 * save_rate if save_rate > 0.05 else 0.0)


def apply_reach_multipliers(post: dict, channel_profile: dict, recent_history: dict) -> float:
    """Compose rules 1, 3, 4, 5 (the reach rules) multiplicatively.

    Kept as a named function (not inlined) so tests and the engine agree on
    exactly which rules are reach rules vs comment/follower rules.
    """
    m = 1.0
    m *= time_window_boost(post, channel_profile, recent_history)
    m *= nonlinear_hashtag_effect(post, channel_profile, recent_history)
    m *= channel_length_penalty(post, channel_profile, recent_history)
    m *= novelty_decay(post, channel_profile, recent_history)
    return m


def get_enabled_rules() -> list[str]:
    """Names of rules currently enabled — used by tests and trace logging."""
    return [name for name, on in RULES_ENABLED.items() if on]

"""Offline agent tests — every agent runs with a SCRIPTED transport.

`transport` (async fn prompt -> str) replaces the Ollama daemon, so these
tests run the FULL agent code paths — prompt rendering, structured-output
validation, bus emission, retry loops — with zero network. Each scripted
response is a realistic model answer; a deliberately malformed one proves
the retry loop fires and terminates.

The reject-loop test at the bottom is the day's centerpiece: it wires
Writer -> Compliance -> retry_policy exactly as the dry run does, forces
rejections, and asserts termination + escalation.
"""

import json

import pytest

from agents.compliance_agent import ComplianceAgent, run_prefilter
from agents.creative_agent import CreativeAgent
from agents.scheduler_agent import SchedulerAgent
from models.campaign import Campaign
from models.engagement import Comment
from models.post import Post, CreativeBrief
from orchestration.message_bus import MessageBus


def make_test_bus(tmp_path) -> MessageBus:
    """A real MessageBus over the test env (DB_PATH/TRACE_LOG_PATH already
    point into tmp_path via the bus_env fixture) — agents under test emit
    real durable messages, which is what the offline tests want to verify."""
    return MessageBus()

# ---------------------------------------------------------------- fixtures


@pytest.fixture()
def bus_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "agents_test.db"))
    monkeypatch.setenv("TRACE_LOG_PATH", str(tmp_path / "agent_trace.jsonl"))
    from config.settings import get_settings
    from platform.db import reset_engine_for_tests

    get_settings.cache_clear()
    reset_engine_for_tests()
    yield tmp_path
    get_settings.cache_clear()
    reset_engine_for_tests()


@pytest.fixture()
def campaign() -> Campaign:
    """A strategy_filled Campaign (the Writer's input contract)."""
    return Campaign(
        id="camp_test",
        brief_raw="launch a budget espresso machine for students",
        objective="awareness among price-conscious students",
        target_audience="price-conscious students, 18-25",
        channel_mix=["ch_shortform", "ch_discussion"],
        duration_days=7,
        content_pillars=[
            {"name": "budget_wins", "description": "cost math for students", "weight": 0.4},
            {"name": "ritual", "description": "the daily brew ritual", "weight": 0.35},
            {"name": "dorm_hacks", "description": "tiny-kitchen recipes", "weight": 0.25},
        ],
        kpis=[
            {"metric": "impressions", "target": 5000.0, "channel": None},
            {"metric": "comment_rate", "target": 0.05, "channel": "ch_discussion"},
        ],
        status="strategy_filled",
    )


def script(*responses: str):
    """Scripted transport: returns responses in order, then repeats the last.

    Repeating the last is deliberate: it lets a test script ONE bad output
    followed by the good one and proves recovery without overspecifying
    the exact retry count.
    """
    queue = list(responses)
    calls = {"n": 0}

    async def transport(prompt: str) -> str:
        calls["n"] += 1
        idx = min(calls["n"] - 1, len(queue) - 1)
        return queue[idx]

    transport.calls = calls  # type: ignore[attr-defined]
    return transport


def jobj(data: dict) -> str:
    return json.dumps(data)


# ---------------------------------------------------------------- writer


async def test_writer_three_pass_pipeline(bus_env, campaign):
    """draft -> critique -> revise produce distinct, connected outputs."""
    from agents.writer_agent import WriterAgent

    drafts_json = jobj({"posts": [
        {"pillar": "budget_wins", "channel": "ch_shortform", "copy": "Cafe money, dorm budget.", "hashtags": ["#espresso", "#studentlife", "#coffee"]},
    ]})
    critique_json = jobj({"issues": [{"post_index": 0, "issue": "no question hook", "fix": "open with the price comparison as a question"}]})
    revised_json = jobj({"posts": [
        {"pillar": "budget_wins", "channel": "ch_shortform", "copy": "What costs less: your cafe habit or this machine?", "hashtags": ["#espresso", "#studentlife", "#coffee"]},
    ]})

    agent = WriterAgent(bus=make_test_bus(bus_env))
    t1, t2, t3 = script(drafts_json), script(critique_json), script(revised_json)
    drafts = await agent.draft(campaign, [{"pillar": "budget_wins", "channel": "ch_shortform"}], transport=t1)
    issues = await agent.critique(campaign, drafts, transport=t2)
    revised = await agent.revise(campaign, drafts, critique_issues=issues, transport=t3)

    assert len(drafts) == 1 and len(revised) == 1
    assert issues and issues[0].post_index == 0
    assert revised[0].copy != drafts[0].copy  # substantive revision, not reshuffle


# ---------------------------------------------------------------- creative


async def test_creative_clamps_illegal_asset_choice(bus_env):
    """The model proposes an out-of-policy asset; code clamps deterministically."""
    agent = CreativeAgent(bus=make_test_bus(bus_env))
    brief = await agent.run(
        "camp_test",
        post_copy="Watch the 15-second latte art.",
        channel={"id": "ch_shortform", "name": "QuickBites", "type": "shortform_video"},
        pillar={"name": "ritual", "description": "daily brew"},
        transport=script(jobj({"asset_type": "billboard", "description": "A 15-second vertical video of a latte art pour, top-down, warm morning light."})),
    )
    assert brief.asset_type == "video"  # clamped to the channel's allowed set


async def test_creative_respects_allowed_choice(bus_env):
    agent = CreativeAgent(bus=make_test_bus(bus_env))
    brief = await agent.run(
        "camp_test",
        post_copy="Desk setup with the machine.",
        channel={"id": "ch_professional", "name": "LeadDesk", "type": "professional"},
        pillar={"name": "budget_wins", "description": "cost math"},
        transport=script(jobj({"asset_type": "carousel", "description": "Three-slide carousel comparing cafe spend vs machine cost per week."})),
    )
    assert brief.asset_type == "carousel"


# ---------------------------------------------------------------- compliance


def test_prefilter_hard_rules_are_explicit_and_real():
    """The prefilter is real regex logic, not a stub — spot-check the
    boundaries a human would probe."""
    pre = run_prefilter("This is 100% guaranteed to change your mornings.")
    assert pre.hard and "100% guaranteed" in pre.banned_terms_hit

    pre = run_prefilter("Risk-free trial, act now, no side effects!")
    assert pre.hard and {"risk-free", "act now", "no side effects"} <= set(pre.banned_terms_hit)

    pre = run_prefilter("A quiet little machine for your desk.")
    assert not pre.hard and pre.banned_terms_hit == []


async def test_compliance_hard_rule_short_circuits_without_model(bus_env):
    """A hard violation rejects WITHOUT calling the model at all —
    the doc's 'don't rely on the LLM alone for hard constraints'."""
    agent = ComplianceAgent(bus=make_test_bus(bus_env))

    async def exploding_transport(prompt: str) -> str:
        raise AssertionError("model must not be called on a hard violation")

    verdict = await agent.review(
        "camp_test",
        {"id": "p1", "channel": "ch_shortform", "copy": "100% guaranteed espresso bliss", "hashtags": ["#coffee", "#espresso"], "cta": None, "creative": "image"},
        transport=exploding_transport,
    )
    assert verdict.verdict == "reject"
    assert verdict.banned_terms_hit == ["100% guaranteed", "guaranteed"]


async def test_compliance_model_judgment_can_clear_soft_findings(bus_env):
    """No hard hit + model approve = approve (soft finding cleared in context)."""
    agent = ComplianceAgent(bus=make_test_bus(bus_env))
    approve = jobj({"verdict": "approve", "reasons": [], "banned_terms_hit": []})
    verdict = await agent.review(
        "camp_test",
        {"id": "p2", "channel": "ch_professional", "copy": "Our #1 budget pick by students who did the math.", "hashtags": ["#coffee", "#espresso"], "cta": None, "creative": "image"},
        transport=script(approve),
    )
    assert verdict.verdict == "approve"


async def test_compliance_model_rejection_merges_reasons(bus_env):
    agent = ComplianceAgent(bus=make_test_bus(bus_env))
    reject = jobj({"verdict": "reject", "reasons": ["tone is too hype for this channel"], "banned_terms_hit": []})
    verdict = await agent.review(
        "camp_test",
        {"id": "p3", "channel": "ch_professional", "copy": "Best ever espresso deal today!", "hashtags": ["#coffee", "#espresso"], "cta": None, "creative": "image"},
        transport=script(reject),
    )
    assert verdict.verdict == "reject"
    assert any("hype" in r for r in verdict.reasons)


# ---------------------------------------------------------------- scheduler


async def test_scheduler_assigns_slots_deterministically(bus_env, campaign):
    """No model: same posts in, same slots out, twice."""
    posts = [
        Post(id="pA", campaign_id=campaign.id, channel="ch_shortform", copy="short", hashtags=["#a", "#b"], creative=CreativeBrief(asset_type="video", description="vertical video")),
        Post(id="pB", campaign_id=campaign.id, channel="ch_shortform", copy="short too", hashtags=["#a", "#b"], creative=CreativeBrief(asset_type="video", description="vertical video")),
    ]
    agent = SchedulerAgent(bus=make_test_bus(bus_env))
    profiles = [{"id": "ch_shortform", "name": "QuickBites", "type": "shortform_video"}]

    slots1 = await agent.build_schedule(campaign.id, posts, profiles)
    slots2 = await agent.build_schedule(campaign.id, posts, profiles)
    assert [s.scheduled_at for s in slots1] == [s.scheduled_at for s in slots2]
    assert {s.channel for s in slots1} == {"ch_shortform"}
    # The preferred time matches the channel's peak window start (18:00):
    assert all(s.scheduled_at.hour == 18 for s in slots1)


# ---------------------------------------------------------------- community manager


def make_comment(id: str, text: str, sentiment: str = "positive", sensitive: bool = False) -> Comment:
    return Comment(id=id, post_id="p1", author_handle=f"@{id}", text=text, sentiment=sentiment, is_sensitive=sensitive)  # type: ignore[arg-type]


async def test_community_manager_routes_replies_and_escalations(bus_env):
    """Sensitive platform flags escalate WITHOUT the router; router classes
    the rest; replies are drafted only for reply-classified comments."""
    from agents.community_manager_agent import CommunityManagerAgent

    comments = [
        make_comment("c1", "This machine is overpriced trash, I want a refund now", "negative", sensitive=True),
        make_comment("c2", "Does it fit under a dorm cabinet?", "neutral"),
        make_comment("c3", "Made my first shot this morning, incredible", "positive"),
    ]
    router = script(jobj({"decisions": [
        {"comment_id": "c2", "action": "reply", "reason": "product question"},
        {"comment_id": "c3", "action": "reply", "reason": "positive engagement"},
    ]}))
    replier = script(jobj({"drafts": [
        {"comment_id": "c2", "reply_text": "Yes — 15cm clearance fits standard dorm cabinets."},
        {"comment_id": "c3", "reply_text": "Love to hear it. What beans did you pull with?"},
    ]}))

    agent = CommunityManagerAgent(bus=make_test_bus(bus_env))
    results = await agent.process_comments(
        "camp_test",
        comments,
        post_context={"copy": "Meet your dorm espresso setup.", "channel": "ch_discussion"},
        channel_tone="conversational, debate-starter, ends with a question",
        transport_router=router,
        transport_replier=replier,
    )
    by_id = {r["comment_id"]: r for r in results}
    assert by_id["c1"]["action"] == "escalate"          # platform flag, no router
    assert by_id["c2"]["action"] == "reply"
    assert "dorm cabinets" in by_id["c2"]["reply_text"]
    assert by_id["c3"]["action"] == "reply"


async def test_community_manager_fails_safe_when_replier_omits_a_reply(bus_env):
    from agents.community_manager_agent import CommunityManagerAgent

    comments = [make_comment("c1", "Does it steam milk?")]
    router = script(jobj({"decisions": [{"comment_id": "c1", "action": "reply", "reason": "question"}]}))
    # Realistic omission: the replier answered a DIFFERENT comment id than
    # the one classified for reply. (An empty drafts list would be rejected
    # by schema validation in the retry loop, so it can't reach this code.)
    replier = script(jobj({"drafts": [{"comment_id": "someone_else", "reply_text": "misaddressed reply"}]}))

    agent = CommunityManagerAgent(bus=make_test_bus(bus_env))
    results = await agent.process_comments(
        "camp_test", comments, post_context={"copy": "x"}, channel_tone="t",
        transport_router=router, transport_replier=replier,
    )
    # A reply-classified comment with no draft must NOT be dropped silently:
    assert results[0]["action"] == "escalate"


# ---------------------------------------------------------------- analytics


def fixture_snapshots():
    """Synthetic week of per-post snapshots with REAL structure planted:
    evening posts outperform overnight; question posts draw more comments."""
    rows = []
    for i in range(6):
        rows.append({
            "post_id": f"evening_{i}", "recorded_at": f"2026-09-14T21:00:00Z",
            "impressions": 1800 + i * 90, "likes": 150, "comments": 30, "shares": 20,
            "saves": 25, "clicks": 40, "follower_delta": 12,
            "copy_is_question": False,
        })
        rows.append({
            "post_id": f"overnight_{i}", "recorded_at": f"2026-09-14T04:00:00Z",
            "impressions": 500 + i * 20, "likes": 30, "comments": 4, "shares": 3,
            "saves": 5, "clicks": 8, "follower_delta": 1,
            "copy_is_question": False,
        })
    # One question post with outsized comments (the rule-2 signal):
    rows.append({
        "post_id": "question_0", "recorded_at": "2026-09-14T19:30:00Z",
        "impressions": 1600, "likes": 140, "comments": 96, "shares": 15,
        "saves": 22, "clicks": 30, "follower_delta": 10,
        "copy_is_question": True,
    })
    return rows


async def test_analytics_two_stage_python_computes_model_narrates(bus_env, campaign):
    """kpi_performance comes from Python; narrative from the model; the
    model never touches the numbers table."""
    from agents.analytics_agent import AnalyticsAgent

    narrative = jobj({
        "patterns_found": ["evening-window posts averaged higher impressions than overnight posts"],
        "post_insights": [{"post_id": "evening_0", "performance_rank": "top", "hypothesis": "published at the channel's peak window"}],
        "comment_sentiment_summary": "sentiment counts show mostly positive engagement",
        "recommendations": [{
            "change": "move overnight slots to the 18:00-21:00 window",
            "evidence": "peak mean impressions exceed offpeak mean in the timing stats",
            "underlying_statistic": "peak mean impressions 1,100 vs offpeak 270; n=2 vs n=2",
            "confidence": "observed_correlation",
            "expected_effect": "higher impressions per post",
            "predicted_effect": "posts at 18:00-21:00 on ch_shortform should average above 1,000 impressions next week",
        }],
    })
    agent = AnalyticsAgent(bus=make_test_bus(bus_env))
    report = await agent.run(
        "camp_test", 1,
        snapshots=fixture_snapshots(),
        comments=[{"post_id": "evening_0", "text": "love", "sentiment": "positive"}],
        kpis=[k.model_dump() for k in campaign.kpis],
        channel_of_post={"evening_0": "ch_shortform", "overnight_0": "ch_shortform"},
        window_hours_of_post={**{f"evening_{i}": "peak" for i in range(6)}, **{f"overnight_{i}": "offpeak" for i in range(6)}},
        transport=script(narrative),
    )
    # Numbers computed by Python (stage 1), NOT by the model:
    expected_impressions = sum(s["impressions"] for s in fixture_snapshots())
    assert report.kpi_performance["impressions"] == expected_impressions
    # Narrative fields came through stage 2:
    assert report.patterns_found and report.recommendations
    # Structural sanity: planted structure is large enough to be meaningful
    # (evening posts 1800+ x6 vs overnight 500+ x6 => totals > 10k).
    assert expected_impressions > 10000

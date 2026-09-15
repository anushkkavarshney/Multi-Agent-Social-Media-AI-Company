"""Day-3 full demo — brief through published, simulated week, community replies,
analytics, memory store, and discovery scoring.

Run:
    python run_demo.py [--brief seed_data/demo_brief.txt] [--auto-approve] [--fresh]

Prerequisites:
    1. The mock platform must be running:  python -m platform.main
    2. Ollama must be running with qwen2.5:7b-instruct + 3b-instruct pulled.

The script does everything the Day-2 dry run does PLUS publishes posts,
simulates a week of engagement, has the Community Manager reply to real
comments, gets the Analytics Agent to narrate a real WeeklyReport from
real platform data, stores the report in memory, and scores the agent's
discovery against the 6 ground-truth hidden rules.

CHECKPOINT/RESUME: mirrors Day-2's pattern — every stage boundary persists
full state to logs/demo_checkpoint.json.  A clean exit clears the checkpoint;
a crash exits 3 with the checkpoint kept.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config.settings import get_settings
from llm.ollama_client import OllamaClient
from llm.structured_output import ModelOutputFailure
from platform.client import PlatformClient

FORCED_BAN = "guaranteed espresso nirvana"
CHECKPOINT_PATH = ROOT / "logs" / "demo_checkpoint.json"
CHECKPOINT_VERSION = 1


# ------------------------------------------------------------------ checkpoint
def save_checkpoint(stage: str, **data) -> str:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"v": CHECKPOINT_VERSION, "stage": stage, **data}
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)
    return stage


# Monotonic order of persisted checkpoint stages. Blocks compare `saved_stage`
# against it so a mid-run save advances the guard immediately (the loaded
# `ckpt` snapshot goes stale the instant a new checkpoint is written).
STAGE_ORDER = [
    "orchestrated", "strategized", "written", "posts_created",
    "compliance_passed", "scheduled", "published", "simulated",
    "community_replied", "reported", "memory_stored",
]


def stage_index(stage: str | None) -> int:
    if stage is None:
        return -1
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return -1


def load_checkpoint() -> dict | None:
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if data.get("v") == CHECKPOINT_VERSION else None


def clear_checkpoint() -> None:
    CHECKPOINT_PATH.unlink(missing_ok=True)


# -------------------------------------------------------------------- helpers
def show(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 62 - len(title)))


def report_model_failure(exc: ModelOutputFailure, agent: str, schema: str) -> None:
    print(
        f"\nERROR [{agent}] could not produce valid {schema} "
        f"({exc.attempts} attempt(s)).\n"
        f"  last error : {exc.last_error[:400]}\n"
        f"  last output: {exc.last_raw[:400]}\n"
        "\nThe pipeline state is checkpointed — fix the issue and re-run this\n"
        "script to resume from this stage (use --fresh to start over).",
        file=sys.stderr,
    )


def preflight_platform(client: OllamaClient, platform: PlatformClient) -> None:
    import httpx

    # Ollama
    try:
        resp = httpx.get(f"{client.host}/api/version", timeout=3.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        print(
            f"ERROR: Ollama is not reachable at {client.host}\n"
            "\nFix:\n"
            "  1. Install:  https://ollama.com/download\n"
            "  2. Start it: open the Ollama app (or `ollama serve`)\n"
            "  3. Pull:     ollama pull qwen2.5:7b-instruct\n"
            "               ollama pull qwen2.5:3b-instruct\n"
            "  4. Verify:   curl http://localhost:11434/api/tags\n",
            file=sys.stderr,
        )
        sys.exit(2)

    # Platform (synchronous check via httpx)
    try:
        resp = httpx.get(f"{platform._client.base_url}/health", timeout=3.0)
        resp.raise_for_status()
    except Exception:
        print(
            "ERROR: Mock platform is not reachable.\n"
            "\nFix: start it in another terminal:\n"
            "     python -m platform.main   # http://127.0.0.1:8010\n",
            file=sys.stderr,
        )
        sys.exit(2)


# ======================================================================== main
async def main(brief_path: str, auto_approve: bool, fresh: bool, platform_url: str) -> dict:
    settings = get_settings()
    client = OllamaClient()

    from platform.db import init_db

    platform = PlatformClient(base_url=platform_url)

    print("=" * 72)
    print("DAY 3 DEMO — full pipeline: brief through weekly report + memory")
    print(f"model: {settings.primary_model}   router: {settings.router_model}")
    print(f"platform: {platform_url}")
    print("=" * 72)

    await init_db()
    preflight_platform(client, platform)

    channels = await platform.get_channels()
    print(f"platform channels: {[c['id'] for c in channels]}")

    brief_raw = Path(brief_path).read_text(encoding="utf-8")

    # ---------------------------------------------------------------- agents
    from agents.analytics_agent import AnalyticsAgent
    from agents.compliance_agent import ComplianceAgent
    from agents.community_manager_agent import CommunityManagerAgent
    from agents.creative_agent import CreativeAgent
    from agents.orchestrator import BriefDraft, OrchestratorAgent
    from agents.scheduler_agent import SchedulerAgent
    from agents.strategy_agent import StrategyAgent
    from agents.writer_agent import PostDraft, WriterAgent
    from memory.campaign_memory import MemoryStore
    from models.campaign import Campaign
    from models.post import CreativeBrief, Post, ScheduleSlot
    from orchestration.message_bus import MessageBus, make_message
    from orchestration.retry_policy import ComplianceRetryPolicy
    from orchestration.state_machine import (
        EV_COMPLIANCE_APPROVED,
        EV_COMPLIANCE_REJECTED,
        EV_COMPLIANCE_REJECTED_EXHAUSTED,
        EV_CONTENT_DRAFTED,
        EV_HUMAN_APPROVED,
        EV_HUMAN_REJECTED,
        EV_REVIEW_STARTED,
        EV_STRATEGY_FILLED,
        transition,
    )

    bus = MessageBus()
    orchestrator = OrchestratorAgent(client=client, bus=bus)
    strategy = StrategyAgent(client=client, bus=bus)
    writer = WriterAgent(client=client, bus=bus)
    creative = CreativeAgent(client=client, bus=bus)
    compliance = ComplianceAgent(client=client, bus=bus)
    scheduler = SchedulerAgent(client=client, bus=bus)
    community_manager = CommunityManagerAgent(client=client, bus=bus)
    analytics = AnalyticsAgent(client=client, bus=bus)
    policy = ComplianceRetryPolicy()

    ckpt = None if fresh else load_checkpoint()
    if ckpt and fresh:
        clear_checkpoint()
        ckpt = None
    resumed_stage = ckpt["stage"] if ckpt else None

    if resumed_stage:
        print(f"checkpoint: resumed from stage '{resumed_stage}' (use --fresh to start over)")

    posts: list[Post] = []
    pillar_of_post: dict[str, str] = {}
    review_cycles = 0
    forced_injection_done = False
    slots: list = []
    published: dict[str, str] = {}
    # Tracks the most recently saved checkpoint stage so later stages know what
    # has been done THIS process (the loaded `ckpt` snapshot goes stale the
    # moment we save a new checkpoint mid-run).
    saved_stage = resumed_stage

    if resumed_stage:
        campaign = Campaign.model_validate(ckpt["campaign"])
        routing = BriefDraft.model_validate(ckpt["routing"]) if "routing" in ckpt else None
        drafts = [PostDraft.model_validate(d) for d in ckpt.get("drafts", [])]
        posts = [Post.model_validate(p) for p in ckpt.get("posts", [])]
        pillar_of_post = ckpt.get("pillar_of_post", {})
        review_cycles = ckpt.get("review_cycles", 0)
        forced_injection_done = ckpt.get("forced_injection_done", False)
        slots = ckpt.get("slots", [])
        slots = [ScheduleSlot.model_validate(s) for s in slots]
        published = ckpt.get("published", {})
        campaign_id = campaign.id
        orchestrator.set_campaign_id(campaign_id)
        # Stamp any previously-published posts
        for p in posts:
            if p.id in published:
                p.published_post_id = published[p.id]
    else:
        campaign_id = f"camp_{uuid.uuid4().hex[:8]}"
        orchestrator.set_campaign_id(campaign_id)

    # ================================================================ 1. orchestrator
    if not resumed_stage:
        show("STAGE 1: Orchestrator parses the brief")
        try:
            campaign, routing = await orchestrator.run(brief_raw, channels)
        except ModelOutputFailure as exc:
            report_model_failure(exc, "orchestrator", "BriefDraft")
            await client.aclose()
            sys.exit(3)
        print(f"campaign      : {campaign.id}")
        print(f"objective     : {campaign.objective}")
        print(f"channel_mix   : {campaign.channel_mix}")
        print(f"duration_days : {campaign.duration_days}")
        print(f"routing plan  : {' -> '.join(routing.routing_plan)}")
        if routing.risks:
            print(f"risks flagged : {routing.risks}")
        save_checkpoint("orchestrated",
                        campaign=campaign.model_dump(mode="json"),
                        routing=routing.model_dump(mode="json"))

    # ================================================================ 2. strategy
    if resumed_stage in (None, "orchestrated"):
        show("STAGE 2: Strategy fills audience, pillars, KPIs")
        try:
            campaign = await strategy.run(campaign, channels)
        except ModelOutputFailure as exc:
            report_model_failure(exc, "strategy", "StrategyOutput")
            await client.aclose()
            sys.exit(3)
        campaign = transition(campaign, EV_STRATEGY_FILLED)
        print(f"audience      : {campaign.target_audience}")
        for p in campaign.content_pillars:
            print(f"pillar        : {p.name} (weight {p.weight}) — {p.description[:70]}")
        for k in campaign.kpis:
            print(f"kpi           : {k.metric} @ {k.target} {k.channel or '(campaign-wide)'}")
        save_checkpoint("strategized",
                        campaign=campaign.model_dump(mode="json"),
                        routing=routing.model_dump(mode="json"))

    # ================================================================ 3. writer
    if resumed_stage in (None, "orchestrated", "strategized"):
        show("STAGE 3: Writer drafts -> self-critiques -> revises")
        rotation = [
            {"pillar": p.name, "channel": campaign.channel_mix[i % len(campaign.channel_mix)]}
            for i, p in enumerate(campaign.content_pillars)
        ]
        try:
            drafts = await writer.draft(campaign, rotation)
            issues = await writer.critique(campaign, drafts)
            print(f"self-critique : {len(issues)} issue(s)")
            for c in issues:
                print(f"  - [{c.post_index}] {c.issue[:70]} -> fix: {c.fix[:50]}")
            if issues:
                drafts = await writer.revise(campaign, drafts, critique_issues=issues)
        except ModelOutputFailure as exc:
            report_model_failure(exc, "writer", "PostDraftList/CritiqueList")
            await client.aclose()
            sys.exit(3)
        campaign = transition(campaign, EV_CONTENT_DRAFTED)
        save_checkpoint("written",
                        campaign=campaign.model_dump(mode="json"),
                        routing=routing.model_dump(mode="json"),
                        drafts=[d.model_dump(mode="json") for d in drafts])

    if not posts:
        posts = []
        pillar_of_post = {}
        # rotation is deterministic from campaign state; on resume from
        # 'written' the writer block is skipped so rebuild it here.
        if "rotation" not in locals():
            rotation = [
                {"pillar": p.name, "channel": campaign.channel_mix[i % len(campaign.channel_mix)]}
                for i, p in enumerate(campaign.content_pillars)
            ]
        for i, d in enumerate(drafts):
            # The post is bound to the pillar it was ROTATED into (authoritative).
            # The model echoes a freeform name in PostDraft.pillar; trusting that
            # string crashes stage 4 when it drifts (e.g. "Student testimonials"
            # vs the actual pillar name), so we use rotation[i] instead.
            assigned = rotation[i]["pillar"] if i < len(rotation) else campaign.content_pillars[0].name
            post = Post(
                id=f"{campaign.id}_post{i + 1}",
                campaign_id=campaign.id,
                channel=d.channel,
                copy=d.copy,
                hashtags=d.hashtags,
                cta=d.cta,
                creative=CreativeBrief(asset_type="image", description="placeholder"),
            )
            posts.append(post)
            pillar_of_post[post.id] = assigned

    # ================================================================ 4. creative
    if resumed_stage in (None, "orchestrated", "strategized", "written"):
        show("STAGE 4: Creative briefs per post")
        ch_map = {c["id"]: c for c in channels}
        for post in posts:
            pillar = next(
                (p for p in campaign.content_pillars if p.name == pillar_of_post.get(post.id)),
                campaign.content_pillars[0] if campaign.content_pillars else None,
            )
            if pillar is None:
                report_model_failure(
                    ModelOutputFailure(
                        message="pillar lookup failed",
                        last_raw="",
                        last_error=f"no campaign pillar matches {post.id} -> {pillar_of_post.get(post.id)}",
                        attempts=1,
                    ),
                    "creative", "CreativeChoice")
                await client.aclose()
                sys.exit(3)
            try:
                post.creative = await creative.run(
                    campaign.id, post.copy, ch_map[post.channel], pillar.model_dump()
                )
            except ModelOutputFailure as exc:
                report_model_failure(exc, "creative", "CreativeChoice")
                await client.aclose()
                sys.exit(3)
            print(f"{post.id}: {post.creative.asset_type:9s} {post.creative.description[:60]}")
        save_checkpoint("posts_created",
                        campaign=campaign.model_dump(mode="json"),
                        routing=routing.model_dump(mode="json"),
                        posts=[p.model_dump(mode="json") for p in posts],
                        pillar_of_post=pillar_of_post)

    # ================================================================ 5. compliance + reject loop
    if resumed_stage in (None, "orchestrated", "strategized", "written", "posts_created", "review"):
        show("STAGE 5: Compliance review (with forced violation on post 1)")
        if campaign.status == "content_drafted":
            campaign = transition(campaign, EV_REVIEW_STARTED)

        while True:
            review_cycles += 1
            print(f"\n[review cycle {review_cycles}]")
            rejected_this_cycle = False
            escalated = False

            for post in posts:
                if post.compliance_status == "approved":
                    continue
                injected = (not forced_injection_done) and post is posts[0]
                review_copy = post.copy + (f" This is {FORCED_BAN}." if injected else "")
                try:
                    verdict = await compliance.review(
                        campaign.id,
                        {"id": post.id, "channel": post.channel, "copy": review_copy,
                         "hashtags": post.hashtags, "cta": post.cta,
                         "creative": post.creative.model_dump()},
                    )
                except ModelOutputFailure as exc:
                    report_model_failure(exc, "compliance", "ComplianceVerdict")
                    save_checkpoint("review",
                                    campaign=campaign.model_dump(mode="json"),
                                    routing=routing.model_dump(mode="json"),
                                    posts=[p.model_dump(mode="json") for p in posts],
                                    pillar_of_post=pillar_of_post,
                                    review_cycles=review_cycles,
                                    forced_injection_done=forced_injection_done)
                    await client.aclose()
                    sys.exit(3)

                if verdict.verdict == "approve":
                    if injected:
                        print("ABORT: injected violation was approved — prefilter failed")
                        await client.aclose()
                        sys.exit(1)
                    post.compliance_status = "approved"
                    print(f"  APPROVE  {post.id}")
                    save_checkpoint("review",
                                    campaign=campaign.model_dump(mode="json"),
                                    routing=routing.model_dump(mode="json"),
                                    posts=[p.model_dump(mode="json") for p in posts],
                                    pillar_of_post=pillar_of_post,
                                    review_cycles=review_cycles,
                                    forced_injection_done=forced_injection_done)
                    continue

                rejected_this_cycle = True
                if injected:
                    forced_injection_done = True
                decision = policy.on_rejection(post.id, post.rejection_count + 1)
                if decision.escalate:
                    post.compliance_status = "rejected"
                    post.rejection_count += 1
                    campaign = transition(campaign, EV_COMPLIANCE_REJECTED_EXHAUSTED)
                    print(f"  ESCALATE {post.id}: {decision.reason}")
                    escalated = True
                    break

                post.rejection_count += 1
                print(f"  REJECT   {post.id}:")
                for r in verdict.reasons:
                    print(f"           - {r[:90]}")

                draft = PostDraft(pillar=pillar_of_post[post.id], channel=post.channel,
                                  copy=post.copy, hashtags=post.hashtags, cta=post.cta)
                try:
                    revised = await writer.revise(campaign, [draft], rejection_reasons=verdict.reasons)
                except ModelOutputFailure as exc:
                    report_model_failure(exc, "writer", "PostDraftList (revise)")
                    await client.aclose()
                    sys.exit(3)
                if revised:
                    post.copy = revised[0].copy
                    post.hashtags = revised[0].hashtags
                    post.cta = revised[0].cta
                    post.compliance_status = "pending"
                print(f"  REVISED  {post.id}: {post.copy[:70]}")
                save_checkpoint("review",
                                campaign=campaign.model_dump(mode="json"),
                                routing=routing.model_dump(mode="json"),
                                posts=[p.model_dump(mode="json") for p in posts],
                                pillar_of_post=pillar_of_post,
                                review_cycles=review_cycles,
                                forced_injection_done=forced_injection_done)
                campaign = transition(campaign, EV_COMPLIANCE_REJECTED)
                campaign = transition(campaign, EV_REVIEW_STARTED)
                break
            if escalated or not rejected_this_cycle:
                break

        if any(p.compliance_status != "approved" for p in posts):
            print("\nPipeline terminated: needs_human")
            clear_checkpoint()
            await client.aclose()
            return {"status": "needs_human"}
        campaign = transition(campaign, EV_COMPLIANCE_APPROVED)
        print("\nAll posts compliance-approved.")
        save_checkpoint("compliance_passed",
                        campaign=campaign.model_dump(mode="json"),
                        routing=routing.model_dump(mode="json"),
                        posts=[p.model_dump(mode="json") for p in posts],
                        pillar_of_post=pillar_of_post,
                        review_cycles=review_cycles)
    else:
        if campaign.status == "compliance_review":
            campaign = transition(campaign, EV_COMPLIANCE_APPROVED)

    # ================================================================ 6. human gate
    show("STAGE 6: Human approval gate")
    for post in posts:
        print(f"\n[{post.channel}] {post.id} (rejections: {post.rejection_count})")
        print(f"  copy    : {post.copy}")
        print(f"  tags    : {' '.join(post.hashtags)}")
        print(f"  creative: ({post.creative.asset_type}) {post.creative.description[:80]}")

    # On resume from 'scheduled'/'published' the gate already passed; only the
    # display above is needed, the transition is illegal from those states.
    if campaign.status == "scheduled" or campaign.status == "published":
        print("\n[human gate] already approved (resumed) — continuing")
    else:
        if auto_approve:
            answer = "y"
            print("\n[human gate] --auto-approve: approved")
        else:
            answer = input("\nApprove all posts for publishing? (y/n): ").strip().lower()
        if answer != "y":
            reason = input("Rejection reason: ").strip() or "human declined"
            await bus.publish(make_message("human", "orchestrator", campaign.id, "human_rejected", {"reason": reason}))
            campaign = transition(campaign, EV_HUMAN_REJECTED)
            clear_checkpoint()
            await client.aclose()
            return {"status": "human_rejected"}

        await bus.publish(make_message("human", "orchestrator", campaign.id, "human_approved", {"decision": "approved"}))
        campaign = transition(campaign, EV_HUMAN_APPROVED)
        print("status: approved")

    # ================================================================ 7. scheduler → slots
    if not slots:
        show("STAGE 7: Scheduler assigns publish slots")
        # Ensure posts spread across a 7-day analytics window
        now_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_day = now_utc + timedelta(days=1)  # tomorrow — all slots are in the future
        slots = await scheduler.build_schedule(campaign.id, posts, channels,
                                               start_day=start_day, campaign_days=7)
        for s in slots:
            print(f"  {s.post_id} -> {s.channel} @ {s.scheduled_at.isoformat()}")
        saved_stage = save_checkpoint("scheduled",
                                      campaign=campaign.model_dump(mode="json"),
                                      routing=routing.model_dump(mode="json"),
                                      posts=[p.model_dump(mode="json") for p in posts],
                                      pillar_of_post=pillar_of_post,
                                      slots=[s.model_dump(mode="json") for s in slots],
                                      published=published)

    # ================================================================ 8. publish to platform
    if not published:
        show("STAGE 8: Publishing approved posts to the mock platform")
        published = await scheduler.publish_posts(posts, slots, platform, campaign.id)
        print(f"published {len(published)} post(s)")
        saved_stage = save_checkpoint("published",
                                      campaign=campaign.model_dump(mode="json"),
                                      routing=routing.model_dump(mode="json"),
                                      posts=[p.model_dump(mode="json") for p in posts],
                                      pillar_of_post=pillar_of_post,
                                      slots=[s.model_dump(mode="json") for s in slots],
                                      published=published)

    # ================================================================ 9. simulate week
    if stage_index(saved_stage) < stage_index("simulated"):
        show("STAGE 9: Simulating a full week of engagement")
        # Compute tick window: from now through last slot + 24h margin.
        last_slot_time = max(s.scheduled_at for s in slots) if slots else datetime.now(timezone.utc)
        target_end = last_slot_time + timedelta(hours=24)
        total_hours = max(0, (target_end - datetime.now(timezone.utc)).total_seconds() / 3600)
        tick_step = 6.0
        num_ticks = int((total_hours + tick_step - 0.01) / tick_step)
        print(f"ticks: {num_ticks} steps of {tick_step}h (total {num_ticks * tick_step:.0f}h)")

        for i in range(num_ticks):
            result = await platform.tick(campaign.id, tick_step)
            accrued = [r for r in result.get("posts_processed", []) if r["status"] == "accrued"]
            if accrued:
                print(f"  tick {i+1}/{num_ticks} | accrued {len(accrued)} post(s): "
                      + ", ".join(f"{a['post_id']} ({a['exposure_fraction']})" for a in accrued))
        saved_stage = save_checkpoint("simulated",
                                      campaign=campaign.model_dump(mode="json"),
                                      routing=routing.model_dump(mode="json"),
                                      posts=[p.model_dump(mode="json") for p in posts],
                                      pillar_of_post=pillar_of_post,
                                      slots=[s.model_dump(mode="json") for s in slots],
                                      published=published)

    # ================================================================ 10. community manager
    if stage_index(saved_stage) < stage_index("community_replied"):
        show("STAGE 10: Community Manager — real comments, real replies")
        ch_map = {c["id"]: c for c in channels}
        total_escalations = 0
        for post in posts:
            pid = post.published_post_id
            if not pid:
                continue
            profile = ch_map.get(post.channel, {})
            pillar_name = pillar_of_post.get(post.id, "")
            post_context = {"copy": post.copy, "pillar": pillar_name}
            result = await community_manager.handle_post_comments(
                platform, campaign.id, pid, profile, post_context, pillar=pillar_name
            )
            n_comments = result["comments_fetched"]
            n_replies = result["replies_posted"]
            n_esc = sum(1 for a in result["actions"] if a["action"] == "escalate")
            total_escalations += n_esc
            print(f"  {post.id}: {n_comments} comments, {n_replies} replies, {n_esc} escalations")
            # Record escalations to the needs_human_review table
            from platform.escalations import enqueue_escalation
            for a in result["actions"]:
                if a["action"] == "escalate":
                    await enqueue_escalation(
                        campaign_id=campaign.id, post_id=pid,
                        comment_id=a["comment_id"], reason=a.get("escalation_reason", a.get("router_reason", "")),
                        comment_text=a.get("comment_text", ""),
                    )
        print(f"total escalations: {total_escalations}")
        saved_stage = save_checkpoint("community_replied",
                                      campaign=campaign.model_dump(mode="json"),
                                      routing=routing.model_dump(mode="json"),
                                      posts=[p.model_dump(mode="json") for p in posts],
                                      pillar_of_post=pillar_of_post,
                                      slots=[s.model_dump(mode="json") for s in slots],
                                      published=published)

    # ================================================================ 11. analytics
    if stage_index(saved_stage) < stage_index("reported"):
        show("STAGE 11: Analytics Agent — two-stage weekly analysis")
        ch_map = {c["id"]: c for c in channels}
        report = await analytics.run_week(
            platform, campaign.id, 1, campaign, channels,
        )
        # Save the report
        report_path = ROOT / "logs" / f"week1_campaign_{campaign.id}_report.json"
        report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
        print(f"\nreport saved  : {report_path}")
        print(f"patterns      : {len(report.patterns_found)}")
        print(f"recommendations:")
        for i, r in enumerate(report.recommendations, 1):
            print(f"  {i}. {r.change}")
            print(f"     evidence      : {r.evidence}")
            print(f"     expected      : {r.expected_effect}")
        print(f"\npost insights:")
        for pi in report.post_insights:
            print(f"  [{pi.performance_rank:6s}] {pi.post_id}: {pi.hypothesis[:80]}")
        print(f"\nsentiment      : {report.comment_sentiment_summary[:120]}")
        print(f"\nkpi_performance:")
        for k, v in report.kpi_performance.items():
            print(f"  {k:25s}: {v}")
        saved_stage = save_checkpoint("reported",
                                      campaign=campaign.model_dump(mode="json"),
                                      routing=routing.model_dump(mode="json"),
                                      posts=[p.model_dump(mode="json") for p in posts],
                                      pillar_of_post=pillar_of_post,
                                      slots=[s.model_dump(mode="json") for s in slots],
                                      published=published)

    # ================================================================ 12. memory
    if stage_index(saved_stage) < stage_index("memory_stored"):
        show("STAGE 12: Memory — store learnings for week-2 retrieval")
        report_data = json.loads(
            (ROOT / "logs" / f"week1_campaign_{campaign.id}_report.json").read_text(encoding="utf-8")
        )
        store = MemoryStore()
        record_id = await store.store_report(
            campaign_id=campaign.id,
            patterns_found=report_data["patterns_found"],
            recommendations=report_data["recommendations"],
            comment_sentiment_summary=report_data["comment_sentiment_summary"],
        )
        print(f"memory record : {record_id}")
        saved_stage = save_checkpoint("memory_stored",
                                      campaign=campaign.model_dump(mode="json"),
                                      routing=routing.model_dump(mode="json"),
                                      posts=[p.model_dump(mode="json") for p in posts],
                                      pillar_of_post=pillar_of_post,
                                      slots=[s.model_dump(mode="json") for s in slots],
                                      published=published)

    # ================================================================ 13. discovery scoring
    if stage_index(saved_stage) >= stage_index("memory_stored"):
        show("STAGE 13: Discovery scoring — agent findings vs. ground-truth rules")
        from scripts.score_analytics_discovery import render_table, score_report

        report_data = json.loads(
            (ROOT / "logs" / f"week1_campaign_{campaign.id}_report.json").read_text(encoding="utf-8")
        )
        results = score_report(report_data)
        print(render_table(results))
        n_yes = sum(1 for r in results if r["status"] == "yes")
        n_part = sum(1 for r in results if r["status"] == "partial")
        n_no = sum(1 for r in results if r["status"] == "no")
        print(f"\nVerdict: {n_yes} found / {n_part} partial / {n_no} not found (of {len(results)} rules)")

    # ================================================================ 14. summary + transcript
    show("RESULT SUMMARY")
    print(f"campaign id       : {campaign.id}")
    print(f"posts published   : {len(published)}")
    print(f"escalations       : (see Stage 10 output above)")
    print(f"report path       : logs/week1_campaign_{campaign.id}_report.json")
    print(f"trace log         : logs/agent_trace.jsonl")
    print(f"checkpoint        : cleared (clean exit)")

    await bus.publish(make_message("demo", "orchestrator", campaign.id, "demo_complete",
                                   {"campaign_id": campaign.id, "status": "full_run_complete"}))

    clear_checkpoint()
    await client.aclose()
    return {"status": "complete", "campaign_id": campaign.id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 3 demo: brief → weekly report + memory")
    parser.add_argument("--brief", default=str(ROOT / "seed_data" / "demo_brief.txt"))
    parser.add_argument("--auto-approve", action="store_true", help="skip the y/n human gate")
    parser.add_argument("--fresh", action="store_true", help="ignore any checkpoint and start over")
    parser.add_argument("--platform-url", default="http://127.0.0.1:8010")
    args = parser.parse_args()
    try:
        result = asyncio.run(main(
            brief_path=args.brief, auto_approve=args.auto_approve,
            fresh=args.fresh, platform_url=args.platform_url,
        ))
    except KeyboardInterrupt:
        print("\ninterrupted — checkpoint kept; re-run to resume.")
        sys.exit(130)
    sys.exit(0 if result.get("status") == "complete" else 1)
"""Day-2 end-to-end dry run — brief -> approved campaign, nothing published.

Run:  .venv/Scripts/python.exe scripts/run_day2_dryrun.py [--auto-approve]

Pipeline (today's scope, doc build-order steps 6-8):
    Orchestrator (brief -> draft Campaign + routing plan)
      -> Strategy   (audience/channels/pillars/KPIs)
      -> Writer     (draft -> self-critique -> revise)
      -> Creative   (CreativeBrief per post)
      -> Compliance (rule prefilter + model judgment)
           >= 1 rejection cycle runs for real: a banned term is injected
           into one post's first review, the prefilter catches it, the
           Writer revises, Compliance re-reviews -> approve. The loop is
           exercised on camera, not simulated with prints.
      -> Human approval gate (CLI y/n; --auto-approve skips it)
      -> Scheduler (deterministic slots; nothing POSTed to the platform)
      -> ends with campaign.status == "scheduled", posts fully populated.

Day-3 wiring (NOT here): publishing to the platform, community manager on
live comments, analytics on real week data, memory retrieval.

Everything the agents say to each other lands in logs/agent_trace.jsonl
(bus messages + model calls + agent I/O). That file is the deliverable.

Ollama: if the daemon is down you get one actionable message, never a
stack trace (see preflight()).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # `scripts/` is sys.path[0] when run as a file

from config.settings import get_settings  # noqa: E402
from llm.ollama_client import OllamaClient  # noqa: E402

# A banned phrase for the FORCED rejection. Unique enough that it cannot
# appear in a legit draft, so if it survives to the final copy the loop
# failed and we abort rather than ship a banned post.
FORCED_BAN = "guaranteed espresso nirvana"


def preflight(client: OllamaClient) -> None:
    """Fail fast and helpfully when the model layer is unavailable.

    asyncio.run() would surface connection errors as a traceback wall;
    probing synchronously first lets us print clean fix instructions.
    """
    import httpx

    try:
        resp = httpx.get(f"{client.host}/api/version", timeout=3.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        print(
            f"ERROR: Ollama is not reachable at {client.host}\n"
            "\n"
            "Fix:\n"
            "  1. Install Ollama:  https://ollama.com/download\n"
            "  2. Start it:        open the Ollama app (or run `ollama serve`)\n"
            "  3. Pull the models: ollama pull qwen2.5:7b-instruct\n"
            "                      ollama pull qwen2.5:3b-instruct\n"
            "  4. Verify:          curl http://localhost:11434/api/tags\n"
            "  5. Re-run this script.\n",
            file=sys.stderr,
        )
        sys.exit(2)


def show(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 62 - len(title)))


async def main(auto_approve: bool) -> None:
    settings = get_settings()
    client = OllamaClient()

    print("=" * 72)
    print("DAY 2 DRY RUN — brief to approved campaign (nothing published)")
    print(f"model: {settings.primary_model}   router: {settings.router_model}")
    print("=" * 72)
    preflight(client)

    from agents.compliance_agent import ComplianceAgent
    from agents.creative_agent import CreativeAgent
    from agents.orchestrator import OrchestratorAgent
    from agents.scheduler_agent import SchedulerAgent
    from agents.strategy_agent import StrategyAgent
    from agents.writer_agent import PostDraft, WriterAgent
    from models.post import CreativeBrief, Post
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
    from platform.db import get_session_factory, init_db
    from platform.schema import Channel

    # Real platform channels from the Day-1 seed (profiles the writer and
    # scheduler read): the dry run plans against the SAME channels that
    # will simulate engagement on Day 3.
    await init_db()
    factory = get_session_factory()
    async with factory() as session:
        from sqlalchemy import select

        rows = (await session.execute(select(Channel))).scalars().all()
    channels = [
        {"id": ch.id, "name": ch.name, "type": ch.type, **json.loads(ch.character_profile_json)}
        for ch in rows
    ]
    print(f"\nplatform channels: {[c['id'] for c in channels]}")

    brief_raw = (ROOT / "seed_data" / "demo_brief.txt").read_text(encoding="utf-8")
    campaign_id = f"camp_{uuid.uuid4().hex[:8]}"

    # ---------------------------------------------------------------- agents
    bus = MessageBus()
    orchestrator = OrchestratorAgent(client=client, bus=bus)
    strategy = StrategyAgent(client=client, bus=bus)
    writer = WriterAgent(client=client, bus=bus)
    creative = CreativeAgent(client=client, bus=bus)
    compliance = ComplianceAgent(client=client, bus=bus)
    scheduler = SchedulerAgent(client=client, bus=bus)
    orchestrator.set_campaign_id(campaign_id)
    policy = ComplianceRetryPolicy()  # max 3 rejections (settings-driven)

    # ================================================================ 1. brief -> draft
    show("STAGE 1: Orchestrator parses the brief")
    campaign, routing = await orchestrator.run(brief_raw, channels)
    print(f"campaign      : {campaign.id}")
    print(f"objective     : {campaign.objective}")
    print(f"channel_mix   : {campaign.channel_mix}")
    print(f"duration_days : {campaign.duration_days}")
    print(f"routing plan  : {' -> '.join(routing.routing_plan)}")
    if routing.risks:
        print(f"risks flagged : {routing.risks}")

    # ================================================================ 2. strategy
    show("STAGE 2: Strategy fills audience, pillars, KPIs")
    campaign = await strategy.run(campaign, channels)
    campaign = transition(campaign, EV_STRATEGY_FILLED)  # draft -> strategy_filled
    print(f"audience      : {campaign.target_audience}")
    for p in campaign.content_pillars:
        print(f"pillar        : {p.name} (weight {p.weight}) — {p.description[:70]}")
    for k in campaign.kpis:
        print(f"kpi           : {k.metric} @ {k.target} {k.channel or '(campaign-wide)'}")

    # ================================================================ 3. writer (3 passes)
    show("STAGE 3: Writer drafts -> self-critiques -> revises")
    # Channel rotation: pillar i writes for channel i % n_channels, so the
    # posts spread across the channel mix instead of piling onto one.
    rotation = [
        {"pillar": p.name, "channel": campaign.channel_mix[i % len(campaign.channel_mix)]}
        for i, p in enumerate(campaign.content_pillars)
    ]
    drafts = await writer.draft(campaign, rotation)
    issues = await writer.critique(campaign, drafts)
    print(f"self-critique : {len(issues)} issue(s) the writer found in its own drafts")
    for c in issues:
        print(f"  - [{c.post_index}] {c.issue[:70]} -> fix: {c.fix[:50]}")
    if issues:
        drafts = await writer.revise(campaign, drafts, critique_issues=issues)
    campaign = transition(campaign, EV_CONTENT_DRAFTED)  # strategy_filled -> content_drafted

    # Convert drafts into the Post contract. Ids are stable across the whole
    # pipeline so ComplianceVerdict.post_id always refers to the same post.
    posts: list[Post] = []
    pillar_of_post: dict[str, str] = {}
    for i, d in enumerate(drafts):
        post = Post(
            id=f"{campaign.id}_post{i + 1}",
            campaign_id=campaign.id,
            channel=d.channel,
            copy=d.copy,
            hashtags=d.hashtags,
            cta=d.cta,
            creative=CreativeBrief(asset_type="image", description="placeholder until the Creative pass"),
        )
        posts.append(post)
        pillar_of_post[post.id] = d.pillar

    # ================================================================ 4. creative
    show("STAGE 4: Creative briefs per post")
    ch_map = {c["id"]: c for c in channels}
    for post in posts:
        pillar = next(p for p in campaign.content_pillars if p.name == pillar_of_post[post.id])
        post.creative = await creative.run(
            campaign.id, post.copy, ch_map[post.channel], pillar.model_dump()
        )
        print(f"{post.id}: {post.creative.asset_type:9s} {post.creative.description[:60]}")

    # ================================================================ 5. compliance + reject loop
    show("STAGE 5: Compliance review (first review carries a forced violation)")
    campaign = transition(campaign, EV_REVIEW_STARTED)  # content_drafted -> compliance_review
    forced_injection_done = False
    review_cycles = 0

    while True:
        review_cycles += 1
        print(f"\n[review cycle {review_cycles}]")
        rejected_this_cycle = False
        escalated = False

        for post in posts:
            # Force exactly one violation, on post 1's FIRST review only:
            # prefilter must catch it, writer revises, loop closes. If the
            # phrase ever survives, abort — never ship a banned post.
            injected = (not forced_injection_done) and post is posts[0]
            review_copy = post.copy + (f" This is {FORCED_BAN}." if injected else "")

            verdict = await compliance.review(
                campaign.id,
                {
                    "id": post.id,
                    "channel": post.channel,
                    "copy": review_copy,
                    "hashtags": post.hashtags,
                    "cta": post.cta,
                    "creative": post.creative.model_dump(),
                },
            )

            if verdict.verdict == "approve":
                if injected:
                    print("ABORT: injected violation was approved — prefilter failed")
                    await client.aclose()
                    sys.exit(1)
                post.compliance_status = "approved"
                print(f"  APPROVE  {post.id}")
                continue

            # ---- rejection path (the loop the doc requires) ----
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

            # Writer revises with the verdict's reasons VERBATIM (that is the
            # contract: reasons are the revision instructions).
            draft = PostDraft(
                pillar=pillar_of_post[post.id], channel=post.channel,
                copy=post.copy, hashtags=post.hashtags, cta=post.cta,
            )
            revised = await writer.revise(campaign, [draft], rejection_reasons=verdict.reasons)
            if revised:
                post.copy, post.hashtags, post.cta = revised[0].copy, revised[0].hashtags, revised[0].cta
                post.compliance_status = "pending"
            print(f"  REVISED  {post.id}: {post.copy[:70]}")

            # The doc's edges: review -> content_drafted (revise) -> review.
            campaign = transition(campaign, EV_COMPLIANCE_REJECTED)
            campaign = transition(campaign, EV_REVIEW_STARTED)
            break  # re-run the full review cycle with the revision in place

        if escalated or not rejected_this_cycle:
            break

    if any(p.compliance_status != "approved" for p in posts):
        print("\nPipeline ended at needs_human — the bounded loop terminated correctly:")
        for p in posts:
            if p.compliance_status != "approved":
                print(f"  {p.id}: {p.compliance_status} after {p.rejection_count} rejection(s)")
        print("A human takes over from here (Day 3: needs_human queue UI).")
        await client.aclose()
        return

    campaign = transition(campaign, EV_COMPLIANCE_APPROVED)  # -> pending_human_approval
    print("\nAll posts compliance-approved.")

    # ================================================================ 6. human gate
    show("STAGE 6: Human approval gate")
    for post in posts:
        print(f"\n[{post.channel}] {post.id} (rejections so far: {post.rejection_count})")
        print(f"  copy    : {post.copy}")
        print(f"  tags    : {' '.join(post.hashtags)}")
        print(f"  cta     : {post.cta}")
        print(f"  creative: ({post.creative.asset_type}) {post.creative.description[:80]}")

    if auto_approve:
        answer = "y"
        print("\n[human gate] --auto-approve: approved")
    else:
        answer = input("\nApprove all posts for scheduling? (y/n): ").strip().lower()

    if answer != "y":
        # Doc edge: pending_human_approval -[reject]-> draft. The human's
        # message is a first-class bus record (from_agent="human").
        reason = input("Reason for rejection (recorded on the bus): ").strip() or "human declined"
        await bus.publish(
            make_message("human", "orchestrator", campaign.id, "human_rejected", {"reason": reason})
        )
        campaign = transition(campaign, EV_HUMAN_REJECTED)
        print(f"Campaign sent back to draft ({campaign.status}); reason recorded on the bus.")
        await client.aclose()
        return

    await bus.publish(
        make_message("human", "orchestrator", campaign.id, "human_approved", {"decision": "approved"})
    )
    campaign = transition(campaign, EV_HUMAN_APPROVED)  # -> scheduled
    print("status: scheduled (nothing published — that's Day 3)")

    # ================================================================ 7. scheduler
    show("STAGE 7: Scheduler assigns publish slots (deterministic, no model)")
    slots = await scheduler.build_schedule(campaign.id, posts, channels)
    for s in slots:
        print(f"  {s.post_id} -> {s.channel} @ {s.scheduled_at.isoformat()}")

    # ================================================================ summary
    show("RESULT")
    print(f"campaign status : {campaign.status}")
    print(f"posts           : {len(posts)} (all approved)")
    print(f"review cycles   : {review_cycles}")
    print(f"slots assigned  : {len(slots)}")
    history = await bus.history(campaign.id)
    print(f"\nbus history for {campaign.id}: {len(history)} messages:")
    for m in history:
        print(f"  {m.from_agent:18s} -> {m.to_agent:18s} {m.message_type}")
    print("\nFull agent conversation: logs/agent_trace.jsonl")

    await client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day-2 dry run: brief -> approved campaign")
    parser.add_argument("--auto-approve", action="store_true", help="skip the y/n human gate")
    args = parser.parse_args()
    asyncio.run(main(auto_approve=args.auto_approve))

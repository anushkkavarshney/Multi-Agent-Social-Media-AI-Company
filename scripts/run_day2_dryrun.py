"""Day-2 end-to-end dry run — brief -> approved campaign, nothing published.

Run:  python scripts/run_day2_dryrun.py [--auto-approve] [--fresh]

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

CHECKPOINT/RESUME (added after the 2026-09-15 crash): on CPU, one run is
30-60 minutes of model calls, and any crash previously lost ALL of it.
Every stage boundary (and every post review inside the compliance loop)
persists full pipeline state to logs/day2_checkpoint.json. Re-running the
script resumes from the last boundary; --fresh starts over. A clean exit
(approve, human-reject, or escalate) clears the checkpoint.

Failure contract: a ModelOutputFailure prints a one-screen digest (which
agent/schema/attempts + the model's last raw output) and exits 3 with the
checkpoint KEPT, so the retry after a fix costs one stage, not the run.
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

# Windows consoles default to legacy codepages; post copy legitimately
# contains emoji. Replace-on-encode guarantees the report always prints
# (mojibake beats a UnicodeEncodeError crash at the finish line).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config.settings import get_settings  # noqa: E402
from llm.ollama_client import OllamaClient  # noqa: E402
from llm.structured_output import ModelOutputFailure  # noqa: E402

# A banned phrase for the FORCED rejection. Unique enough that it cannot
# appear in a legit draft, so if it survives to the final copy the loop
# failed and we abort rather than ship a banned post.
FORCED_BAN = "guaranteed espresso nirvana"

CHECKPOINT_PATH = ROOT / "logs" / "day2_checkpoint.json"
CHECKPOINT_VERSION = 3  # bump when the checkpoint payload shape changes


# ------------------------------------------------------------------ checkpoint

def save_checkpoint(stage: str, **data) -> None:
    """Persist full pipeline state at a stage boundary (atomic-ish write)."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"v": CHECKPOINT_VERSION, "stage": stage, **data}
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(CHECKPOINT_PATH)  # os.replace is atomic on Windows+POSIX


def load_checkpoint() -> dict | None:
    """Return the saved checkpoint, or None (missing / stale shape / corrupt)."""
    if not CHECKPOINT_PATH.exists():
        return None
    try:
        data = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("v") != CHECKPOINT_VERSION:
        return None
    return data


def clear_checkpoint() -> None:
    CHECKPOINT_PATH.unlink(missing_ok=True)


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


def report_model_failure(exc: ModelOutputFailure, agent: str, schema: str) -> None:
    """One-screen digest of a structured-output failure. No stack trace.

    The checkpoint is deliberately KEPT: the caller re-runs this script to
    resume from the failing stage after whatever fix is needed.
    """
    print(
        f"\nERROR [{agent}] could not produce valid {schema} "
        f"({exc.attempts} attempt(s)).\n"
        f"  last error : {exc.last_error[:400]}\n"
        f"  last output: {exc.last_raw[:400]}\n"
        "\nThe pipeline state is checkpointed — fix the issue and re-run this\n"
        "script to resume from this stage (use --fresh to start over).",
        file=sys.stderr,
    )


async def main(auto_approve: bool, fresh: bool) -> None:
    settings = get_settings()
    client = OllamaClient()

    print("=" * 72)
    print("DAY 2 DRY RUN — brief to approved campaign (nothing published)")
    print(f"model: {settings.primary_model}   router: {settings.router_model}")
    print("=" * 72)
    preflight(client)

    from agents.compliance_agent import ComplianceAgent
    from agents.creative_agent import CreativeAgent
    from agents.orchestrator import BriefDraft, OrchestratorAgent
    from agents.scheduler_agent import SchedulerAgent
    from agents.strategy_agent import StrategyAgent
    from agents.writer_agent import PostDraft, WriterAgent
    from models.campaign import Campaign
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

    if fresh:
        clear_checkpoint()
        print("checkpoint: cleared (--fresh)")
    else:
        ckpt = load_checkpoint()
        if ckpt:
            print(f"checkpoint: found stage '{ckpt['stage']}' — resuming "
                  f"(use --fresh to start over)")

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

    # ---------------------------------------------------------------- agents
    bus = MessageBus()
    orchestrator = OrchestratorAgent(client=client, bus=bus)
    strategy = StrategyAgent(client=client, bus=bus)
    writer = WriterAgent(client=client, bus=bus)
    creative = CreativeAgent(client=client, bus=bus)
    compliance = ComplianceAgent(client=client, bus=bus)
    scheduler = SchedulerAgent(client=client, bus=bus)
    policy = ComplianceRetryPolicy()  # max 3 rejections (settings-driven)

    ckpt = None if fresh else load_checkpoint()
    resumed_stage = ckpt["stage"] if ckpt else None

    # `posts` / `pillar_of_post` default to empty on a FRESH run; the resume
    # branch below overrides them from the checkpoint. (Bug fixed 2026-09-15:
    # a fresh run previously crashed with UnboundLocalError at the
    # draft->Post conversion because `posts` was only defined in the resume
    # branch.)
    posts: list[Post] = []
    pillar_of_post: dict[str, str] = {}
    review_cycles = 0
    forced_injection_done = False

    # ================================================================ 1. brief -> draft
    if resumed_stage:
        # Everything below rebuilds from the checkpoint (single resume block
        # per stage keeps the happy-path code linear and readable).
        campaign = Campaign.model_validate(ckpt["campaign"])
        routing = BriefDraft.model_validate(ckpt["routing"]) if "routing" in ckpt else None
        drafts = [PostDraft.model_validate(d) for d in ckpt.get("drafts", [])]
        posts = [Post.model_validate(p) for p in ckpt.get("posts", [])]
        pillar_of_post: dict[str, str] = ckpt.get("pillar_of_post", {})
        review_cycles = ckpt.get("review_cycles", 0)
        forced_injection_done = ckpt.get("forced_injection_done", False)
        campaign_id = campaign.id
        orchestrator.set_campaign_id(campaign_id)
        print(f"resumed campaign: {campaign_id} at stage '{resumed_stage}'")
    else:
        campaign_id = f"camp_{uuid.uuid4().hex[:8]}"
        orchestrator.set_campaign_id(campaign_id)

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
        save_checkpoint(
            "orchestrated",
            campaign=campaign.model_dump(mode="json"),
            routing=routing.model_dump(mode="json"),
        )

    # ================================================================ 2. strategy
    if resumed_stage in (None, "orchestrated"):
        show("STAGE 2: Strategy fills audience, pillars, KPIs")
        try:
            campaign = await strategy.run(campaign, channels)
        except ModelOutputFailure as exc:
            report_model_failure(exc, "strategy", "StrategyOutput")
            await client.aclose()
            sys.exit(3)
        campaign = transition(campaign, EV_STRATEGY_FILLED)  # draft -> strategy_filled
        print(f"audience      : {campaign.target_audience}")
        for p in campaign.content_pillars:
            print(f"pillar        : {p.name} (weight {p.weight}) — {p.description[:70]}")
        for k in campaign.kpis:
            print(f"kpi           : {k.metric} @ {k.target} {k.channel or '(campaign-wide)'}")
        save_checkpoint(
            "strategized",
            campaign=campaign.model_dump(mode="json"),
            routing=routing.model_dump(mode="json"),
        )

    # ================================================================ 3. writer (3 passes)
    if resumed_stage in (None, "orchestrated", "strategized"):
        show("STAGE 3: Writer drafts -> self-critiques -> revises")
        # Channel rotation: pillar i writes for channel i % n_channels, so the
        # posts spread across the channel mix instead of piling onto one.
        rotation = [
            {"pillar": p.name, "channel": campaign.channel_mix[i % len(campaign.channel_mix)]}
            for i, p in enumerate(campaign.content_pillars)
        ]
        try:
            drafts = await writer.draft(campaign, rotation)
            issues = await writer.critique(campaign, drafts)
            print(f"self-critique : {len(issues)} issue(s) the writer found in its own drafts")
            for c in issues:
                print(f"  - [{c.post_index}] {c.issue[:70]} -> fix: {c.fix[:50]}")
            if issues:
                drafts = await writer.revise(campaign, drafts, critique_issues=issues)
        except ModelOutputFailure as exc:
            report_model_failure(exc, "writer", "PostDraftList/CritiqueList")
            await client.aclose()
            sys.exit(3)
        campaign = transition(campaign, EV_CONTENT_DRAFTED)  # -> content_drafted
        save_checkpoint(
            "written",
            campaign=campaign.model_dump(mode="json"),
            routing=routing.model_dump(mode="json"),
            drafts=[d.model_dump(mode="json") for d in drafts],
        )

    # Convert drafts into the Post contract. Ids are stable across the whole
    # pipeline so ComplianceVerdict.post_id always refers to the same post.
    if not posts:
        posts = []
        pillar_of_post = {}
        for i, d in enumerate(drafts):
            post = Post(
                id=f"{campaign.id}_post{i + 1}",
                campaign_id=campaign.id,
                channel=d.channel,
                copy=d.copy,
                hashtags=d.hashtags,
                cta=d.cta,
                creative=CreativeBrief(
                    asset_type="image", description="placeholder until the Creative pass"
                ),
            )
            posts.append(post)
            pillar_of_post[post.id] = d.pillar

    # ================================================================ 4. creative
    if resumed_stage in (None, "orchestrated", "strategized", "written"):
        show("STAGE 4: Creative briefs per post")
        ch_map = {c["id"]: c for c in channels}
        for post in posts:
            pillar = next(p for p in campaign.content_pillars if p.name == pillar_of_post[post.id])
            try:
                post.creative = await creative.run(
                    campaign.id, post.copy, ch_map[post.channel], pillar.model_dump()
                )
            except ModelOutputFailure as exc:
                report_model_failure(exc, "creative", "CreativeChoice")
                await client.aclose()
                sys.exit(3)
            print(f"{post.id}: {post.creative.asset_type:9s} {post.creative.description[:60]}")
        save_checkpoint(
            "posts_created",
            campaign=campaign.model_dump(mode="json"),
            routing=routing.model_dump(mode="json"),
            posts=[p.model_dump(mode="json") for p in posts],
            pillar_of_post=pillar_of_post,
        )

    # ================================================================ 5. compliance + reject loop
    if resumed_stage in (None, "orchestrated", "strategized", "written", "posts_created", "review"):
        show("STAGE 5: Compliance review (first review carries a forced violation)")
        # Resume note: a "review" checkpoint is taken while the campaign is
        # ALREADY in compliance_review — re-firing review_started there would
        # be an invalid transition, so only enter from content_drafted.
        if campaign.status == "content_drafted":
            campaign = transition(campaign, EV_REVIEW_STARTED)  # -> compliance_review

        while True:
            review_cycles += 1
            print(f"\n[review cycle {review_cycles}]")
            rejected_this_cycle = False
            escalated = False

            for post in posts:
                if post.compliance_status == "approved":
                    continue  # resume-safe: approved posts are never re-reviewed

                # Force exactly one violation, on post 1's FIRST review only:
                # prefilter must catch it, writer revises, loop closes. If the
                # phrase ever survives, abort — never ship a banned post.
                injected = (not forced_injection_done) and post is posts[0]
                review_copy = post.copy + (f" This is {FORCED_BAN}." if injected else "")

                try:
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
                except ModelOutputFailure as exc:
                    report_model_failure(exc, "compliance", "ComplianceVerdict")
                    # Checkpoint BEFORE exiting: this post's earlier approvals
                    # in this cycle are already persisted below on approve.
                    await client.aclose()
                    sys.exit(3)

                if verdict.verdict == "approve":
                    if injected:
                        print("ABORT: injected violation was approved — prefilter failed")
                        await client.aclose()
                        sys.exit(1)
                    post.compliance_status = "approved"
                    print(f"  APPROVE  {post.id}")
                    save_checkpoint(
                        "review",
                        campaign=campaign.model_dump(mode="json"),
                        routing=routing.model_dump(mode="json"),
                        posts=[p.model_dump(mode="json") for p in posts],
                        pillar_of_post=pillar_of_post,
                        review_cycles=review_cycles,
                        forced_injection_done=forced_injection_done,
                    )
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
                try:
                    revised = await writer.revise(
                        campaign, [draft], rejection_reasons=verdict.reasons
                    )
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
                save_checkpoint(
                    "review",
                    campaign=campaign.model_dump(mode="json"),
                    routing=routing.model_dump(mode="json"),
                    posts=[p.model_dump(mode="json") for p in posts],
                    pillar_of_post=pillar_of_post,
                    review_cycles=review_cycles,
                    forced_injection_done=forced_injection_done,
                )

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
            clear_checkpoint()  # clean exit — no resume past a terminated loop
            await client.aclose()
            return
        campaign = transition(campaign, EV_COMPLIANCE_APPROVED)  # -> pending_human_approval
        print("\nAll posts compliance-approved.")
        save_checkpoint(
            "compliance_passed",
            campaign=campaign.model_dump(mode="json"),
            routing=routing.model_dump(mode="json"),
            posts=[p.model_dump(mode="json") for p in posts],
            pillar_of_post=pillar_of_post,
            review_cycles=review_cycles,
        )
    else:
        # resumed at compliance_passed or later: state carries approved posts
        campaign = transition(campaign, EV_COMPLIANCE_APPROVED) if campaign.status == "compliance_review" else campaign
        print("\nAll posts compliance-approved. (resumed)")

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
        clear_checkpoint()  # clean exit
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

    clear_checkpoint()  # success — nothing to resume
    await client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day-2 dry run: brief -> approved campaign")
    parser.add_argument("--auto-approve", action="store_true", help="skip the y/n human gate")
    parser.add_argument("--fresh", action="store_true", help="ignore any checkpoint and start over")
    args = parser.parse_args()
    try:
        asyncio.run(main(auto_approve=args.auto_approve, fresh=args.fresh))
    except KeyboardInterrupt:
        print("\ninterrupted — checkpoint kept; re-run this script to resume.")
        sys.exit(130)

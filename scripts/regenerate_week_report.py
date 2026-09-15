"""Regenerate Week N's WeeklyReport under the Day-4 auditable schema.

Usage:
    python scripts/regenerate_week_report.py CAMP_ID [--week N] [--platform-url URL]

Rebuilds the campaign from the bus history's ``campaign_strategy_filled``
payload (so KPIs/channels/pillars match the live run), fetches the week's
posts + comments from the running platform, and re-runs the two-stage
analytics pipeline (compute in Python -> narrate over precomputed stats).

The regenerated report overwrites ``logs/week{N}_campaign_{id}_report.json``.
Print the before/after diff yourself: the original report is left as
``..._report.before_day4.json`` next to it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.analytics_agent import AnalyticsAgent
from llm.ollama_client import OllamaClient
from models.campaign import Campaign
from orchestration.message_bus import MessageBus
from platform.client import PlatformClient
from platform.db import get_session_factory


async def find_campaign_payload(campaign_id: str) -> dict:
    bus = MessageBus(session_factory=get_session_factory())
    hist = await bus.history(campaign_id)
    for m in hist:
        if getattr(m, "message_type", None) == "campaign_strategy_filled":
            campaign = m.payload.get("campaign", {})
            if campaign.get("id") == campaign_id:
                return campaign
    raise SystemExit(f"no campaign_strategy_filled for {campaign_id} in bus history")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("campaign_id")
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--platform-url", default="http://127.0.0.1:8010")
    args = ap.parse_args()

    campaign_data = await find_campaign_payload(args.campaign_id)
    campaign = Campaign.model_validate(campaign_data)

    platform = PlatformClient(base_url=args.platform_url)
    health = await platform.health()
    print(f"platform     : {health}")
    channels = await platform.get_channels()

    bus = MessageBus(session_factory=get_session_factory())
    analytics = AnalyticsAgent(client=OllamaClient(timeout=2400.0), bus=bus)

    report = await analytics.run_week(
        platform, campaign.id, args.week, campaign, channels
    )

    out = ROOT / "logs" / f"week{args.week}_campaign_{campaign.id}_report.json"
    before = out.with_name(out.stem + ".before_day4.json")
    if out.exists() and not before.exists():
        shutil.copy2(out, before)
        print(f"backup       : {before}")

    out.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(f"regenerated  : {out}")
    print(f"patterns     : {len(report.patterns_found)}")
    print("recommendations:")
    for i, r in enumerate(report.recommendations, 1):
        print(f"  {i}. {r.change}")
        print(f"     statistic     : {r.underlying_statistic}")
        print(f"     confidence    : {r.confidence}")
        print(f"     predicted     : {r.predicted_effect}")


if __name__ == "__main__":
    asyncio.run(main())
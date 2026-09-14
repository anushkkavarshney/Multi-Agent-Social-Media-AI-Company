"""Orchestrator Agent — brief in, draft Campaign + routing out.

Role (doc section 7): decompose the human's raw brief into a draft Campaign,
decide the task routing plan, and drive the state machine forward. The
Orchestrator is the ONLY agent that owns Campaign.status transitions (single
writer principle — agents never mutate status, they emit events the
orchestrator applies via transition()).

Day-2 scope: brief -> draft campaign (status=draft) + routing plan. It does
NOT run the pipeline itself in this module — scripts/run_day2_dryrun.py
composes agents explicitly so each handoff is visible on the bus.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from models.campaign import Campaign


class BriefDraft(BaseModel):
    """Orchestrator's structured decomposition of the raw brief.

    Deliberately NOT a Campaign: the Orchestrator has no authority over
    strategy fields (audience/pillars/KPIs are the Strategy Agent's output —
    models/campaign.py enforces that at the validator level). Fields here are
    only what the BRIEF itself supports.
    """

    objective: str = Field(..., min_length=1, description="Campaign goal, <= 15 words, brief's own framing.")
    target_audience_hint: str = Field(..., min_length=1)
    duration_days: int = Field(..., ge=1, le=90)
    channel_mix: list[str] = Field(..., min_length=1)
    content_focus: list[str] = Field(..., min_length=1, description="Concrete themes; feeds Strategy's pillar draft.")
    risks: list[str] = Field(default_factory=list)
    routing_plan: list[str] = Field(..., min_length=1, description="Ordered agent handoffs, e.g. [strategy, writer, creative, compliance, human_gate].")


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"

    async def run(self, brief_raw: str, available_channels: list[dict], *, transport=None) -> tuple[Campaign, BriefDraft]:
        """Parse `brief_raw` into a draft Campaign + routing plan.

        Emits: brief_received (on run start), campaign_drafted (with the
        routing plan) on success. The routing plan is data, not code: the
        runner (dry-run script) reads it to decide which agents to invoke —
        the doc's "function-call-like JSON dispatch" without dynamic imports.
        """
        # One id for the whole run: the runner sets it BEFORE run() so the
        # pre-campaign brief_received message groups with everything else.
        # (The Campaign object gets the same id back from the model output.)
        cid = self._campaign_id
        self.log_io(
            "input",
            stage_detail="brief_received",
            brief_chars=len(brief_raw),
            brief_preview=brief_raw[:120],
        )
        await self.emit(
            to_agent="strategy",
            campaign_id=cid,
            message_type="brief_received",
            payload={"brief_preview": brief_raw[:200]},
        )

        prompt = self.render_prompt(
            "orchestrator.jinja",
            brief_raw=brief_raw,
            available_channels=available_channels,
        )
        draft = await self.call_model(
            instruction=prompt,
            schema_cls=BriefDraft,
            transport=transport,
            campaign_id=cid,
        )

        campaign = Campaign(
            id=cid,
            brief_raw=brief_raw,
            objective=draft.objective,
            target_audience=draft.target_audience_hint,  # Strategy refines
            channel_mix=draft.channel_mix,
            duration_days=draft.duration_days,
            content_pillars=[],  # Strategy fills (contract allows empty only in draft)
            kpis=[],  # Strategy fills
            status="draft",
        )
        self.log_io(
            "output",
            stage_detail="campaign_drafted",
            campaign_id=campaign.id,
            objective=campaign.objective,
            channel_mix=campaign.channel_mix,
            duration_days=campaign.duration_days,
            risks=draft.risks,
        )
        await self.emit(
            to_agent="strategy",
            campaign_id=campaign.id,
            message_type="campaign_drafted",
            payload={
                "campaign": campaign.model_dump(mode="json"),
                "routing_plan": draft.routing_plan,
                "content_focus": draft.content_focus,
            },
        )
        return campaign, draft

    # Id used for ALL messages of this run, set by the runner before run()
    # so trace grouping works from message 1. Default only for unit tests.
    _campaign_id: str = "camp_pending"

    def set_campaign_id(self, campaign_id: str) -> None:
        self._campaign_id = campaign_id

"""Strategy Agent — turns the draft Campaign into a measurable plan.

Fills: target_audience, channel_mix, duration_days, content_pillars,
kpis (doc section 7). Emits campaign_strategy_filled with the full Campaign
so the runner can advance the state machine (draft -> strategy_filled).

MEMORY (doc section 7: "RAG-lite: retrieve top-k past campaign insights"):
Day-2 scope is a deliberate stub — retrieve_learnings() returns [] and the
prompt template renders the "none available" branch. Day 3 wires
memory/campaign_memory.py in; the stub exists so the call site and the
data contract are already correct (one-line change later, no restructuring).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from models.campaign import Campaign


class PillarOut(BaseModel):
    """Strategy's pillar proposal (pre-validation shape)."""

    name: str
    description: str
    weight: float = Field(..., gt=0.0, le=1.0)


class KPIOut(BaseModel):
    """Strategy's KPI proposal (pre-validation shape)."""

    metric: str
    target: float
    channel: str | None = None


class StrategyOutput(BaseModel):
    """Strategy's structured plan, validated against the campaign contract.

    Field constraints mirror models/campaign.py on purpose: invalid plans
    are rejected HERE (at the agent boundary) with the model's own output in
    the retry loop, instead of failing at Campaign construction with an
    error the model never got a chance to fix.
    """

    target_audience: str = Field(..., min_length=1)
    channel_mix: list[str] = Field(..., min_length=1)
    duration_days: int = Field(..., ge=1, le=90)
    pillars: list[PillarOut] = Field(..., min_length=3, max_length=3)
    kpis: list[KPIOut] = Field(..., min_length=2)


class StrategyAgent(BaseAgent):
    name = "strategy"

    # --- memory stub (Day 3 wires memory/campaign_memory.py here) -----------
    async def retrieve_learnings(self, objective: str, channels: list[str]) -> list[str]:
        """Return top-k past-campaign learnings relevant to this plan.

        Day-2 stub: memory/ is a Day-3 deliverable (doc build order step 9).
        Signature is final — replace the body with a vector-store query and
        nothing else in this file changes.
        """
        # TODO(Day 3): memory.campaign_memory.retrieve_top_k(objective, channels, k=3)
        return []

    async def run(
        self, campaign: Campaign, channel_profiles: list[dict], *, transport=None
    ) -> Campaign:
        """Produce the full strategy; returns a NEW strategy_filled Campaign.

        channel_profiles are the REAL platform channel rows (from
        platform/db.py seeds) — strategy plans against actual audiences.
        """
        self.log_io(
            "input",
            campaign_id=campaign.id,
            objective=campaign.objective,
            channels=channel_profiles and [c["id"] for c in channel_profiles],
        )

        learnings = await self.retrieve_learnings(campaign.objective, campaign.channel_mix)

        prompt = self.render_prompt(
            "strategy.jinja",
            campaign=campaign.model_dump_json(indent=2),
            channel_profiles=channel_profiles,
            memory_learnings=learnings,
        )
        plan = await self.call_model(
            instruction=prompt,
            schema_cls=StrategyOutput,
            transport=transport,
            campaign_id=campaign.id,
        )

        # Rebuild the Campaign through the Pydantic contract: weight-sum and
        # KPI-shape rules are re-validated HERE, not trusted from the model.
        # Dict-overlay + model_validate (NOT model_copy(update=...)): model_copy
        # bypasses validation, which left content_pillars/kpis as raw dicts and
        # produced pydantic serializer warnings on every later dump. Going
        # through model_validate constructs real ContentPillar/KPISet objects
        # and re-runs the strategy-fields-required validator — the plan is
        # structurally complete or this raises (never half-filled status).
        data = campaign.model_dump()
        data.update(
            {
                "target_audience": plan.target_audience,
                "channel_mix": plan.channel_mix,
                "duration_days": plan.duration_days,
                "content_pillars": [
                    {"name": p.name, "description": p.description, "weight": p.weight}
                    for p in plan.pillars
                ],
                "kpis": [
                    {"metric": k.metric, "target": k.target, "channel": k.channel}
                    for k in plan.kpis
                ],
                # status NOT touched: transitions belong to the runner/orchestrator
            }
        )
        filled = type(campaign).model_validate(data)

        self.log_io(
            "output",
            campaign_id=campaign.id,
            audience=filled.target_audience,
            pillars=[p.name for p in filled.content_pillars],
            kpi_count=len(filled.kpis),
            memory_used=len(learnings),
        )
        await self.emit(
            to_agent="writer",
            campaign_id=campaign.id,
            message_type="campaign_strategy_filled",
            payload={"campaign": filled.model_dump(mode="json")},
        )
        return filled

"""Campaign-level data contracts.

DATA FLOW (who produces -> who consumes):
    KPISet          produced by Strategy Agent       -> consumed by Campaign, Analytics Agent
    ContentPillar   produced by Strategy Agent       -> consumed by Campaign, Writer Agent
    Campaign        drafted by Orchestrator, filled by Strategy Agent
                                                        -> consumed by Writer/Creative/Scheduler
                                                          (and by Analytics Agent for week reports)

These are the objects that cross agent boundaries — every handoff is one of
these Pydantic models, never free text (architecture doc section 4).
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Valid campaign lifecycle states (state machine lives in orchestration/,
# Day 2 — this Literal is the contract the state machine must respect).
CampaignStatus = Literal["draft", "pending_human_approval", "approved", "running", "reviewed"]


class KPISet(BaseModel):
    """One measurable target for the campaign.

    Produced by the Strategy Agent when it fills a Campaign; consumed by the
    Analytics Agent as the yardstick for the weekly report ("performance vs
    KPIs"). `channel=None` means the target applies campaign-wide.
    """

    metric: str = Field(..., description='e.g. "click_through_rate", "impressions", "follower_growth"')
    target: float = Field(..., description="Numeric target the metric must reach.")
    channel: str | None = Field(None, description="Restrict KPI to one channel; None = campaign-wide.")

    @field_validator("metric")
    @classmethod
    def metric_not_blank(cls, v: str) -> str:
        # A blank metric name would make kpi_performance dict keys meaningless
        # in WeeklyReport; reject it at the boundary instead of at report time.
        if not v.strip():
            raise ValueError("metric must be a non-empty string")
        return v.strip()


class ContentPillar(BaseModel):
    """A recurring content theme with a share-of-voice weight.

    Produced by the Strategy Agent; consumed by the Content Writer Agent (which
    rotates post topics across pillars proportionally to `weight`).
    """

    name: str
    description: str
    # % of posts allocated to this pillar; weights across a campaign's pillars
    # must sum to 1.0 (validated on Campaign, where the full list is visible).
    weight: float = Field(..., gt=0.0, le=1.0)

    @field_validator("name", "description")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("pillar name/description must be non-empty")
        return v.strip()


class Campaign(BaseModel):
    """The root object a human brief becomes; everything else hangs off it.

    Produced by: Orchestrator (draft with brief_raw + objective),
    Strategy Agent (audience, channels, pillars, KPIs, duration).
    Consumed by: Writer, Creative, Scheduler, Analytics.
    """

    id: str = Field(..., min_length=1, description="Unique campaign id, e.g. 'camp_espresso_w1'.")
    brief_raw: str = Field(..., min_length=1, description="The original human brief, kept verbatim for traceability.")
    objective: str = Field(..., min_length=1, description="What this campaign must achieve, e.g. 'awareness'.")
    target_audience: str
    channel_mix: list[str] = Field(..., min_length=1, description="Channel ids/names the campaign runs on.")
    duration_days: int = Field(..., gt=0, description="Campaign length in days.")
    content_pillars: list[ContentPillar] = Field(..., min_length=1)
    kpis: list[KPISet] = Field(..., min_length=1)
    status: CampaignStatus = "draft"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("channel_mix")
    @classmethod
    def channels_non_empty_strings(cls, v: list[str]) -> list[str]:
        if not v or any(not c.strip() for c in v):
            raise ValueError("channel_mix must be a non-empty list of non-empty channel names")
        return [c.strip() for c in v]

    @model_validator(mode="after")
    def pillar_weights_sum_to_one(self) -> "Campaign":
        # Weights are share-of-voice: the Writer Agent uses them to allocate
        # posts, so a sum != 1.0 silently skews the calendar. Fail at the
        # boundary (floating point tolerated within 1e-6).
        total = sum(p.weight for p in self.content_pillars)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"content_pillar weights must sum to 1.0 across pillars (got {total:.6f}); "
                f"weights={[p.weight for p in self.content_pillars]}"
            )
        return self

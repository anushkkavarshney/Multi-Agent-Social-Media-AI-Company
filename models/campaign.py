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

# Campaign lifecycle states — the exact set from architecture doc section 8.
# MUST stay in sync with the transition table in orchestration/state_machine.py,
# which is the runtime source of truth; this Literal is the contract layer that
# validates Campaign.status against it. NOTE: expanded on Day 2 from the
# original 5-state draft to the full doc lifecycle (see SETUP_LOG.md).
CampaignStatus = Literal[
    "draft",                   # brief parsed; Strategy not yet run
    "strategy_filled",         # audience/channels/pillars/KPIs populated
    "content_drafted",         # Writer produced Post drafts
    "compliance_review",       # posts in Compliance review (incl. reject loops)
    "needs_human",             # compliance loop exhausted (3 rejections) or escalated
    "pending_human_approval",  # compliance-approved; awaiting the human gate
    "scheduled",               # human approved; Scheduler assigned publish slots
    "published",               # posts live on the mock platform
    "simulating",              # engagement accruing under the hidden rules
    "week_reviewed",           # WeeklyReport produced; recommendations feed week 2
]


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
    # Default empty (NOT min_length=1): the ORCHESTRATOR's draft Campaign
    # legitimately has no pillars/KPIs yet — the Strategy Agent fills them in
    # the strategy_filled step. Enforced below per-status instead of at the
    # field level, so a non-draft Campaign still cannot ship without them.
    # (Day 2 contract change from min_length=1; see SETUP_LOG.md.)
    content_pillars: list[ContentPillar] = Field(default_factory=list)
    kpis: list[KPISet] = Field(default_factory=list)
    status: CampaignStatus = "draft"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("channel_mix")
    @classmethod
    def channels_non_empty_strings(cls, v: list[str]) -> list[str]:
        if not v or any(not c.strip() for c in v):
            raise ValueError("channel_mix must be a non-empty list of non-empty channel names")
        return [c.strip() for c in v]

    @model_validator(mode="after")
    def strategy_fields_required_once_out_of_draft(self) -> "Campaign":
        # A `draft` Campaign is the Orchestrator's raw parse: pillars/KPIs are
        # the STRATEGY AGENT's output and arrive later. Any later state means
        # Strategy has run, so they must exist — this keeps the empty-list
        # allowance above from leaking into the filled stages.
        if self.status != "draft":
            missing = []
            if not self.content_pillars:
                missing.append("content_pillars")
            if not self.kpis:
                missing.append("kpis")
            if missing:
                raise ValueError(
                    f"a {self.status!r} campaign must have {missing} "
                    "(the Strategy Agent fills these before leaving draft)"
                )
        return self

    @model_validator(mode="after")
    def pillar_weights_sum_to_one(self) -> "Campaign":
        # Weights are share-of-voice: the Writer Agent uses them to allocate
        # posts, so a sum != 1.0 silently skews the calendar. Fail at the
        # boundary (floating point tolerated within 1e-6).
        # Empty pillars skip this check: a draft Campaign has none yet (the
        # Strategy Agent adds them), and an empty list must not masquerade as
        # a weighting error.
        if not self.content_pillars:
            return self
        total = sum(p.weight for p in self.content_pillars)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"content_pillar weights must sum to 1.0 across pillars (got {total:.6f}); "
                f"weights={[p.weight for p in self.content_pillars]}"
            )
        return self

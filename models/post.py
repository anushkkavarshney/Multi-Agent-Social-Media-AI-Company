"""Post-level data contracts.

DATA FLOW:
    CreativeBrief   produced by Creative Agent   -> consumed by Post (embedded), Compliance Agent
    Post            produced by Writer Agent (copy/hashtags/CTA) + Creative Agent (creative)
                                                        -> consumed by Compliance Agent (review),
                                                           Scheduler Agent (publish to mock platform),
                                                           and the platform simulation engine (engagement)
    ScheduleSlot    produced by Strategy/Scheduler planning -> consumed by Scheduler Agent

A Post's life: drafted -> compliance-reviewed -> human-approved -> published
(on the mock platform, which stamps `published_post_id`).
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

AssetType = Literal["image", "video", "carousel", "text_only"]
ComplianceStatus = Literal["pending", "approved", "rejected"]


class CreativeBrief(BaseModel):
    """What the visual/asset for a post should look like — text only, no real
    image generation (architecture doc section 4: "text brief, no real image
    gen required").

    Produced by the Creative Agent; consumed by humans at the approval gate and
    by the simulation engine via Post.creative.asset_type (asset_type drives
    the novelty-decay hidden rule: same format 3+ days running -> decay).
    """

    asset_type: AssetType
    description: str = Field(..., min_length=1, description="Concrete text brief, e.g. '15s vertical video, latte art timelapse'.")

    @field_validator("description")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("creative description must be non-empty")
        return v.strip()


class Post(BaseModel):
    """One piece of content on one channel.

    Produced by the Content Writer Agent (copy, hashtags, cta) and Creative
    Agent (creative brief); consumed by the Compliance Agent (verdict),
    the Scheduler Agent (publish), and the mock platform (engagement).
    """

    id: str = Field(..., min_length=1)
    campaign_id: str = Field(..., min_length=1)
    channel: str = Field(..., min_length=1, description="Channel name on the mock platform.")
    copy: str = Field(..., min_length=1, description="The post text itself.")
    # Non-empty enforced: hashtag count is a hidden-rule input (non-linear reach
    # effect), so "no hashtags" must be a deliberate list, not an accident.
    hashtags: list[str] = Field(..., min_length=1)
    cta: str | None = None
    creative: CreativeBrief
    scheduled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    compliance_status: ComplianceStatus = "pending"
    rejection_count: int = Field(default=0, ge=0)
    # Filled by the Scheduler Agent once the mock platform accepts the post;
    # None means "not live yet". The platform is the source of truth for the id.
    published_post_id: str | None = None

    @field_validator("hashtags")
    @classmethod
    def hashtags_normalized(cls, v: list[str]) -> list[str]:
        # Normalize to canonical "#tag" form so the simulation engine and the
        # Analytics Agent never see "espresso" vs "#espresso" as different tags.
        normalized = []
        for tag in v:
            if not tag.strip():
                raise ValueError("hashtags must not contain empty strings")
            tag = tag.strip().lstrip("#")
            normalized.append(f"#{tag}")
        return normalized

    @model_validator(mode="after")
    def rejection_count_consistent(self) -> "Post":
        # A rejected post must have been rejected at least once; conversely a
        # post with rejections on record cannot still be "pending".
        if self.compliance_status == "rejected" and self.rejection_count < 1:
            raise ValueError("compliance_status='rejected' requires rejection_count >= 1")
        if self.rejection_count > 0 and self.compliance_status == "pending":
            raise ValueError("rejection_count > 0 is inconsistent with compliance_status='pending'")
        return self


class ScheduleSlot(BaseModel):
    """A planned publish time for a post (repo layout lists it under models/post.py).

    Produced by planning (Strategy/Orchestrator); consumed by the Scheduler
    Agent when it decides when to call POST /posts on the mock platform.
    Kept separate from Post so timing can change without touching approved copy.
    """

    post_id: str
    channel: str
    scheduled_at: datetime

    @model_validator(mode="after")
    def not_in_far_past(self) -> "ScheduleSlot":
        # Guard against timezone bugs producing 1970/9999 dates; allow small
        # clock skew (1 day) so "just published" slots remain valid.
        now = datetime.now(timezone.utc)
        scheduled = self.scheduled_at if self.scheduled_at.tzinfo else self.scheduled_at.replace(tzinfo=timezone.utc)
        if scheduled < now.replace(year=now.year - 1) or scheduled > now.replace(year=now.year + 1):
            raise ValueError(f"scheduled_at {self.scheduled_at} is implausibly far from now ({now})")
        return self

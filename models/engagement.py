"""Engagement data contracts (platform -> agents direction).

DATA FLOW:
    EngagementSnapshot   produced by the mock platform simulation engine (one per post, per recording)
                             -> consumed by the Analytics Agent (weekly aggregation) and
                                the Scheduler/Orchestrator (live pulse checks).
    Comment              produced by the mock platform (simulated comment text)
                             -> consumed by the Community Manager Agent (draft replies,
                                escalation flags) and the Analytics Agent (sentiment themes).

NOTE: the platform's *storage* models (platform/schema.py ORM) are richer than
these Pydantic contracts — e.g. the ORM Comment row carries `author_handle` and
`replied` bookkeeping. These Pydantic models are the *agent-facing* shape only.
"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class EngagementSnapshot(BaseModel):
    """One metric recording for one post.

    Produced by the simulation engine at each simulate-tick; consumed by the
    Analytics Agent. Counts are cumulative since publish; `follower_delta` is
    the followers gained attributable to this post in the recorded window.
    """

    post_id: str
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    impressions: int = Field(..., ge=0)
    likes: int = Field(..., ge=0)
    comments: int = Field(..., ge=0)
    shares: int = Field(..., ge=0)
    saves: int = Field(..., ge=0)
    clicks: int = Field(..., ge=0)
    follower_delta: int = Field(..., ge=0, description="Followers gained from this post (>=0; unfollows ignored).")

    @model_validator(mode="after")
    def counts_within_impressions(self) -> "EngagementSnapshot":
        # Physical sanity: nobody can like/save/share/click more often than the
        # post was seen. (Comments are exempt — a viewer may comment multiple times.)
        if self.likes > self.impressions or self.shares > self.impressions or self.saves > self.impressions or self.clicks > self.impressions:
            raise ValueError(
                f"engagement counts exceed impressions={self.impressions} "
                f"(likes={self.likes}, shares={self.shares}, saves={self.saves}, clicks={self.clicks})"
            )
        return self


class Comment(BaseModel):
    """A simulated (or real, if replied) comment on a post.

    Produced by the platform's comment-text generator; consumed by the
    Community Manager Agent, which posts replies via the platform API and may
    escalate `is_sensitive=True` comments to the human.
    """

    id: str
    post_id: str
    author_handle: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    sentiment: Literal["positive", "neutral", "negative"]
    # Set by the platform's sensitivity pre-filter (price complaints, competitor
    # mentions, harassment...) so the Community Manager knows to escalate
    # instead of auto-replying.
    is_sensitive: bool = False
    replied: bool = False

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("comment text must be non-empty")
        return v.strip()

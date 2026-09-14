"""Weekly report data contracts.

DATA FLOW:
    PostInsight      produced by Analytics Agent  -> consumed by WeeklyReport.post_insights
    Recommendation   produced by Analytics Agent  -> consumed by WeeklyReport.recommendations,
                                                     fed back into the Strategy Agent for week 2
                                                     and persisted by the memory layer
    WeeklyReport     produced by Analytics Agent  -> consumed by Strategy Agent (week 2 planning),
                                                     the human approval gate, and memory/campaign_memory

The Analytics Agent is a two-stage pipeline (architecture doc section 7):
plain Python computes the stats; the LLM turns computed numbers into the
narrative strings here. It never invents numbers — every numeric claim lives
in `kpi_performance`, which is computed, not generated.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PostInsight(BaseModel):
    """Why a specific post over/under-performed, from the Analytics Agent."""

    post_id: str
    performance_rank: Literal["top", "bottom", "mid"]
    # Must be a *hypothesis* referencing a candidate cause (timing, length,
    # format, hashtags, CTA) — the write-up maps these onto the hidden rules.
    hypothesis: str = Field(..., min_length=1)

    @field_validator("hypothesis")
    @classmethod
    def hypothesis_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("hypothesis must be non-empty")
        return v.strip()


class Recommendation(BaseModel):
    """One concrete next-week change, with evidence and an expected effect.

    The project brief's quality bar for these strings is e.g.
    "shift Channel B posts from 09:00 to 18:30, cut copy under 80 words" —
    NOT "post more engaging content". `change` and `expected_effect` carry that
    specificity; `evidence` must cite the computed numbers that justify it.
    """

    change: str = Field(..., min_length=1)
    evidence: str = Field(..., min_length=1)
    expected_effect: str = Field(..., min_length=1)

    @field_validator("change", "evidence", "expected_effect")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("recommendation fields must be non-empty")
        return v.strip()


class WeeklyReport(BaseModel):
    """The full weekly analysis the Analytics Agent emits.

    Produced by the Analytics Agent at the end of each simulated week;
    consumed by the Strategy Agent (recommendations -> week 2 plan), the
    memory layer (patterns/recommendations embedded for future retrieval),
    and the human.
    """

    campaign_id: str = Field(..., min_length=1)
    week_number: int = Field(..., ge=1)
    # metric name -> measured value for the week (computed, never hallucinated).
    kpi_performance: dict[str, float]
    post_insights: list[PostInsight] = Field(..., min_length=1)
    patterns_found: list[str] = Field(..., min_length=1)
    comment_sentiment_summary: str = Field(..., min_length=1)
    recommendations: list[Recommendation] = Field(..., min_length=1)

    @field_validator("kpi_performance")
    @classmethod
    def kpi_values_finite(cls, v: dict[str, float]) -> dict[str, float]:
        # NaN/inf would silently poison week-2 comparisons; reject at the boundary.
        for k, val in v.items():
            if not val == val or val in (float("inf"), float("-inf")):
                raise ValueError(f"kpi_performance['{k}'] is not a finite number")
        return v

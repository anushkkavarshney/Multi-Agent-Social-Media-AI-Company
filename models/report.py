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

    Day 4 makes every recommendation AUDITABLE, mirroring how a credible
    analytics report states its scoring criteria before the score:
      - ``underlying_statistic`` names the exact groups and sample sizes the
        claim rests on (e.g. "mean impressions 15,542 peak vs 3,075 off-peak;
        2 posts vs 1 post"), so the reader can re-check the number.
      - ``confidence`` is a two-valued qualifier: ``observed_correlation``
        (both comparison groups cleared the sample floor, n >= 2 each, and
        the direction matches the data) vs ``hypothesis_unverified`` (the
        effect direction is plausible but the sample floor or evidence is
        not yet met). It is NOT free narrative — the report assembly
        downgrades any claim whose stated sample is too small.
      - ``predicted_effect`` is a falsifiable statement of what should
        happen in Week 2's data if the recommendation is correct, so the
        next report (and the Week 1 vs Week 2 write-up) can test it.
    """

    change: str = Field(..., min_length=1)
    evidence: str = Field(..., min_length=1)
    underlying_statistic: str = Field(..., min_length=1)
    confidence: Literal["observed_correlation", "hypothesis_unverified"]
    expected_effect: str = Field(..., min_length=1)
    predicted_effect: str = Field(..., min_length=1)

    @field_validator("change", "evidence", "underlying_statistic", "expected_effect", "predicted_effect")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("recommendation fields must be non-empty")
        return v.strip()


# The fixed methodology preamble every WeeklyReport leads with. One honest
# place where the scoring bar is stated so the reader can audit the findings
# below against it instead of trusting the narrative blindly.
METHODOLOGY_PARAGRAPH = """\
METHODOLOGY (stated before findings so the report can be audited against it):

1. Sample floor — a pattern is reported as a FINDING only when each comparison
   group contains at least 2 posts (n >= 2). Comparisons with n = 1 in either
   group are allowed only as a labeled hypothesis, never as a finding.
2. Effect floor — a group difference counts as an effect only when the better
   group's mean is >= 25% above the worse group's mean (relative), so small
   quirks from noise are not reported as discovered rules.
3. Source of every number — all figures come from the platform's per-post
   metric snapshots and its own comment records via GET /analytics/week; the
   narrative layer may only restate numbers that appear in the computed stats,
   never invent them.
4. Confidence labels — \"observed_correlation\" requires both groups above the
   sample floor (n >= 2 each); anything smaller is \"hypothesis_unverified\",
   even if the direction looks real.
5. Falsifiability — every recommendation carries a predicted_effect that states
   exactly what Week 2 data should show if the recommendation works. The
   Week 1 vs Week 2 comparison checks each one and reports honestly whether it
   held."""


class WeeklyReport(BaseModel):
    """The full weekly analysis the Analytics Agent emits.

    Produced by the Analytics Agent at the end of each simulated week;
    consumed by the Strategy Agent (recommendations -> week 2 plan), the
    memory layer (patterns/recommendations embedded for future retrieval),
    and the human.
    """

    campaign_id: str = Field(..., min_length=1)
    week_number: int = Field(..., ge=1)
    # Fixed methodology preamble — the same audited bar for every report.
    methodology: str = Field(default=METHODOLOGY_PARAGRAPH, min_length=1)
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

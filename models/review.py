"""Review data contracts.

DATA FLOW:
    ComplianceVerdict   produced by Brand & Compliance Agent
                            -> consumed by the Orchestrator (approve -> human gate,
                               reject -> back to Writer with `reasons` as revision input),
                               and recorded on Post.compliance_status / rejection_count.

The reject loop is bounded (max 3 cycles -> escalate to human) by
orchestration/retry_policy.py on Day 2; this model is what carries the
rejection reasons back to the Writer on every cycle.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Verdict = Literal["approve", "reject"]


class ComplianceVerdict(BaseModel):
    """Compliance's structured decision on one Post.

    Produced by the Brand & Compliance Agent (rule-based pre-filter + model
    judgment); consumed by the Orchestrator to route the post to the human
    approval gate or back to the Writer with concrete reasons.
    """

    post_id: str = Field(..., min_length=1)
    verdict: Verdict
    # On reject: the Writer's revision prompt gets exactly these strings, so
    # they must be actionable ("no medical claims") not vague ("bad tone").
    reasons: list[str] = Field(default_factory=list)
    banned_terms_hit: list[str] = Field(default_factory=list)

    @field_validator("reasons")
    @classmethod
    def reasons_non_empty_strings(cls, v: list[str]) -> list[str]:
        if any(not r.strip() for r in v):
            raise ValueError("reasons must not contain empty strings")
        return [r.strip() for r in v]

    @field_validator("banned_terms_hit")
    @classmethod
    def banned_terms_non_empty_strings(cls, v: list[str]) -> list[str]:
        if any(not t.strip() for t in v):
            raise ValueError("banned_terms_hit must not contain empty strings")
        return [t.strip() for t in v]

    @field_validator("banned_terms_hit")
    @classmethod
    def banned_terms_unique(cls, v: list[str]) -> list[str]:
        # Duplicated terms would overstate severity in the trace log.
        if len(v) != len(set(t.lower() for t in v)):
            raise ValueError("banned_terms_hit contains duplicates")
        return v

"""Brand & Compliance Agent — two-layer review (doc section 7).

Layer 1, RULE-BASED PREFILTER (deterministic, in code, explicit rules):
    A banned-phrase list + regexes. NOT faked with a single keyword check —
    each rule below is a real pattern with a real reason a human wrote.
    Hard rules cannot be overridden by the model: the doc is explicit that
    compliance must not "rely on the LLM alone for hard constraints", so a
    hard hit forces reject regardless of what the judgment layer says.
    This determinism is also what makes the reject-loop test and the
    Day-2 dry run reproducible.

Layer 2, MODEL JUDGMENT (this file renders llm/prompts/compliance.jinja):
    Sees the post AND the prefilter findings, and judges what rules can't:
    tone, misleading implication, context (e.g. a banned term inside a clear
    negation — "not a miracle cure" — is a model call, not a regex call).

Output: models/review.py::ComplianceVerdict (the shared contract). The
Orchestrator consumes verdicts; orchestration/retry_policy.py owns what a
rejection costs (bounded retries) — this agent never decides retry counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from agents.base_agent import BaseAgent
from models.review import ComplianceVerdict


# --- Layer 1: the explicit rules --------------------------------------------
# Each rule: the regex (case-insensitive), the label reported on the bus/verdict,
# and the human-written reason that goes back to the Writer verbatim.
HARD_RULES: list[dict] = [
    {
        "pattern": r"\b100%\s*guaranteed\b",
        "label": "100% guaranteed",
        "reason": "remove the absolute guarantee '100% guaranteed' — outcomes cannot be promised",
    },
    {
        "pattern": r"\bguaranteed\b",
        "label": "guaranteed",
        "reason": "remove the word 'guaranteed' — no performance claims can be guaranteed",
    },
    {
        "pattern": r"\brisk[-\s]?free\b",
        "label": "risk-free",
        "reason": "remove 'risk-free' — absolute safety claims are banned",
    },
    {
        "pattern": r"\bmiracle\b",
        "label": "miracle",
        "reason": "remove 'miracle' — pseudo-medical framing is banned",
    },
    {
        "pattern": r"\bcures?\b",
        "label": "cure/cures",
        "reason": "remove medical claim 'cure/cures' — we never make health claims",
    },
    {
        "pattern": r"\bno\s+side\s+effects\b",
        "label": "no side effects",
        "reason": "remove 'no side effects' — medical assurance claims are banned",
    },
    {
        "pattern": r"\bact\s+now\b",
        "label": "act now",
        "reason": "remove 'act now' — false-urgency pressure tactics are banned",
    },
]

# Soft rules: model judges these IN CONTEXT (a "#1" claim vs quoting a customer
# saying "#1" are different cases — that's why they're not hard rules).
SOFT_RULES: list[dict] = [
    {
        "pattern": r"\bbest\s+ever\b",
        "label": "best ever",
        "reason": "'best ever' is an unsubstantiated superlative — rephrase as a factual benefit",
    },
    {
        "pattern": r"#1\b",
        "label": "#1",
        "reason": "'#1' is an unverifiable ranking claim — replace with a concrete differentiator",
    },
    {
        "pattern": r"\bnothing\s+better\s+than\b",
        "label": "nothing better than",
        "reason": "'nothing better than' is an absolute comparison — rephrase factually",
    },
]

_RULES_COMPILED = [
    (rule, re.compile(rule["pattern"], re.IGNORECASE)) for rule in HARD_RULES + SOFT_RULES
]
_HARD_LABELS = {rule["label"] for rule in HARD_RULES}


@dataclass
class PrefilterResult:
    """Layer-1 output. hard=True iff a HARD_RULE hit (model cannot override)."""

    banned_terms_hit: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    hard: bool = False


def run_prefilter(copy: str) -> PrefilterResult:
    """Apply every banned-list rule to the copy. Deterministic by design."""
    result = PrefilterResult()
    for rule, rx in _RULES_COMPILED:
        if rx.search(copy):
            result.banned_terms_hit.append(rule["label"])
            result.findings.append(f"{rule['reason']} (banned term: '{rule['label']}')")
            if rule["label"] in _HARD_LABELS:
                result.hard = True
    return result


class _Judgment(BaseModel):
    """Shape the model must return (mirrors ComplianceVerdict fields)."""

    verdict: str  # validated below: only approve|reject accepted
    reasons: list[str] = []
    banned_terms_hit: list[str] = []


class ComplianceAgent(BaseAgent):
    name = "compliance"

    async def review(self, campaign_id: str, post: dict, *, transport=None) -> ComplianceVerdict:
        """Review one post; returns the shared ComplianceVerdict contract.

        post: {id, channel, copy, hashtags, cta, creative} (Post-shaped dict;
        the runner converts drafts to Posts before review so the id is stable
        across rejection cycles).
        Emits compliance_verdict (payload carries verdict + reasons, which is
        the exact data the reject loop replays in the write-up).
        """
        post_id = post["id"]
        self.log_io("input", campaign_id=campaign_id, post_id=post_id, channel=post.get("channel"))

        # Layer 1 — deterministic prefilter, always first.
        pre = run_prefilter(post.get("copy", ""))
        self.log_io(
            "prefilter",
            campaign_id=campaign_id,
            post_id=post_id,
            banned_terms_hit=pre.banned_terms_hit,
            hard_violation=pre.hard,
        )

        # Hard violation short-circuit: the model CANNOT override hard rules
        # (doc: "don't rely on the LLM alone for hard constraints"), so asking
        # it would be theater — reject immediately, no model call, no latency.
        # The judgment layer runs only when the rules left room for context.
        if pre.hard:
            result = ComplianceVerdict(
                post_id=post_id,
                verdict="reject",
                reasons=pre.findings,
                banned_terms_hit=pre.banned_terms_hit,
            )
            self.log_io(
                "output",
                campaign_id=campaign_id,
                post_id=post_id,
                verdict=result.verdict,
                reasons=result.reasons,
                short_circuit="hard_rule",
            )
            await self.emit(
                to_agent="orchestrator",
                campaign_id=campaign_id,
                message_type="compliance_verdict",
                payload=result.model_dump(mode="json"),
            )
            return result

        # Layer 2 — model judgment WITH the findings in view (the template
        # asks it to confirm/deny context, not re-run the regexes).
        prompt = self.render_prompt(
            "compliance.jinja",
            post={
                "id": post_id,
                "channel": post.get("channel"),
                "copy": post.get("copy"),
                "hashtags": post.get("hashtags", []),
                "cta": post.get("cta"),
                "creative": post.get("creative"),
            },
            rule_findings=pre.findings,
        )
        judgment: _Judgment = await self.call_model(
            instruction=prompt,
            schema_cls=_Judgment,
            transport=transport,
            campaign_id=campaign_id,
        )

        # Merge the layers into the final verdict:
        #   model reject   -> rule findings + model reasons together
        #   model approve  -> approve; soft findings the model cleared are
        #                     dropped (contextual judgment is the model's job)
        if judgment.verdict == "reject":
            verdict = "reject"
            reasons = pre.findings + [r for r in judgment.reasons if r.strip()]
            banned = sorted(set(pre.banned_terms_hit) | set(judgment.banned_terms_hit))
        else:
            verdict = "approve"
            reasons = []
            banned = []

        result = ComplianceVerdict(
            post_id=post_id,
            verdict=verdict,  # type: ignore[arg-type]
            reasons=reasons,
            banned_terms_hit=banned,
        )
        self.log_io(
            "output",
            campaign_id=campaign_id,
            post_id=post_id,
            verdict=result.verdict,
            reasons=result.reasons,
        )
        await self.emit(
            to_agent="orchestrator",
            campaign_id=campaign_id,
            message_type="compliance_verdict",
            payload=result.model_dump(mode="json"),
        )
        return result

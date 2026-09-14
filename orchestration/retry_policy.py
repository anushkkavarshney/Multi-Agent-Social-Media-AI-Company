"""Bounded-retry policy for the Compliance reject loop (doc section 7).

The rule this file exists to enforce:

    Compliance rejects -> Writer receives `rejection_reasons`, revises,
    resubmits. Max 3 rejection cycles per post -> auto-escalate to human
    queue. Do not loop indefinitely.

WHY a policy object instead of `if count < 3` inside the compliance agent:
  1. The limit is TUNED, not owned — it comes from .env
     (MAX_COMPLIANCE_RETRIES) via config.settings; an agent that owns the
     constant would bypass the config layer.
  2. The same escalation shape will be reused for model-output failures
     (Day 3+: a post whose Writer call exhausts ModelOutputFailure retries
     also lands in needs_human) — one tested object, two call sites.
  3. "Here is the exact code that prevents an infinite loop" is a one-file
     answer in the write-up; hunting an if-statement across agents is not.

The policy is deliberately dumb (pure function of a counter, no I/O, no
clock) so it is trivially testable — orchestration/agents.py owns *when* it
is consulted.
"""

from dataclasses import dataclass

from config.settings import get_settings


@dataclass(frozen=True)
class RetryDecision:
    """Outcome of consulting the policy for one rejection.

    Why a tiny result type instead of a bare bool: the CALLER also needs the
    human-readable reason when escalating (it goes on the bus and into the
    needs_human queue), and this keeps that string next to the decision
    that produced it.
    """

    should_retry: bool
    escalate: bool
    reason: str  # populated (and meaningful) only when escalate=True


class ComplianceRetryPolicy:
    """Bounds how many times a single post may bounce Writer -> Compliance.

    Semantics: `rejections_so_far` counts COMPLETED rejections (a post that
    was rejected once and revised once arrives with rejections_so_far == 1).
    The (N+1)th rejection is allowed only if N < max_retries, so with the
    default max_retries=3 a post gets 3 revision chances before escalation.
    """

    def __init__(self, max_retries: int | None = None):
        # None => read from settings so .env retunes the loop without code
        # changes. Explicit values exist for tests (e.g. max_retries=1).
        if max_retries is None:
            max_retries = get_settings().max_compliance_retries
        if max_retries < 1:
            # Zero retries would make the whole loop dead code; reject the
            # misconfiguration loudly instead of silently never retrying.
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self.max_retries = max_retries

    def on_rejection(self, post_id: str, rejection_count: int) -> RetryDecision:
        """Decide what happens to `post_id` after another rejection.

        `rejection_count` is the post's NEW total (after the rejection that
        just happened). Callers get this from Post.rejection_count, which
        models/post.py already keeps consistent with compliance_status.
        """
        if rejection_count < self.max_retries:
            return RetryDecision(
                should_retry=True,
                escalate=False,
                reason="",
            )
        return RetryDecision(
            should_retry=False,
            escalate=True,
            reason=(
                f"post {post_id} rejected {rejection_count} times "
                f"(max_compliance_retries={self.max_retries}); "
                "escalating to needs_human instead of looping again"
            ),
        )

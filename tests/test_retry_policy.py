"""Retry policy tests — the bounded-loop guarantee, in isolation.

The policy is a pure function of a counter, so these tests are exhaustive
at the boundary: the LAST allowed rejection retries, the FIRST over-budget
rejection escalates, and the escalation reason is human-readable (it goes
on the bus and into the needs_human queue verbatim).
"""

import pytest

from orchestration.retry_policy import ComplianceRetryPolicy


class TestBoundary:
    def test_first_of_three_rejections_retries(self):
        policy = ComplianceRetryPolicy(max_retries=3)
        decision = policy.on_rejection("p1", rejection_count=1)
        assert decision.should_retry and not decision.escalate

    def test_last_allowed_rejection_still_retries(self):
        """Rejection #2 of 3: one revision chance remains."""
        policy = ComplianceRetryPolicy(max_retries=3)
        decision = policy.on_rejection("p1", rejection_count=2)
        assert decision.should_retry and not decision.escalate

    def test_rejection_at_limit_escalates(self):
        """Rejection #3 of 3: budget spent -> needs_human, never a 4th cycle."""
        policy = ComplianceRetryPolicy(max_retries=3)
        decision = policy.on_rejection("p1", rejection_count=3)
        assert decision.escalate and not decision.should_retry
        # The reason must be self-explanatory on the bus / needs_human queue:
        assert "3 times" in decision.reason
        assert "p1" in decision.reason

    def test_escalation_is_sticky_beyond_limit(self):
        """An off-by-one caller passing 4 (or 10) must still escalate —
        escalation is monotonic in rejection_count."""
        policy = ComplianceRetryPolicy(max_retries=3)
        for count in (4, 10):
            assert policy.on_rejection("p1", count).escalate

    def test_env_tuned_limit_is_honored(self, monkeypatch):
        """The limit is config-driven (.env MAX_COMPLIANCE_RETRIES), so the
        policy must follow settings, not a hardcoded 3."""
        monkeypatch.setenv("MAX_COMPLIANCE_RETRIES", "5")
        from config.settings import get_settings

        get_settings.cache_clear()
        try:
            policy = ComplianceRetryPolicy()
            assert policy.max_retries == 5
            assert not policy.on_rejection("p1", 4).escalate
            assert policy.on_rejection("p1", 5).escalate
        finally:
            get_settings.cache_clear()


class TestConstruction:
    def test_zero_retries_is_rejected_loudly(self):
        """max_retries=0 would make the loop dead code; misconfiguration
        must raise at construction, not silently never-retry."""
        with pytest.raises(ValueError):
            ComplianceRetryPolicy(max_retries=0)

    def test_one_retry_is_a_legal_configuration(self):
        policy = ComplianceRetryPolicy(max_retries=1)
        assert policy.on_rejection("p1", 1).escalate  # single strike, then human

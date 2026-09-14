"""State machine tests (doc build order step 8: "test with
tests/test_state_machine.py — reject loop terminates correctly").

These tests ARE the lifecycle documentation executed: the happy path walks
the doc's diagram edge by edge; the rejection tests prove the loop both
continues and terminates; the table-exhaustiveness check proves every
state has an outgoing edge EXCEPT the true dead end (needs_human — which
only a human unblocks, by design).
"""

import pytest

from models.campaign import Campaign
from orchestration import state_machine as sm


def make_campaign(status: str = "draft") -> Campaign:
    """Campaign factory at any lifecycle stage.

    Non-draft statuses need pillars/KPIs (the contract requires it), so we
    fill minimal valid ones — the state machine doesn't care about content,
    only status.
    """
    kwargs = dict(
        id="camp_test",
        brief_raw="test brief",
        objective="awareness",
        target_audience="students",
        channel_mix=["ch_shortform"],
        duration_days=7,
    )
    if status != "draft":
        kwargs.update(
            content_pillars=[{"name": "p1", "description": "d", "weight": 1.0}],
            kpis=[{"metric": "impressions", "target": 1000.0, "channel": None}],
        )
    return Campaign(status=status, **kwargs)


def walk(campaign: Campaign, events: list[str]) -> Campaign:
    """Apply events in sequence, asserting each is legal before applying."""
    for event in events:
        assert sm.can_transition(campaign.status, event), (
            f"{event!r} not legal from {campaign.status!r}"
        )
        campaign = sm.transition(campaign, event)
    return campaign


HAPPY_PATH = [
    sm.EV_STRATEGY_FILLED,
    sm.EV_CONTENT_DRAFTED,
    sm.EV_REVIEW_STARTED,
    sm.EV_COMPLIANCE_APPROVED,
    sm.EV_HUMAN_APPROVED,
    sm.EV_POSTS_PUBLISHED,
    sm.EV_SIMULATION_WINDOW_CLOSED,
    sm.EV_REPORT_GENERATED,
]


class TestHappyPath:
    def test_full_lifecycle_walks_the_doc_diagram(self):
        """draft -> ... -> week_reviewed, one doc edge at a time."""
        c = walk(make_campaign(), HAPPY_PATH)
        assert c.status == "week_reviewed"

    def test_week2_loop_closure(self):
        """week_reviewed -[recommendations applied]-> strategy_filled."""
        c = walk(make_campaign(), HAPPY_PATH)
        c = sm.transition(c, sm.EV_RECOMMENDATIONS_APPLIED)
        assert c.status == "strategy_filled"


class TestRejectLoop:
    def test_rejection_with_retries_returns_to_content_drafted(self):
        """The doc edge: compliance_review -[reject, retries<3]-> content_drafted."""
        c = walk(make_campaign(), [sm.EV_STRATEGY_FILLED, sm.EV_CONTENT_DRAFTED, sm.EV_REVIEW_STARTED])
        c = sm.transition(c, sm.EV_COMPLIANCE_REJECTED)
        assert c.status == "content_drafted"
        # And the campaign can re-enter review for the revised drafts:
        c = sm.transition(c, sm.EV_REVIEW_STARTED)
        assert c.status == "compliance_review"

    def test_exhausted_rejection_goes_to_needs_human(self):
        c = walk(make_campaign(), [sm.EV_STRATEGY_FILLED, sm.EV_CONTENT_DRAFTED, sm.EV_REVIEW_STARTED])
        c = sm.transition(c, sm.EV_COMPLIANCE_REJECTED_EXHAUSTED)
        assert c.status == "needs_human"

    def test_rejection_loop_terminates_full_cycle_then_approves(self):
        """Two full rejection cycles (the dry run's shape), then approval."""
        c = walk(make_campaign(), [sm.EV_STRATEGY_FILLED, sm.EV_CONTENT_DRAFTED])
        for _ in range(2):
            c = sm.transition(c, sm.EV_REVIEW_STARTED)
            c = sm.transition(c, sm.EV_COMPLIANCE_REJECTED)
            assert c.status == "content_drafted"
        c = sm.transition(c, sm.EV_REVIEW_STARTED)
        c = sm.transition(c, sm.EV_COMPLIANCE_APPROVED)
        assert c.status == "pending_human_approval"

    def test_needs_human_is_a_dead_end(self):
        """No event leaves needs_human — only a human action outside the
        machine (re-drafting the campaign) unblocks it. This is the bounded
        loop made structural."""
        c = sm.transition(
            walk(make_campaign(), [sm.EV_STRATEGY_FILLED, sm.EV_CONTENT_DRAFTED, sm.EV_REVIEW_STARTED]),
            sm.EV_COMPLIANCE_REJECTED_EXHAUSTED,
        )
        for event in (
            sm.EV_COMPLIANCE_APPROVED, sm.EV_COMPLIANCE_REJECTED,
            sm.EV_HUMAN_APPROVED, sm.EV_STRATEGY_FILLED, sm.EV_REVIEW_STARTED,
        ):
            assert not sm.can_transition("needs_human", event), event


class TestInvalidTransitions:
    def test_illegal_event_raises_and_leaves_campaign_untouched(self):
        c = make_campaign("draft")
        with pytest.raises(sm.InvalidTransitionError) as excinfo:
            sm.transition(c, sm.EV_HUMAN_APPROVED)
        # The error must be actionable: name the event, the state, and what
        # IS allowed — this is the "clear error, not silent failure" rule.
        assert "human_approved" in str(excinfo.value)
        assert "draft" in str(excinfo.value)
        # And the original object is unmutated:
        assert c.status == "draft"

    def test_cannot_skip_the_human_gate(self):
        c = walk(make_campaign(), [sm.EV_STRATEGY_FILLED, sm.EV_CONTENT_DRAFTED, sm.EV_REVIEW_STARTED])
        with pytest.raises(sm.InvalidTransitionError):
            sm.transition(c, sm.EV_HUMAN_APPROVED)  # still in compliance_review

    def test_cannot_approve_from_pending_without_human_event(self):
        c = walk(make_campaign(), HAPPY_PATH[:4])  # ... -> pending_human_approval
        with pytest.raises(sm.InvalidTransitionError):
            sm.transition(c, sm.EV_POSTS_PUBLISHED)  # must pass human gate first

    def test_unknown_state_value_raises_loudly(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            make_campaign("status_that_does_not_exist")


class TestTableExhaustiveness:
    def test_every_state_has_outgoing_edges_except_needs_human(self):
        """Every state can do SOMETHING except needs_human (the terminal
        dead end). If a future edit adds a state without edges, this fails —
        structural protection against orphan states."""
        states_with_edges = {cur for cur, _, _ in sm._TRANSITIONS}
        all_states = set(sm.CampaignState)
        orphans = all_states - states_with_edges
        assert orphans == {sm.CampaignState.NEEDS_HUMAN}, orphans

    def test_transition_table_rows_are_unique(self):
        """A duplicate (state, event) row would shadow the earlier one
        silently — the set structure makes that impossible, but this test
        documents the invariant in case the table becomes a dict."""
        rows = [(cur, ev) for cur, ev, _ in sm._TRANSITIONS]
        assert len(rows) == len(set(rows))

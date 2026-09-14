"""Campaign lifecycle state machine (architecture doc section 8).

The lifecycle, verbatim from the doc:

    draft -> strategy_filled -> content_drafted -> compliance_review
       compliance_review -[reject, retries<3]-> content_drafted
       compliance_review -[reject, retries>=3]-> needs_human
       compliance_review -[approve]-> pending_human_approval
       pending_human_approval -[reject]-> draft   (human can send back)
       pending_human_approval -[approve]-> scheduled
       scheduled -> published -> simulating -> week_reviewed
       week_reviewed -[recommendations applied]-> strategy_filled (week 2)

Design decisions (this file is the one the reviewer scrutinizes):

- ENUM + TABLE, not if/else: every legal transition is one visible row in
  _TRANSITIONS. Auditing "what can happen to a campaign in needs_human?"
  is reading two table entries, not tracing control flow across agents.
- The retry boundary (retries<3 vs >=3) lives HERE as two distinct events
  (compliance_rejected / compliance_rejected_exhausted) because the doc
  writes it into the diagram itself. The policy that DECIDES which event
  to fire is orchestration/retry_policy.py; this file only enforces that
  no other route into needs_human exists.
- Rejections leave compliance_review and return to content_drafted (the
  doc's edge), so every revision cycle is: content_drafted ->
  compliance_review -> content_drafted. Review is an ACTIVITY the campaign
  re-enters each cycle, not a flag set once.
- transition() takes and returns a Campaign (doc-shaped signature) and
  mutates a copy — callers never see a half-transitioned object on error,
  because the check happens BEFORE any write.
"""

from __future__ import annotations

from enum import Enum

from models.campaign import Campaign


class CampaignState(str, Enum):
    """The 10 lifecycle states, named exactly as in doc section 8."""

    DRAFT = "draft"
    STRATEGY_FILLED = "strategy_filled"
    CONTENT_DRAFTED = "content_drafted"
    COMPLIANCE_REVIEW = "compliance_review"
    NEEDS_HUMAN = "needs_human"
    PENDING_HUMAN_APPROVAL = "pending_human_approval"
    SCHEDULED = "scheduled"
    PUBLISHED = "published"
    SIMULATING = "simulating"
    WEEK_REVIEWED = "week_reviewed"


class InvalidTransitionError(Exception):
    """Raised on any state/event pair the lifecycle does not allow.

    Deliberately not a ValueError: callers (orchestrator, agents) catch this
    specifically to log a bus event and halt gracefully — a bare ValueError
    would risk being swallowed by generic exception handlers.
    """

    def __init__(self, current: "CampaignState", event: str):
        self.current = current
        self.event = event
        allowed = sorted(t[1] for t in _TRANSITIONS if t[0] is current)
        super().__init__(
            f"illegal transition: event '{event}' is not valid from state "
            f"'{current.value}' (allowed events here: {allowed or 'none — terminal'})"
        )


# --- Events -----------------------------------------------------------------
# Plain string constants (not an Enum) so agents emit them as readable bus
# message_types without importing anything special.
EV_STRATEGY_FILLED = "strategy_filled"
EV_CONTENT_DRAFTED = "content_drafted"
EV_REVIEW_STARTED = "compliance_review_started"               # content_drafted -> compliance_review
EV_COMPLIANCE_REJECTED = "compliance_rejected"                # retries remain
EV_COMPLIANCE_REJECTED_EXHAUSTED = "compliance_rejected_exhausted"  # retries gone
EV_COMPLIANCE_APPROVED = "compliance_approved"
EV_HUMAN_REJECTED = "human_rejected"
EV_HUMAN_APPROVED = "human_approved"
EV_SCHEDULING_DONE = "scheduling_done"
EV_POSTS_PUBLISHED = "posts_published"
EV_SIMULATION_WINDOW_CLOSED = "simulation_window_closed"
EV_REPORT_GENERATED = "report_generated"
EV_RECOMMENDATIONS_APPLIED = "recommendations_applied"

# --- The transition table: the entire lifecycle, one row per legal move ------
# (current_state, event) -> next_state. If a pair is absent, it is illegal.
_TRANSITIONS: set[tuple[CampaignState, str, CampaignState]] = {
    # (current, event, next)                                  # why it exists
    (CampaignState.DRAFT, EV_STRATEGY_FILLED, CampaignState.STRATEGY_FILLED),
    # ^ Strategy Agent fills audience/channels/pillars/KPIs (doc section 7).
    (CampaignState.STRATEGY_FILLED, EV_CONTENT_DRAFTED, CampaignState.CONTENT_DRAFTED),
    # ^ Writer produces post drafts from the filled strategy.
    (CampaignState.CONTENT_DRAFTED, EV_REVIEW_STARTED, CampaignState.COMPLIANCE_REVIEW),
    # ^ Drafts enter review. A DISTINCT state (not an action on content_drafted)
    #   because the doc draws it as an edge — and because rejections must have
    #   an unambiguous state to return FROM.
    (CampaignState.COMPLIANCE_REVIEW, EV_COMPLIANCE_REJECTED, CampaignState.CONTENT_DRAFTED),
    # ^ Reject WITH retries left: back to the Writer (doc edge, exactly).
    #   The retry_policy object decides WHICH reject event fires; the table
    #   guarantees these are the only routes out of compliance_review.
    (CampaignState.COMPLIANCE_REVIEW, EV_COMPLIANCE_REJECTED_EXHAUSTED, CampaignState.NEEDS_HUMAN),
    # ^ Reject WITHOUT retries: bounded-loop guarantee. needs_human is reached
    #   ONLY here — no other edge leads into it.
    (CampaignState.COMPLIANCE_REVIEW, EV_COMPLIANCE_APPROVED, CampaignState.PENDING_HUMAN_APPROVAL),
    # ^ Compliance approves: the doc REQUIRES a human gate before publishing,
    #   so approval never jumps straight to scheduled.
    (CampaignState.PENDING_HUMAN_APPROVAL, EV_HUMAN_REJECTED, CampaignState.DRAFT),
    # ^ Human bounces the whole campaign back to the start (doc: "human can
    #   send back to Strategy"). Draft — not strategy_filled — so the human's
    #   reasons reach the Strategy Agent, not just the Writer.
    (CampaignState.PENDING_HUMAN_APPROVAL, EV_HUMAN_APPROVED, CampaignState.SCHEDULED),
    # ^ The human gate is the ONLY route from pending to scheduled.
    (CampaignState.SCHEDULED, EV_POSTS_PUBLISHED, CampaignState.PUBLISHED),
    # ^ Scheduler posts to the mock platform (Day 3).
    (CampaignState.PUBLISHED, EV_SIMULATION_WINDOW_CLOSED, CampaignState.SIMULATING),
    # ^ Engagement accrues under the hidden rules for the campaign window.
    (CampaignState.SIMULATING, EV_REPORT_GENERATED, CampaignState.WEEK_REVIEWED),
    # ^ Analytics Agent produces the WeeklyReport.
    (CampaignState.WEEK_REVIEWED, EV_RECOMMENDATIONS_APPLIED, CampaignState.STRATEGY_FILLED),
    # ^ Week-2 loop closure: recommendations applied => strategy re-filled.
    #   NOT draft: the objective/brief persist; only the plan regenerates.
}


def can_transition(current_state: CampaignState | str, event: str) -> bool:
    """Is `event` legal from `current_state`? Pure predicate, never raises.

    Accepts the raw string form too ("draft"), because bus messages carry
    strings and callers should not need to import the enum to ask a question.
    """
    state = CampaignState(current_state)  # ValueError on an unknown state — loud is right
    return any(cur is state and ev == event for cur, ev, _ in _TRANSITIONS)


def transition(campaign: Campaign, event: str) -> Campaign:
    """Apply `event` to `campaign`; return the NEW Campaign with the next state.

    Contract:
    - Checks the table FIRST: an illegal pair raises InvalidTransitionError
      and the original Campaign is untouched (copy is taken after the check).
    - The rejected/exhausted distinction is the CALLER's job (retry_policy):
      this function will happily refuse the wrong one, but it cannot know
      rejection counts — single responsibility, kept honest by tests.
    """
    state = CampaignState(campaign.status)
    for cur, ev, nxt in _TRANSITIONS:
        if cur is state and ev == event:
            updated = campaign.model_copy(deep=True)  # never mutate the caller's object
            updated.status = nxt.value
            return updated
    raise InvalidTransitionError(state, event)

"""Community Manager Agent — draft replies + escalation flags (doc section 7).

Design (doc): "Sentiment/sensitivity classification via small router model
before reply generation" — so this agent makes TWO structured calls per
batch:
    1. ROUTER (uses_router_model=True -> qwen2.5:3b): per-comment
       action class. The router does NOT write text; it only classifies.
       `is_sensitive` from the platform pre-escapes the router entirely:
       platform-flagged comments escalate without a model opinion.
    2. REPLIER (primary model): writes reply text ONLY for comments the
       router classified as reply-worthy.

Day-2 scope: runs against FIXTURE comment data (models/engagement.py
Comment objects built in tests/fixtures) because nothing is published yet;
Day 3 wires it to GET /posts/{id}/comments on the live platform. The input
is already the real contract, so rewiring is a data-source change, not an
agent rewrite.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from config.settings import get_settings
from models.engagement import Comment


class RouterDecision(BaseModel):
    """The small model's classification of one comment. No text generation."""

    comment_id: str
    action: str  # "reply" | "escalate" — validated below
    reason: str = Field(..., min_length=1, description="One-line justification, logged for audit.")


class RouterDecisions(BaseModel):
    decisions: list[RouterDecision] = Field(..., min_length=1)


class ReplyDraft(BaseModel):
    comment_id: str
    reply_text: str = Field(..., min_length=1, max_length=280, description="On-tone public reply, <= 40 words target.")


class ReplyDrafts(BaseModel):
    drafts: list[ReplyDraft] = Field(..., min_length=1)


class CommunityManagerAgent(BaseAgent):
    name = "community_manager"
    # The router-model default applies to the CLASSIFICATION call; the
    # replier call explicitly overrides to the primary model.
    uses_router_model = True

    async def process_comments(
        self,
        campaign_id: str,
        comments: list[Comment],
        post_context: dict,
        channel_tone: str,
        *,
        transport_router=None,
        transport_replier=None,
    ) -> list[dict]:
        """Classify + draft replies for a batch of comments.

        Returns one dict per comment:
          {comment_id, action, reply_text?, escalation_reason?, router_reason}
        Escalations carry escalation_reason; replies carry reply_text.

        Fixture-driven today (tests pass synthetic Comment objects); the
        signature takes the models.Comment contract so Day 3's platform
        wiring changes only where the list comes FROM.
        """
        self.log_io("input", campaign_id=campaign_id, comments=len(comments))

        # Stage 0 — platform pre-filter: sensitivity is a PLATFORM flag, not a
        # model opinion. Escalate before the router sees it (cheaper AND
        # deterministic: sensitive handling is policy, not judgment).
        platform_escalations = [c for c in comments if c.is_sensitive]
        router_input = [c for c in comments if not c.is_sensitive]

        decisions: dict[str, RouterDecision] = {}
        if router_input:
            prompt = self.render_prompt(
                "community_manager.jinja",
                comments=[c.model_dump(mode="json") for c in router_input],
                post_context=post_context,
                channel_tone=channel_tone,
            )
            router_result: RouterDecisions = await self.call_model(
                instruction=prompt,
                schema_cls=RouterDecisions,
                transport=transport_router,
                campaign_id=campaign_id,
            )
            decisions = {d.comment_id: d for d in router_result.decisions}

        # Platform-flagged comments: escalation reason is policy, not judgment.
        for c in platform_escalations:
            decisions[c.id] = RouterDecision(
                comment_id=c.id,
                action="escalate",
                reason="flagged sensitive by platform pre-filter (policy: human handles)",
            )

        # Stage 2 — draft replies ONLY for reply-classified comments.
        to_reply = [
            c for c in comments if decisions.get(c.id) and decisions[c.id].action == "reply"
        ]
        drafts: dict[str, ReplyDraft] = {}
        if to_reply:
            reply_prompt = (
                "You are the Community Manager. Write one public reply per comment.\n"
                f"Channel tone: {channel_tone}\n"
                f"Post context: {post_context}\n"
                "Rules: <= 40 words, one emoji max, no promises, continue the "
                "conversation (a question back is good on discussion channels).\n"
                "COMMENTS:\n"
                + "\n".join(f'- id={c.id} by {c.author_handle}: "{c.text}"' for c in to_reply)
            )
            replier_result: ReplyDrafts = await self.call_model(
                instruction=reply_prompt,
                schema_cls=ReplyDrafts,
                model=get_settings().primary_model,  # generation = primary model
                transport=transport_replier,
                campaign_id=campaign_id,
            )
            drafts = {d.comment_id: d for d in replier_result.drafts}

        # Assemble the per-comment action list in input order.
        results: list[dict] = []
        for c in comments:
            d = decisions.get(c.id)
            if d is None:
                # Router omitted a comment it was given: fail safe to human
                # queue rather than silently ignoring a community member.
                results.append(
                    {
                        "comment_id": c.id,
                        "action": "escalate",
                        "escalation_reason": "router omitted this comment; failing safe to human",
                        "router_reason": "",
                    }
                )
                continue
            entry: dict = {
                "comment_id": c.id,
                "action": d.action,
                "router_reason": d.reason,
            }
            if d.action == "reply":
                draft = drafts.get(c.id)
                if draft is None:
                    # Router said reply but the replier produced nothing:
                    # fail safe, never leave a reply-classified comment hanging.
                    entry["action"] = "escalate"
                    entry["escalation_reason"] = "reply draft missing; failing safe to human"
                else:
                    entry["reply_text"] = draft.reply_text
            else:
                entry["escalation_reason"] = d.reason
            results.append(entry)

        self.log_io(
            "output",
            campaign_id=campaign_id,
            replies=sum(1 for r in results if r["action"] == "reply"),
            escalations=sum(1 for r in results if r["action"] == "escalate"),
        )
        await self.emit(
            to_agent="orchestrator",
            campaign_id=campaign_id,
            message_type="community_actions",
            payload={"actions": results},
        )
        return results

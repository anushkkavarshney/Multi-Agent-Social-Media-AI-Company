"""Creative Agent — one CreativeBrief per drafted post (doc section 7).

"Simple, constrained generation — lowest-risk agent": the model chooses the
asset type inside a per-channel-type whitelist and writes a one-sitting
designer brief. The whitelist is enforced IN CODE after generation (the
model's choice is a proposal; the channel constraint is ours).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from models.post import CreativeBrief


class CreativeChoice(BaseModel):
    asset_type: str  # validated against the whitelist below, not the Literal —
    # so a bad choice becomes a retryable validation error with a clear
    # message rather than a Pydantic literal error the model can't interpret.
    description: str = Field(..., min_length=20, description="Executable designer brief.")


# Allowed asset types per platform channel type (mirrors the constraint
# stated in llm/prompts/creative.jinja — prompt proposes, code disposes).
_ALLOWED_BY_CHANNEL_TYPE: dict[str, set[str]] = {
    "shortform_video": {"video"},
    "discussion": {"text_only", "image"},
    "professional": {"image", "carousel"},
}
_FALLBACK: dict[str, str] = {
    # Deterministic fallback if the model proposes something illegal after
    # retries: the pipeline must not stall on the lowest-risk agent.
    "shortform_video": "video",
    "discussion": "text_only",
    "professional": "image",
}


class CreativeAgent(BaseAgent):
    name = "creative"

    async def run(
        self,
        campaign_id: str,
        post_copy: str,
        channel: dict,
        pillar: dict,
        *,
        transport=None,
    ) -> CreativeBrief:
        """Produce the CreativeBrief for one post's copy.

        channel: {id, name, type}; pillar: {name, description}.
        Emits creative_brief_ready per post (the bus shows every handoff).
        """
        self.log_io("input", campaign_id=campaign_id, channel=channel["id"], copy_chars=len(post_copy))
        prompt = self.render_prompt(
            "creative.jinja",
            post={"copy": post_copy},
            channel=channel,
            pillar=pillar,
        )
        choice: CreativeChoice = await self.call_model(
            instruction=prompt,
            schema_cls=CreativeChoice,
            transport=transport,
            campaign_id=campaign_id,
        )

        asset_type = choice.asset_type.strip().lower()
        allowed = _ALLOWED_BY_CHANNEL_TYPE.get(channel.get("type", ""), {"image", "text_only"})
        fallback = _FALLBACK.get(channel.get("type", ""), "image")
        if asset_type not in allowed:
            # Deterministic clamp, logged: never let the constrained-choice
            # agent produce an out-of-policy asset for the channel.
            self.log_io(
                "clamp",
                campaign_id=campaign_id,
                proposed=choice.asset_type,
                clamped_to=fallback,
                channel_type=channel.get("type"),
            )
            asset_type = fallback

        brief = CreativeBrief(asset_type=asset_type, description=choice.description)  # type: ignore[arg-type]
        self.log_io(
            "output",
            campaign_id=campaign_id,
            asset_type=brief.asset_type,
            description_chars=len(brief.description),
        )
        await self.emit(
            to_agent="compliance",
            campaign_id=campaign_id,
            message_type="creative_brief_ready",
            payload={"channel": channel["id"], "asset_type": brief.asset_type},
        )
        return brief

"""Content Writer Agent — drafts and revises post copy (doc section 7).

The three-pass loop, implemented with ONE template at three renderings:
    1. draft     -> PostDrafts from the strategy's pillars
    2. critique  -> self-critique against the charter checklist (the doc's
                    "generate -> critique own output -> revise once before
                    sending to Compliance" — the critique pass is REQUIRED,
                    not decorative, so its issues are fed into the revision)
    3. revise    -> final drafts, fixing critique issues and/or Compliance's
                    rejection_reasons (on the rejection path reasons are
                    binding: the Compliance agent's wording goes verbatim
                    into the prompt)

Output schema: PostDraftList — plain dicts here; conversion into the Post
contract (with hashtags normalization + validation) happens in the runner/
orchestrator, because Post requires an id/creative that are produced by
other steps in the pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from models.campaign import Campaign


class PostDraft(BaseModel):
    """One drafted post (pre-Post-contract shape)."""

    pillar: str = Field(..., min_length=1)
    channel: str = Field(..., min_length=1)
    copy: str = Field(..., min_length=1)
    hashtags: list[str] = Field(..., min_length=1)
    cta: str | None = None


class PostDraftList(BaseModel):
    """Wrapper model for the model-call boundary (Ollama returns one JSON)."""

    posts: list[PostDraft] = Field(..., min_length=1)


class CritiqueIssue(BaseModel):
    post_index: int = Field(..., ge=0, description="Index into the drafts list being critiqued.")
    issue: str = Field(..., min_length=1)
    fix: str = Field(..., min_length=1)


class CritiqueList(BaseModel):
    issues: list[CritiqueIssue] = Field(default_factory=list)


class WriterAgent(BaseAgent):
    name = "writer"

    async def draft(
        self,
        campaign: Campaign,
        channel_assignments: list[dict],
        posts_per_pillar: int = 1,
        *,
        transport=None,
    ) -> list[PostDraft]:
        """Pass 1: first drafts. One post per (pillar, assigned channel).

        channel_assignments: [{pillar: str, channel: str}] — the rotation
        plan; the writer alternates channels so every campaign channel
        gets content (doc: posts run across the campaign's channel mix).
        """
        self.log_io(
            "input",
            campaign_id=campaign.id,
            pass_name="draft",
            posts_requested=len(channel_assignments),
        )
        prompt = self.render_prompt(
            "writer.jinja",
            writing_pass="draft",
            campaign_summary=campaign.model_dump_json(indent=2),
            posts_per_pillar=posts_per_pillar,
        )
        # The channel rotation is instruction, not freeform: the model is
        # told the exact (pillar, channel) pairs to write for.
        prompt += "\n\nCHANNEL ASSIGNMENTS (write exactly these posts, in order):\n" + "\n".join(
            f"- {a['pillar']} -> {a['channel']}" for a in channel_assignments
        )
        result: PostDraftList = await self.call_model(
            instruction=prompt,
            schema_cls=PostDraftList,
            transport=transport,
            campaign_id=campaign.id,
        )
        self.log_io("output", campaign_id=campaign.id, pass_name="draft", posts=len(result.posts))
        return result.posts

    async def critique(self, campaign: Campaign, drafts: list[PostDraft], *, transport=None) -> list[CritiqueIssue]:
        """Pass 2: self-critique against the charter checklist."""
        self.log_io("input", campaign_id=campaign.id, pass_name="critique", posts=len(drafts))
        prompt = self.render_prompt(
            "writer.jinja",
            writing_pass="critique",
            campaign_summary=campaign.model_dump_json(indent=2),
            drafts_for_critique="\n".join(
                f"[{i}] {d.channel}: {d.copy}" for i, d in enumerate(drafts)
            ),
        )
        result: CritiqueList = await self.call_model(
            instruction=prompt,
            schema_cls=CritiqueList,
            transport=transport,
            campaign_id=campaign.id,
        )
        self.log_io(
            "output",
            campaign_id=campaign.id,
            pass_name="critique",
            issues_found=len(result.issues),
        )
        return result.issues

    async def revise(
        self,
        campaign: Campaign,
        previous_drafts: list[PostDraft],
        *,
        critique_issues: list[CritiqueIssue] | None = None,
        rejection_reasons: list[str] | None = None,
        transport=None,
    ) -> list[PostDraft]:
        """Pass 3: revision. Used BOTH for the self-critique fixes (before
        Compliance) and for Compliance rejections (reasons verbatim)."""
        self.log_io(
            "input",
            campaign_id=campaign.id,
            pass_name="revise",
            critique_issues=len(critique_issues or []),
            rejection_reasons=rejection_reasons or [],
        )
        prompt = self.render_prompt(
            "writer.jinja",
            writing_pass="revise",
            campaign_summary=campaign.model_dump_json(indent=2),
            critique_issues=[c.model_dump() for c in (critique_issues or [])],
            rejection_reasons=rejection_reasons or [],
            previous_posts="\n".join(
                f"[{i}] {d.channel}: {d.copy}" for i, d in enumerate(previous_drafts)
            ),
            posts_per_pillar=1,  # template needs it in the draft branch only
        )
        result: PostDraftList = await self.call_model(
            instruction=prompt,
            schema_cls=PostDraftList,
            transport=transport,
            campaign_id=campaign.id,
        )
        self.log_io("output", campaign_id=campaign.id, pass_name="revise", posts=len(result.posts))
        return result.posts

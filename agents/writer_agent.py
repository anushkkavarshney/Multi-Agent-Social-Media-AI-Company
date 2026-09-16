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
from config.settings import get_settings
from llm.structured_output import ModelOutputFailure
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

        # Exact-count guard: a schema-valid PostDraftList can still silently
        # under-produce (live: 2026-09-16 run, 1 of 3 requested returned with
        # min_length=1 passing validation). Count mismatch is a retryable
        # semantic failure, not a schema one, so retry with the mismatch
        # spelled out (bounded by the same max_structured_retries budget).
        total_attempts = get_settings().max_structured_retries
        for attempt in range(1, total_attempts + 1):
            result: PostDraftList = await self.call_model(
                instruction=prompt,
                schema_cls=PostDraftList,
                transport=transport,
                campaign_id=campaign.id,
            )
            expected = len(channel_assignments) * posts_per_pillar
            if len(result.posts) == expected:
                break
            prompt += (
                f"\n\nIMPORTANT: You returned {len(result.posts)} post(s), but "
                f"{expected} were requested (one per assignment line above). "
                f"Return EXACTLY {expected} posts, one per assignment, in the "
                f"same order as the assignments."
            )
        else:
            last_raw = result.model_dump_json()
            raise ModelOutputFailure(
                f"{self.name} produced {len(result.posts)} posts, expected {expected}, "
                f"after {total_attempts} attempts",
                last_raw=last_raw,
                last_error=(
                    f"exact-count guard: returned {len(result.posts)} of {expected} "
                    f"requested posts"
                ),
                attempts=total_attempts,
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
        # Guard: the model sometimes returns FEWER posts than it was given
        # (2026-09-15 live run dropped 3 -> 1 on the revise pass). Never allow
        # a silent data-loss: posts missing from the model output are kept
        # from the previous round, preserving order, so the pipeline always
        # carries the full assignment forward.
        revision = result.posts
        if len(revision) < len(previous_drafts):
            merged: list[PostDraft] = []
            for i, prev in enumerate(previous_drafts):
                merged.append(revision[i] if i < len(revision) else prev)
            self.log_io(
                "output",
                campaign_id=campaign.id,
                pass_name="revise",
                posts_requested=len(previous_drafts),
                posts_returned=len(result.posts),
                posts_merged_from_previous=len(previous_drafts) - len(result.posts),
            )
            revision = merged
        elif len(revision) > len(previous_drafts):
            self.log_io(
                "output",
                campaign_id=campaign.id,
                pass_name="revise",
                posts_requested=len(previous_drafts),
                posts_returned=len(result.posts),
                posts_truncated_to_requested=len(previous_drafts),
            )
            revision = revision[: len(previous_drafts)]
        self.log_io(
            "output",
            campaign_id=campaign.id,
            pass_name="revise",
            posts=len(revision),
        )
        return revision

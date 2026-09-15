"""Escalation queue persistence helpers (Day 3).

The Community Manager decides WHICH comments escalate (policy + router), but
per the codebase layering rule ("agents never touch the DB directly") the
*write* happens here — called by the pipeline runner after the agent returns
its actions. Reading the queue is what the CLI `queue` viewer and any humans
do.

Kept as plain functions on the session factory so both run_demo.py and
cli/viewer.py use one code path.
"""

from __future__ import annotations

import uuid

from platform.schema import NeedsHumanReview


async def enqueue_escalation(
    *,
    campaign_id: str,
    post_id: str,
    comment_id: str,
    reason: str,
    comment_text: str = "",
    session_factory=None,
) -> str:
    """Record one escalation to the needs_human_review table. Returns its id."""
    from platform.db import get_session_factory

    factory = session_factory or get_session_factory()
    row_id = f"esc_{uuid.uuid4().hex[:12]}"
    await _ensure_table()
    async with factory() as session:
        session.add(
            NeedsHumanReview(
                id=row_id,
                campaign_id=campaign_id,
                post_id=post_id,
                comment_id=comment_id,
                reason=reason,
                comment_text=comment_text,
            )
        )
        await session.commit()
    return row_id


async def _ensure_table() -> None:
    """Create the needs_human_review table idempotently (mirrors MemoryStore)."""
    from platform.db import get_engine

    async with get_engine().begin() as conn:
        await conn.run_sync(
            lambda sync_conn: NeedsHumanReview.__table__.create(sync_conn, checkfirst=True)
        )


async def list_escalations(campaign_id: str | None = None, *, session_factory=None) -> list[dict]:
    """Return open (unresolved) escalations, newest first."""
    from sqlalchemy import select

    from platform.db import get_session_factory

    factory = session_factory or get_session_factory()
    async with factory() as session:
        query = (
            select(NeedsHumanReview)
            .where(NeedsHumanReview.resolved == 0)
            .order_by(NeedsHumanReview.created_at.desc())
        )
        if campaign_id:
            query = query.where(NeedsHumanReview.campaign_id == campaign_id)
        rows = (await session.execute(query)).scalars().all()
    return [
        {
            "id": r.id,
            "campaign_id": r.campaign_id,
            "post_id": r.post_id,
            "comment_id": r.comment_id,
            "reason": r.reason,
            "comment_text": r.comment_text,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
"""Comment routes: fetch a post's comments + post a Community Manager reply."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from platform.routes.posts import get_db
from platform.schema import Comment, Post, as_utc

router = APIRouter()


def _comment_to_dict(c: Comment) -> dict:
    return {
        "id": c.id,
        "post_id": c.post_id,
        "author_handle": c.author_handle,
        "text": c.text,
        "sentiment": c.sentiment,
        "is_sensitive": c.is_sensitive,
        "replied": c.replied,
    }


@router.get("/posts/{post_id}/comments")
async def get_comments(
    post_id: str,
    session: AsyncSession = Depends(get_db),
) -> dict:
    """**Fetch simulated comments for a post** (required endpoint)."""
    post = (
        await session.execute(select(Post).where(Post.id == post_id))
    ).scalar_one_or_none()
    if post is None:
        raise HTTPException(status_code=404, detail=f"unknown post: {post_id}")
    comments = (
        await session.execute(
            select(Comment).where(Comment.post_id == post_id).order_by(Comment.id)
        )
    ).scalars().all()
    return {
        "post_id": post_id,
        "comments": [_comment_to_dict(c) for c in comments],
        "unreplied_sensitive": sum(1 for c in comments if c.is_sensitive and not c.replied),
    }


@router.post("/posts/{post_id}/comments/{comment_id}/reply", status_code=201)
async def post_reply(
    post_id: str,
    comment_id: str,
    payload: dict,
    session: AsyncSession = Depends(get_db),
) -> dict:
    """Community Manager posts a reply to a comment (required endpoint).

    Body: {"reply_text": str, "author": str (optional, defaults to the
    platform's manager persona)}. Marks the comment `replied=True` and records
    the reply text in the response for trace logging on the agent side.
    """
    if not isinstance(payload.get("reply_text"), str) or not payload["reply_text"].strip():
        raise HTTPException(status_code=400, detail="missing or empty field: reply_text")

    comment = (
        await session.execute(
            select(Comment).where(Comment.id == comment_id, Comment.post_id == post_id)
        )
    ).scalar_one_or_none()
    if comment is None:
        raise HTTPException(status_code=404, detail=f"unknown comment: {comment_id}")

    reply = Comment(
        id=f"cmt_{uuid.uuid4().hex[:12]}",
        post_id=post_id,
        author_handle=payload.get("author", "brand_team"),
        text=payload["reply_text"].strip(),
        # A reply from the brand is stored as a neutral comment row — this
        # keeps one table for the whole thread (matches the doc's schema).
        sentiment="neutral",
        is_sensitive=False,
        replied=True,
    )
    session.add(reply)
    comment.replied = True
    await session.commit()
    return {"status": "replied", "reply": _comment_to_dict(reply)}

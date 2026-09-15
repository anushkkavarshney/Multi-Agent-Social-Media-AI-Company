"""Local memory store for past campaign learnings (architecture doc section 10).

Embedding strategy: a pure-Python, deterministic char-n-gram hash →
normalized vector.  No hosted embedding API, no extra daemon pull needed for
tests.  The architecture doc permits this ("local embedding or offline
deterministic"); the Ollama nomic-embed-text model is the intended upgrade
path once the demo's plumbing is proven.  When used for retrieval, cosine
similarity between the query vector and each stored record vector drives
ranking.

Persistence: SQLite table ``memory_records`` via platform/schema.py (created
idempotently on first use, mirroring MessageBus._ensure_schema).  Keeping the
table in the same DB keeps the system single-file and avoids yet another
migration path.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import uuid
from datetime import datetime, timezone

from platform.db import get_session_factory, get_engine
from platform.schema import MemoryRecord


# --- deterministic local embedding -----------------------------------------

_EMBED_DIM = 256  # fixed vector size; cosine needs consistent dims


def _embed(text: str) -> list[float]:
    """Deterministic local embedding via char 3-gram hashing.

    Each unique 3-gram maps to a dimension index; the sign of its SHA-256
    determines +1/-1.  The resulting vector is L2-normalized.  The quality is
    far below a learned embedding, but it is (a) deterministic across runs,
    (b) requires zero external dependencies, and (c) the retrieval quality
    threshold for "1 campaign report out of 1" is trivially met today.
    """
    vec = [0.0] * _EMBED_DIM
    grams = [text[i : i + 3] for i in range(len(text) - 2)]
    if not grams:
        return vec
    for g in grams:
        h = hashlib.sha256(g.encode()).digest()
        idx = struct.unpack("<I", h[:4])[0] % _EMBED_DIM
        vec[idx] += 1.0 if h[4] & 0x80 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    # inputs are L2-normalized so dot == cosine; clamp for float safety
    return max(-1.0, min(1.0, dot))


def _payload_vector(payload) -> list[float]:
    """Vector_json is a JSON column — SQLAlchemy already returns it as a dict."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        return payload.get("vector", []) or []
    return []


# --- public API -------------------------------------------------------------

class MemoryStore:
    def __init__(self, session_factory=None):
        self._factory = session_factory or get_session_factory()

    async def _ensure_table(self) -> None:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: MemoryRecord.__table__.create(sync_conn, checkfirst=True)
            )

    async def store_report(
        self,
        campaign_id: str,
        patterns_found: list[str],
        recommendations: list[dict],
        comment_sentiment_summary: str,
        week_number: int = 1,
    ) -> str:
        """Embed a report's learnings and persist them.  Returns the record id."""
        await self._ensure_table()

        text = (
            "PATTERNS FOUND:\n" + "\n".join(patterns_found) + "\n\n"
            "RECOMMENDATIONS:\n"
            + "\n".join(
                f"- {r['change']} (evidence: {r['evidence']}; expected: {r['expected_effect']})"
                for r in recommendations
            )
            + f"\n\nSENTIMENT: {comment_sentiment_summary}"
        )
        vec = _embed(text)
        record_id = f"mem_{uuid.uuid4().hex[:12]}"
        async with self._factory() as session:
            session.add(
                MemoryRecord(
                    id=record_id,
                    campaign_id=campaign_id,
                    source_week=week_number,
                    text=text,
                    vector_json={"vector": vec},
                )
            )
            await session.commit()
        return record_id

    async def retrieve_top_k(
        self,
        query: str,
        channels: list[str] | None = None,
        k: int = 3,
    ) -> list[str]:
        """Return up to k most-relevant learning texts (cosine ranking).

        For k >= current record count all records are returned ranked.
        ``channels`` is accepted for future filtering but currently unused
        (only 1 report exists on Day 3).
        """
        await self._ensure_table()
        from sqlalchemy import select

        async with self._factory() as session:
            rows = (
                await session.execute(select(MemoryRecord).order_by(MemoryRecord.created_at.desc()))
            ).scalars().all()

        if not rows:
            return []
        q_vec = _embed(query)
        scored = [
            (_cosine(q_vec, _payload_vector(r.vector_json)), r)
            for r in rows
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [r.text for _, r in scored[:k]]
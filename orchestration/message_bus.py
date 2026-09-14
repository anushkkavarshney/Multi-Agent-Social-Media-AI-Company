"""Agent-to-agent message bus (architecture doc section 7, base_agent spec).

Contract (from the task spec):
    - publish(message), subscribe(agent_name), history(campaign_id)
    - Every message is a STRUCTURED record:
      {from_agent, to_agent, campaign_id, payload, timestamp, message_type}
      — never free text, so the conversation is machine-replayable.

Why persistence lives in TWO places:
    - SQLite (`message_bus` table, platform/schema.py) is the SOURCE OF
      TRUTH: append-only rows survive restarts, and history() replays the
      full conversation for any campaign. This is what makes the agent
      conversation auditable after the fact.
    - logs/agent_trace.jsonl gets a mirror line per message because that
      file is the deliverable the write-up quotes ("show the agent
      conversation"). It is written via llm/trace.py so model calls and bus
      traffic interleave in one chronological file.

Live delivery: subscribe() hands the agent an asyncio.Queue that publish()
pushes to when the consumer runs in this process. The queue is a
CONVENIENCE, not the record — a slow or absent subscriber never loses a
message, because the DB write happens first.

Ordering: SQLite's autoincrement `seq` is the global order. history()
returns messages ordered by seq, which under the sequential agent runner
(the only execution model on Day 2) is exactly the causal conversation
order.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from llm.trace import append_event
from platform.schema import MessageBusRecord


class AgentMessage(BaseModel):
    """One structured bus record. This is the unit of agent conversation."""

    message_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    from_agent: str = Field(..., min_length=1)
    to_agent: str = Field(..., min_length=1)
    campaign_id: str = Field(..., min_length=1)
    message_type: str = Field(
        ...,
        min_length=1,
        description='e.g. "brief_received", "posts_rejected", "verdict" — the verb of the handoff.',
    )
    # MUST be JSON-serializable. Pydantic producers pass model_dump(mode="json")
    # output; keeping the bus schema-free here (dict, not BaseModel) is
    # deliberate: the bus must never reject a legitimate payload because a
    # contract evolved mid-conversation.
    payload: dict = Field(default_factory=dict)


class MessageBus:
    """Append-only bus backed by SQLite, mirrored to the JSONL trace."""

    def __init__(self, session_factory=None):
        # Injectable session factory (tests point it at a throwaway DB);
        # default is the platform's process-wide factory.
        if session_factory is None:
            from platform.db import get_session_factory

            session_factory = get_session_factory()
        self._session_factory = session_factory
        # agent_name -> asyncio.Queue of AgentMessage (in-process live delivery)
        self._subscribers: dict[str, asyncio.Queue] = {}
        # Schema bootstrap flag: the bus creates its own table on first use
        # (idempotent, checkfirst=True) so ANY consumer — tests, the dry-run
        # script, the API — can publish without booting the FastAPI app first.
        self._schema_ready = False

    async def _ensure_schema(self) -> None:
        """Create the message_bus table if missing. Exactly once per instance.

        Deliberately creates ONLY its own table (not the whole platform
        schema): the bus must not be the thing that seeds channels — that is
        the platform's startup responsibility.
        """
        if self._schema_ready:
            return
        from platform.db import get_engine

        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: MessageBusRecord.__table__.create(sync_conn, checkfirst=True)
            )
        self._schema_ready = True

    # --- publish -------------------------------------------------------------

    async def publish(self, message: AgentMessage) -> AgentMessage:
        """Record one message durably, mirror it to the trace, wake subscribers.

        Order matters: DB first (the record exists even if the trace file is
        unwritable), then JSONL mirror (observability), then queues (live
        delivery). A failure in the later steps must not lose the record.
        """
        await self._ensure_schema()
        async with self._session_factory() as session:
            session.add(
                MessageBusRecord(
                    message_id=message.message_id,
                    timestamp=message.timestamp,
                    from_agent=message.from_agent,
                    to_agent=message.to_agent,
                    campaign_id=message.campaign_id,
                    message_type=message.message_type,
                    payload_json=message.payload,
                )
            )
            await session.commit()

        append_event(
            {
                "event": "bus_message",
                "ts": message.timestamp.isoformat(),
                "message_id": message.message_id,
                "from_agent": message.from_agent,
                "to_agent": message.to_agent,
                "campaign_id": message.campaign_id,
                "message_type": message.message_type,
                "payload": message.payload,
            }
        )

        queue = self._subscribers.get(message.to_agent)
        if queue is not None:
            # A subscriber that never drains its queue must not block the
            # sender — drop-oldest keeps memory bounded in the demo runner.
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(message)
        return message

    # --- subscribe -----------------------------------------------------------

    def subscribe(self, agent_name: str) -> asyncio.Queue:
        """Register an in-process queue for live delivery to `agent_name`.

        Returns the (existing) queue on repeat subscription so double
        registration in a test setup can't silently fork delivery.
        """
        if agent_name not in self._subscribers:
            self._subscribers[agent_name] = asyncio.Queue(maxsize=256)
        return self._subscribers[agent_name]

    # --- history / replay ------------------------------------------------------

    async def history(self, campaign_id: str) -> list[AgentMessage]:
        """Replay the conversation for one campaign, in global seq order.

        Reads from SQLite, NOT from any in-memory state — this is what makes
        the conversation survive a restart (the dry-run script can replay
        yesterday's messages with no other process running).
        """
        await self._ensure_schema()
        from sqlalchemy import select

        async with self._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(MessageBusRecord)
                        .where(MessageBusRecord.campaign_id == campaign_id)
                        .order_by(MessageBusRecord.seq.asc())
                    )
                )
                .scalars()
                .all()
            )
        return [
            AgentMessage(
                message_id=r.message_id,
                timestamp=r.timestamp.replace(tzinfo=timezone.utc) if r.timestamp.tzinfo is None else r.timestamp,
                from_agent=r.from_agent,
                to_agent=r.to_agent,
                campaign_id=r.campaign_id,
                message_type=r.message_type,
                payload=r.payload_json or {},
            )
            for r in rows
        ]

    async def count(self, campaign_id: str | None = None) -> int:
        """Row count, optionally per campaign — used by tests and the viewer CLI."""
        from sqlalchemy import func, select

        async with self._session_factory() as session:
            query = select(func.count(MessageBusRecord.message_id))
            if campaign_id is not None:
                query = query.where(MessageBusRecord.campaign_id == campaign_id)
            return (await session.execute(query)).scalar_one()


def make_message(
    from_agent: str,
    to_agent: str,
    campaign_id: str,
    message_type: str,
    payload: dict | None = None,
) -> AgentMessage:
    """Convenience constructor keeping call sites one-liners."""
    return AgentMessage(
        from_agent=from_agent,
        to_agent=to_agent,
        campaign_id=campaign_id,
        message_type=message_type,
        payload=payload or {},
    )

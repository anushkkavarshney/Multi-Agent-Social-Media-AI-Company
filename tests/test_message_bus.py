"""Message bus tests.

The bus is the "show the agent conversation" requirement made real, so the
tests assert its three critical properties:
    1. history(campaign_id) returns ONLY that campaign's messages, in
       publish order (this is what the write-up replays).
    2. The conversation survives a RESTART: a brand-new bus instance (fresh
       engine, same DB file) reads the same history — SQLite is the record,
       not process memory.
    3. subscribe() queues get live delivery without blocking or losing the
       durable record.
Plus: the JSONL mirror line lands in agent_trace.jsonl interleaved with
LLM-call events (the file the reviewer actually reads).
"""

import json

import pytest

from orchestration.message_bus import AgentMessage, MessageBus, make_message


@pytest.fixture()
def bus_env(tmp_path, monkeypatch):
    """Isolated DB + trace file per test (the bus writes to both)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "bus_test.db"))
    monkeypatch.setenv("TRACE_LOG_PATH", str(tmp_path / "agent_trace.jsonl"))
    from config.settings import get_settings
    from platform.db import reset_engine_for_tests

    get_settings.cache_clear()
    reset_engine_for_tests()  # a previous test's engine may hold the old path
    yield tmp_path
    get_settings.cache_clear()
    reset_engine_for_tests()


async def test_history_returns_this_campaign_in_publish_order(bus_env):
    bus = MessageBus()
    for i in range(3):
        await bus.publish(
            make_message("orchestrator", "writer", "camp_A", f"msg_{i}", {"step": i})
        )
    # An interleaved message for a DIFFERENT campaign must not leak in:
    await bus.publish(make_message("orchestrator", "writer", "camp_B", "msg_other", {}))

    history = await bus.history("camp_A")
    assert [m.message_type for m in history] == ["msg_0", "msg_1", "msg_2"]
    assert all(m.campaign_id == "camp_A" for m in history)
    assert [m.payload["step"] for m in history] == [0, 1, 2]


async def test_history_survives_a_restart(bus_env):
    """The core durability claim: fresh bus object, same DB file, full history."""
    bus = MessageBus()
    await bus.publish(make_message("writer", "compliance", "camp_X", "posts_drafted", {"n": 1}))
    await bus.publish(make_message("compliance", "orchestrator", "camp_X", "compliance_verdict", {"verdict": "reject"}))

    # Simulate a restart: new MessageBus (and a fresh engine over the file).
    from platform.db import reset_engine_for_tests

    reset_engine_for_tests()
    reborn_bus = MessageBus()
    history = await reborn_bus.history("camp_X")
    assert [m.message_type for m in history] == ["posts_drafted", "compliance_verdict"]
    assert history[1].payload["verdict"] == "reject"


async def test_subscribe_queues_receive_published_messages(bus_env):
    bus = MessageBus()
    queue = bus.subscribe("writer")
    # Double subscription returns the SAME queue (no forked delivery):
    assert bus.subscribe("writer") is queue

    sent = await bus.publish(make_message("orchestrator", "writer", "camp_A", "campaign_strategy_filled"))
    received = queue.get_nowait()
    assert received.message_id == sent.message_id
    assert received.payload == {}  # published empty, received empty


async def test_unsubscribed_recipient_loses_nothing_durable(bus_env):
    """Publishing to an agent with no live queue still records the message —
    delivery is a convenience; the record is the contract."""
    bus = MessageBus()
    await bus.publish(make_message("a", "b", "camp_A", "hello", {}))
    assert await bus.count("camp_A") == 1
    # And the count() without a campaign filter counts everything:
    assert await bus.count() == 1


async def test_jsonl_mirror_records_the_handoff(bus_env):
    bus = MessageBus()
    await bus.publish(
        make_message("compliance", "writer", "camp_A", "posts_rejected", {"reasons": ["no guarantees"]})
    )
    trace_file = bus_env / "agent_trace.jsonl"
    events = [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]
    bus_events = [e for e in events if e.get("event") == "bus_message"]
    assert len(bus_events) == 1
    assert bus_events[0]["from_agent"] == "compliance"
    assert bus_events[0]["to_agent"] == "writer"
    assert bus_events[0]["message_type"] == "posts_rejected"
    assert bus_events[0]["payload"]["reasons"] == ["no guarantees"]


async def test_message_contract_shape(bus_env):
    """Every bus record carries the full structured envelope — the spec's
    {from_agent, to_agent, campaign_id, payload, timestamp, message_type}."""
    msg = make_message("a", "b", "camp_A", "t", {"k": "v"})
    for field in ("from_agent", "to_agent", "campaign_id", "payload", "timestamp", "message_type"):
        assert hasattr(msg, field), field
    stored = await MessageBus().publish(msg)
    assert stored.timestamp.tzinfo is not None  # aware UTC timestamps only

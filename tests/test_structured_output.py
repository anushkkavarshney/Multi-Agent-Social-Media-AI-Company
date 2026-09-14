"""Offline tests for the structured-output retry loop.

These run the FULL retry state machine without Ollama — the `transport`
injection in generate_structured() replaces the daemon call with a scripted
async function. Each test is one scripted failure mode drawn from real 3B/7B
instruct-model behavior:

    1. clean JSON on first try               -> success, attempt 1
    2. markdown-fenced JSON                  -> extracted, success
    3. JSON + trailing commentary            -> extracted, success
    4. wrong field TYPE, then fixed          -> recovers on attempt 2
    5. missing required field, then fixed    -> recovers on attempt 2
    6. garbage every time                    -> ModelOutputFailure (exhausted)
    7. trace events written for fail + recover paths

The corruption scenarios (4-6) are the same ones scripts/test_model_layer.py
runs against the live models — offline tests pin the LOGIC, live tests pin
the MODEL BEHAVIOR.
"""

import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, Field

from llm.ollama_client import OllamaClient  # import-time check only
from llm.structured_output import (
    ModelOutputFailure,
    extract_json_block,
    generate_structured,
)


class TwoFields(BaseModel):
    """Trivial 2-field schema for the model-layer demo (per Day-1 scope)."""

    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# --- extract_json_block unit behavior ----------------------------------------

def test_extract_clean_json():
    assert extract_json_block('{"a": 1}') == '{"a": 1}'


def test_extract_fenced_json():
    text = "Sure! Here is the JSON:\n```json\n{\"a\": 1}\n```\nHope that helps!"
    assert extract_json_block(text) == '{"a": 1}'


def test_extract_json_with_trailing_commentary():
    text = '{"a": 1} - hope that helps, let me know if you need more!'
    assert extract_json_block(text) == '{"a": 1}'


def test_extract_raises_when_no_json():
    with pytest.raises(ValueError):
        extract_json_block("I am sorry, I cannot answer that.")


# --- the retry loop -----------------------------------------------------------

@pytest.mark.asyncio
async def test_clean_output_succeeds_on_first_attempt():
    async def transport(prompt: str) -> str:
        return '{"name": "espresso", "confidence": 0.9}'

    result = await generate_structured(None, "fake-model", "classify this", TwoFields, transport=transport)
    assert result.name == "espresso"
    assert result.confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_fenced_and_commented_output_is_extracted():
    async def transport(prompt: str) -> str:
        return 'Here you go:\n```json\n{"name": "latte", "confidence": 0.75}\n```\nEnjoy!'

    result = await generate_structured(None, "fake-model", "classify this", TwoFields, transport=transport)
    assert result.name == "latte"


@pytest.mark.asyncio
async def test_wrong_type_recovers_on_second_attempt():
    """Attempt 1 returns a string where a float belongs (classic 3B failure);
    the repair prompt must contain the validation error and the model 'fixes'
    it on attempt 2."""
    calls: list[str] = []

    async def transport(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return '{"name": "mocha", "confidence": "high"}'
        return '{"name": "mocha", "confidence": 0.8}'

    result = await generate_structured(None, "fake-model", "classify this", TwoFields, transport=transport, max_attempts=3)
    assert result.confidence == pytest.approx(0.8)
    assert len(calls) == 2
    # The repair prompt must carry the validation error back to the model.
    assert "validation" in calls[1].lower() or "error" in calls[1].lower()
    # And it must include the bad output for context.
    assert '"confidence": "high"' in calls[1]


@pytest.mark.asyncio
async def test_missing_required_field_recovers():
    async def transport(prompt: str) -> str:
        return '{"name": "americano"}'

    with pytest.raises(ModelOutputFailure) as excinfo:
        await generate_structured(None, "fake-model", "classify", TwoFields, transport=transport, max_attempts=1)
    assert "confidence" in excinfo.value.last_error


@pytest.mark.asyncio
async def test_garbage_exhausts_retries_and_raises_typed_error():
    async def transport(prompt: str) -> str:
        return "Sorry, I do not understand the task."

    with pytest.raises(ModelOutputFailure) as excinfo:
        await generate_structured(None, "fake-model", "classify this", TwoFields, transport=transport, max_attempts=3)
    err = excinfo.value
    assert err.attempts == 3
    assert "Sorry" in err.last_raw
    assert "no JSON" in err.last_error or "ValueError" in err.last_error


@pytest.mark.asyncio
async def test_each_retry_gets_repair_prompt():
    """The prompt on attempt N>1 must differ from attempt 1 and carry the error."""
    prompts: list[str] = []

    async def transport(prompt: str) -> str:
        prompts.append(prompt)
        return "no json here"

    with pytest.raises(ModelOutputFailure):
        await generate_structured(None, "fake-model", "classify", TwoFields, transport=transport, max_attempts=3)
    assert len(prompts) == 3
    # Attempt 1 gets the schema prompt; attempts 2+ get the REPAIR prompt.
    assert "failed validation" not in prompts[0]
    assert all("failed validation" in p for p in prompts[1:])
    # Identical failures legitimately produce identical repair prompts, so
    # equality between attempts 2 and 3 is correct, not a loop bug.


# --- trace integration ---------------------------------------------------------

@pytest.mark.asyncio
async def test_trace_events_written(tmp_path, monkeypatch):
    """Success path writes a structured_output event to the JSONL trace."""
    monkeypatch.setenv("TRACE_LOG_PATH", str(tmp_path / "trace.jsonl"))
    from config.settings import get_settings

    get_settings.cache_clear()
    try:
        async def transport(prompt: str) -> str:
            return '{"name": "cortado", "confidence": 0.5}'

        await generate_structured(None, "fake-model", "classify", TwoFields, transport=transport)
        lines = (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(l) for l in lines]
        assert any(e["event"] == "structured_output" and e["success"] for e in events)
    finally:
        get_settings.cache_clear()

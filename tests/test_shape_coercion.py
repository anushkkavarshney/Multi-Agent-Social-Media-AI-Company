"""Regression tests for the shape-coercion layer and example-bearing prompts.

These pin the EXACT failure modes observed live on 2026-09-15, when
qwen2.5:7b-instruct failed the writer's revise pass 3/3:

    1. arrays emitted as numeric-keyed objects: {"posts": [{"0": {...}}]}
    2. list fields emitted as one string:        "hashtags": "#a #b #c"
    3. both at once; recovered only after the schema was shown as an EXAMPLE

The coercion tests assert the layer fixes CONTAINERS only — content is never
invented (an empty string for a list field stays invalid on purpose). The
prompt tests assert the model always SEES the nested shape: the field-list
hint was the root cause (it described types but never showed the nesting).
"""

import json

import pytest
from pydantic import BaseModel, Field

from agents.writer_agent import PostDraftList
from llm.structured_output import (
    _coerce,
    _shape_example,
    build_repair_prompt,
    build_schema_prompt,
    generate_structured,
)

# The real failing schema — not a toy model, so these tests guard the exact
# contract the live run broke.
POST_DRAFT_DEFS = PostDraftList.model_json_schema().get("$defs", {})


@pytest.fixture(autouse=True)
def _trace_to_tmp(tmp_path, monkeypatch):
    """Keep retry/coercion trace events out of the repo's real trace log."""
    monkeypatch.setenv("TRACE_LOG_PATH", str(tmp_path / "trace.jsonl"))
    from config.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --- _coerce unit behavior ------------------------------------------------------


def test_numeric_keyed_object_becomes_list():
    data = {"0": {"a": 1}, "1": {"a": 2}, "2": {"a": 3}}
    out, changed = _coerce(data, {"type": "array", "items": {}}, {})
    assert out == [{"a": 1}, {"a": 2}, {"a": 3}]  # numeric order, not dict order luck
    assert changed is True


def test_numeric_keys_nested_inside_valid_container():
    """The exact live shape: posts was a real list, items were numeric-keyed."""
    data = {"posts": [{"0": {"pillar": "p", "copy": "c"}}]}
    schema = {
        "type": "object",
        "properties": {
            "posts": {"type": "array", "items": {"$ref": "#/$defs/PostDraft"}}
        },
    }
    out, changed = _coerce(data, schema, POST_DRAFT_DEFS)
    assert out["posts"][0]["copy"] == "c"
    assert changed is True


def test_comma_string_becomes_list_for_array_fields():
    # Uses the REAL PostDraft $defs: the coercion reads the field spec from
    # the resolved item schema, so items must carry `properties` (a bare
    # {"type": "object"} has none — that fixture mistake, not the layer).
    data = {"posts": [{"pillar": "p", "channel": "ch", "copy": "c", "hashtags": "#a, #b, #c"}]}
    schema = {
        "type": "object",
        "properties": {
            "posts": {"type": "array", "items": {"$ref": "#/$defs/PostDraft"}}
        },
    }
    out, changed = _coerce(data, schema, POST_DRAFT_DEFS)
    assert out["posts"][0]["hashtags"] == ["#a", "#b", "#c"]
    assert changed is True


def test_empty_string_list_field_stays_invalid():
    """Coercion must not fabricate an empty list — that hides a real failure."""
    data = {"posts": [{"pillar": "p", "copy": "c", "hashtags": ""}]}
    schema = {
        "type": "object",
        "properties": {"posts": {"type": "array", "items": {"type": "object"}}},
    }
    out, changed = _coerce(data, schema, {})
    assert out["posts"][0]["hashtags"] == ""  # untouched -> validation still fails
    assert changed is False


def test_content_is_never_modified():
    data = {"posts": [{"pillar": "P", "copy": "keep  me", "hashtags": ["#x"]}]}
    schema = {
        "type": "object",
        "properties": {
            "posts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pillar": {"type": "string"},
                        "copy": {"type": "string"},
                        "hashtags": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    }
    out, changed = _coerce(data, schema, {})
    assert out == data
    assert changed is False


# --- prompts show the full nested shape -------------------------------------------


def test_shape_example_renders_nested_arrays_of_objects():
    example = json.loads(_shape_example(PostDraftList))
    assert isinstance(example["posts"], list)
    item = example["posts"][0]
    # The inner fields MUST be visible — their absence was the root cause.
    assert set(item.keys()) == {"pillar", "channel", "copy", "hashtags", "cta"}
    assert item["hashtags"] == ["first item", "second item"]


def test_schema_prompt_states_array_rules_and_example():
    prompt = build_schema_prompt("write posts", PostDraftList)
    assert '"posts":' in prompt
    assert "NEVER" in prompt  # the numeric-keyed-object prohibition
    assert '"cta"' in prompt


def test_repair_prompt_repeats_the_example():
    """The model cannot be assumed to remember attempt 1's layout — the
    repair prompt must show the exact shape again next to the error."""
    base = build_schema_prompt("write posts", PostDraftList)
    repair = build_repair_prompt(base, '{"posts": [{"0": {}}]}', "posts.0.pillar: Field required")
    assert '{"posts": [{"0": {}}]}' in repair  # bad output echoed for context
    assert "EXACT required shape" in repair
    assert '"pillar"' in repair


def test_repair_error_is_short_not_a_pydantic_dump():
    """Live evidence: the raw 12-error pydantic string pushed the model into
    repeating the mistake. The repair error must be a few plain lines."""
    from pydantic import ValidationError

    bad = {"posts": [{"0": {"hashtags": ""}}]}
    with pytest.raises(ValidationError) as excinfo:
        PostDraftList.model_validate(bad)
    # _short_error is exercised via generate_structured; assert its raw input
    # is the noisy thing and that our summary compresses it.
    assert "input_value" in str(excinfo.value)  # the noise we do NOT forward


# --- the full retry loop over scripted live shapes ---------------------------------


@pytest.mark.asyncio
async def test_exact_live_revise_failure_recovers_on_attempt_2():
    """Attempt 1 replays the EXACT model output from the 2026-09-15 crash
    (nested numeric keys + hashtags as a string). Coercion cannot rescue the
    empty string, so the loop must repair-prompt and succeed on attempt 2."""
    calls: list[str] = []

    bad = json.dumps(
        {
            "posts": [
                {
                    "0": {
                        "pillar": "Product Introduction",
                        "channel": "ch_shortform",
                        "copy": "Sip on savings: our budget espresso machine!",
                        "hashtags": "",
                    }
                }
            ]
        },
        ensure_ascii=False,
    )
    good = json.dumps(
        {
            "posts": [
                {
                    "pillar": "Product Introduction",
                    "channel": "ch_shortform",
                    "copy": "Sip on savings: our budget espresso machine!",
                    "hashtags": ["#espresso", "#studentlife", "#coffeetok"],
                    "cta": "Tag your study buddy",
                }
            ]
        }
    )

    async def transport(prompt: str) -> str:
        calls.append(prompt)
        return bad if len(calls) == 1 else good

    result = await generate_structured(
        None, "fake-model", "revise", PostDraftList, transport=transport
    )
    assert len(result.posts) == 1
    assert result.posts[0].hashtags == ["#espresso", "#studentlife", "#coffeetok"]
    assert len(calls) == 2
    # Repair prompt carries the bad output AND the shape example again.
    assert '"0"' in calls[1]
    assert '"pillar"' in calls[1]


@pytest.mark.asyncio
async def test_pure_container_mistake_rescued_without_second_call():
    """Numeric keys with otherwise-valid content: coercion rescues it on
    attempt 1 — no second model call, which on CPU saves ~2 minutes."""
    calls: list[str] = []

    rescueable = json.dumps(
        {
            "posts": [
                {
                    "0": {
                        "pillar": "p",
                        "channel": "ch_discussion",
                        "copy": "c",
                        "hashtags": ["#a", "#b"],
                        "cta": None,
                    }
                }
            ]
        }
    )

    async def transport(prompt: str) -> str:
        calls.append(prompt)
        return rescueable

    result = await generate_structured(
        None, "fake-model", "revise", PostDraftList, transport=transport
    )
    assert result.posts[0].copy == "c"
    assert len(calls) == 1  # the whole point: no retry burned


@pytest.mark.asyncio
async def test_repair_prompt_carries_the_coerced_error_not_stale_container_error():
    """Regression for the 2026-09-15 live failure: coercion unwraps {'0': {...}}
    containers but a field error remains (hashtags: ""). The repair prompt MUST
    tell the model the REAL remaining problem (hashtags) — the bug was it
    recycled the ORIGINAL container error ("posts.0.pillar Field required, input
    {'0': ...}"), which coercion had already fixed, so the model re-emitted the
    same container 3x and the retries were byte-identical."""
    calls: list[str] = []

    # Always the SAME bad shape: containers coercible, hashtags irrecoverable.
    bad = json.dumps(
        {
            "posts": [
                {
                    "0": {
                        "pillar": "p",
                        "channel": "ch",
                        "copy": "c",
                        "hashtags": "",
                    }
                }
            ]
        }
    )

    async def transport(prompt: str) -> str:
        calls.append(prompt)
        return bad

    with pytest.raises(Exception) as excinfo:
        await generate_structured(
            None, "fake-model", "revise", PostDraftList,
            transport=transport, max_attempts=2,
        )
    # The 2nd call is the repair prompt. It must carry the COERCED error
    # (hashtags), not the stale container error (pillar inside '0').
    assert "hashtags" in calls[1].lower()
    assert "valid list" in calls[1].lower()


# --- native format constraint -------------------------------------------------------


class _FakeClient:
    """Captures generate() kwargs so tests can assert the daemon-side contract."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict] = []

    async def generate(self, model, prompt, **kwargs):
        self.calls.append({"model": model, "prompt": prompt, **kwargs})
        return {"response": self.response}


@pytest.mark.asyncio
async def test_native_format_passes_nested_json_schema():
    """Ollama's `format` gets the FULL schema incl. $defs — that is what makes
    {"0": {...}} structurally impossible on daemons that support it."""
    good = json.dumps(
        {
            "posts": [
                {
                    "pillar": "p",
                    "channel": "ch",
                    "copy": "c",
                    "hashtags": ["#a"],
                    "cta": None,
                }
            ]
        }
    )
    client = _FakeClient(good)
    result = await generate_structured(
        client, "fake-model", "write", PostDraftList, transport=None
    )
    assert result.posts[0].copy == "c"
    sent = client.calls[0]["format_schema"]
    assert "$defs" in sent and "PostDraft" in sent["$defs"]
    assert sent["properties"]["posts"]["type"] == "array"


@pytest.mark.asyncio
async def test_native_format_can_be_disabled():
    """Older daemons / mock transports: use_native_format=False must omit it."""
    good = json.dumps(
        {
            "posts": [
                {
                    "pillar": "p",
                    "channel": "ch",
                    "copy": "c",
                    "hashtags": ["#a"],
                    "cta": None,
                }
            ]
        }
    )
    client = _FakeClient(good)
    await generate_structured(
        client, "fake-model", "write", PostDraftList, use_native_format=False
    )
    assert client.calls[0]["format_schema"] is None

"""Structured-output enforcement: generate -> parse -> validate -> retry.

The loop (architecture doc section 9, item 2):
    1. Build a prompt that shows the model the exact JSON schema.
    2. Ask Ollama for a response (passing the JSON schema via Ollama's native
       `format` parameter where supported, PLUS our own parse-validate pass —
       local models still emit prose, markdown fences, or trailing commentary
       around the JSON, so client-side validation remains mandatory).
    3. On ValidationError: re-prompt with THE VALIDATION ERROR APPENDED so the
       model can correct itself, up to MAX attempts (settings.max_structured_retries).
    4. On final failure: raise ModelOutputFailure (typed) — the caller (Day 2
       orchestrator) catches it and routes the unit to a human/queue instead
       of crashing the pipeline.

Extract-before-parse order matters: strip markdown fences, then scan for the
outermost {...} or [...] block. This handles the two failure modes that cover
most malformed outputs from 3B/7B instruct models.
"""

import json
import re
from datetime import datetime, timezone

from pydantic import BaseModel, ValidationError

from config.settings import get_settings
from llm.ollama_client import OllamaClient, OllamaError
from llm.trace import append_event

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class ModelOutputFailure(Exception):
    """All retry attempts exhausted without schema-valid output.

    Carries the last raw text + last validation error so callers can log a
    precise failure record (and the write-up can quote real failure modes).
    """

    def __init__(self, message: str, last_raw: str, last_error: str, attempts: int):
        super().__init__(message)
        self.last_raw = last_raw
        self.last_error = last_error
        self.attempts = attempts


def extract_json_block(text: str) -> str:
    """Return the outermost JSON object/array substring found in `text`.

    Order of attempts: fenced ```json block first, then raw scan. Raises
    ValueError when nothing JSON-like exists at all.
    """
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    raise ValueError("no JSON object or array found in model output")


def _schema_hint(model_cls: type[BaseModel]) -> str:
    """Compact human-readable field list from the Pydantic model."""
    lines = ["{"]
    props = model_cls.model_json_schema().get("properties", {})
    required = set(model_cls.model_json_schema().get("required", []))
    for name, spec in props.items():
        typ = spec.get("type", json.dumps(spec.get("anyOf", "?")))
        req = "required" if name in required else "optional"
        lines.append(f'  "{name}": <{typ}>  // {req}')
    lines.append("}")
    return "\n".join(lines)


def build_schema_prompt(instruction: str, model_cls: type[BaseModel]) -> str:
    """The generation prompt: instruction + exact schema + hard output rule."""
    return (
        f"{instruction}\n\n"
        "Respond with ONLY a JSON object matching this schema exactly "
        "(no markdown, no commentary, no code fences):\n"
        f"{_schema_hint(model_cls)}\n"
        "Output the raw JSON object and nothing else."
    )


def build_repair_prompt(previous_prompt: str, bad_output: str, validation_error: str) -> str:
    """Re-prompt: original task + the bad output + the precise validation error."""
    return (
        f"{previous_prompt}\n\n"
        f"Your previous response was:\n{bad_output}\n\n"
        f"It failed validation with this error:\n{validation_error}\n\n"
        "Return the corrected JSON object only. Same schema, no commentary."
    )


async def generate_structured(
    client: OllamaClient,
    model: str,
    instruction: str,
    schema_cls: type[BaseModel],
    *,
    max_attempts: int | None = None,
    options: dict | None = None,
    transport=None,
) -> BaseModel:
    """Generate, parse, validate against `schema_cls`; retry with the error.

    Args:
        client: an OllamaClient (or any object with .generate()).
        model: Ollama model tag.
        instruction: the task prompt.
        schema_cls: the Pydantic model to validate against.
        max_attempts: override settings.max_structured_retries (tests do).
        transport: injectable generation function
            (async (prompt:str) -> str) for tests — replaces the daemon call.

    Returns: a validated instance of schema_cls.
    Raises: ModelOutputFailure after `max_attempts` failed attempts;
            OllamaError propagates (connection/model problems are NOT retried —
            they are environmental, not output-format problems).
    """
    if max_attempts is None:
        max_attempts = get_settings().max_structured_retries

    prompt = build_schema_prompt(instruction, schema_cls)
    last_raw, last_error = "", ""

    for attempt in range(1, max_attempts + 1):
        try:
            if transport is not None:
                raw = await transport(prompt)
            else:
                response = await client.generate(model, prompt, raw=True, options=options)
                raw = response.get("response", "")
        except OllamaError:
            # Environmental failure — re-raising is correct; retrying a down
            # daemon inside a format-retry loop would just burn 3x latency.
            raise

        last_raw = raw
        try:
            json_text = extract_json_block(raw)
            data = json.loads(json_text)
            validated = schema_cls.model_validate(data)
            append_event(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "structured_output",
                    "model": model,
                    "schema": schema_cls.__name__,
                    "attempt": attempt,
                    "success": True,
                }
            )
            return validated
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            last_error = (
                f"{exc.__class__.__name__}: {exc}"
                if not isinstance(exc, ValidationError)
                else str(exc)
            )
            append_event(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": "structured_output_retry",
                    "model": model,
                    "schema": schema_cls.__name__,
                    "attempt": attempt,
                    "success": False,
                    "error": last_error[:500],
                    "raw_preview": raw[:300],
                }
            )
            if attempt < max_attempts:
                prompt = build_repair_prompt(
                    build_schema_prompt(instruction, schema_cls), raw, last_error
                )

    raise ModelOutputFailure(
        f"model {model} failed to produce schema-valid {schema_cls.__name__} "
        f"after {max_attempts} attempts",
        last_raw=last_raw,
        last_error=last_error,
        attempts=max_attempts,
    )

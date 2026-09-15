"""Structured-output enforcement: generate -> parse -> coerce -> validate -> retry.

The loop (architecture doc section 9, item 2):
    1. Build a prompt that shows the model the EXACT JSON shape — as a filled-in
       example, not a field list (small models copy examples far more reliably
       than they read schema descriptions).
    2. Constrain generation with Ollama's native `format` (JSON-schema grammar)
       where supported — makes malformed JSON structurally impossible on the
       daemon side. Client-side validation REMAINS mandatory: it is the only
       guard that works on older daemons, mock transports, and semantic
       (not just syntactic) violations.
    3. On ValidationError: try a shape-coercion pass (see _coerce) — two live
       failure modes from qwen2.5 on 2026-09-15 motivated this: arrays emitted
       as {"0": {...}, "1": {...}} and list fields emitted as "a, b" strings.
       Coercion fixes CONTAINERS only; content is never invented.
    4. Still failing: re-prompt with a SHORT plain-language error (a raw
       12-line pydantic dump is noise to a 7B) plus the exact shape example.
       Up to MAX attempts (settings.max_structured_retries).
    5. On final failure: raise ModelOutputFailure (typed) — the caller catches
       it and routes the unit to a human/queue instead of crashing the pipeline.

Extract-before-parse order matters: strip markdown fences, then scan for the
outermost {...} or [...] block. With native format the output is already clean
JSON, so extraction is a no-op there — the scan exists for transports that
can't constrain output (mocks in tests, older daemons).
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


# --- schema example rendering -------------------------------------------------
#
# Why an EXAMPLE and not a field list: the 2026-09-15 live run proved a 7B
# will emit the right fields but the wrong CONTAINER ({"0": {...}} instead of
# [ {...} ]) when the prompt describes types but never shows the nesting.
# A filled example demonstrates the nesting; "copy the shape" is the single
# easiest instruction an instruct model can follow.

def _resolve_ref(ref: str, defs: dict) -> dict:
    """Resolve a JSON-schema $ref ("#/$defs/PostDraft") against $defs."""
    return defs.get(ref.split("/")[-1], {})


def _example_for(spec: dict, defs: dict, depth: int = 0):
    """Minimal JSON example value for one schema node.

    depth guards against pathological recursive schemas (a model referencing
    itself) — 5 levels is far deeper than any contract in this repo.
    """
    if depth > 5:
        return "..."
    if "$ref" in spec:
        return _example_for(_resolve_ref(spec["$ref"], defs), defs, depth + 1)
    if "anyOf" in spec:
        # Optional fields render as their non-null variant: the example should
        # teach the PRIMARY shape, not the null shape.
        non_null = [s for s in spec["anyOf"] if s.get("type") != "null"]
        return _example_for(non_null[0], defs, depth + 1) if non_null else None
    t = spec.get("type")
    if t == "string":
        return "..."
    if t == "integer":
        return 1
    if t == "number":
        return 1.0
    if t == "boolean":
        return True
    if t == "array":
        items = spec.get("items", {})
        if items.get("type") == "string":
            # Two placeholder items teach "list of strings" better than [].
            return ["first item", "second item"]
        return [_example_for(items, defs, depth + 1)]
    if t == "object":
        return {
            name: _example_for(prop, defs, depth + 1)
            for name, prop in spec.get("properties", {}).items()
        }
    return "..."  # unknown node: keep the example syntactically valid


def _shape_example(model_cls: type[BaseModel]) -> str:
    """Pretty-printed minimal example of `model_cls`'s full JSON shape."""
    schema = model_cls.model_json_schema()
    example = _example_for(schema, schema.get("$defs", {}))
    return json.dumps(example, indent=2, ensure_ascii=False)


_SHAPE_RULES = (
    "Shape rules:\n"
    '- Arrays are JSON arrays of objects: [ { ... }, { ... } ] — NEVER an\n'
    '  object with numeric keys like { "0": { ... } }.\n'
    '- List fields (e.g. hashtags) are JSON arrays of strings: ["#tag1", "#tag2"].\n'
    "  Never a single string like \"#tag1 #tag2\".\n"
    "- Every field shown above is required in every object unless marked optional."
)


def build_schema_prompt(instruction: str, model_cls: type[BaseModel]) -> str:
    """The generation prompt: instruction + exact example + hard output rules."""
    return (
        f"{instruction}\n\n"
        "Respond with ONLY one JSON object of EXACTLY this shape "
        "(no markdown, no commentary, no code fences):\n"
        f"{_shape_example(model_cls)}\n\n"
        f"{_SHAPE_RULES}\n"
        "Output the raw JSON object and nothing else."
    )


def _short_error(exc: Exception) -> str:
    """Human-digestible error summary for the repair prompt.

    The raw str(ValidationError) for a list-of-objects schema is a dozen
    'Field required' blocks with input echoes — observed live to push qwen2.5
    into repeating the same mistake. Four plain lines plus the shape example
    (re-added by build_repair_prompt) is what actually gets corrected.
    """
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        lines = []
        for e in errors[:4]:
            loc = ".".join(str(x) for x in e.get("loc", [])) or "(root)"
            lines.append(f"{loc}: {e.get('msg', 'invalid')}")
        summary = "\n".join(lines)
        if len(errors) > 4:
            summary += f"\n... and {len(errors) - 4} more error(s) of the same kind."
        return summary
    return f"{exc.__class__.__name__}: {exc}"


def build_repair_prompt(previous_prompt: str, bad_output: str, validation_error: str) -> str:
    """Re-prompt: original task + bad output + SHORT error + the shape again.

    The example is re-included because the most common repair (fixing the
    container shape) needs the target nesting in view, and the model cannot
    be assumed to remember attempt 1's prompt layout.
    """
    return (
        f"{previous_prompt}\n\n"
        f"Your previous response was:\n{bad_output}\n\n"
        f"It failed validation with this error:\n{validation_error}\n\n"
        f"The EXACT required shape is:\n"
        "Return the corrected JSON object only. Same schema, no commentary."
    )


# --- shape coercion --------------------------------------------------------------

def _coerce(data, schema: dict, defs: dict):
    """Best-effort structural normalization of known small-model shapes.

    Returns (new_data, changed). Fixes CONTAINERS only — content is never
    invented:
        - {"0": x, "1": y} where an ARRAY belongs -> [x, y]  (live: qwen2.5:7b)
        - {"0": x} where an OBJECT belongs      -> x         (single-key unwrap;
            the live revise failure put ONE post inside a numeric envelope:
            "posts": [ {"0": {...post...}} ])
        - "a, b, c" in a list field             -> ["a", "b", "c"]
    An EMPTY string for a list field is deliberately left invalid: splitting it
    would fabricate an empty list where the schema demands items, hiding a real
    model failure from the retry loop. Ambiguous shapes (multiple numeric keys
    where a single object belongs) are left for validation to reject.
    """
    # Normalize the node: $ref -> target schema; anyOf (Optional fields) ->
    # the non-null variant, so the type checks below see the primary shape.
    if "$ref" in schema:
        schema = _resolve_ref(schema["$ref"], defs)
    if "anyOf" in schema:
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            schema = non_null[0]

    t = schema.get("type")

    # Numeric-keyed dicts: a mis-keyed sequence at ANY depth.
    if isinstance(data, dict) and data and all(str(k).isdigit() for k in data):
        ordered_values = [data[k] for k in sorted(data, key=lambda x: int(x))]
        if t == "array":
            items_schema = schema.get("items", {})
            return [_coerce(v, items_schema, defs)[0] for v in ordered_values], True
        if t == "object" and len(ordered_values) == 1:
            # One numeric envelope around one object: unwrap it.
            coerced, _ = _coerce(ordered_values[0], schema, defs)
            return coerced, True
        # Ambiguous — leave it; validation will reject and the loop re-prompts.
        return data, False

    if t == "object" and isinstance(data, dict):
        props = schema.get("properties", {})
        out: dict = {}
        changed = False
        for key, value in data.items():
            spec = props.get(key, {})
            if spec.get("type") == "array" and isinstance(value, str):
                # "#a, #b" -> ["#a", "#b"] (live failure: hashtags as one string)
                parts = [p.strip() for p in value.split(",") if p.strip()]
                if parts:
                    out[key], changed = parts, True
                else:
                    out[key] = value  # stay invalid on purpose; see docstring
            else:
                out[key], ch = _coerce(value, spec, defs)
                changed = changed or ch
        return out, changed
    if t == "array" and isinstance(data, list):
        items_schema = schema.get("items", {})
        out = []
        changed = False
        for item in data:
            fixed, ch = _coerce(item, items_schema, defs)
            out.append(fixed)
            changed = changed or ch
        return out, changed
    return data, False


# --- the retry loop ---------------------------------------------------------------

async def generate_structured(
    client: OllamaClient,
    model: str,
    instruction: str,
    schema_cls: type[BaseModel],
    *,
    max_attempts: int | None = None,
    options: dict | None = None,
    transport=None,
    use_native_format: bool = True,
) -> BaseModel:
    """Generate, parse, coerce, validate against `schema_cls`; retry with the error.

    Args:
        client: an OllamaClient (or any object with .generate()).
        model: Ollama model tag.
        instruction: the task prompt.
        schema_cls: the Pydantic model to validate against.
        max_attempts: override settings.max_structured_retries (tests do).
        transport: injectable generation function
            (async (prompt:str) -> str) for tests — replaces the daemon call.
        use_native_format: pass the JSON schema to Ollama's `format` parameter
            (daemon-side grammar constraint). Disable for daemons that reject
            `format` or transports that ignore it.

    Returns: a validated instance of schema_cls.
    Raises: ModelOutputFailure after `max_attempts` failed attempts;
            OllamaError propagates (connection/model problems are NOT retried —
            they are environmental, not output-format problems).
    """
    if max_attempts is None:
        max_attempts = get_settings().max_structured_retries

    schema_root = schema_cls.model_json_schema()
    defs = schema_root.get("$defs", {})
    prompt = build_schema_prompt(instruction, schema_cls)
    last_raw, last_error = "", ""

    for attempt in range(1, max_attempts + 1):
        try:
            if transport is not None:
                raw = await transport(prompt)
            else:
                response = await client.generate(
                    model,
                    prompt,
                    raw=True,
                    options=options,
                    format_schema=schema_root if use_native_format else None,
                )
                raw = response.get("response", "")
        except OllamaError:
            # Environmental failure — re-raising is correct; retrying a down
            # daemon inside a format-retry loop would just burn 3x latency.
            raise

        last_raw = raw
        try:
            json_text = extract_json_block(raw)
            data = json.loads(json_text)
            try:
                validated = schema_cls.model_validate(data)
            except ValidationError as raw_err:
                # One structural fix-up pass before burning a retry: if the
                # model had the right CONTENT in the wrong CONTAINER, coercion
                # rescues it without another 2-minute CPU inference call.
                coerced, changed = _coerce(data, schema_root, defs)
                if not changed:
                    raise raw_err
                try:
                    validated = schema_cls.model_validate(coerced)
                except ValidationError as coerced_err:
                    # Report the error against the COERCED data, not the
                    # original. Coercion only fixes containers; the errors
                    # that remain are field-level ("hashtags must be a valid
                    # list"), and that is exactly what the repair prompt must
                    # tell the model. Re-raising `raw_err` here recycles a
                    # stale container error ("pillar missing, input {'0':...}")
                    # that the coercion already fixed — observed live on
                    # 2026-09-15 as two byte-identical failed retries of the
                    # writer's revise pass.
                    raise coerced_err
                append_event(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "event": "structured_output_coerced",
                        "model": model,
                        "schema": schema_cls.__name__,
                        "attempt": attempt,
                        "success": True,
                        "raw_preview": raw[:200],
                    }
                )
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
            last_error = _short_error(exc)
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

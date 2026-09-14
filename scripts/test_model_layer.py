"""Live test of the LLM layer against Ollama (Day-1 item 5).

What it does, per model (primary + router from settings):
    CASE A "clean"      — well-formed task; expect success on attempt 1.
    CASE B "corrupted"  — the FIRST daemon response is deliberately corrupted
                          (truncated mid-JSON, closing brace dropped) before
                          it reaches the parser; the retry loop must catch the
                          validation error, re-prompt, and recover on the
                          model's second response. This proves the retry path
                          works against the REAL client, not a mock.
    CASE C "hostile"    — a prompt engineered to elicit malformed output
                          (asks for prose around the JSON). Whatever happens,
                          the layer must either validate or fail TYPED — never
                          crash with a raw stack trace.

Usage (from repo root, venv active):
    python scripts/test_model_layer.py
    python scripts/test_model_layer.py --models qwen2.5:3b-instruct

Exit codes: 0 ok | 2 Ollama unreachable | 3 models missing | 1 case failures
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Runnable-script bootstrap: when executed as a FILE, sys.path[0] is scripts/
# and the project root is not importable. Insert the repo root explicitly so
# `python scripts/test_model_layer.py` works from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings  # noqa: E402
from llm.ollama_client import (
    ModelNotFoundError,
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
)
from llm.structured_output import ModelOutputFailure, generate_structured
from llm.trace import _log_path  # reusing the settings-backed path helper
from pydantic import BaseModel, Field


class SentimentLabel(BaseModel):
    """The trivial 2-field schema required by the Day-1 scope."""

    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


INSTRUCTION = (
    "Classify the sentiment of this product review as positive or negative. "
    'Respond with JSON: {"label": "positive"|"negative", "confidence": 0.0-1.0}.\n'
    'Review: "The espresso machine arrived fast, pulls great shots, and the '
    'cleanup takes two minutes. Absolutely thrilled with it."'
)

HOSTILE_INSTRUCTION = (
    INSTRUCTION
    + "\n\nIMPORTANT: start your reply with a friendly one-sentence introduction "
    "before the JSON, and add a closing remark after it. Be conversational."
)


def corrupt(text: str) -> str:
    """Deterministic corruption: truncate mid-JSON and drop the closing brace.

    Simulates the most common real failure mode: generation cut off / model
    rambles past the JSON and the tail gets mangled.
    """
    cut = max(10, int(len(text) * 0.6))
    return text[:cut]


async def run_case(client: OllamaClient, model: str, name: str, instruction: str, corrupt_first: bool) -> dict:
    """Run one structured-output case; returns a result record for the table."""
    calls = {"n": 0}
    tokens = {"prompt": 0, "completion": 0}

    async def transport(prompt: str) -> str:
        calls["n"] += 1
        response = await client.generate(model, prompt, raw=True)
        tokens["prompt"] += response.get("prompt_eval_count") or 0
        tokens["completion"] += response.get("eval_count") or 0
        text = response.get("response", "")
        if corrupt_first and calls["n"] == 1:
            text = corrupt(text)
        return text

    try:
        result = await generate_structured(client, model, instruction, SentimentLabel, transport=transport, max_attempts=3)
        return {
            "case": name,
            "ok": True,
            "attempts": calls["n"],
            "tokens": dict(tokens),
            "answer": result.model_dump(),
            "error": None,
        }
    except ModelOutputFailure as exc:
        return {
            "case": name,
            "ok": False,
            "attempts": exc.attempts,
            "tokens": dict(tokens),
            "answer": None,
            "error": f"{exc} | last_error={exc.last_error[:120]}",
        }


def print_preflight_error(exc: Exception, models: list[str]) -> int:
    print("\n" + "=" * 70)
    print("MODEL LAYER TEST — CANNOT START")
    print("=" * 70)
    print(f"{exc.__class__.__name__}: {exc}")
    if isinstance(exc, ModelNotFoundError):
        return 3
    return 2


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None, help="override model list")
    args = parser.parse_args()

    settings = get_settings()
    models = args.models or [settings.primary_model, settings.router_model]

    async with OllamaClient() as client:
        # --- preflight: reachable + models pulled (fail with ACTIONABLE msg) ---
        try:
            await client._ensure_reachable()
            tags = await client._client.get("/api/tags")
            available = {m.get("name") for m in tags.json().get("models", [])}
            missing = [m for m in models if m not in available and m.split(":")[0] not in {a.split(":")[0] for a in available}]
            if missing:
                raise ModelNotFoundError(
                    "missing models: " + ", ".join(missing)
                    + "\nFix:\n" + "\n".join(f"  ollama pull {m}" for m in missing)
                )
        except (OllamaConnectionError, ModelNotFoundError) as exc:
            print_preflight_error(exc, models)
            return 2 if isinstance(exc, OllamaConnectionError) else 3

        results = []
        for model in models:
            print(f"\n=== {model} ===")
            cases = [
                ("clean", INSTRUCTION, False),
                ("corrupted-first-response", INSTRUCTION, True),
                ("hostile-prose-around-json", HOSTILE_INSTRUCTION, False),
            ]
            for name, instruction, do_corrupt in cases:
                r = await run_case(client, model, name, instruction, do_corrupt)
                results.append({**r, "model": model})
                status = "OK  " if r["ok"] else "FAIL"
                print(
                    f"  [{status}] {name:<28} attempts={r['attempts']} "
                    f"tokens(p/c)={r['tokens']['prompt']}/{r['tokens']['completion']}"
                )
                if r["ok"]:
                    print(f"          -> {json.dumps(r['answer'])}")
                else:
                    print(f"          -> {r['error']}")

        # --- summary + trace tail -------------------------------------------
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        passed = sum(1 for r in results if r["ok"])
        print(f"{passed}/{len(results)} cases succeeded")
        print("\nExpectations:")
        print("  - 'clean' cases: OK on attempt 1")
        print("  - 'corrupted-first-response': OK on attempt >=2 (retry recovered)")
        print("  - 'hostile': OK via extraction, or a TYPED ModelOutputFailure (never a crash)")
        print("\nLast 12 trace events (logs/agent_trace.jsonl):")
        try:
            lines = _log_path().read_text(encoding="utf-8").strip().splitlines()
            for line in lines[-12:]:
                e = json.loads(line)
                print(f"  {e['ts'][11:19]} {e['event']:<24} {e.get('model','')[:24]:<24}"
                      f" ok={e.get('success')} attempt={e.get('attempt')}"
                      f" p={e.get('prompt_tokens','')} c={e.get('completion_tokens','')}")
        except FileNotFoundError:
            print("  (no trace file found — was TRACE_LOG_ENABLED=false?)")

        # A corrupted case that ends OK proves the retry path; a failed clean
        # case means the model itself is failing — surface that as exit code 1.
        corrupted_recovered = any(
            r["case"] == "corrupted-first-response" and r["ok"] and r["attempts"] >= 2
            for r in results
        )
        if not corrupted_recovered:
            print("\nWARNING: no corrupted case recovered via retry — inspect above.")
        return 0 if passed == len(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)

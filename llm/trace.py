"""JSONL trace logging for every LLM call.

This file is a first-class deliverable, not a debug log (project brief
section 6.2: "Log/trace the agent conversation — explicitly called out as one
of the most valuable things to include"). One JSON object per line, appended
to logs/agent_trace.jsonl:

    {"ts": "...", "event": "llm_call", "model": "...", "endpoint": "generate",
     "prompt_tokens": 210, "completion_tokens": 84, "latency_ms": 1422,
     "success": true, "error": null, "attempt": 1, "schema": "CreativeBrief"}

Design:
- Appends are line-buffered + flushed per call so a crashed run still leaves
  a readable trace; single-writer assumption (agents run sequentially).
- Logging failures must never take down the pipeline: write errors are
  swallowed by design (a trace log is observability, not a dependency).
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def _log_path():
    from config.settings import get_settings

    return Path(get_settings().trace_log_path)


def append_event(record: dict) -> None:
    """Append a generic event dict as one JSONL line (shared writer logic).

    Used by structured_output.py for validation/retry events so the trace
    shows not just the transport call but the parse-validate-retry cycle —
    the failure-mode section of the write-up reads straight from this.
    """
    try:
        from config.settings import get_settings

        settings = get_settings()
        if not settings.trace_log_enabled:
            return
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def log_llm_call(
    *,
    model: str,
    endpoint: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    success: bool,
    error: str | None = None,
    attempt: int = 1,
    schema: str | None = None,
) -> None:
    """Append one llm_call event. Never raises."""
    try:
        settings = get_settings()
        if not settings.trace_log_enabled:
            return
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "llm_call",
            "model": model,
            "endpoint": endpoint,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": round(latency_ms, 1),
            "success": success,
            "error": error,
            "attempt": attempt,
            "schema": schema,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        # Observability must never crash the pipeline (deliberate swallow).
        pass


class Timer:
    """Latency stopwatch — keeps call sites readable."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.start) * 1000
        return False

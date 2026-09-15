"""Thin async wrapper over the local Ollama daemon (architecture doc section 9).

Covers the doc's required behaviors:
    1. Streaming from /api/generate and /api/chat
    2. Token accounting (Ollama returns counts in the final stream chunk /
       final response object) -> llm/trace.py -> logs/agent_trace.jsonl
    3. Typed, catchable failures instead of raw stack traces
    4. Actionable error when Ollama is down or a model isn't pulled

Deliberately NOT used by structured_output.py, which needs the *final*
response object per attempt (token counts, done_reason) — the streaming path
here is for interactive/CLI use (Day 2+) and for the trace-log demonstration.
"""

import json
from typing import AsyncIterator

import httpx

from config.settings import get_settings
from llm.trace import Timer, log_llm_call


class OllamaError(Exception):
    """Base class for model-layer failures. Callers catch this, not httpx."""


class OllamaConnectionError(OllamaError):
    """Ollama daemon unreachable — message tells the user how to fix it."""


class ModelNotFoundError(OllamaError):
    """Model name not present in the daemon's local model list."""


class OllamaAPIError(OllamaError):
    """Daemon returned a non-200 status for a well-formed request."""


# Models explicitly known to need a manual pull (kept in sync with .env.example)
_KNOWN_MODELS = {"qwen2.5:7b-instruct", "qwen2.5:3b-instruct", "nomic-embed-text"}
_MODEL_HINT = (
    "model not found locally. Pull it first:\n"
    "  ollama pull qwen2.5:7b-instruct\n"
    "  ollama pull qwen2.5:3b-instruct\n"
    "(list what you have with: ollama list)"
)


class OllamaClient:
    """One client per process; methods are safe to call concurrently."""

    def __init__(self, host: str | None = None, timeout: float | None = None):
        settings = get_settings()
        self.host = (host or settings.ollama_host).rstrip("/")
        # Default comes from settings (OLLAMA_TIMEOUT_SECONDS) so CPU-only hosts
        # can allow for slow first-call model loading; explicit arg wins for tests.
        self.timeout = timeout if timeout is not None else settings.ollama_timeout_seconds
        # One connection pool; limits kept modest — Ollama serializes GPU runs.
        self._client = httpx.AsyncClient(base_url=self.host, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # --- diagnostics ---------------------------------------------------------

    async def _ensure_reachable(self) -> None:
        try:
            resp = await self._client.get("/api/version")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaConnectionError(
                f"cannot reach Ollama at {self.host} ({exc.__class__.__name__}).\n"
                "Fix: start the daemon ('ollama serve' or launch the Ollama app),\n"
                "then verify with: curl http://localhost:11434/api/tags"
            ) from exc

    async def _ensure_model(self, model: str) -> None:
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            names = {m.get("name") for m in resp.json().get("models", [])}
        except httpx.HTTPError as exc:
            raise OllamaConnectionError(
                f"cannot reach Ollama at {self.host} ({exc.__class__.__name__})."
            ) from exc
        # Tag names may appear with or without an explicit ':latest'; normalize.
        base = model.split(":")[0]
        if model not in names and not any(n.split(":")[0] == base for n in names):
            hint = _MODEL_HINT if model in _KNOWN_MODELS else f"model not found locally: {model}"
            raise ModelNotFoundError(hint)

    # --- generation ----------------------------------------------------------

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        format_schema: dict | None = None,
        options: dict | None = None,
        raw: bool = False,
    ) -> dict:
        """POST /api/generate — returns the FULL final response dict.

        `format_schema` passes a JSON schema to Ollama's structured-output
        mode (supported natively on modern Ollama); `raw` disables the
        daemon's implicit prompt templating (used by the retry loop in
        structured_output.py so our own formatting reaches the model intact).
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if format_schema is not None:
            payload["format"] = format_schema
        if options:
            payload["options"] = options
        # `raw` was previously accepted but never forwarded — structured_output
        # passes raw=True so OUR prompt reaches the model untemplated (the
        # daemon otherwise wraps it in the model's chat template, which mangles
        # the embedded JSON contract). Fixed 2026-09-15 after the writer's
        # revise pass produced {posts:[{"0":...}]} envelopes in the trace.
        if raw:
            payload["raw"] = True

        await self._ensure_reachable()
        await self._ensure_model(model)
        with Timer() as t:
            try:
                resp = await self._client.post("/api/generate", json=payload)
            except httpx.HTTPError as exc:
                raise OllamaConnectionError(f"request to {self.host} failed: {exc}") from exc
        if resp.status_code != 200:
            raise OllamaAPIError(f"/api/generate returned {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        log_llm_call(
            model=model,
            endpoint="generate",
            prompt_tokens=data.get("prompt_eval_count") or 0,
            completion_tokens=data.get("eval_count") or 0,
            latency_ms=t.ms,
            success=True,
        )
        return data

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        format_schema: dict | None = None,
        options: dict | None = None,
    ) -> dict:
        """POST /api/chat — returns the FULL final response dict."""
        payload: dict = {"model": model, "messages": messages, "stream": False}
        if format_schema is not None:
            payload["format"] = format_schema
        if options:
            payload["options"] = options

        await self._ensure_reachable()
        await self._ensure_model(model)
        with Timer() as t:
            try:
                resp = await self._client.post("/api/chat", json=payload)
            except httpx.HTTPError as exc:
                raise OllamaConnectionError(f"request to {self.host} failed: {exc}") from exc
        if resp.status_code != 200:
            raise OllamaAPIError(f"/api/chat returned {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        log_llm_call(
            model=model,
            endpoint="chat",
            prompt_tokens=data.get("prompt_eval_count") or 0,
            completion_tokens=data.get("eval_count") or 0,
            latency_ms=t.ms,
            success=True,
        )
        return data

    # --- streaming -----------------------------------------------------------

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        options: dict | None = None,
    ) -> AsyncIterator[dict]:
        """Yield /api/generate stream chunks; final chunk carries token counts.

        Streams NDJSON lines. On success the caller has consumed the whole
        response, so we log with the final chunk's eval counts.
        """
        payload: dict = {"model": model, "prompt": prompt, "stream": True}
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        await self._ensure_reachable()
        await self._ensure_model(model)
        prompt_tokens = completion_tokens = 0
        with Timer() as t:
            try:
                async with self._client.stream("POST", "/api/generate", json=payload) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        raise OllamaAPIError(
                            f"/api/generate (stream) returned {resp.status_code}: {body[:300]}"
                        )
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        if chunk.get("error"):
                            raise OllamaAPIError(f"stream error from daemon: {chunk['error']}")
                        prompt_tokens = chunk.get("prompt_eval_count") or prompt_tokens
                        completion_tokens = chunk.get("eval_count") or completion_tokens
                        yield chunk
            except httpx.HTTPError as exc:
                raise OllamaConnectionError(f"stream to {self.host} failed: {exc}") from exc
        log_llm_call(
            model=model,
            endpoint="generate_stream",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=t.ms,
            success=True,
        )

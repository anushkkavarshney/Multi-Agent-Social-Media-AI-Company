"""Project-wide configuration via pydantic-settings.

Single source of truth for every tunable value (model names, Ollama host,
retry limits, simulation seed). Nothing in the codebase hardcodes these —
the architecture doc (section 1, "Env/config" row) requires config-driven
behavior so the whole system can be re-tuned without touching code.

Values load from `.env` (created from `.env.example`) and fall back to
sensible defaults so the platform can boot even before `.env` exists.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Fields mirror `.env.example` exactly."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Tolerate stray whitespace in .env values (common manual-edit mistake).
        extra="ignore",
    )

    # --- Ollama ------------------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    primary_model: str = "qwen2.5:7b-instruct"
    router_model: str = "qwen2.5:3b-instruct"
    embedding_model: str = "nomic-embed-text"
    # HTTP read timeout for the Ollama client. 600s because on CPU-only hosts
    # the FIRST call for a model can spend minutes loading weights into RAM
    # before the first token arrives (7B ≈ 4.7 GB). If this times out you'll
    # see OllamaConnectionError wrapping httpx.ReadTimeout — raise it here.
    ollama_timeout_seconds: float = 600.0

    # --- Structured-output retry policy -------------------------------------
    # Number of *total* attempts before raising ModelOutputFailure (doc section 9:
    # "max 3 attempts, then fall back / fail loudly").
    max_structured_retries: int = 3

    # --- Compliance reject loop ---------------------------------------------
    # Max times Compliance may reject a post and send it back to the Writer
    # before the loop escalates to needs_human (doc section 7: "Max 3 rejection
    # cycles per post -> auto-escalate to human queue"). Owned by
    # orchestration/retry_policy.py; mirrored here so .env can re-tune it.
    max_compliance_retries: int = 3

    # --- Mock platform -------------------------------------------------------
    db_path: str = "./platform.db"
    create_tables_on_startup: bool = True

    # --- Simulation ----------------------------------------------------------
    # Fixed default seed keeps the platform's statistical behavior reproducible
    # across runs; individual tests override it per-call.
    sim_seed: int = 42

    # --- Trace logging -------------------------------------------------------
    trace_log_path: str = "./logs/agent_trace.jsonl"
    trace_log_enabled: bool = True


@lru_cache
def get_settings() -> Settings:
    """Cached accessor — one Settings instance per process.

    Cached because several modules import this at call time; re-reading
    `.env` on every call would silently mask later edits and slow hot paths.
    Call `get_settings.cache_clear()` in tests that mutate env vars.
    """
    return Settings()

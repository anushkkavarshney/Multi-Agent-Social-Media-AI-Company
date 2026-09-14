"""BaseAgent — shared machinery for every agent (architecture doc section 7).

The doc: "Each agent inherits agents/base_agent.py, which provides
call_model(), validate_or_retry(), emit(message) to the bus, and automatic
trace logging." This file is that base; individual agents stay small —
their whole job is choosing inputs, prompts, and output schemas.

Division of responsibility (why subclasses stay thin):
    call_model()   -> llm/structured_output.py owns generate -> parse ->
                      validate -> re-prompt-with-error (max 3) -> raise
                      ModelOutputFailure. The base layer adds: default model
                      selection, agent-identity trace records, and the doc
                      section 9 rule that a terminal model failure becomes a
                      bus message (status="failed") rather than a crash.
    emit()         -> orchestration/message_bus.py owns durability (SQLite +
                      JSONL mirror). Agents never touch the DB directly.
    trace logging  -> llm/trace.py owns the JSONL format; the base layer owns
                      WHEN: every agent logs its input summary before acting
                      and its output summary after, so the trace reads as a
                      conversation, not just a stream of token counts.

Agents are async (the Ollama client is async end-to-end) but stateless
beyond injected dependencies — safe to construct per pipeline run.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel

from config.settings import get_settings
from llm.ollama_client import OllamaClient
from llm.structured_output import ModelOutputFailure, generate_structured
from llm.trace import append_event
from orchestration.message_bus import AgentMessage, MessageBus, make_message

# Prompts live in llm/prompts/*.jinja per the doc's repo layout (section 3).
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


class BaseAgent:
    """Subclass contract: set `name`, optionally `uses_router_model`,
    implement run()-style methods that compose call_model/emit."""

    name: str = "base_agent"
    # Community-Manager-style agents run classification through the small
    # router model; generation agents use the primary model. Overridable per
    # call — this is only the default (config-driven, not hardcoded magic).
    uses_router_model: bool = False

    def __init__(
        self,
        client: OllamaClient | None = None,
        bus: MessageBus | None = None,
    ):
        # Injected dependencies make every agent testable without a daemon
        # (tests pass client=None and use transport= injection in call_model).
        self.client = client if client is not None else OllamaClient()
        self.bus = bus if bus is not None else MessageBus()

    # --- prompts -------------------------------------------------------------

    def render_prompt(self, template_name: str, **context) -> str:
        """Render an llm/prompts/<template_name> with StrictUndefined.

        StrictUndefined (not the Jinja default) is deliberate: a typo'd
        context key must raise HERE, at the agent that owns the bug, instead
        of silently rendering an empty slot the model then improvises around.
        """
        env = Environment(
            loader=FileSystemLoader(PROMPTS_DIR),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            # Prompts are prose for an LLM, not HTML: no escaping (would
            # corrupt quotes/braces the model needs to see verbatim).
            autoescape=False,
        )
        return env.get_template(template_name).render(**context)

    # --- model calls -----------------------------------------------------------

    def _default_model(self) -> str:
        settings = get_settings()
        return settings.router_model if self.uses_router_model else settings.primary_model

    async def call_model(
        self,
        *,
        instruction: str,
        schema_cls: type[BaseModel],
        model: str | None = None,
        options: dict | None = None,
        transport=None,
        campaign_id: str | None = None,
    ) -> BaseModel:
        """Structured model call with the built-in retry loop.

        `transport` (async fn prompt -> str) exists for offline tests; in
        production it is None and the real Ollama daemon is called.

        Raises ModelOutputFailure after the retry budget; that failure is
        ALSO recorded as a bus `agent_failed` message (doc section 9, item 4:
        "agent emits a status='failed' message ... orchestrator handles this
        gracefully") when `campaign_id` is supplied, because the bus record
        requires a campaign context. Agents doing one-shot work outside a
        campaign (rare) may omit it.
        """
        chosen_model = model or self._default_model()
        try:
            result = await generate_structured(
                self.client,
                chosen_model,
                instruction,
                schema_cls,
                options=options,
                transport=transport,
            )
        except ModelOutputFailure as exc:
            append_event(
                {
                    "event": "agent_model_failure",
                    "agent": self.name,
                    "model": chosen_model,
                    "schema": schema_cls.__name__,
                    "attempts": exc.attempts,
                    "last_error": exc.last_error[:300],
                }
            )
            if campaign_id:
                await self.emit(
                    to_agent="orchestrator",
                    campaign_id=campaign_id,
                    message_type="agent_failed",
                    payload={
                        "agent": self.name,
                        "status": "failed",
                        "schema": schema_cls.__name__,
                        "error": exc.last_error[:300],
                    },
                )
            raise
        return result

    # --- bus + trace -----------------------------------------------------------

    async def emit(
        self,
        *,
        to_agent: str,
        campaign_id: str,
        message_type: str,
        payload: dict,
    ) -> AgentMessage:
        """Publish one structured message to the bus (durable + traced)."""
        message = make_message(
            from_agent=self.name,
            to_agent=to_agent,
            campaign_id=campaign_id,
            message_type=message_type,
            payload=payload,
        )
        return await self.bus.publish(message)

    def log_io(self, stage: str, **fields) -> None:
        """Append an agent-scoped event to the trace JSONL.

        Every agent calls this with its input summary BEFORE acting and its
        output summary AFTER — the "automatic trace logging of every
        input/output pair" the doc requires. Never raises (trace is
        observability, not a dependency).
        """
        append_event({"event": "agent_io", "agent": self.name, "stage": stage, **fields})

from uuid import UUID

import structlog
from pika.adapters.blocking_connection import BlockingChannel
from pydantic import ValidationError

from agentforge_worker.config import settings
from agentforge_worker.contracts import (
    AgentTaskRequested,
    TaskCompletedEvent,
    TaskFailedEvent,
)
from agentforge_worker.llm import invoke_llm
from agentforge_worker.messaging.producer import publish_event

log = structlog.get_logger(__name__)


def _classify_error(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "llm_timeout"
    if "connect" in name:
        return "llm_connection_error"
    return "llm_api_error"


def _publish_parse_failure(channel: BlockingChannel, raw_body: bytes, exc: Exception) -> None:
    # We don't know task_id / conversation_id / agent_role — use zero UUIDs as sentinels
    # so the orchestrator can detect unparseable messages without a schema mismatch.
    sentinel = UUID(int=0)
    event = TaskFailedEvent(
        task_id=sentinel,
        conversation_id=sentinel,
        agent_role="unknown",
        error_type="parse_error",
        error_message=f"{type(exc).__name__}: {exc}",
    )
    publish_event(channel, settings.errors_queue, event)
    log.error("task.parse_failed", error=str(exc), raw_preview=raw_body[:200].decode("utf-8", errors="replace"))


def handle_task(channel: BlockingChannel, raw_body: bytes) -> None:
    try:
        task = AgentTaskRequested.model_validate_json(raw_body)
    except ValidationError as exc:
        _publish_parse_failure(channel, raw_body, exc)
        return

    bound = log.bind(
        task_id=str(task.task_id),
        conversation_id=str(task.conversation_id),
        agent_role=task.agent_role,
    )
    bound.info("task.received", context_size=len(task.context))

    model = task.model or settings.default_model
    temperature = task.temperature if task.temperature is not None else settings.default_temperature

    try:
        response = invoke_llm(
            model=model,
            temperature=temperature,
            system_prompt=task.system_prompt,
            user_prompt=task.user_prompt,
            context=task.context,
        )
    except Exception as exc:  # noqa: BLE001 — we classify then re-surface via errors_queue
        failure = TaskFailedEvent(
            task_id=task.task_id,
            conversation_id=task.conversation_id,
            agent_role=task.agent_role,
            error_type=_classify_error(exc),
            error_message=f"{type(exc).__name__}: {exc}",
        )
        publish_event(channel, settings.errors_queue, failure)
        bound.error("task.failed", error_type=failure.error_type, error=str(exc))
        return

    completed = TaskCompletedEvent(
        task_id=task.task_id,
        conversation_id=task.conversation_id,
        agent_role=task.agent_role,
        output=response.output,
        model_used=response.model_used,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
    )
    publish_event(channel, settings.results_queue, completed)
    bound.info(
        "task.completed",
        model=response.model_used,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        output_length=len(response.output),
    )

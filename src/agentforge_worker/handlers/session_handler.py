import asyncio
from uuid import UUID

import structlog
from langchain_core.messages import AIMessage, HumanMessage
from pika.adapters.blocking_connection import BlockingChannel
from pydantic import ValidationError

from agentforge_worker.config import settings
from agentforge_worker.contracts import (
    AgentEventOccurred,
    AgentSessionCompleted,
    AgentSessionFailed,
    AgentSessionRequested,
    ErrorType,
)
from agentforge_worker.graph import build_graph
from agentforge_worker.handlers.event_mapper import SessionAccumulator, map_event
from agentforge_worker.messaging.publisher import (
    publish_completed,
    publish_event,
    publish_failed,
)
from agentforge_worker.tools import ToolContext

log = structlog.get_logger(__name__)

_SENTINEL_UUID = UUID(int=0)


class _MaxIterationsExceeded(Exception):
    pass


def _classify_error(exc: Exception) -> ErrorType:
    if isinstance(exc, _MaxIterationsExceeded):
        return "max_iterations_exceeded"
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "llm_timeout"
    if "connect" in name:
        return "llm_connection_error"
    if "tool" in name:
        return "tool_error"
    if "validation" in name or "parse" in name:
        return "parse_error"
    if "ollama" in name or "httpx" in name:
        return "llm_api_error"
    return "unknown"


def _publish_parse_failure(
    channel: BlockingChannel, raw_body: bytes, exc: Exception
) -> None:
    failed = AgentSessionFailed(
        session_id=_SENTINEL_UUID,
        conversation_id=_SENTINEL_UUID,
        error_type="parse_error",
        error_message=f"{type(exc).__name__}: {exc}",
    )
    publish_failed(channel, failed)
    log.error(
        "session.parse_failed",
        error=str(exc),
        raw_preview=raw_body[:200].decode("utf-8", errors="replace"),
    )


def _build_initial_messages(req: AgentSessionRequested) -> list:
    messages = []
    for h in req.history:
        if h.role == "user":
            messages.append(HumanMessage(content=h.content))
        else:
            prefix = f"[{h.agent_role}]: " if h.agent_role else ""
            messages.append(AIMessage(content=prefix + h.content))
    messages.append(HumanMessage(content=req.user_prompt))
    return messages


async def _run_session_async(
    channel: BlockingChannel,
    req: AgentSessionRequested,
    acc: SessionAccumulator,
    bound_log,
) -> None:
    ctx = ToolContext(session_id=req.session_id, scratchpad={})
    graph = build_graph(req, settings, ctx)
    initial_state = {
        "session_id": req.session_id,
        "conversation_id": req.conversation_id,
        "messages": _build_initial_messages(req),
        "iterations": 0,
        "next_agent": None,
        "last_reasoning": None,
        "scratchpad": ctx.scratchpad,
        "trace": [],
    }

    publish_event(
        channel,
        AgentEventOccurred(
            session_id=req.session_id,
            conversation_id=req.conversation_id,
            event_type="session_started",
            payload={
                "agents": [a.role for a in req.team.agents],
                "max_iterations": req.team.max_iterations,
            },
        ),
    )

    stop_reason = "supervisor_end"
    async for ev in graph.astream_events(initial_state, version="v2"):
        mapped = map_event(ev, acc)
        if mapped is not None:
            publish_event(channel, mapped)

    if (
        acc.iterations >= req.team.max_iterations
        and acc.last_supervisor_next not in (None, "END")
    ):
        stop_reason = "max_iterations"

    publish_event(
        channel,
        AgentEventOccurred(
            session_id=req.session_id,
            conversation_id=req.conversation_id,
            event_type="session_finished",
            payload={"iterations": acc.iterations, "stop_reason": stop_reason},
        ),
    )

    if stop_reason == "max_iterations":
        raise _MaxIterationsExceeded(
            f"supervisor did not reach END within {req.team.max_iterations} iterations"
        )

    final = acc.final_output or "(no agent produced output)"
    completed = AgentSessionCompleted(
        session_id=req.session_id,
        conversation_id=req.conversation_id,
        final_output=final,
        trace=acc.trace,
        iterations=acc.iterations,
    )
    publish_completed(channel, completed)
    bound_log.info(
        "session.completed",
        iterations=acc.iterations,
        trace_size=len(acc.trace),
        output_length=len(final),
    )


def handle_session(channel: BlockingChannel, raw_body: bytes) -> None:
    try:
        req = AgentSessionRequested.model_validate_json(raw_body)
    except ValidationError as exc:
        _publish_parse_failure(channel, raw_body, exc)
        return

    bound = log.bind(
        session_id=str(req.session_id),
        conversation_id=str(req.conversation_id),
        agents=[a.role for a in req.team.agents],
    )
    bound.info(
        "session.received",
        history_size=len(req.history),
        max_iterations=req.team.max_iterations,
    )

    acc = SessionAccumulator(
        session_id=req.session_id, conversation_id=req.conversation_id
    )
    try:
        # pika's BlockingConnection callback is sync; LangGraph streaming is async — bridge here.
        # prefetch=1 means this thread is dedicated to a single session, so blocking is fine.
        asyncio.run(_run_session_async(channel, req, acc, bound))
    except Exception as exc:  # noqa: BLE001
        error_type = _classify_error(exc)
        publish_failed(
            channel,
            AgentSessionFailed(
                session_id=req.session_id,
                conversation_id=req.conversation_id,
                error_type=error_type,
                error_message=f"{type(exc).__name__}: {exc}",
                partial_trace=acc.trace,
            ),
        )
        bound.error("session.failed", error_type=error_type, error=str(exc))

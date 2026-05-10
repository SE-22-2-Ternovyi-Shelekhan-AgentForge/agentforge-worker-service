"""Map LangGraph stream events → AgentEventOccurred + accumulate trace."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage

from agentforge_worker.contracts import AgentEventOccurred, TraceEntry

_PREVIEW_LIMIT = 200


def _preview(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= _PREVIEW_LIMIT else text[:_PREVIEW_LIMIT] + "..."


def _node_role(node_name: str) -> str | None:
    if node_name and node_name.startswith("agent_"):
        return node_name[len("agent_") :]
    return None


def _extract_message_content(msg: BaseMessage | None) -> str:
    if msg is None:
        return ""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


@dataclass
class _AgentRun:
    role: str
    tools_used: list[str] = field(default_factory=list)
    last_output: str = ""


@dataclass
class SessionAccumulator:
    """Collects per-agent traces while a session streams."""

    session_id: UUID
    conversation_id: UUID
    iterations: int = 0
    last_supervisor_reasoning: str = ""
    last_supervisor_next: str | None = None
    trace: list[TraceEntry] = field(default_factory=list)
    final_output: str = ""
    _current_agent: dict[str, _AgentRun] = field(default_factory=dict)

    def make_event(
        self,
        event_type: str,
        agent_role: str | None,
        payload: dict[str, Any],
    ) -> AgentEventOccurred:
        return AgentEventOccurred(
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            event_type=event_type,  # type: ignore[arg-type]
            agent_role=agent_role,
            payload=payload,
        )


def map_event(ev: dict, acc: SessionAccumulator) -> AgentEventOccurred | None:
    """Convert a single LangGraph astream_events item into AgentEventOccurred (or None to skip)."""

    kind = ev.get("event")
    name = ev.get("name", "")
    data = ev.get("data", {}) or {}
    metadata = ev.get("metadata", {}) or {}
    langgraph_node = metadata.get("langgraph_node")

    if kind == "on_chain_end" and name == "supervisor":
        output = data.get("output") or {}
        nxt = output.get("next_agent")
        reasoning = output.get("last_reasoning") or ""
        if nxt is not None:
            acc.last_supervisor_next = nxt
            acc.last_supervisor_reasoning = reasoning
            iters = output.get("iterations")
            if isinstance(iters, int):
                acc.iterations = iters
            if nxt == "END" and not acc.final_output and reasoning:
                acc.final_output = reasoning
            return acc.make_event(
                "supervisor_routed",
                None,
                {"next": nxt, "reasoning": reasoning},
            )
        return None

    if kind == "on_chain_start" and name.startswith("agent_"):
        role = _node_role(name)
        if role:
            acc._current_agent[role] = _AgentRun(role=role)
            return acc.make_event("agent_started", role, {})
        return None

    if kind == "on_tool_start":
        role = _node_role(langgraph_node) if langgraph_node else None
        if role and role in acc._current_agent:
            acc._current_agent[role].tools_used.append(name)
        return acc.make_event(
            "tool_called",
            role,
            {"tool": name, "input": data.get("input")},
        )

    if kind == "on_tool_end":
        role = _node_role(langgraph_node) if langgraph_node else None
        out = data.get("output")
        return acc.make_event(
            "tool_result",
            role,
            {"tool": name, "output_preview": _preview(str(out))},
        )

    if kind == "on_chain_end" and name.startswith("agent_"):
        role = _node_role(name)
        output = data.get("output") or {}
        msgs = output.get("messages") or []
        last_msg = msgs[-1] if msgs else None
        if isinstance(last_msg, AIMessage):
            text = _extract_message_content(last_msg)
        elif last_msg is not None:
            text = _extract_message_content(last_msg)
        else:
            text = ""
        run = acc._current_agent.get(role) if role else None
        tools_used = list(run.tools_used) if run else []
        if role and run:
            run.last_output = text
            acc.trace.append(
                TraceEntry(agent_role=role, output=text, tools_used=tools_used)
            )
            acc.final_output = text
        return acc.make_event(
            "agent_finished",
            role,
            {"output_preview": _preview(text), "tools_used": tools_used},
        )

    return None

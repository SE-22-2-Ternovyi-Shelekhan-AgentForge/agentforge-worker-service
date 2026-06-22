from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from agentforge_worker.handlers.event_mapper import SessionAccumulator, _AgentRun, map_event


def _make_acc() -> SessionAccumulator:
    return SessionAccumulator(session_id=uuid4(), conversation_id=uuid4())


def test_supervisor_routed_event():
    acc = _make_acc()
    ev = {
        "event": "on_chain_end",
        "name": "supervisor",
        "data": {
            "output": {
                "next_agent": "researcher",
                "last_reasoning": "needs research",
                "iterations": 1,
            }
        },
        "metadata": {},
    }
    result = map_event(ev, acc)

    assert result is not None
    assert result.event_type == "supervisor_routed"
    assert result.payload == {"next": "researcher", "reasoning": "needs research"}
    assert acc.last_supervisor_next == "researcher"
    assert acc.iterations == 1


def test_tool_called_event():
    acc = _make_acc()
    acc._current_agent["researcher"] = _AgentRun(role="researcher")
    ev = {
        "event": "on_tool_start",
        "name": "web_search",
        "data": {"input": {"query": "AI news"}},
        "metadata": {"langgraph_node": "agent_researcher"},
    }
    result = map_event(ev, acc)

    assert result is not None
    assert result.event_type == "tool_called"
    assert result.agent_role == "researcher"
    assert result.payload["tool"] == "web_search"
    assert "web_search" in acc._current_agent["researcher"].tools_used


def test_agent_finished_event():
    acc = _make_acc()
    acc._current_agent["researcher"] = _AgentRun(role="researcher")
    ai_msg = AIMessage(content="[researcher]: Research complete")
    ev = {
        "event": "on_chain_end",
        "name": "agent_researcher",
        "data": {"output": {"messages": [ai_msg]}},
        "metadata": {},
    }
    result = map_event(ev, acc)

    assert result is not None
    assert result.event_type == "agent_finished"
    assert result.agent_role == "researcher"
    assert len(acc.trace) == 1
    assert acc.trace[0].agent_role == "researcher"
    assert acc.final_output == "[researcher]: Research complete"


def test_skip_irrelevant_event():
    acc = _make_acc()
    ev = {
        "event": "on_unknown_event",
        "name": "something",
        "data": {},
        "metadata": {},
    }
    assert map_event(ev, acc) is None


def test_agent_started_event():
    acc = _make_acc()
    ev = {
        "event": "on_chain_start",
        "name": "agent_writer",
        "data": {},
        "metadata": {},
    }
    result = map_event(ev, acc)

    assert result is not None
    assert result.event_type == "agent_started"
    assert result.agent_role == "writer"
    assert "writer" in acc._current_agent


def test_tool_result_event():
    acc = _make_acc()
    ev = {
        "event": "on_tool_end",
        "name": "web_search",
        "data": {"output": "Some search result"},
        "metadata": {"langgraph_node": "agent_researcher"},
    }
    result = map_event(ev, acc)

    assert result is not None
    assert result.event_type == "tool_result"
    assert result.agent_role == "researcher"
    assert result.payload["tool"] == "web_search"
    assert "Some search result" in result.payload["output_preview"]


def test_supervisor_end_sets_final_output_from_reasoning():
    acc = _make_acc()
    ev = {
        "event": "on_chain_end",
        "name": "supervisor",
        "data": {
            "output": {
                "next_agent": "END",
                "last_reasoning": "The answer is 42",
                "iterations": 1,
            }
        },
        "metadata": {},
    }
    map_event(ev, acc)

    assert acc.final_output == "The answer is 42"


def test_preview_truncates_long_text():
    from agentforge_worker.handlers.event_mapper import _preview
    long_text = "x" * 300
    result = _preview(long_text)
    assert len(result) == 203
    assert result.endswith("...")


def test_supervisor_event_skipped_when_no_next_agent():
    acc = _make_acc()
    ev = {
        "event": "on_chain_end",
        "name": "supervisor",
        "data": {"output": {}},
        "metadata": {},
    }
    result = map_event(ev, acc)
    assert result is None

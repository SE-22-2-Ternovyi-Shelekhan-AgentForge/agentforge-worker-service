"""
E2E scenarios using the real build_graph() with mocked LLMs.

Mock strategy:
- agentforge_worker.graph.supervisor.make_chat_model → supervisor mock
  The returned mock's .with_structured_output().invoke() yields SupervisorDecision
  objects in sequence (side_effect list).
- agentforge_worker.graph.agent_node.make_chat_model → FakeAgentLLM
  A real BaseChatModel subclass that returns AIMessage without tool_calls,
  causing create_react_agent to terminate immediately.
"""
from typing import Any, List, Optional
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agentforge_worker.contracts import AgentConfig, AgentSessionRequested, TeamConfig
from agentforge_worker.graph.builder import build_graph
from agentforge_worker.graph.supervisor import SupervisorDecision
from agentforge_worker.tools import ToolContext


class FakeAgentLLM(BaseChatModel):
    response: str = "Fake agent response"

    @property
    def _llm_type(self) -> str:
        return "fake-agent"

    def _generate(
        self,
        messages: List,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.response))]
        )


def _make_supervisor_llm(mocker, decisions: list[SupervisorDecision]):
    chain = mocker.MagicMock()
    chain.invoke.side_effect = decisions
    llm = mocker.MagicMock()
    llm.with_structured_output.return_value = chain
    return llm


def _patch_llms(mocker, decisions: list[SupervisorDecision], agent_response: str):
    sup_llm = _make_supervisor_llm(mocker, decisions)
    agent_llm = FakeAgentLLM(response=agent_response)
    mocker.patch("agentforge_worker.graph.supervisor.make_chat_model", return_value=sup_llm)
    mocker.patch("agentforge_worker.graph.agent_node.make_chat_model", return_value=agent_llm)


def _build(req: AgentSessionRequested, fake_settings):
    ctx = ToolContext(session_id=req.session_id, scratchpad={})
    return build_graph(req, fake_settings, ctx)


def _initial_state(req: AgentSessionRequested) -> dict:
    return {
        "session_id": req.session_id,
        "conversation_id": req.conversation_id,
        "messages": [HumanMessage(content=req.user_prompt)],
        "iterations": 0,
        "next_agent": None,
        "last_reasoning": None,
        "scratchpad": {},
        "trace": [],
    }


def test_linear_session(mocker, fake_settings):
    """Supervisor delegates to researcher once, then ends."""
    req = AgentSessionRequested(
        session_id=uuid4(),
        conversation_id=uuid4(),
        user_prompt="Summarise AI news",
        team=TeamConfig(
            agents=[AgentConfig(role="researcher", system_prompt="Research things " * 5)],
            max_iterations=5,
        ),
    )
    _patch_llms(
        mocker,
        decisions=[
            SupervisorDecision(next="researcher", reasoning="delegate to researcher"),
            SupervisorDecision(next="END", reasoning="task complete"),
        ],
        agent_response="[researcher]: Here are the latest AI developments.",
    )

    graph = _build(req, fake_settings)
    result = graph.invoke(_initial_state(req))

    contents = [m.content for m in result["messages"] if hasattr(m, "content")]
    assert any("[researcher]:" in c for c in contents)
    assert result["iterations"] >= 1


def test_repeated_interaction(mocker, fake_settings):
    """Supervisor routes: researcher → writer → END (two agent handoffs)."""
    req = AgentSessionRequested(
        session_id=uuid4(),
        conversation_id=uuid4(),
        user_prompt="Research and write an article",
        team=TeamConfig(
            agents=[
                AgentConfig(role="researcher", system_prompt="Research things " * 5),
                AgentConfig(role="writer", system_prompt="Write things " * 5),
            ],
            max_iterations=10,
        ),
    )

    researcher_response = "[researcher]: Here are the facts."
    writer_response = "[writer]: Here is the article."

    sup_llm = _make_supervisor_llm(
        mocker,
        decisions=[
            SupervisorDecision(next="researcher", reasoning="research first"),
            SupervisorDecision(next="writer", reasoning="now write"),
            SupervisorDecision(next="END", reasoning="done"),
        ],
    )
    mocker.patch("agentforge_worker.graph.supervisor.make_chat_model", return_value=sup_llm)

    call_count = {"n": 0}
    responses = [researcher_response, writer_response]

    class SequentialFakeLLM(FakeAgentLLM):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            resp = responses[min(call_count["n"], len(responses) - 1)]
            call_count["n"] += 1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=resp))])

    mocker.patch(
        "agentforge_worker.graph.agent_node.make_chat_model",
        return_value=SequentialFakeLLM(response=""),
    )

    graph = _build(req, fake_settings)
    result = graph.invoke(_initial_state(req))

    contents = " ".join(m.content for m in result["messages"] if hasattr(m, "content"))
    assert "[researcher]:" in contents
    assert "[writer]:" in contents
    assert result["iterations"] >= 2


def test_loop_protection(mocker, fake_settings):
    """Supervisor always routes to researcher; max_iterations=2 stops the graph."""
    req = AgentSessionRequested(
        session_id=uuid4(),
        conversation_id=uuid4(),
        user_prompt="Keep going",
        team=TeamConfig(
            agents=[AgentConfig(role="researcher", system_prompt="Research things " * 5)],
            max_iterations=2,
        ),
    )
    _patch_llms(
        mocker,
        decisions=[SupervisorDecision(next="researcher", reasoning="again")] * 10,
        agent_response="[researcher]: Still working.",
    )

    graph = _build(req, fake_settings)
    result = graph.invoke(_initial_state(req))

    assert result["iterations"] >= 2

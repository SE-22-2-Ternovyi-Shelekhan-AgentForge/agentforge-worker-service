"""
E2E scenarios using the real build_graph() with a mocked agent LLM.

The team coordinator (supervisor) is deterministic — it dispatches each agent per
round and decides "another round or done?" from the reviewer's verdict — so only the
agent LLM is mocked here:
- agentforge_worker.graph.agent_node.make_chat_model → FakeAgentLLM
  A real BaseChatModel subclass that returns an AIMessage without tool_calls,
  causing create_react_agent to terminate immediately.
"""
from typing import Any, List, Optional
from uuid import uuid4

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agentforge_worker.contracts import AgentConfig, AgentSessionRequested, TeamConfig
from agentforge_worker.graph.builder import build_graph
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


def _patch_llms(mocker, agent_response: str):
    agent_llm = FakeAgentLLM(response=agent_response)
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
        "round": 1,
        "next_agent": None,
        "last_reasoning": None,
        "scratchpad": {},
        "trace": [],
        "agents_visited": [],
    }


async def test_linear_session(mocker, fake_settings):
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
    _patch_llms(mocker, agent_response="[researcher]: Here are the latest AI developments.")

    graph = _build(req, fake_settings)
    result = await graph.ainvoke(_initial_state(req))

    contents = [m.content for m in result["messages"] if hasattr(m, "content")]
    assert any("[researcher]:" in c for c in contents)
    assert result["iterations"] >= 1


async def test_repeated_interaction(mocker, fake_settings):
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
    result = await graph.ainvoke(_initial_state(req))

    contents = " ".join(m.content for m in result["messages"] if hasattr(m, "content"))
    assert "[researcher]:" in contents
    assert "[writer]:" in contents
    assert result["iterations"] >= 2


async def test_loop_protection(mocker, fake_settings):
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
    _patch_llms(mocker, agent_response="[researcher]: Still working.")

    graph = _build(req, fake_settings)
    result = await graph.ainvoke(_initial_state(req))

    assert result["iterations"] >= 2

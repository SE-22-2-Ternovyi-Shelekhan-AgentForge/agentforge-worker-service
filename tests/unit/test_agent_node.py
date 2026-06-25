from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from agentforge_worker.contracts import AgentConfig
from agentforge_worker.graph.agent_node import make_agent_node
from agentforge_worker.tools import ToolContext
from tests.conftest import make_graph_state


async def test_agent_with_tool(mocker, fake_settings):
    cfg = AgentConfig(role="researcher", system_prompt="Research things", tools=["current_time"])
    ctx = ToolContext(session_id=uuid4(), scratchpad={})

    prebuilt_mock = mocker.MagicMock()
    prebuilt_mock.ainvoke = mocker.AsyncMock(
        return_value={"messages": [AIMessage(content="The time is now")]}
    )
    mocker.patch("agentforge_worker.graph.agent_node.make_chat_model", return_value=mocker.MagicMock())
    mocker.patch("agentforge_worker.graph.agent_node.create_react_agent", return_value=prebuilt_mock)

    node = make_agent_node(cfg, fake_settings, ctx)
    result = await node(make_graph_state())

    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content == "[researcher]: The time is now"
    assert msg.additional_kwargs["agent_role"] == "researcher"


async def test_agent_without_tool(mocker, fake_settings):
    cfg = AgentConfig(role="writer", system_prompt="Write things", tools=[])
    ctx = ToolContext(session_id=uuid4(), scratchpad={})

    prebuilt_mock = mocker.MagicMock()
    prebuilt_mock.ainvoke = mocker.AsyncMock(
        return_value={"messages": [AIMessage(content="Here is the essay")]}
    )
    mocker.patch("agentforge_worker.graph.agent_node.make_chat_model", return_value=mocker.MagicMock())
    mocker.patch("agentforge_worker.graph.agent_node.create_react_agent", return_value=prebuilt_mock)

    node = make_agent_node(cfg, fake_settings, ctx)
    result = await node(make_graph_state())

    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content == "[writer]: Here is the essay"

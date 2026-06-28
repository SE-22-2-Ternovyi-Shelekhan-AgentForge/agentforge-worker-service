from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from agentforge_worker.config import Settings
from agentforge_worker.contracts import AgentConfig, AgentSessionRequested, TeamConfig
from agentforge_worker.graph.state import GraphState


@pytest.fixture
def fake_settings():
    return Settings(
        rabbitmq_url="amqp://localhost/",
        default_model="fake-model",
        supervisor_model="fake-model",
        supervisor_provider="ollama",
        default_provider="ollama",
    )


@pytest.fixture
def simple_team():
    return TeamConfig(
        agents=[
            AgentConfig(role="researcher", system_prompt="Research things " * 5),
            AgentConfig(role="writer", system_prompt="Write things " * 5),
        ]
    )


@pytest.fixture
def session_request(simple_team):
    return AgentSessionRequested(
        session_id=uuid4(),
        conversation_id=uuid4(),
        user_prompt="Hello agents",
        team=simple_team,
    )


def make_graph_state(messages=None, agents_visited=None, round=1) -> GraphState:
    return GraphState(
        session_id=uuid4(),
        conversation_id=uuid4(),
        messages=messages or [HumanMessage(content="Hello")],
        iterations=0,
        round=round,
        next_agent=None,
        last_reasoning=None,
        scratchpad={},
        trace=[],
        agents_visited=agents_visited if agents_visited is not None else [],
    )

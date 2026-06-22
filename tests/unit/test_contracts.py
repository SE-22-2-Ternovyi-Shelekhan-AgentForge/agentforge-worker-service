from uuid import UUID

import pytest
from pydantic import ValidationError

from agentforge_worker.contracts import (
    AgentConfig,
    AgentSessionRequested,
    ContextMessage,
    TeamConfig,
)


def test_round_trip_serialization():
    original = AgentSessionRequested(
        session_id=UUID("12345678-1234-5678-1234-567812345678"),
        conversation_id=UUID("87654321-4321-8765-4321-876543218765"),
        user_prompt="Hello agents!",
        history=[
            ContextMessage(role="user", content="Previous message"),
            ContextMessage(role="assistant", content="Previous reply", agent_role="researcher"),
        ],
        team=TeamConfig(
            agents=[
                AgentConfig(role="researcher", system_prompt="Research things", tools=["web_search"])
            ],
            max_iterations=5,
        ),
    )

    restored = AgentSessionRequested.model_validate_json(original.model_dump_json())

    assert restored.session_id == original.session_id
    assert restored.conversation_id == original.conversation_id
    assert restored.user_prompt == original.user_prompt
    assert len(restored.history) == 2
    assert restored.history[1].agent_role == "researcher"
    assert restored.team.max_iterations == 5
    assert restored.team.agents[0].tools == ["web_search"]


def test_validation_errors():
    with pytest.raises(ValidationError):
        TeamConfig(agents=[])

    with pytest.raises(ValidationError):
        AgentConfig(system_prompt="Missing role field")

    with pytest.raises(ValidationError):
        ContextMessage(role="invalid_role", content="test")

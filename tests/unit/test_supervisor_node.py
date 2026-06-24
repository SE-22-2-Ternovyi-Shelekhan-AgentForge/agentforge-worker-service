import pytest
from langchain_core.messages import HumanMessage

from agentforge_worker.graph.supervisor import SupervisorDecision, make_supervisor_node
from tests.conftest import make_graph_state


def _build_node(mocker, fake_settings, simple_team, decision: SupervisorDecision):
    chain_mock = mocker.MagicMock()
    chain_mock.invoke.return_value = decision
    llm_mock = mocker.MagicMock()
    llm_mock.with_structured_output.return_value = chain_mock
    mocker.patch("agentforge_worker.graph.supervisor.make_chat_model", return_value=llm_mock)
    return make_supervisor_node(simple_team, fake_settings)


def test_routes_to_valid_agent(mocker, fake_settings, simple_team):
    node = _build_node(
        mocker, fake_settings, simple_team,
        SupervisorDecision(next="researcher", reasoning="needs research"),
    )
    result = node(make_graph_state(agents_visited=["researcher", "writer"]))

    assert result["next_agent"] == "researcher"
    assert result["iterations"] == 1
    assert result["last_reasoning"] == "needs research"


def test_hallucination_falls_back_to_END(mocker, fake_settings, simple_team):
    node = _build_node(
        mocker, fake_settings, simple_team,
        SupervisorDecision(next="nonexistent_role", reasoning="confused"),
    )
    result = node(make_graph_state(agents_visited=["researcher", "writer"]))

    assert result["next_agent"] == "END"


def test_routes_to_END(mocker, fake_settings, simple_team):
    node = _build_node(
        mocker, fake_settings, simple_team,
        SupervisorDecision(next="END", reasoning="task complete"),
    )
    result = node(make_graph_state(agents_visited=["researcher", "writer"]))

    assert result["next_agent"] == "END"
    assert result["last_reasoning"] == "task complete"

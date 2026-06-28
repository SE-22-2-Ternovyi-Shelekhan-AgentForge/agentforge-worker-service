from langchain_core.messages import AIMessage

from agentforge_worker.contracts import AgentConfig, TeamConfig
from agentforge_worker.graph.supervisor import SupervisorDecision, make_supervisor_node
from tests.conftest import make_graph_state


def _team_with_prompt(prompt: str) -> TeamConfig:
    return TeamConfig(
        supervisor_prompt=prompt,
        agents=[
            AgentConfig(role="researcher", system_prompt="Research " * 5),
            AgentConfig(role="writer", system_prompt="Write " * 5),
        ],
    )


def _reviewer_msg(text: str, role: str = "writer") -> AIMessage:
    """A tagged reviewer message, as agent_node would produce it."""
    return AIMessage(content=f"[{role}]: {text}", additional_kwargs={"agent_role": role})


def test_fast_path_dispatches_first_unvisited(fake_settings, simple_team):
    node = make_supervisor_node(simple_team, fake_settings)
    result = node(make_graph_state(agents_visited=["researcher"]))

    assert result["next_agent"] == "writer"   # the remaining unvisited agent
    assert result["iterations"] == 1


def test_new_round_when_reviewer_not_satisfied(fake_settings, simple_team):
    node = make_supervisor_node(simple_team, fake_settings)
    state = make_graph_state(
        messages=[_reviewer_msg("Є помилка у функції, виправ обробку нуля.")],
        agents_visited=["researcher", "writer"],
        round=1,
    )
    result = node(state)

    assert result["next_agent"] == "researcher"   # back to the first agent
    assert result["round"] == 2
    assert result["agents_visited"] == []


def test_ends_when_reviewer_approves(fake_settings, simple_team):
    node = make_supervisor_node(simple_team, fake_settings)
    state = make_graph_state(
        messages=[_reviewer_msg("Все добре, схвалюю роботу.")],
        agents_visited=["researcher", "writer"],
        round=1,
    )
    result = node(state)

    assert result["next_agent"] == "END"


def test_round_cap_ends(fake_settings, simple_team):
    """At the round ceiling we stop even if the reviewer still complains."""
    node = make_supervisor_node(simple_team, fake_settings)
    state = make_graph_state(
        messages=[_reviewer_msg("Досі є зауваження.")],
        agents_visited=["researcher", "writer"],
        round=2,   # simple_team.max_rounds == 2
    )
    result = node(state)

    assert result["next_agent"] == "END"


def test_solo_team_ends_after_one_pass(fake_settings):
    solo = TeamConfig(agents=[AgentConfig(role="researcher", system_prompt="Research " * 5)])
    node = make_supervisor_node(solo, fake_settings)
    result = node(make_graph_state(agents_visited=["researcher"], round=1))

    assert result["next_agent"] == "END"


def _patch_supervisor_llm(mocker, decision: SupervisorDecision):
    chain = mocker.MagicMock()
    chain.invoke.return_value = decision
    llm = mocker.MagicMock()
    llm.with_structured_output.return_value = chain
    mocker.patch("agentforge_worker.graph.supervisor.make_chat_model", return_value=llm)
    return chain


def test_llm_mode_used_when_supervisor_prompt_set(mocker, fake_settings):
    """With a supervisor_prompt, the end-of-round decision goes through the LLM."""
    chain = _patch_supervisor_llm(
        mocker, SupervisorDecision(next="END", reasoning="готово")
    )
    node = make_supervisor_node(_team_with_prompt("Ти координатор. Вирішуй."), fake_settings)
    result = node(make_graph_state(agents_visited=["researcher", "writer"], round=1))

    chain.invoke.assert_called_once()
    assert result["next_agent"] == "END"
    assert result["last_reasoning"] == "готово"


def test_llm_mode_continues_when_not_end(mocker, fake_settings):
    _patch_supervisor_llm(
        mocker, SupervisorDecision(next="researcher", reasoning="ще раунд")
    )
    node = make_supervisor_node(_team_with_prompt("Ти координатор."), fake_settings)
    result = node(make_graph_state(agents_visited=["researcher", "writer"], round=1))

    assert result["next_agent"] == "researcher"
    assert result["round"] == 2
    assert result["agents_visited"] == []

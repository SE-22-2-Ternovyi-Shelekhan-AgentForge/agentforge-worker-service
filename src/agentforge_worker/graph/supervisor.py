"""Hybrid team coordinator.

Within a round the coordinator is always deterministic: it dispatches every agent
that hasn't spoken yet (registration order). The "another round or done?" decision
after a full pass has two modes:

- If the team defines a ``supervisor_prompt`` → LLM mode: a small structured-output
  model judges, guided by that prompt, whether to END or run another round.
- Otherwise → deterministic mode: look at the reviewer's (last agent's) final message
  and continue until it reads as an approval or ``max_rounds`` is reached. This is the
  robust default — on weak local models the LLM judge tends to always answer END, so
  the review loop never runs.
"""

from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from agentforge_worker.config import Settings
from agentforge_worker.contracts import TeamConfig
from agentforge_worker.graph.state import GraphState
from agentforge_worker.llm import make_chat_model

# Substrings that mark the reviewer being satisfied (UA + EN, matched case-insensitively).
APPROVAL_MARKERS = (
    "схвал",            # схвалюю / схвалено
    "немає зауважень",
    "зауважень немає",
    "без зауважень",
    "не маю зауважень",
    "все добре",
    "все гаразд",
    "approve",
    "lgtm",
    "looks good",
    "no issues",
)


class SupervisorDecision(BaseModel):
    next: str
    reasoning: str


def _is_approval(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in APPROVAL_MARKERS)


def make_supervisor_node(team: TeamConfig, settings: Settings):
    agent_order = [a.role for a in team.agents]
    first_role = agent_order[0]
    reviewer_role = agent_order[-1]   # by convention the last agent is the reviewer
    max_rounds = team.max_rounds
    valid_roles = {a.role for a in team.agents}

    # LLM mode is enabled only when the team provides a supervisor prompt.
    supervisor_prompt = (team.supervisor_prompt or "").strip()
    use_llm = bool(supervisor_prompt)
    llm = None
    if use_llm:
        llm = make_chat_model(
            provider=settings.supervisor_provider,
            model=settings.supervisor_model,
            temperature=0.0,
            timeout=settings.llm_timeout_seconds,
            settings=settings,
        ).with_structured_output(SupervisorDecision)

    def _last_output_of(state: GraphState, role: str) -> str:
        for msg in reversed(state.get("messages") or []):
            if getattr(msg, "additional_kwargs", {}).get("agent_role") == role:
                content = getattr(msg, "content", "")
                return content if isinstance(content, str) else str(content)
        return ""

    def _start_new_round(state: GraphState, rnd: int, reasoning: str) -> dict:
        return {
            "next_agent": first_role,
            "agents_visited": [],
            "round": rnd + 1,
            "last_reasoning": reasoning,
            "iterations": state["iterations"] + 1,
        }

    def _end(state: GraphState, reasoning: str) -> dict:
        return {
            "next_agent": "END",
            "last_reasoning": reasoning,
            "iterations": state["iterations"] + 1,
        }

    def node(state: GraphState) -> dict:
        rnd = state.get("round") or 1
        visited = set(state.get("agents_visited") or [])
        unvisited = [r for r in agent_order if r not in visited]

        # Still agents to hear from this round — dispatch the next one.
        if unvisited:
            nxt = unvisited[0]
            return {
                "next_agent": nxt,
                "last_reasoning": f"раунд {rnd}: черга агента «{nxt}»",
                "iterations": state["iterations"] + 1,
            }

        # Full pass complete. A solo team has no review loop.
        if len(agent_order) < 2:
            return _end(state, "єдиний агент завершив роботу")

        if rnd >= max_rounds:
            return _end(state, f"досягнуто межі раундів ({max_rounds}) — завершую")

        # Decide: another round or done?
        if use_llm:
            decision: SupervisorDecision = llm.invoke(
                [SystemMessage(content=supervisor_prompt), *state["messages"]]
            )
            nxt = decision.next.strip()
            if nxt == "END" or nxt not in valid_roles:
                return _end(state, decision.reasoning)
            return _start_new_round(state, rnd, decision.reasoning)

        # Deterministic mode: continue until the reviewer approves.
        verdict = _last_output_of(state, reviewer_role)
        if _is_approval(verdict):
            return _end(state, f"рецензент схвалив роботу — завершую (раунд {rnd})")
        return _start_new_round(
            state, rnd, f"рецензент має зауваження — раунд {rnd + 1} на виправлення"
        )

    return node

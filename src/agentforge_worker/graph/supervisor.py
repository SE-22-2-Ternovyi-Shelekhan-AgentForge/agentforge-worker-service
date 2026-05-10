from langchain_core.messages import SystemMessage
from pydantic import BaseModel

from agentforge_worker.config import Settings
from agentforge_worker.contracts import TeamConfig
from agentforge_worker.graph.state import GraphState
from agentforge_worker.llm import make_chat_model

DEFAULT_SUPERVISOR_PROMPT = """\
Ти — координатор команди агентів. Твоя задача: на основі поточної історії розмови і запиту користувача обрати наступного агента, який має продовжити роботу. Якщо задача виконана — відповідай END.

Доступні агенти:
{agents_description}

Правила:
- Обирай тільки одного агента або END.
- Якщо користувач уже отримав достатню відповідь, обирай END.
- Не повторюй того самого агента поспіль без явної потреби.

Відповідай у JSON: {{"next": "<agent_role>" або "END", "reasoning": "коротке обґрунтування"}}.
"""


class SupervisorDecision(BaseModel):
    next: str
    reasoning: str


def make_supervisor_node(team: TeamConfig, settings: Settings):
    agents_desc = "\n".join(
        f"- {a.role}: {a.system_prompt[:120]}..." for a in team.agents
    )
    prompt = (team.supervisor_prompt or DEFAULT_SUPERVISOR_PROMPT).format(
        agents_description=agents_desc
    )
    llm = make_chat_model(
        provider=settings.supervisor_provider,
        model=settings.supervisor_model,
        temperature=0.0,
        timeout=settings.llm_timeout_seconds,
        settings=settings,
    ).with_structured_output(SupervisorDecision)

    valid_roles = {a.role for a in team.agents}

    def node(state: GraphState) -> dict:
        decision: SupervisorDecision = llm.invoke(
            [SystemMessage(content=prompt), *state["messages"]]
        )
        nxt = decision.next.strip()
        if nxt != "END" and nxt not in valid_roles:
            nxt = "END"
        return {
            "next_agent": nxt,
            "last_reasoning": decision.reasoning,
            "iterations": state["iterations"] + 1,
        }

    return node

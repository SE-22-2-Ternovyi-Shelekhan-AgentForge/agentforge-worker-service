from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentforge_worker.config import Settings
from agentforge_worker.contracts import TeamConfig
from agentforge_worker.graph.state import GraphState
from agentforge_worker.llm import make_chat_model

# Role tag used for the synthesized final answer. The orchestrator/web client
# render this as "Підсумок команди".
SUMMARY_ROLE = "summary"

DEFAULT_SYNTH_PROMPT = """\
Ти — координатор команди агентів. Вище наведено запит користувача та внески \
кожного агента, зроблені в межах їхніх компетенцій.

Твоя задача — сформувати ЄДИНУ цілісну підсумкову відповідь користувачу, яка:
- прямо й повно відповідає на початковий запит;
- інтегрує найкорисніше з кожного внеску й узгоджує суперечності;
- структурована та зрозуміла (використовуй markdown, списки, блоки коду за потреби);
- написана від імені команди, без згадок про внутрішню кухню (не пиши «агент X сказав»).

Не додавай нічого зайвого поза підсумковою відповіддю.
"""


def make_synthesizer_node(team: TeamConfig, settings: Settings):
    prompt = team.supervisor_prompt or DEFAULT_SYNTH_PROMPT
    llm = make_chat_model(
        provider=settings.supervisor_provider,
        model=settings.supervisor_model,
        temperature=0.3,
        timeout=settings.llm_timeout_seconds,
        settings=settings,
    )

    def node(state: GraphState) -> dict:
        # The full message list holds the user request + each agent's tagged
        # contribution; hand it all to the coordinator to synthesize from.
        messages = [
            SystemMessage(content=prompt),
            *state["messages"],
            HumanMessage(
                content="Сформуй фінальну підсумкову відповідь користувачу "
                "на основі запиту та внесків команди вище."
            ),
        ]
        resp = llm.invoke(messages)
        text = resp.content if isinstance(resp, AIMessage) else str(resp)
        final = AIMessage(
            content=text,
            additional_kwargs={"agent_role": SUMMARY_ROLE, "is_final": True},
        )
        return {"messages": [final]}

    return node

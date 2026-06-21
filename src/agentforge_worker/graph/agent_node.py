from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agentforge_worker.config import Settings
from agentforge_worker.contracts import AgentConfig
from agentforge_worker.graph.state import GraphState
from agentforge_worker.llm import make_chat_model
from agentforge_worker.tools import ToolContext, get_tools

# Appended to every agent's own system prompt so it behaves as one member of a
# collaborating team rather than a lone assistant. This is what makes each
# agent answer strictly from its own competence and build on its colleagues'
# work instead of repeating it.
_TEAM_CONTEXT = """\

---
Ти працюєш у складі команди агентів над спільним запитом користувача.
Твоя роль у команді: «{role}».
{peers_line}
Вище в історії можуть бути внески колег. Дай відповідь СУВОРО в межах своєї \
компетенції ({role}): доповни, уточни, виправ або перевір роботу колег, не \
дублюючи вже сказане. Будь конкретним і корисним.
Не намагайся підсумувати роботу всієї команди — фінальний підсумок зробить \
координатор наприкінці.
"""


def make_agent_node(
    agent: AgentConfig,
    settings: Settings,
    ctx: ToolContext,
    peers: list[str] | None = None,
):
    model_name = agent.model or settings.default_model
    temperature = (
        agent.temperature if agent.temperature is not None else settings.default_temperature
    )
    tools = get_tools(agent.tools, ctx)
    llm = make_chat_model(
        provider=agent.provider,
        model=model_name,
        temperature=temperature,
        timeout=settings.llm_timeout_seconds,
        settings=settings,
    )

    other_roles = [p for p in (peers or []) if p != agent.role]
    peers_line = (
        f"Колеги в команді (їх компетенції): {', '.join(other_roles)}."
        if other_roles
        else "Ти єдиний агент у команді."
    )
    full_prompt = agent.system_prompt + _TEAM_CONTEXT.format(
        role=agent.role, peers_line=peers_line
    )

    prebuilt = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=full_prompt),
    )
    role = agent.role

    def node(state: GraphState) -> dict:
        result = prebuilt.invoke({"messages": state["messages"]})
        new_msg = result["messages"][-1]
        if isinstance(new_msg, AIMessage):
            tagged = AIMessage(
                content=f"[{role}]: {new_msg.content}",
                additional_kwargs={**new_msg.additional_kwargs, "agent_role": role},
            )
        else:
            tagged = new_msg
        return {"messages": [tagged]}

    return node

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agentforge_worker.config import Settings
from agentforge_worker.contracts import AgentConfig
from agentforge_worker.graph.state import GraphState
from agentforge_worker.llm import make_chat_model
from agentforge_worker.tools import ToolContext, get_tools


def make_agent_node(agent: AgentConfig, settings: Settings, ctx: ToolContext, teammates: list | None = None):
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
    system_prompt = agent.system_prompt
    if teammates:
        peers = "\n".join(f"- {a.role}: {a.system_prompt[:80]}..." for a in teammates)
        system_prompt += f"\n\n---\nТвоя команда:\n{peers}\nТи можеш посилатися на роботу колег у своїй відповіді."

    prebuilt = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=system_prompt),
    )
    role = agent.role

    def _turn_instruction(rnd: int) -> HumanMessage:
        # Explicit "your turn" instruction. Without it, the last message in history
        # is a teammate's tagged AIMessage (e.g. "[developer]: ...") and the model
        # tends to continue that role instead of acting in its own — producing empty
        # or echoed output. Used only for the local invoke, NOT persisted to state.
        if rnd > 1:
            body = (
                f"Це раунд правок №{rnd}. Ти агент із роллю «{role}». Уважно прочитай "
                f"зауваження колег вище. Якщо ти автор роботи — виправ свою попередню "
                f"відповідь з урахуванням цих зауважень і наведи оновлений результат "
                f"повністю. Якщо ти рецензент — перевір, чи враховано твої попередні "
                f"зауваження, і якщо все гаразд, чітко напиши, що схвалюєш роботу."
            )
        else:
            body = (
                f"Тепер твоя черга — ти агент із роллю «{role}». Виконай своє завдання "
                f"відповідно до цієї ролі, спираючись на повідомлення вище. Не повторюй і "
                f"не продовжуй чужі відповіді — дай власний внесок як «{role}»."
            )
        return HumanMessage(content=body)

    async def node(state: GraphState) -> dict:
        rnd = state.get("round") or 1
        result = await prebuilt.ainvoke(
            {"messages": list(state["messages"]) + [_turn_instruction(rnd)]}
        )
        new_msg = result["messages"][-1]
        if isinstance(new_msg, AIMessage):
            prefix = f"[{role}]: "
            content = new_msg.content
            # The model sometimes self-prepends its own role tag (mimicking the
            # "[role]: ..." format it sees in history). Strip it so we don't
            # produce a doubled "[role]: [role]: ..." prefix.
            if isinstance(content, str) and content.startswith(prefix):
                content = content[len(prefix):]
            tagged = AIMessage(
                content=f"{prefix}{content}",
                additional_kwargs={**new_msg.additional_kwargs, "agent_role": role},
            )
        else:
            tagged = new_msg
        visited = list(state.get("agents_visited") or [])
        if role not in visited:
            visited = visited + [role]
        return {"messages": [tagged], "agents_visited": visited}

    return node

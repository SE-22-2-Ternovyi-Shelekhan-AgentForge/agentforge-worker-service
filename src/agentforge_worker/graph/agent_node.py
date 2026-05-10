from langchain_core.messages import AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agentforge_worker.config import Settings
from agentforge_worker.contracts import AgentConfig
from agentforge_worker.graph.state import GraphState
from agentforge_worker.llm import make_chat_model
from agentforge_worker.tools import ToolContext, get_tools


def make_agent_node(agent: AgentConfig, settings: Settings, ctx: ToolContext):
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
    prebuilt = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=agent.system_prompt),
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

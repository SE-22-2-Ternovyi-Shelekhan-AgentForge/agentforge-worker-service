from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentforge_worker.config import Settings
from agentforge_worker.contracts import AgentSessionRequested
from agentforge_worker.graph.agent_node import make_agent_node
from agentforge_worker.graph.state import GraphState
from agentforge_worker.graph.supervisor import make_supervisor_node
from agentforge_worker.tools import ToolContext


def build_graph(req: AgentSessionRequested, settings: Settings, ctx: ToolContext) -> CompiledStateGraph:
    g = StateGraph(GraphState)
    g.add_node("supervisor", make_supervisor_node(req.team, settings))

    agent_node_names = []
    for agent in req.team.agents:
        node_name = f"agent_{agent.role}"
        agent_node_names.append(node_name)
        g.add_node(node_name, make_agent_node(agent, settings, ctx))

    g.set_entry_point("supervisor")

    max_iter = req.team.max_iterations

    def route(state: GraphState):
        if state["iterations"] >= max_iter:
            return END
        nxt = state.get("next_agent")
        if nxt in (None, "END"):
            return END
        target = f"agent_{nxt}"
        if target not in agent_node_names:
            return END
        return target

    branches: dict = {n: n for n in agent_node_names}
    branches[END] = END
    g.add_conditional_edges("supervisor", route, branches)
    for n in agent_node_names:
        g.add_edge(n, "supervisor")

    return g.compile()

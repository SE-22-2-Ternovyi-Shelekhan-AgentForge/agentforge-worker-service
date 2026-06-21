from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentforge_worker.config import Settings
from agentforge_worker.contracts import AgentSessionRequested
from agentforge_worker.graph.agent_node import make_agent_node
from agentforge_worker.graph.state import GraphState
from agentforge_worker.graph.synthesizer import make_synthesizer_node
from agentforge_worker.tools import ToolContext


def build_graph(req: AgentSessionRequested, settings: Settings, ctx: ToolContext) -> CompiledStateGraph:
    """Build a deterministic collaboration pipeline.

    Every agent in the team contributes in turn, each seeing the user request
    and the contributions of the colleagues before it, answering strictly from
    its own competence. A final synthesizer node merges all contributions into
    one coherent answer for the user:

        START → agent_1 → agent_2 → … → agent_N → synthesizer → END
    """
    g = StateGraph(GraphState)

    roles = [a.role for a in req.team.agents]
    node_names: list[str] = []
    for agent in req.team.agents:
        node_name = f"agent_{agent.role}"
        node_names.append(node_name)
        g.add_node(node_name, make_agent_node(agent, settings, ctx, peers=roles))

    g.add_node("synthesizer", make_synthesizer_node(req.team, settings))

    # Wire the agents in order, then into the synthesizer.
    g.add_edge(START, node_names[0])
    for prev, nxt in zip(node_names, node_names[1:]):
        g.add_edge(prev, nxt)
    g.add_edge(node_names[-1], "synthesizer")
    g.add_edge("synthesizer", END)

    return g.compile()

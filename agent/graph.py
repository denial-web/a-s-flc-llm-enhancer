# A-S-FLC LangGraph Agent
#
# Orchestrates the Planner → Simulator → Navigator pipeline as a LangGraph StateGraph.
# This turns next-token prediction into force-guided, chain-aware, loop-stable planning.

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agent.nodes import navigator_node, planner_node, simulator_node
from config import A_S_FLC_Config
from core.types import ChainPath, DecisionOutput, ForceBreakdown, LoopState


class AgentState(TypedDict, total=False):
    query: str
    config: A_S_FLC_Config
    raw_analysis: Dict[str, Any]
    paths: List[ChainPath]
    breakdowns: List[ForceBreakdown]
    loop_results: List[tuple[ForceBreakdown, List[LoopState], float]]
    reasoning_steps: List[str]
    decision: DecisionOutput


def build_graph() -> StateGraph:
    """Construct the A-S-FLC agent graph: plan → simulate → navigate."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("simulator", simulator_node)
    graph.add_node("navigator", navigator_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "simulator")
    graph.add_edge("simulator", "navigator")
    graph.add_edge("navigator", END)

    return graph.compile()


def run_agent(
    query: str,
    raw_analysis: Dict[str, Any],
    config: Optional[A_S_FLC_Config] = None,
) -> DecisionOutput:
    """Run the full A-S-FLC agent pipeline.

    Args:
        query: The user's original query.
        raw_analysis: Parsed LLM output with keys 'positives', 'negatives', 'chains'.
        config: Optional config override.

    Returns:
        DecisionOutput with the chosen action and full breakdown.
    """
    if config is None:
        config = A_S_FLC_Config()

    app = build_graph()
    initial_state: AgentState = {
        "query": query,
        "config": config,
        "raw_analysis": raw_analysis,
        "reasoning_steps": [],
    }

    final_state = app.invoke(initial_state)
    return final_state["decision"]

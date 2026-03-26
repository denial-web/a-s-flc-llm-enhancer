# A-S-FLC Agent Nodes
#
# Planner, Simulator, and Navigator nodes for the LangGraph agent.
# Each node transforms the shared state according to A-S-FLC principles:
# - Positives = exact, 100% trusted
# - Negatives = estimated + buffer δ
# - Event chains = path-dependent branches
# - Loops = forward simulation until stable

from typing import Any, Dict, List

from config import A_S_FLC_Config
from core.chains import build_tree_from_llm_output, enumerate_paths
from core.forces import rank_chains
from core.loops import run_all_chains
from core.types import DecisionOutput, ForceBreakdown


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract forces and build event chains from the LLM's initial analysis.

    Expects state keys:
        - raw_analysis: dict with 'positives', 'negatives', 'chains' from LLM
        - config: A_S_FLC_Config
    """
    config: A_S_FLC_Config = state["config"]
    raw = state["raw_analysis"]

    root = build_tree_from_llm_output(raw["chains"])
    paths = enumerate_paths(root, config)
    breakdowns = rank_chains(paths, config)

    state["paths"] = paths
    state["breakdowns"] = breakdowns
    state["reasoning_steps"] = [
        f"Identified {len(raw.get('positives', []))} exact positives",
        f"Identified {len(raw.get('negatives', []))} estimated negatives",
        f"Built {len(paths)} event-chain paths",
        f"Top chain net score: {breakdowns[0].net if breakdowns else 'N/A'}",
    ]
    return state


def simulator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run LCDI forward-simulation loops on all chains.

    Expects state keys:
        - paths: List[ChainPath]
        - config: A_S_FLC_Config
    """
    config: A_S_FLC_Config = state["config"]
    paths = state["paths"]

    results = run_all_chains(paths, config)

    state["loop_results"] = results
    state["reasoning_steps"].extend([
        f"Ran LCDI simulation on {len(paths)} chains",
        f"Best post-loop net: {results[0][0].net if results else 'N/A'}",
        f"Best stability score: {results[0][2] if results else 'N/A'}",
    ])
    return state


def navigator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Select the highest stable-net chain and produce the final DecisionOutput.

    Expects state keys:
        - loop_results: List[tuple[ForceBreakdown, List[LoopState], float]]
        - reasoning_steps: List[str]
    """
    results = state["loop_results"]

    if not results:
        state["decision"] = DecisionOutput(
            chosen_action="No viable chains found",
            breakdown=ForceBreakdown(
                positives=0, negatives_estimated=0, negatives_buffered=0,
                net=0, chain_id="none", events=[],
            ),
            reasoning_steps=state["reasoning_steps"],
            stability_score=0.0,
        )
        return state

    best_breakdown, best_history, best_stability = results[0]
    all_breakdowns = [r[0] for r in results]

    state["decision"] = DecisionOutput(
        chosen_action=f"Execute chain {best_breakdown.chain_id}: {' → '.join(best_breakdown.events[:3])}",
        breakdown=best_breakdown,
        all_chains=all_breakdowns,
        reasoning_steps=state["reasoning_steps"] + [
            f"Selected {best_breakdown.chain_id} with net={best_breakdown.net}, stability={best_stability}",
        ],
        stability_score=best_stability,
    )
    return state

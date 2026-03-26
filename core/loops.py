# A-S-FLC LCDI (Loop-Chain-Delta-Iteration) Engine
#
# Loops = self-reinforcing simulation (1–3 iterations until net stabilizes).
# Stabilized Net = LCDI iteration until |Δnet| < ε

from typing import Callable, List, Optional, Tuple

from config import A_S_FLC_Config
from core.types import ChainPath, ForceBreakdown, LoopState
from core.forces import score_chain


def iterate_until_stable(
    chain: ChainPath,
    config: A_S_FLC_Config,
    perturbation_fn: Optional[Callable[[ChainPath, int], ChainPath]] = None,
) -> Tuple[ForceBreakdown, List[LoopState], float]:
    """Run LCDI forward-simulation loops on a single chain.

    Args:
        chain: The event-chain path to simulate.
        config: Framework configuration (max_iterations, epsilon, etc.).
        perturbation_fn: Optional function that mutates the chain between iterations
            to simulate forward effects (e.g. updated negatives after new info).

    Returns:
        (final_breakdown, loop_history, stability_score)
    """
    history: List[LoopState] = []
    prev_net: Optional[float] = None
    current_chain = chain

    for i in range(config.max_iterations):
        breakdown = score_chain(current_chain, config)

        delta = abs(breakdown.net - prev_net) if prev_net is not None else None
        converged = delta is not None and delta < config.epsilon

        state = LoopState(
            iteration=i + 1,
            net_score=breakdown.net,
            delta_from_previous=delta,
            converged=converged,
        )
        history.append(state)

        if converged:
            break

        prev_net = breakdown.net

        if perturbation_fn is not None:
            current_chain = perturbation_fn(current_chain, i + 1)

    nets = [s.net_score for s in history]
    if len(nets) < 2:
        stability = 1.0
    else:
        variance = sum((n - nets[-1]) ** 2 for n in nets) / len(nets)
        stability = max(0.0, 1.0 - variance)

    return breakdown, history, round(stability, 4)


def run_all_chains(
    chains: List[ChainPath],
    config: A_S_FLC_Config,
    perturbation_fn: Optional[Callable[[ChainPath, int], ChainPath]] = None,
) -> List[Tuple[ForceBreakdown, List[LoopState], float]]:
    """Run LCDI on every chain, return results sorted by net (descending)."""
    results = [
        iterate_until_stable(c, config, perturbation_fn) for c in chains
    ]
    results.sort(key=lambda r: r[0].net, reverse=True)
    return results

# A-S-FLC Force Calculators
#
# Mathematical Foundation:
#   Optimistic Base = Σ(exact positives × P(transition))
#   Buffered Negatives = est_neg + δ × uncertainty_factor
#   Net Score = Optimistic Base - Buffered Negatives

from typing import List

import numpy as np

from config import A_S_FLC_Config
from core.types import ChainPath, EventNode, ForceBreakdown


def compute_exact_positives(nodes: List[EventNode]) -> float:
    """Sum exact positives weighted by transition probability along the chain."""
    total = 0.0
    cumulative_prob = 1.0
    for node in nodes:
        cumulative_prob *= node.transition_prob
        total += node.positives * cumulative_prob
    return total


def compute_estimated_negatives(nodes: List[EventNode]) -> float:
    """Sum raw estimated negatives along the chain."""
    return sum(node.negatives for node in nodes)


def compute_uncertainty_factor(nodes: List[EventNode]) -> float:
    """Variance of per-node negatives as a measure of estimation uncertainty.
    Returns 1.0 when there is insufficient data."""
    negs = [node.negatives for node in nodes]
    if len(negs) < 2:
        return 1.0
    return float(np.var(negs)) + 1.0


def apply_buffer(
    estimated_neg: float,
    uncertainty_factor: float,
    config: A_S_FLC_Config,
) -> float:
    """Buffered Negatives = est_neg + δ × uncertainty_factor."""
    return estimated_neg + config.buffer_delta * uncertainty_factor


def score_chain(chain: ChainPath, config: A_S_FLC_Config) -> ForceBreakdown:
    """Compute a full ForceBreakdown for a single event-chain path."""
    positives = compute_exact_positives(chain.nodes)
    est_neg = compute_estimated_negatives(chain.nodes)
    uf = compute_uncertainty_factor(chain.nodes)
    buffered_neg = apply_buffer(est_neg, uf, config)
    net = positives - buffered_neg

    return ForceBreakdown(
        positives=round(positives, 4),
        negatives_estimated=round(est_neg, 4),
        negatives_buffered=round(buffered_neg, 4),
        net=round(net, 4),
        chain_id=chain.chain_id,
        events=[n.description for n in chain.nodes],
    )


def rank_chains(
    chains: List[ChainPath],
    config: A_S_FLC_Config,
) -> List[ForceBreakdown]:
    """Score and rank all chains by net score (descending)."""
    breakdowns = [score_chain(c, config) for c in chains]
    breakdowns.sort(key=lambda b: b.net, reverse=True)
    return breakdowns

# A-S-FLC Evaluation Metrics
#
# - Positive exactness: must be 100% match to ground truth.
# - Net alignment: |predicted_net − actual_outcome_net| / scale.
# - Loop stability: variance of net after 3 iterations < 0.05.
# - Chain regret: net of best possible chain minus chosen net.

from typing import Dict, List, Optional

import numpy as np

from core.types import DecisionOutput, ForceBreakdown, LoopState


def positive_exactness(predicted_positives: float, ground_truth_positives: float) -> float:
    """Measure how precisely the predicted positives match ground truth.

    Returns 1.0 for perfect match, 0.0 for completely wrong.
    Positives must be 100% trusted, so any deviation is penalized heavily.
    """
    if ground_truth_positives == 0:
        return 1.0 if predicted_positives == 0 else 0.0
    error = abs(predicted_positives - ground_truth_positives) / abs(ground_truth_positives)
    return max(0.0, 1.0 - error)


def net_alignment(predicted_net: float, actual_outcome_net: float, scale: float = 10.0) -> float:
    """Normalized error between predicted and actual net outcome.

    Returns a value in [0, 1] where 1 = perfect alignment.
    """
    error = abs(predicted_net - actual_outcome_net) / scale
    return max(0.0, 1.0 - error)


def loop_stability(loop_history: List[LoopState], threshold: float = 0.05):
    """Variance of net scores across LCDI iterations.

    Returns (variance, is_stable) where is_stable = variance < threshold.
    """
    nets = [s.net_score for s in loop_history]
    if len(nets) < 2:
        return 0.0, True
    var = float(np.var(nets))
    return var, var < threshold


def chain_regret(
    chosen_breakdown: ForceBreakdown,
    all_breakdowns: List[ForceBreakdown],
) -> float:
    """Regret = best possible chain net - chosen chain net.

    Returns 0.0 if the chosen chain is the best.
    """
    if not all_breakdowns:
        return 0.0
    best_net = max(b.net for b in all_breakdowns)
    return max(0.0, best_net - chosen_breakdown.net)


def evaluate_decision(
    decision: DecisionOutput,
    ground_truth_positives: float,
    actual_outcome_net: float,
    loop_history: Optional[List[LoopState]] = None,
) -> Dict:
    """Run all evaluation metrics on a single DecisionOutput.

    Returns a dict with all metric values.
    """
    pe = positive_exactness(decision.breakdown.positives, ground_truth_positives)
    na = net_alignment(decision.breakdown.net, actual_outcome_net)
    regret = chain_regret(decision.breakdown, decision.all_chains)

    ls_var, ls_stable = (0.0, True)
    if loop_history:
        ls_var, ls_stable = loop_stability(loop_history)

    return {
        "positive_exactness": round(pe, 4),
        "net_alignment": round(na, 4),
        "loop_stability_variance": round(ls_var, 4),
        "loop_stable": ls_stable,
        "chain_regret": round(regret, 4),
        "stability_score": decision.stability_score,
    }

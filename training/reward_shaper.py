# A-S-FLC Signed Reward Shaper for RLHF / GRPO
#
# Converts A-S-FLC net scores into reward signals for fine-tuning.
# Net score = optimistic_positive_sum − (estimated_negatives + buffer).
# The reward preserves asymmetry: positives are exact, negatives are conservatively buffered.

from typing import List, Optional

import numpy as np

from config import A_S_FLC_Config


def trajectory_uncertainty(trajectory: List[dict]) -> float:
    """Estimate uncertainty across a trajectory of steps.

    Each step dict should have a 'negatives' key.
    Returns the standard deviation of negatives (proxy for estimation noise).
    """
    negs = [step.get("negatives", 0.0) for step in trajectory]
    if len(negs) < 2:
        return 1.0
    return float(np.std(negs)) + 1.0


def signed_reward(
    trajectory: List[dict],
    positives_exact: float,
    negatives_est: float,
    config: Optional[A_S_FLC_Config] = None,
) -> float:
    """Compute the A-S-FLC signed reward for a trajectory.

    This is the core reward signal for GRPO/PPO fine-tuning:
        reward = positives_exact - (negatives_est + δ × uncertainty)

    Args:
        trajectory: List of step dicts with 'negatives' values.
        positives_exact: Known exact positive reward (100% trusted).
        negatives_est: Raw estimated total negatives.
        config: Optional config; uses default if None.

    Returns:
        Signed reward value.
    """
    if config is None:
        config = A_S_FLC_Config()

    uf = trajectory_uncertainty(trajectory)
    buffered_neg = negatives_est + config.buffer_delta * uf
    return positives_exact - buffered_neg


def batch_rewards(
    trajectories: List[List[dict]],
    positives_list: List[float],
    negatives_list: List[float],
    config: Optional[A_S_FLC_Config] = None,
) -> List[float]:
    """Compute signed rewards for a batch of trajectories."""
    return [
        signed_reward(traj, pos, neg, config)
        for traj, pos, neg in zip(trajectories, positives_list, negatives_list)
    ]


def normalize_rewards(rewards: List[float], clip_range: float = 5.0) -> List[float]:
    """Z-normalize rewards and clip to [-clip_range, clip_range].

    Useful for stable PPO/GRPO training.
    """
    arr = np.array(rewards)
    if arr.std() < 1e-8:
        return [0.0] * len(rewards)
    normalized = (arr - arr.mean()) / arr.std()
    clipped = np.clip(normalized, -clip_range, clip_range)
    return clipped.tolist()

# A-S-FLC Core Principles:
# - Positives = exact, 100% trusted (goals, known rewards, attractive forces).
# - Negatives = estimated to nearest accurate value + conservative buffer δ.
# - Net score = optimistic_positive_sum − (estimated_negatives + buffer).
# - Event chains = path-dependent branches (one action → locked sequence of follow-on events).
# - Loops = self-reinforcing simulation (1–3 iterations until net stabilizes).
# - Navigation = choose highest stable net path.

from dataclasses import dataclass


@dataclass
class A_S_FLC_Config:
    buffer_delta: float = 0.15
    max_iterations: int = 3
    max_branches: int = 5
    epsilon: float = 0.01
    llm_provider: str = "groq"
    model_name: str = "llama-3.3-70b-versatile"

# A-S-FLC Core Principles:
# - Positives = exact, 100% trusted (goals, known rewards, attractive forces).
# - Negatives = estimated to nearest accurate value + conservative buffer δ.
# - Net score = optimistic_positive_sum − (estimated_negatives + buffer).
# - Event chains = path-dependent branches.
# - Loops = self-reinforcing simulation (1–3 iterations until net stabilizes).
# - Navigation = choose highest stable net path.

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ForceBreakdown(BaseModel):
    positives: float = Field(description="Exact positive score (100% trusted)")
    negatives_estimated: float = Field(description="Raw estimated negatives")
    negatives_buffered: float = Field(description="Negatives after applying buffer δ")
    net: float = Field(description="Net score = positives - negatives_buffered")
    chain_id: str = Field(description="Unique identifier for this event chain")
    events: List[str] = Field(
        default_factory=list,
        description="Ordered sequence of co-occurring events in this chain",
    )


class DecisionOutput(BaseModel):
    chosen_action: str = Field(description="Selected action from the best chain")
    breakdown: ForceBreakdown = Field(description="Force breakdown for the chosen chain")
    all_chains: List[ForceBreakdown] = Field(
        default_factory=list,
        description="Force breakdowns for all explored chains",
    )
    reasoning_steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step reasoning trace",
    )
    stability_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How stable the net score is after LCDI loops (0-1)",
    )
    what_if_summary: Optional[str] = Field(
        default=None,
        description="What-if stress test result on the top chain",
    )
    risk_flags: List[str] = Field(
        default_factory=list,
        description="Risk warnings from what-if analysis",
    )
    # Security / routing extensions (optional; fine-tuning stages 1b+)
    risk_level: Optional[str] = Field(
        default=None,
        description="SAFE | SUSPICIOUS | DANGEROUS",
    )
    threat_type: Optional[str] = Field(
        default=None,
        description="e.g. safe, phishing, scam, injection, impersonation, fraud",
    )
    decision_route: Optional[str] = Field(
        default=None,
        description="LOCAL | BLOCK | MEMORY_STORE | MEMORY_RETRIEVE | ESCALATE (Stage 2+)",
    )
    memory_action: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Stage 2: {"op": store|retrieve|skip, "key": str, "reason": str}',
    )
    knowledge_request: Optional[str] = Field(
        default=None,
        description="One-sentence missing fact for large-model lookup, or null",
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="low_confidence | unknown_threat | complex_reasoning | null",
    )
    source: Optional[str] = Field(
        default="small",
        description="small | large_knowledge",
    )


class EventNode(BaseModel):
    """Single node in an event-chain tree."""

    id: str
    description: str
    positives: float = 0.0
    negatives: float = 0.0
    transition_prob: float = 1.0
    children: List["EventNode"] = Field(default_factory=list)


class ChainPath(BaseModel):
    """A fully-resolved path through the event-chain tree."""

    chain_id: str
    nodes: List[EventNode]
    total_positives: float = 0.0
    total_negatives: float = 0.0
    path_probability: float = 1.0


class LoopState(BaseModel):
    """Tracks the state of a single LCDI iteration."""

    iteration: int
    net_score: float
    delta_from_previous: Optional[float] = None
    converged: bool = False

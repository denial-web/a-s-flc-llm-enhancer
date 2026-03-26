# A-S-FLC Example: Travel Planner
#
# Demonstrates the framework end-to-end using a synthetic event tree
# (no LLM call required). Forces the system to use exact positives
# (budget, comfort), buffered negatives (delays, hidden costs), and
# chains (flight → layover → hotel events).

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from core.chains import enumerate_paths
from core.forces import rank_chains
from core.loops import run_all_chains
from core.types import EventNode


def build_travel_tree() -> EventNode:
    """Singapore → Tokyo trip with 3 route options."""
    direct_flight = EventNode(
        id="direct",
        description="Direct flight SIN→NRT",
        positives=8.0,
        negatives=5.5,
        transition_prob=0.95,
        children=[
            EventNode(
                id="direct-hotel",
                description="Book hotel near Shinjuku",
                positives=6.0,
                negatives=3.0,
                transition_prob=0.9,
            ),
        ],
    )

    layover_flight = EventNode(
        id="layover",
        description="Cheap flight SIN→KUL→NRT (layover)",
        positives=4.0,
        negatives=2.0,
        transition_prob=0.85,
        children=[
            EventNode(
                id="layover-delay",
                description="Potential layover delay risk",
                positives=0.0,
                negatives=3.5,
                transition_prob=0.6,
                children=[
                    EventNode(
                        id="layover-hotel",
                        description="Budget hotel near station",
                        positives=4.0,
                        negatives=1.5,
                        transition_prob=0.9,
                    ),
                ],
            ),
            EventNode(
                id="layover-smooth",
                description="Smooth connection, arrive on time",
                positives=5.0,
                negatives=0.5,
                transition_prob=0.4,
                children=[
                    EventNode(
                        id="layover-smooth-hotel",
                        description="Mid-range hotel in Shibuya",
                        positives=5.5,
                        negatives=2.5,
                        transition_prob=0.9,
                    ),
                ],
            ),
        ],
    )

    train_route = EventNode(
        id="train",
        description="Multi-leg train/ferry adventure",
        positives=9.0,
        negatives=6.0,
        transition_prob=0.7,
        children=[
            EventNode(
                id="train-hotel",
                description="Ryokan experience in Kyoto (stopover)",
                positives=7.0,
                negatives=4.0,
                transition_prob=0.8,
            ),
        ],
    )

    root = EventNode(
        id="root",
        description="Singapore to Tokyo trip planning",
        children=[direct_flight, layover_flight, train_route],
    )
    return root


def main():
    config = A_S_FLC_Config()
    root = build_travel_tree()

    print("=== A-S-FLC Travel Planner Example ===\n")

    paths = enumerate_paths(root, config)
    print(f"Enumerated {len(paths)} event chains:\n")
    for p in paths:
        events = " → ".join(n.description for n in p.nodes)
        print(f"  [{p.chain_id}] {events}")

    print("\n--- Force Rankings (pre-loop) ---\n")
    breakdowns = rank_chains(paths, config)
    for b in breakdowns:
        print(f"  {b.chain_id}: net={b.net:+.4f}  (+{b.positives} / -{b.negatives_buffered})")

    print("\n--- LCDI Simulation ---\n")
    results = run_all_chains(paths, config)
    for breakdown, history, stability in results:
        print(f"  {breakdown.chain_id}: final_net={breakdown.net:+.4f}  stability={stability}")
        for loop in history:
            tag = " [CONVERGED]" if loop.converged else ""
            delta_str = f"Δ={loop.delta_from_previous:.4f}" if loop.delta_from_previous is not None else "Δ=N/A"
            print(f"    iter {loop.iteration}: net={loop.net_score:+.4f}  {delta_str}{tag}")

    best = results[0]
    print(f"\n=== CHOSEN: {best[0].chain_id} ===")
    print(f"  Net: {best[0].net:+.4f}")
    print(f"  Stability: {best[2]}")
    print(f"  Events: {' → '.join(best[0].events)}")


if __name__ == "__main__":
    main()

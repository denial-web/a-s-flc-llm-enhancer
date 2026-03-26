"""A-S-FLC Edge-Case Synthetic Trees

Four hand-crafted event trees that stress-test the framework's core
assumptions. Each tree runs end-to-end through forces + LCDI loops
and asserts the expected winner — no LLM required.

Run:  python examples/edge_case_trees.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from core.chains import enumerate_paths
from core.forces import rank_chains
from core.loops import run_all_chains
from core.types import EventNode


def _run_tree(name: str, root: EventNode, config: A_S_FLC_Config):
    """Enumerate, rank, and LCDI-simulate a tree. Returns (paths, results)."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    paths = enumerate_paths(root, config)
    print(f"\n  {len(paths)} chains enumerated:")
    for p in paths:
        events = " -> ".join(n.id for n in p.nodes)
        print(f"    [{p.chain_id}] {events}")

    breakdowns = rank_chains(paths, config)
    print(f"\n  Pre-loop rankings:")
    for b in breakdowns:
        print(f"    {b.chain_id}: net={b.net:+.4f}  (+{b.positives:.2f} / -{b.negatives_buffered:.2f})")

    results = run_all_chains(paths, config)
    print(f"\n  Post-LCDI rankings:")
    for bd, history, stability in results:
        tag = "STABLE" if stability >= 0.99 else f"stability={stability:.4f}"
        print(f"    {bd.chain_id}: net={bd.net:+.4f}  [{tag}]")

    return paths, results


# ---------------------------------------------------------------------------
# Tree 1: High positives + high uncertainty negatives
# The "looks great on paper" option should be penalised by the buffer.
# ---------------------------------------------------------------------------

def test_high_uncertainty():
    config = A_S_FLC_Config(buffer_delta=0.3)

    risky = EventNode(
        id="risky",
        description="Risky: high reward, wild cost variance",
        positives=12.0, negatives=1.0, transition_prob=0.95,
        children=[
            EventNode(id="risky-end", description="Risky final outcome",
                      positives=2.0, negatives=9.0, transition_prob=0.9),
        ],
    )
    safe = EventNode(
        id="safe",
        description="Safe: moderate reward, predictable costs",
        positives=7.0, negatives=3.0, transition_prob=0.95,
        children=[
            EventNode(id="safe-end", description="Safe final outcome",
                      positives=4.0, negatives=3.5, transition_prob=0.9),
        ],
    )
    root = EventNode(id="root", description="High-uncertainty test", children=[risky, safe])

    _, results = _run_tree("HIGH-UNCERTAINTY: buffer should penalise risky chain", root, config)
    winner_id = results[0][0].chain_id

    risky_bd = next(r[0] for r in results if any("risky" in e.lower() for e in r[0].events))
    safe_bd = next(r[0] for r in results if any("safe" in e.lower() for e in r[0].events))

    assert safe_bd.net > risky_bd.net, (
        f"Expected safe chain to win after buffering, but risky net={risky_bd.net:.4f} > safe net={safe_bd.net:.4f}"
    )
    print(f"\n  PASS: safe chain wins (risky net={risky_bd.net:+.4f}, safe net={safe_bd.net:+.4f})")


# ---------------------------------------------------------------------------
# Tree 2: All-equal paths
# Every chain should have nearly identical net; stability should be ~1.0.
# ---------------------------------------------------------------------------

def test_all_equal():
    config = A_S_FLC_Config(buffer_delta=0.15)

    children = [
        EventNode(
            id=f"opt-{i}",
            description=f"Option {i}",
            positives=5.0, negatives=2.0, transition_prob=0.9,
        )
        for i in range(4)
    ]
    root = EventNode(id="root", description="All-equal test", children=children)

    _, results = _run_tree("ALL-EQUAL: nets should be identical", root, config)
    nets = [r[0].net for r in results]
    spread = max(nets) - min(nets)
    assert spread < 0.001, f"Expected identical nets, but spread={spread:.6f}"

    for _, _, stability in results:
        assert stability >= 0.99, f"Expected stability ~1.0 but got {stability}"

    print(f"\n  PASS: all nets within {spread:.6f}, all stable")


# ---------------------------------------------------------------------------
# Tree 3: Deep chain vs wide (shallow) chains
# Deep chain accumulates probability decay; despite higher raw positives
# it should lose to a shallow high-confidence path.
# ---------------------------------------------------------------------------

def test_deep_vs_wide():
    config = A_S_FLC_Config(buffer_delta=0.15)

    # Deep: 5 nodes, each with prob 0.8 -> cumulative ~0.33 at the end
    deep_tip = EventNode(id="deep-5", description="Deep leaf", positives=3.0, negatives=1.0, transition_prob=0.8)
    deep_4 = EventNode(id="deep-4", description="Deep step 4", positives=3.0, negatives=1.0, transition_prob=0.8, children=[deep_tip])
    deep_3 = EventNode(id="deep-3", description="Deep step 3", positives=3.0, negatives=1.0, transition_prob=0.8, children=[deep_4])
    deep_2 = EventNode(id="deep-2", description="Deep step 2", positives=3.0, negatives=1.0, transition_prob=0.8, children=[deep_3])
    deep_1 = EventNode(id="deep-1", description="Deep chain start", positives=3.0, negatives=1.0, transition_prob=0.8, children=[deep_2])

    # Wide: single-step, high confidence
    wide = EventNode(id="wide", description="Quick win", positives=7.0, negatives=2.0, transition_prob=0.95)

    root = EventNode(id="root", description="Deep-vs-wide test", children=[deep_1, wide])

    paths, results = _run_tree("DEEP-vs-WIDE: probability decay should penalise deep chain", root, config)

    deep_result = next(r for r in results if len([n for n in paths if n.chain_id == r[0].chain_id][0].nodes) > 3)
    wide_result = next(r for r in results if r is not deep_result)

    assert wide_result[0].net > deep_result[0].net, (
        f"Expected wide chain to win, but deep net={deep_result[0].net:+.4f} >= wide net={wide_result[0].net:+.4f}"
    )
    print(f"\n  PASS: wide chain wins (deep net={deep_result[0].net:+.4f}, wide net={wide_result[0].net:+.4f})")


# ---------------------------------------------------------------------------
# Tree 4: Adversarial — looks best pre-buffer, flips to worst after
# This is the key proof that asymmetric signing matters.
# ---------------------------------------------------------------------------

def test_adversarial_flip():
    config = A_S_FLC_Config(buffer_delta=0.5)

    # Trap chain: high positives, but negatives have huge variance -> big buffer penalty
    # Raw: pos=15+2=17, neg=0.5+8=8.5, raw_net=8.5  (looks great)
    # After buffer: neg variance is huge -> buffered_neg >> raw, net drops below honest
    trap = EventNode(
        id="trap",
        description="Trap: high reward, chaotic costs",
        positives=15.0, negatives=0.5, transition_prob=1.0,
        children=[
            EventNode(id="trap-end", description="Hidden cost spike",
                      positives=2.0, negatives=8.0, transition_prob=1.0),
        ],
    )

    # Honest chain: moderate everything, low variance
    # Raw: pos=8+3=11, neg=3+3=6, raw_net=5.0  (looks worse)
    # After buffer: neg variance is tiny -> buffered_neg barely grows, net stays healthy
    honest = EventNode(
        id="honest",
        description="Honest: moderate and predictable",
        positives=8.0, negatives=3.0, transition_prob=1.0,
        children=[
            EventNode(id="honest-end", description="Predictable outcome",
                      positives=3.0, negatives=3.0, transition_prob=1.0),
        ],
    )

    root = EventNode(id="root", description="Adversarial flip test", children=[trap, honest])

    # Pre-buffer check: trap should look better by raw positives - raw negatives
    trap_raw_net = (15.0 + 2.0) - (0.5 + 8.0)   # 8.5
    honest_raw_net = (8.0 + 3.0) - (3.0 + 3.0)   # 5.0
    print(f"\n  Raw (no buffer) -> trap net={trap_raw_net:+.1f}, honest net={honest_raw_net:+.1f}")
    assert trap_raw_net > honest_raw_net, "Trap should look better without buffer"

    _, results = _run_tree("ADVERSARIAL FLIP: trap chain should lose after buffering", root, config)

    trap_bd = next(r[0] for r in results if any("trap" in e.lower() for e in r[0].events))
    honest_bd = next(r[0] for r in results if any("honest" in e.lower() for e in r[0].events))

    assert honest_bd.net > trap_bd.net, (
        f"Expected honest chain to win after buffer, but trap net={trap_bd.net:+.4f} >= honest net={honest_bd.net:+.4f}"
    )
    print(f"\n  PASS: honest chain wins after buffer (trap net={trap_bd.net:+.4f}, honest net={honest_bd.net:+.4f})")
    print(f"  This proves asymmetric signing flips the naive ranking.")


def main():
    test_high_uncertainty()
    test_all_equal()
    test_deep_vs_wide()
    test_adversarial_flip()
    print(f"\n{'='*60}")
    print("  ALL EDGE-CASE TREES PASSED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

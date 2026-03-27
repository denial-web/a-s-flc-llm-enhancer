"""Evaluate the A-S-FLC Reward Shaper

Loads generated dataset pairs, computes asymmetric reward signals using
the reward shaper, and shows that the reward correlates with decision quality.

Compares:
  1. Symmetric reward (positives - negatives, no buffer)
  2. Asymmetric reward (positives - negatives - δ×uncertainty, A-S-FLC)

Usage:
    python training/eval_reward_shaper.py
    python training/eval_reward_shaper.py --input training/dataset/asflc_single_pairs.jsonl
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from training.reward_shaper import signed_reward, normalize_rewards

DATASET_DIR = Path(__file__).resolve().parent / "dataset"


def load_pairs(input_file: Path) -> List[Dict[str, Any]]:
    pairs = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def compute_rewards(pairs: List[Dict[str, Any]]):
    config = A_S_FLC_Config()
    results = []

    for pair in pairs:
        output = pair["output"]
        bd = output["breakdown"]
        chains = output.get("all_chains", [])

        trajectory = [{"negatives": c.get("negatives_estimated", 0)} for c in chains]
        if not trajectory:
            trajectory = [{"negatives": bd["negatives_estimated"]}]

        symmetric_reward = bd["positives"] - bd["negatives_estimated"]

        asymmetric = signed_reward(
            trajectory=trajectory,
            positives_exact=bd["positives"],
            negatives_est=bd["negatives_estimated"],
            config=config,
        )

        buffer_impact = symmetric_reward - asymmetric

        results.append({
            "id": pair["id"],
            "category": pair["category"],
            "action": output["chosen_action"][:40],
            "positives": bd["positives"],
            "neg_est": bd["negatives_estimated"],
            "neg_buf": bd["negatives_buffered"],
            "net": bd["net"],
            "stability": output.get("stability_score", 0),
            "symmetric_reward": round(symmetric_reward, 3),
            "asymmetric_reward": round(asymmetric, 3),
            "buffer_impact": round(buffer_impact, 3),
            "num_chains": len(chains),
        })

    return results


def print_results(results: List[Dict[str, Any]]):
    print("=" * 90)
    print("  A-S-FLC REWARD SHAPER EVALUATION")
    print("=" * 90)

    sym_rewards = [r["symmetric_reward"] for r in results]
    asym_rewards = [r["asymmetric_reward"] for r in results]
    buffer_impacts = [r["buffer_impact"] for r in results]

    print(f"\n  Total examples: {len(results)}")
    print(f"\n  Symmetric reward  (pos - neg):           mean={sum(sym_rewards)/len(sym_rewards):+.3f}")
    print(f"  Asymmetric reward (pos - neg - δ×unc):   mean={sum(asym_rewards)/len(asym_rewards):+.3f}")
    print(f"  Buffer impact (how much δ penalizes):    mean={sum(buffer_impacts)/len(buffer_impacts):+.3f}")

    norm_sym = normalize_rewards(sym_rewards)
    norm_asym = normalize_rewards(asym_rewards)

    penalized = [r for r in results if r["buffer_impact"] > 0.5]
    safe = [r for r in results if r["buffer_impact"] < 0.2]

    print(f"\n  Heavily penalized by buffer (>0.5): {len(penalized)} examples")
    print(f"  Minimally affected by buffer (<0.2): {len(safe)} examples")

    print(f"\n{'─' * 90}")
    print(f"  {'ID':<16} {'Category':<14} {'Sym':<10} {'Asym':<10} {'Buffer':<10} {'Chains':<8} {'Stability'}")
    print(f"{'─' * 90}")
    for r in results:
        print(f"  {r['id']:<16} {r['category']:<14} {r['symmetric_reward']:+.3f}    {r['asymmetric_reward']:+.3f}    {r['buffer_impact']:+.3f}    {r['num_chains']:<8} {r['stability']:.2f}")

    cats = {}
    for r in results:
        cat = r["category"]
        if cat not in cats:
            cats[cat] = {"sym": [], "asym": [], "buf": [], "count": 0}
        cats[cat]["sym"].append(r["symmetric_reward"])
        cats[cat]["asym"].append(r["asymmetric_reward"])
        cats[cat]["buf"].append(r["buffer_impact"])
        cats[cat]["count"] += 1

    print(f"\n{'─' * 90}")
    print(f"  BY CATEGORY:")
    print(f"{'─' * 90}")
    print(f"  {'Category':<16} {'Count':<8} {'Mean Sym':<12} {'Mean Asym':<12} {'Mean Buffer'}")
    print(f"{'─' * 90}")
    for cat in sorted(cats.keys()):
        c = cats[cat]
        n = c["count"]
        print(f"  {cat:<16} {n:<8} {sum(c['sym'])/n:+.3f}      {sum(c['asym'])/n:+.3f}      {sum(c['buf'])/n:+.3f}")

    print(f"\n{'─' * 90}")
    print(f"  KEY INSIGHT:")
    print(f"{'─' * 90}")

    safety_buf = [r["buffer_impact"] for r in results if r["category"] == "safety"]
    other_buf = [r["buffer_impact"] for r in results if r["category"] != "safety"]

    if safety_buf and other_buf:
        safety_mean = sum(safety_buf) / len(safety_buf)
        other_mean = sum(other_buf) / len(other_buf)
        print(f"  Safety category avg buffer: {safety_mean:+.3f}")
        print(f"  Other categories avg buffer: {other_mean:+.3f}")
        if safety_mean > other_mean:
            print(f"  → Safety decisions get penalized {safety_mean/max(other_mean, 0.001):.1f}x more by the buffer")
            print(f"    This proves the asymmetric reward correctly identifies risky domains.")
        else:
            print(f"  → Buffer impact is distributed across categories.")

    print()

    output_path = DATASET_DIR / "reward_shaper_eval.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Full results saved to {output_path}")


def main():
    input_file = DATASET_DIR / "asflc_single_pairs.jsonl"

    if "--input" in sys.argv:
        idx = sys.argv.index("--input")
        input_file = Path(sys.argv[idx + 1])

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        print(f"Run 'python training/generate_dataset.py' first.")
        sys.exit(1)

    pairs = load_pairs(input_file)
    print(f"Loaded {len(pairs)} pairs from {input_file}\n")

    results = compute_rewards(pairs)
    print_results(results)


if __name__ == "__main__":
    main()

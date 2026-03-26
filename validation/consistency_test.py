"""Consistency / Reliability Test for A-S-FLC.

Runs the same queries N times and measures how deterministic the
FG-CoT prompt makes the model's output: action agreement rate,
stability score variance, chain set overlap, and net score spread.

Usage:
    python validation/consistency_test.py [--runs N] [--cases N]
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from inference.wrapper import A_S_FLC_Wrapper
from validation.utils import (
    check_api_key,
    load_test_cases,
    print_table,
    save_results,
)


def run_consistency(
    config: A_S_FLC_Config,
    num_runs: int = 10,
    max_cases: Optional[int] = 3,
    hybrid: bool = False,
) -> List[Dict[str, Any]]:
    if not check_api_key(config):
        sys.exit(1)

    wrapper = A_S_FLC_Wrapper(config)
    cases = load_test_cases()[:max_cases]

    mode_label = "hybrid" if hybrid else "single-shot"
    print(f"Mode: {mode_label}")

    all_results = []
    for ci, case in enumerate(cases, 1):
        query = case["query"]
        print(f"\n[Case {ci}/{len(cases)}] {case['id']}: {query[:60]}...")

        runs = []
        parse_failures = 0
        for r in range(num_runs):
            try:
                decision = wrapper.decide_hybrid(query) if hybrid else wrapper.decide(query)
                runs.append({
                    "run": r + 1,
                    "chosen_action": decision.chosen_action,
                    "stability_score": decision.stability_score,
                    "net": decision.breakdown.net,
                    "chain_ids": sorted([c.chain_id for c in decision.all_chains]),
                    "num_chains": len(decision.all_chains),
                    "success": True,
                })
                sys.stdout.write(f"  run {r+1}/{num_runs}: net={decision.breakdown.net:+.4f}  "
                                 f"action={decision.chosen_action[:40]}\n")
            except Exception as e:
                parse_failures += 1
                runs.append({"run": r + 1, "success": False, "error": str(e)})
                sys.stdout.write(f"  run {r+1}/{num_runs}: FAILED ({e})\n")

        good_runs = [r for r in runs if r["success"]]

        if not good_runs:
            all_results.append({
                "case_id": case["id"],
                "total_runs": num_runs,
                "parse_failures": parse_failures,
                "action_agreement": 0.0,
                "stability_variance": None,
                "net_range": None,
                "chain_consistency": 0.0,
            })
            continue

        # Action agreement: normalize whitespace/casing before comparing
        def _normalize_action(a):
            return " ".join(a.lower().split())
        actions_raw = [r["chosen_action"] for r in good_runs]
        actions_norm = [_normalize_action(a) for a in actions_raw]
        mode_action_norm, mode_count = Counter(actions_norm).most_common(1)[0]
        action_agreement = mode_count / len(good_runs)
        mode_action = next(a for a, n in zip(actions_raw, actions_norm) if n == mode_action_norm)

        # Stability score variance
        stabilities = [r["stability_score"] for r in good_runs]
        stability_var = float(np.var(stabilities))

        # Net score range
        nets = [r["net"] for r in good_runs]
        net_range = max(nets) - min(nets)

        # Chain set consistency: Jaccard on chain_ids, with chain-count fallback
        chain_sets = [frozenset(r["chain_ids"]) for r in good_runs]
        if len(chain_sets) >= 2:
            union = frozenset.union(*chain_sets)
            intersection = frozenset.intersection(*chain_sets)
            if union:
                chain_consistency = len(intersection) / len(union)
            else:
                chain_consistency = 1.0
        else:
            chain_consistency = 1.0

        # Fallback: if Jaccard is 0 (free-form IDs), measure by chain count agreement
        chain_counts = [r["num_chains"] for r in good_runs]
        mode_count_chains = Counter(chain_counts).most_common(1)[0][1]
        chain_count_agreement = mode_count_chains / len(good_runs)

        result = {
            "case_id": case["id"],
            "total_runs": num_runs,
            "successful_runs": len(good_runs),
            "parse_failures": parse_failures,
            "action_agreement": round(action_agreement, 4),
            "mode_action": mode_action,
            "stability_variance": round(stability_var, 6),
            "stability_mean": round(float(np.mean(stabilities)), 4),
            "net_range": round(net_range, 4),
            "net_mean": round(float(np.mean(nets)), 4),
            "net_std": round(float(np.std(nets)), 4),
            "chain_consistency": round(chain_consistency, 4),
            "chain_count_agreement": round(chain_count_agreement, 4),
            "runs": runs,
        }
        all_results.append(result)

    return all_results


def print_summary(results: List[Dict[str, Any]]):
    print("\n" + "=" * 70)
    print("  CONSISTENCY TEST SUMMARY")
    print("=" * 70)

    headers = ["Case", "Action%", "StabVar", "NetRange", "NetStd", "ChainID", "ChainCnt", "Fails"]
    rows = []
    for r in results:
        chain_jacc = r.get("chain_consistency")
        chain_cnt = r.get("chain_count_agreement")
        rows.append([
            r["case_id"],
            f"{r['action_agreement']:.0%}" if r.get("action_agreement") else "N/A",
            f"{r['stability_variance']:.6f}" if r.get("stability_variance") is not None else "N/A",
            f"{r['net_range']:.4f}" if r.get("net_range") is not None else "N/A",
            f"{r.get('net_std', 'N/A')}",
            f"{chain_jacc:.2f}" if chain_jacc is not None else "N/A",
            f"{chain_cnt:.0%}" if chain_cnt is not None else "N/A",
            str(r.get("parse_failures", "?")),
        ])
    print_table(headers, rows, col_width=14)

    valid = [r for r in results if r.get("action_agreement") is not None]
    if valid:
        mean_agree = np.mean([r["action_agreement"] for r in valid])
        print(f"\n  Overall action agreement: {mean_agree:.0%}")


def main():
    num_runs = 10
    max_cases = 3
    hybrid = "--hybrid" in sys.argv

    if "--runs" in sys.argv:
        idx = sys.argv.index("--runs")
        num_runs = int(sys.argv[idx + 1])
    if "--cases" in sys.argv:
        idx = sys.argv.index("--cases")
        max_cases = int(sys.argv[idx + 1])

    config = A_S_FLC_Config()
    results = run_consistency(config, num_runs=num_runs, max_cases=max_cases, hybrid=hybrid)
    print_summary(results)
    suffix = "consistency_hybrid" if hybrid else "consistency"
    save_results(results, suffix)


if __name__ == "__main__":
    main()

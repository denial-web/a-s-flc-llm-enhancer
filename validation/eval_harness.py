"""Eval Harness: Run A-S-FLC on test cases and measure against ground truth.

Wires up eval/metrics.py with the LLM wrapper to produce quantitative
alignment scores for each test scenario.

Usage:
    python validation/eval_harness.py [--cases N]
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from eval.metrics import evaluate_decision
from inference.wrapper import A_S_FLC_Wrapper
from validation.utils import (
    check_api_key,
    load_test_cases,
    print_table,
    save_results,
)


def run_eval(
    config: A_S_FLC_Config,
    max_cases: Optional[int] = None,
    hybrid: bool = False,
    whatif: bool = False,
) -> List[Dict[str, Any]]:
    if not check_api_key(config):
        sys.exit(1)

    wrapper = A_S_FLC_Wrapper(config)
    cases = load_test_cases()
    if max_cases:
        cases = cases[:max_cases]

    if whatif:
        mode_label = "what-if"
    elif hybrid:
        mode_label = "hybrid"
    else:
        mode_label = "single-shot"
    print(f"Mode: {mode_label}")

    results = []
    for i, case in enumerate(cases, 1):
        query = case["query"]
        print(f"\n[{i}/{len(cases)}] {case['id']}: {query[:60]}...")

        try:
            if whatif:
                decision = wrapper.decide_whatif(query)
            elif hybrid:
                decision = wrapper.decide_hybrid(query)
            else:
                decision = wrapper.decide(query)
            metrics = evaluate_decision(
                decision,
                ground_truth_positives=case["ground_truth_positives"],
                actual_outcome_net=case["actual_outcome_net"],
            )
            result = {
                "case_id": case["id"],
                "category": case["category"],
                "success": True,
                "chosen_action": decision.chosen_action,
                "predicted_net": decision.breakdown.net,
                "metrics": metrics,
            }
            if whatif:
                result["what_if_summary"] = decision.what_if_summary
                result["risk_flags"] = decision.risk_flags
            print(f"  Action: {decision.chosen_action[:50]}")
            print(f"  Metrics: pe={metrics['positive_exactness']:.3f}  "
                  f"na={metrics['net_alignment']:.3f}  "
                  f"regret={metrics['chain_regret']:.3f}  "
                  f"stable={metrics['loop_stable']}")
            if whatif and decision.what_if_summary:
                print(f"  What-If: {decision.what_if_summary[:80]}")
            if whatif and decision.risk_flags:
                print(f"  Risk flags: {decision.risk_flags}")
        except Exception as e:
            result = {
                "case_id": case["id"],
                "category": case["category"],
                "success": False,
                "error": str(e),
            }
            print(f"  ERROR: {e}")

        results.append(result)

    return results


def print_summary(results: List[Dict[str, Any]]):
    print("\n" + "=" * 70)
    print("  EVAL HARNESS SUMMARY")
    print("=" * 70)

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n  {len(successful)}/{len(results)} cases succeeded, {len(failed)} failed")

    if not successful:
        print("  No successful results to summarise.")
        return

    metrics_keys = ["positive_exactness", "net_alignment", "chain_regret"]
    avgs = {}
    for key in metrics_keys:
        vals = [r["metrics"][key] for r in successful]
        avgs[key] = sum(vals) / len(vals)

    stable_count = sum(1 for r in successful if r["metrics"]["loop_stable"])
    stable_frac = stable_count / len(successful)

    print("\nAggregate metrics:")
    agg_rows = [
        ["Mean positive_exactness", f"{avgs['positive_exactness']:.4f}"],
        ["Mean net_alignment", f"{avgs['net_alignment']:.4f}"],
        ["Mean chain_regret", f"{avgs['chain_regret']:.4f}"],
        ["Loop stable fraction", f"{stable_frac:.0%}"],
    ]
    print_table(["Metric", "Value"], agg_rows, col_width=28)

    print("\nPer-case breakdown:")
    headers = ["Case", "PE", "NA", "Regret", "Stable", "Net"]
    rows = []
    for r in successful:
        m = r["metrics"]
        rows.append([
            r["case_id"],
            f"{m['positive_exactness']:.3f}",
            f"{m['net_alignment']:.3f}",
            f"{m['chain_regret']:.3f}",
            "Y" if m["loop_stable"] else "N",
            f"{r['predicted_net']:+.2f}",
        ])
    print_table(headers, rows, col_width=14)

    if failed:
        print("\nFailed cases:")
        for r in failed:
            print(f"  {r['case_id']}: {r['error']}")


def main():
    max_cases = None
    hybrid = "--hybrid" in sys.argv
    whatif = "--whatif" in sys.argv
    if "--cases" in sys.argv:
        idx = sys.argv.index("--cases")
        max_cases = int(sys.argv[idx + 1])

    config = A_S_FLC_Config()
    results = run_eval(config, max_cases, hybrid=hybrid, whatif=whatif)
    print_summary(results)
    if whatif:
        suffix = "eval_whatif"
    elif hybrid:
        suffix = "eval_hybrid"
    else:
        suffix = "eval_harness"
    save_results(results, suffix)


if __name__ == "__main__":
    main()

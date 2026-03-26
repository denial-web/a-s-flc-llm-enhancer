"""A-S-FLC Ablation Study

Tests whether each component of A-S-FLC contributes to performance by
systematically removing one component at a time and measuring the impact.

Conditions:
  1. full        — Full A-S-FLC (δ=0.15, 3 iterations, 3-5 chains)
  2. no_buffer   — Remove asymmetric buffer (δ=0)
  3. no_loops    — Remove LCDI simulation (max_iterations=1)
  4. single_chain— Only 1 chain (no alternative comparison)
  5. std_cot     — Standard CoT baseline ("think step by step")

Usage:
    python validation/ablation_study.py
    python validation/ablation_study.py --cases 5
"""

import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from core.types import DecisionOutput, ForceBreakdown
from eval.metrics import evaluate_decision
from inference.fg_cot_prompt import FG_COT_SYSTEM_PROMPT, FG_COT_USER_TEMPLATE
from inference.wrapper import A_S_FLC_Wrapper, _extract_json
from validation.utils import (
    call_llm,
    check_api_key,
    create_llm_client,
    load_test_cases,
    print_table,
    save_results,
)

# -- Ablated prompts ----------------------------------------------------------

NO_BUFFER_SYSTEM = FG_COT_SYSTEM_PROMPT.replace(
    "apply buffer δ={buffer_delta}",
    "do NOT apply any buffer (δ=0, use raw negatives as-is)",
).replace(
    "ESTIMATED NEGATIVES: List costs/obstacles on the 0–10 scale + apply buffer δ={buffer_delta}.",
    "ESTIMATED NEGATIVES: List costs/obstacles on the 0–10 scale. Use them as-is with NO buffer.",
).replace(
    "buffered_neg = est_neg + δ×uncertainty",
    "buffered_neg = est_neg (no buffer applied)",
).replace(
    "negatives_buffered\": <float: est + δ×uncertainty>",
    "negatives_buffered\": <float: same as negatives_estimated, no buffer>",
)

NO_LOOPS_SYSTEM = FG_COT_SYSTEM_PROMPT.replace(
    "Loops = self-reinforcing simulation (1–3 iterations until net stabilizes).",
    "Loops = DISABLED. Score each chain exactly once, no re-simulation.",
).replace(
    "LOOP ITERATION: Simulate 1–3 steps ahead, update forces, re-score until net changes < {epsilon}.",
    "SKIP loop iteration. Use the first-pass score directly.",
)

SINGLE_CHAIN_SYSTEM = FG_COT_SYSTEM_PROMPT.replace(
    "BUILD 3–5 EVENT CHAINS",
    "BUILD EXACTLY 1 EVENT CHAIN (the single best option you can identify)",
).replace(
    "Choose highest stable net chain.",
    "Report the score for your single chain.",
)

STD_COT_SYSTEM = """You are a helpful decision-making assistant. Think step by step.

For any decision query:
1. List the pros and cons of each option.
2. Weigh them carefully.
3. Make a clear recommendation.

Score your analysis on a 0–10 scale:
- positives: overall benefit score (0–10)
- negatives_estimated: overall cost/risk score (0–10)
- net: positives minus negatives_estimated

Output ONLY valid JSON matching this schema. No other text.
{{
  "chosen_action": "<string>",
  "breakdown": {{
    "positives": <float 0-10>,
    "negatives_estimated": <float 0-10>,
    "negatives_buffered": <float: same as negatives_estimated>,
    "net": <float>,
    "chain_id": "chain-0",
    "events": ["<string>", ...]
  }},
  "all_chains": [
    {{ same schema }}
  ],
  "reasoning_steps": ["<string>", ...],
  "stability_score": <float 0-1>
}}
"""

# -- Condition definitions -----------------------------------------------------

CONDITIONS = [
    {
        "name": "full",
        "label": "Full A-S-FLC",
        "description": "All components enabled (δ=0.15, 3 iter, 3-5 chains)",
        "system_prompt": FG_COT_SYSTEM_PROMPT,
        "config_overrides": {},
    },
    {
        "name": "no_buffer",
        "label": "No Buffer (δ=0)",
        "description": "Asymmetric signing disabled — negatives used as-is",
        "system_prompt": NO_BUFFER_SYSTEM,
        "config_overrides": {"buffer_delta": 0.0},
    },
    {
        "name": "no_loops",
        "label": "No LCDI Loops",
        "description": "Loop simulation disabled — single-pass scoring only",
        "system_prompt": NO_LOOPS_SYSTEM,
        "config_overrides": {"max_iterations": 1},
    },
    {
        "name": "single_chain",
        "label": "Single Chain",
        "description": "Only 1 event chain — no alternative comparison",
        "system_prompt": SINGLE_CHAIN_SYSTEM,
        "config_overrides": {"max_branches": 1},
    },
    {
        "name": "std_cot",
        "label": "Standard CoT",
        "description": "Plain 'think step by step' baseline — no A-S-FLC",
        "system_prompt": STD_COT_SYSTEM,
        "config_overrides": {},
    },
]


def _normalize_decision_json(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in missing ForceBreakdown fields so Pydantic validation succeeds.

    Standard CoT and other weak baselines often return simplified JSON
    missing chain-level detail. This backfills defaults so we can still
    compute metrics.
    """
    bd = parsed.get("breakdown", {})
    bd.setdefault("positives", 0.0)
    bd.setdefault("negatives_estimated", 0.0)
    bd.setdefault("negatives_buffered", bd.get("negatives_estimated", 0.0))
    bd.setdefault("net", bd.get("positives", 0.0) - bd.get("negatives_buffered", 0.0))
    bd.setdefault("chain_id", "chain-0")
    bd.setdefault("events", [])
    parsed["breakdown"] = bd

    chains = parsed.get("all_chains", [])
    normalized_chains = []
    for i, chain in enumerate(chains):
        if isinstance(chain, dict):
            chain.setdefault("positives", chain.get("breakdown", {}).get("positives", 0.0))
            chain.setdefault("negatives_estimated", chain.get("breakdown", {}).get("negatives_estimated", 0.0))
            chain.setdefault("negatives_buffered", chain.get("negatives_estimated", 0.0))
            chain.setdefault("net", chain.get("positives", 0.0) - chain.get("negatives_buffered", 0.0))
            chain.setdefault("chain_id", f"chain-{i}")
            chain.setdefault("events", chain.get("breakdown", {}).get("events", []))
            normalized_chains.append(chain)
    parsed["all_chains"] = normalized_chains if normalized_chains else [bd]

    parsed.setdefault("reasoning_steps", [])
    parsed.setdefault("stability_score", 0.0)
    parsed.setdefault("chosen_action", "unknown")

    return parsed


def _run_condition(
    condition: Dict[str, Any],
    client,
    config: A_S_FLC_Config,
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run one ablation condition across all test cases."""
    name = condition["name"]
    system_tpl = condition["system_prompt"]

    # Build config with overrides
    cfg = deepcopy(config)
    for k, v in condition["config_overrides"].items():
        setattr(cfg, k, v)

    # Format system prompt
    try:
        system = system_tpl.format(buffer_delta=cfg.buffer_delta, epsilon=cfg.epsilon)
    except KeyError:
        system = system_tpl

    results = []
    for i, case in enumerate(cases, 1):
        query = case["query"]
        user_prompt = FG_COT_USER_TEMPLATE.format(query=query)

        try:
            raw = call_llm(client, cfg, system, user_prompt)
            cleaned = _extract_json(raw)
            parsed = json.loads(cleaned)
            parsed = _normalize_decision_json(parsed)
            decision = DecisionOutput.model_validate(parsed)

            metrics = evaluate_decision(
                decision,
                ground_truth_positives=case["ground_truth_positives"],
                actual_outcome_net=case["actual_outcome_net"],
            )
            results.append({
                "case_id": case["id"],
                "success": True,
                "chosen_action": decision.chosen_action,
                "predicted_net": decision.breakdown.net,
                "num_chains": len(decision.all_chains),
                "stability_score": decision.stability_score,
                "metrics": metrics,
            })
            print(f"  [{name}] {case['id']}: net={decision.breakdown.net:+.2f}  "
                  f"pe={metrics['positive_exactness']:.3f}  "
                  f"na={metrics['net_alignment']:.3f}  "
                  f"regret={metrics['chain_regret']:.3f}")
        except Exception as e:
            results.append({
                "case_id": case["id"],
                "success": False,
                "error": str(e),
            })
            print(f"  [{name}] {case['id']}: FAILED ({e})")

    successful = [r for r in results if r["success"]]
    n = len(successful) if successful else 1

    return {
        "condition": name,
        "label": condition["label"],
        "description": condition["description"],
        "success_rate": f"{len(successful)}/{len(results)}",
        "mean_pe": round(sum(r["metrics"]["positive_exactness"] for r in successful) / n, 4),
        "mean_na": round(sum(r["metrics"]["net_alignment"] for r in successful) / n, 4),
        "mean_regret": round(sum(r["metrics"]["chain_regret"] for r in successful) / n, 4),
        "mean_chains": round(sum(r.get("num_chains", 0) for r in successful) / n, 1),
        "loop_stable_frac": round(sum(1 for r in successful if r["metrics"]["loop_stable"]) / n, 2),
        "per_case": results,
    }


def run_ablation(config: A_S_FLC_Config, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
    if not check_api_key(config):
        sys.exit(1)

    client = create_llm_client(config)
    cases = load_test_cases()
    if max_cases:
        cases = cases[:max_cases]

    print(f"Running ablation study: {len(CONDITIONS)} conditions × {len(cases)} cases")
    print(f"Model: {config.llm_provider}/{config.model_name}\n")

    all_results = []
    for condition in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"  {condition['label']}: {condition['description']}")
        print(f"{'='*60}")
        result = _run_condition(condition, client, config, cases)
        all_results.append(result)

    return all_results


def print_ablation_summary(results: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("  ABLATION STUDY RESULTS")
    print("=" * 80)

    headers = ["Condition", "Success", "PE", "NA", "Regret", "Chains", "Stable"]
    rows = []
    for r in results:
        rows.append([
            r["label"],
            r["success_rate"],
            f"{r['mean_pe']:.4f}",
            f"{r['mean_na']:.4f}",
            f"{r['mean_regret']:.4f}",
            f"{r['mean_chains']:.1f}",
            f"{r['loop_stable_frac']:.0%}",
        ])
    print_table(headers, rows, col_width=16)

    # Compute deltas from full
    full = results[0]
    print("\nDelta from Full A-S-FLC:")
    headers = ["Condition", "ΔPE", "ΔNA", "ΔRegret", "Impact"]
    rows = []
    for r in results[1:]:
        dpe = r["mean_pe"] - full["mean_pe"]
        dna = r["mean_na"] - full["mean_na"]
        dreg = r["mean_regret"] - full["mean_regret"]

        if dpe < -0.05 or dna < -0.05 or dreg > 0.05:
            impact = "HURTS"
        elif abs(dpe) < 0.02 and abs(dna) < 0.02 and abs(dreg) < 0.02:
            impact = "minimal"
        else:
            impact = "mixed"

        rows.append([
            r["label"],
            f"{dpe:+.4f}",
            f"{dna:+.4f}",
            f"{dreg:+.4f}",
            impact,
        ])
    print_table(headers, rows, col_width=16)

    print("\nKey:")
    print("  PE = Positive Exactness (higher is better, 1.0 = perfect)")
    print("  NA = Net Alignment (higher is better, 1.0 = perfect)")
    print("  Regret = Chain Regret (lower is better, 0.0 = always picks best)")
    print("  HURTS = removing this component degrades performance")
    print("  minimal = removing this component has negligible effect")


def main():
    max_cases = None
    if "--cases" in sys.argv:
        idx = sys.argv.index("--cases")
        max_cases = int(sys.argv[idx + 1])

    config = A_S_FLC_Config()
    results = run_ablation(config, max_cases)
    print_ablation_summary(results)
    save_results(results, "ablation")


if __name__ == "__main__":
    main()

"""Three-Way Comparison: No-CoT vs Standard CoT vs FG-CoT.

Runs each test case through three prompting strategies on the same model
and scores them with automated heuristics: structure, risk awareness,
reasoning depth, and JSON parse success (FG-CoT only).

Usage:
    python validation/three_way_compare.py [--cases N]
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from core.types import DecisionOutput
from eval.metrics import evaluate_decision
from inference.fg_cot_prompt import FG_COT_SYSTEM_PROMPT, FG_COT_USER_TEMPLATE
from validation.utils import (
    call_llm,
    check_api_key,
    create_llm_client,
    load_test_cases,
    print_table,
    save_results,
)

NO_COT_SYSTEM = (
    "You are a helpful assistant. Answer directly with your recommendation. "
    "Be concise."
)

STANDARD_COT_SYSTEM = (
    "You are a helpful assistant. Think step by step before answering. "
    "Consider pros and cons, then give your final recommendation."
)


def _extract_json(text: str) -> Optional[Dict]:
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    raw = fence.group(1).strip() if fence else None
    if raw is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        raw = brace.group(0) if brace else None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _score_output(text: str) -> Dict[str, Any]:
    """Compute automated rubric scores for a single output.

    Structure scoring works for both prose (keyword detection) and
    JSON outputs (detects force breakdowns, multiple chains, reasoning steps).
    """
    # Prose-based structure: pro/con keywords
    pro_con_terms = [
        r"\bpro\b", r"\bcon\b", r"\badvantage\b", r"\bdisadvantage\b",
        r"\bbenefit\b", r"\bdrawback\b", r"\brisk\b", r"\breward\b",
        r"\bupside\b", r"\bdownside\b",
    ]
    prose_structure = sum(len(re.findall(p, text, re.IGNORECASE)) for p in pro_con_terms)

    # JSON-based structure: detect A-S-FLC schema elements
    json_structure = 0
    has_breakdown = bool(re.search(r'"positives"\s*:', text))
    has_negatives = bool(re.search(r'"negatives_buffered"\s*:', text))
    has_chains = bool(re.search(r'"all_chains"\s*:', text))
    has_reasoning = bool(re.search(r'"reasoning_steps"\s*:', text))
    has_net = bool(re.search(r'"net"\s*:', text))
    has_stability = bool(re.search(r'"stability_score"\s*:', text))

    if has_breakdown:
        json_structure += 2
    if has_negatives:
        json_structure += 2
    if has_chains:
        chain_count = len(re.findall(r'"chain_id"\s*:', text))
        json_structure += chain_count
    if has_reasoning:
        json_structure += 2
    if has_net:
        json_structure += 1
    if has_stability:
        json_structure += 1

    structure_count = prose_structure + json_structure

    # Risk awareness: keywords in prose OR negatives fields in JSON
    risk_terms = [
        r"\brisk\b", r"\bdanger\b", r"\buncertain\b", r"\bvolatil\b",
        r"\bdownside\b", r"\bcost\b", r"\bloss\b", r"\bfail\b",
        r"\bobstacle\b", r"\bthreat\b",
    ]
    risk_count = sum(len(re.findall(p, text, re.IGNORECASE)) for p in risk_terms)
    if has_negatives:
        risk_count += len(re.findall(r'"negatives_estimated"\s*:\s*[\d.]+', text))

    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]
    reasoning_depth = len(sentences)

    return {
        "structure_score": structure_count,
        "prose_structure": prose_structure,
        "json_structure": json_structure,
        "risk_mentions": risk_count,
        "reasoning_depth": reasoning_depth,
        "word_count": len(text.split()),
    }


def run_three_way(config: A_S_FLC_Config, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
    if not check_api_key(config):
        sys.exit(1)

    client = create_llm_client(config)
    cases = load_test_cases()
    if max_cases:
        cases = cases[:max_cases]

    fg_system = FG_COT_SYSTEM_PROMPT.format(
        buffer_delta=config.buffer_delta,
        epsilon=config.epsilon,
    )

    results = []
    for i, case in enumerate(cases, 1):
        query = case["query"]
        print(f"\n[{i}/{len(cases)}] {case['id']}: {query[:60]}...")

        # --- No-CoT ---
        no_cot_out = call_llm(client, config, NO_COT_SYSTEM, query)
        no_cot_scores = _score_output(no_cot_out)

        # --- Standard CoT ---
        std_cot_out = call_llm(client, config, STANDARD_COT_SYSTEM, query)
        std_cot_scores = _score_output(std_cot_out)

        # --- FG-CoT ---
        fg_user = FG_COT_USER_TEMPLATE.format(query=query)
        fg_out = call_llm(client, config, fg_system, fg_user)
        fg_scores = _score_output(fg_out)

        # Try parsing FG-CoT as DecisionOutput
        fg_json = _extract_json(fg_out)
        fg_parsed = False
        eval_metrics = None
        if fg_json:
            try:
                decision = DecisionOutput.model_validate(fg_json)
                fg_parsed = True
                fg_scores["chains_enumerated"] = len(decision.all_chains)
                fg_scores["stability_score"] = decision.stability_score

                eval_metrics = evaluate_decision(
                    decision,
                    ground_truth_positives=case["ground_truth_positives"],
                    actual_outcome_net=case["actual_outcome_net"],
                )
            except Exception:
                pass

        fg_scores["json_parse_success"] = fg_parsed

        result = {
            "case_id": case["id"],
            "category": case["category"],
            "no_cot": {"output": no_cot_out, **no_cot_scores},
            "standard_cot": {"output": std_cot_out, **std_cot_scores},
            "fg_cot": {"output": fg_out, **fg_scores},
        }
        if eval_metrics:
            result["fg_cot"]["eval_metrics"] = eval_metrics

        results.append(result)

        print(f"  No-CoT:  {no_cot_scores['word_count']}w  "
              f"struct={no_cot_scores['structure_score']}  "
              f"risk={no_cot_scores['risk_mentions']}  "
              f"depth={no_cot_scores['reasoning_depth']}")
        print(f"  Std-CoT: {std_cot_scores['word_count']}w  "
              f"struct={std_cot_scores['structure_score']}  "
              f"risk={std_cot_scores['risk_mentions']}  "
              f"depth={std_cot_scores['reasoning_depth']}")
        status = "VALID" if fg_parsed else "FAIL"
        print(f"  FG-CoT:  {fg_scores['word_count']}w  "
              f"struct={fg_scores['structure_score']}  "
              f"risk={fg_scores['risk_mentions']}  "
              f"depth={fg_scores['reasoning_depth']}  "
              f"json={status}")

    return results


def print_summary(results: List[Dict[str, Any]]):
    print("\n" + "=" * 70)
    print("  THREE-WAY COMPARISON SUMMARY")
    print("=" * 70)

    conditions = ["no_cot", "standard_cot", "fg_cot"]
    labels = ["No-CoT", "Std-CoT", "FG-CoT"]

    # Aggregate per condition
    print("\nAggregate scores (averaged across all cases):")
    headers = ["Metric"] + labels
    metrics_to_show = ["structure_score", "risk_mentions", "reasoning_depth", "word_count"]
    rows = []
    for metric in metrics_to_show:
        row = [metric]
        for cond in conditions:
            vals = [r[cond][metric] for r in results if metric in r[cond]]
            avg = sum(vals) / len(vals) if vals else 0
            row.append(f"{avg:.1f}")
        rows.append(row)

    # FG-CoT specific
    fg_parse_rate = sum(1 for r in results if r["fg_cot"].get("json_parse_success")) / len(results)
    rows.append(["FG-CoT parse rate", "—", "—", f"{fg_parse_rate:.0%}"])

    print_table(headers, rows, col_width=18)

    # Per-case comparison
    print("\nPer-case: structure_score (No / Std / FG)")
    headers = ["Case", "No-CoT", "Std-CoT", "FG-CoT", "FG parsed"]
    rows = []
    for r in results:
        rows.append([
            r["case_id"],
            str(r["no_cot"]["structure_score"]),
            str(r["standard_cot"]["structure_score"]),
            str(r["fg_cot"]["structure_score"]),
            "YES" if r["fg_cot"].get("json_parse_success") else "NO",
        ])
    print_table(headers, rows, col_width=14)

    # Winner analysis
    print("\nWinner by structure_score per case:")
    for r in results:
        scores = {
            "No-CoT": r["no_cot"]["structure_score"],
            "Std-CoT": r["standard_cot"]["structure_score"],
            "FG-CoT": r["fg_cot"]["structure_score"],
        }
        winner = max(scores, key=scores.get)
        print(f"  {r['case_id']}: {winner} ({scores[winner]})")


def main():
    max_cases = None
    if "--cases" in sys.argv:
        idx = sys.argv.index("--cases")
        max_cases = int(sys.argv[idx + 1])

    config = A_S_FLC_Config()
    results = run_three_way(config, max_cases)
    print_summary(results)
    save_results(results, "three_way")


if __name__ == "__main__":
    main()

"""A/B Comparison: Vanilla LLM vs FG-CoT (A-S-FLC) prompt.

Runs each test case through two system prompts on the same model and
compares structure, chain count, reasoning depth, and JSON parse success.

Usage:
    python validation/ab_compare.py [--cases N]
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from core.types import DecisionOutput
from inference.fg_cot_prompt import FG_COT_SYSTEM_PROMPT, FG_COT_USER_TEMPLATE
from validation.utils import (
    call_llm,
    check_api_key,
    create_llm_client,
    load_test_cases,
    print_table,
    save_results,
)

VANILLA_SYSTEM = (
    "You are a helpful planning assistant. Analyse the query carefully, "
    "weigh the pros and cons, and provide your recommendation with reasoning."
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


def _count_pros_cons(text: str) -> int:
    patterns = [r"\bpro\b", r"\bcon\b", r"\badvantage\b", r"\bdisadvantage\b",
                r"\bbenefit\b", r"\bdrawback\b", r"\brisk\b", r"\breward\b"]
    return sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)


def run_ab(config: A_S_FLC_Config, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
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

        # --- Control: vanilla ---
        vanilla_out = call_llm(client, config, VANILLA_SYSTEM, query)

        # --- Treatment: FG-CoT ---
        fg_user = FG_COT_USER_TEMPLATE.format(query=query)
        fg_out = call_llm(client, config, fg_system, fg_user)

        # --- Analysis ---
        fg_json = _extract_json(fg_out)
        fg_parsed = False
        fg_chains = 0
        fg_steps = 0
        fg_stability = None
        if fg_json:
            try:
                decision = DecisionOutput.model_validate(fg_json)
                fg_parsed = True
                fg_chains = len(decision.all_chains)
                fg_steps = len(decision.reasoning_steps)
                fg_stability = decision.stability_score
            except Exception:
                pass

        result = {
            "case_id": case["id"],
            "category": case["category"],
            "vanilla": {
                "output": vanilla_out,
                "word_count": len(vanilla_out.split()),
                "pros_cons_mentions": _count_pros_cons(vanilla_out),
            },
            "fg_cot": {
                "output": fg_out,
                "json_parse_success": fg_parsed,
                "chains_enumerated": fg_chains,
                "reasoning_steps": fg_steps,
                "stability_score": fg_stability,
                "word_count": len(fg_out.split()),
            },
        }
        results.append(result)

        status = "VALID JSON" if fg_parsed else "PARSE FAIL"
        print(f"  Vanilla: {result['vanilla']['word_count']} words, "
              f"{result['vanilla']['pros_cons_mentions']} pro/con mentions")
        print(f"  FG-CoT:  {status}, {fg_chains} chains, {fg_steps} reasoning steps")

    return results


def print_summary(results: List[Dict[str, Any]]):
    print("\n" + "=" * 70)
    print("  A/B COMPARISON SUMMARY")
    print("=" * 70)

    parse_rate = sum(1 for r in results if r["fg_cot"]["json_parse_success"]) / len(results)
    avg_chains = sum(r["fg_cot"]["chains_enumerated"] for r in results) / len(results)
    avg_steps = sum(r["fg_cot"]["reasoning_steps"] for r in results) / len(results)
    avg_vanilla_pros = sum(r["vanilla"]["pros_cons_mentions"] for r in results) / len(results)

    rows = [
        ["FG-CoT parse rate", f"{parse_rate:.0%}"],
        ["Avg chains enumerated", f"{avg_chains:.1f}"],
        ["Avg reasoning steps", f"{avg_steps:.1f}"],
        ["Avg vanilla pro/con mentions", f"{avg_vanilla_pros:.1f}"],
    ]
    print_table(["Metric", "Value"], rows, col_width=30)

    print("\nPer-case breakdown:")
    headers = ["Case", "Vanilla words", "FG-CoT parsed", "Chains", "Steps"]
    rows = []
    for r in results:
        rows.append([
            r["case_id"],
            str(r["vanilla"]["word_count"]),
            "YES" if r["fg_cot"]["json_parse_success"] else "NO",
            str(r["fg_cot"]["chains_enumerated"]),
            str(r["fg_cot"]["reasoning_steps"]),
        ])
    print_table(headers, rows, col_width=16)


def main():
    max_cases = None
    if "--cases" in sys.argv:
        idx = sys.argv.index("--cases")
        max_cases = int(sys.argv[idx + 1])

    config = A_S_FLC_Config()
    results = run_ab(config, max_cases)
    print_summary(results)
    save_results(results, "ab_compare")


if __name__ == "__main__":
    main()

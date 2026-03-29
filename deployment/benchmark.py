"""Performance Benchmark — Measure inference speed against mobile targets.

Runs a fixed set of queries across device tiers and compares against
the performance budget defined in mobile_config.py.

Usage:
    python deployment/benchmark.py --model path/to/model.gguf
    python deployment/benchmark.py --model path/to/model.gguf --tiers high_end mid_range
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deployment.local_inference import LocalRunner
from deployment.mobile_config import (
    DEVICE_TARGETS,
    PERFORMANCE_BUDGET,
    get_config_for_device,
)

BENCH_QUERIES = [
    {
        "query": "Should I buy a used car for $8000 or lease a new one for $300/month?",
        "mode": "single",
        "label": "basic_decision",
    },
    {
        "query": "I got an email from my bank asking me to verify my account by clicking a link.",
        "mode": "security",
        "label": "phishing_check",
    },
    {
        "query": "I am allergic to peanuts. Remember this for food recommendations.",
        "mode": "memory",
        "label": "memory_store",
    },
    {
        "query": (
            "I have $50k savings. Should I invest in index funds, buy rental "
            "property, or start a small business? I'm 32 with stable income."
        ),
        "mode": "single",
        "label": "complex_multi_option",
    },
    {
        "query": "Accept a remote job at 20% less pay or stay in-office at current salary?",
        "mode": "single",
        "label": "career_tradeoff",
    },
]


def run_benchmark(model_path: str, tiers: List[str]) -> Dict[str, Any]:
    all_results = {}

    for tier in tiers:
        target = DEVICE_TARGETS.get(tier)
        config = get_config_for_device(tier)
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {tier} — {target.name if target else 'unknown'}")
        print(f"  n_ctx={config.n_ctx} max_tokens={config.max_tokens} "
              f"n_threads={config.n_threads}")
        print(f"{'='*70}")

        runner = LocalRunner(model_path, config)
        tier_results = []

        for bq in BENCH_QUERIES:
            label = bq["label"]
            print(f"\n  [{label}] ({bq['mode']})")

            latencies = []
            tokens_list = []
            valid_count = 0
            runs = 2

            for r in range(runs):
                result = runner.generate(bq["query"], mode=bq["mode"])
                if result["blocked"]:
                    print(f"    run {r+1}: BLOCKED")
                    continue
                latencies.append(result["latency_ms"])
                tokens_list.append(result["tokens"])
                if result["valid_json"]:
                    valid_count += 1
                print(f"    run {r+1}: {result['latency_ms']:.0f}ms, "
                      f"{result['tokens_per_sec']:.1f} tok/s, "
                      f"valid={result['valid_json']}")

            if latencies:
                avg_lat = sum(latencies) / len(latencies)
                avg_tok = sum(tokens_list) / len(tokens_list)
                avg_tok_sec = avg_tok / (avg_lat / 1000) if avg_lat > 0 else 0
            else:
                avg_lat = avg_tok = avg_tok_sec = 0

            entry = {
                "label": label,
                "mode": bq["mode"],
                "avg_latency_ms": round(avg_lat, 1),
                "avg_tokens": round(avg_tok, 1),
                "avg_tok_per_sec": round(avg_tok_sec, 1),
                "valid_rate": valid_count / runs if runs else 0,
            }
            tier_results.append(entry)

        budget = PERFORMANCE_BUDGET
        pass_count = 0
        fail_count = 0
        for tr in tier_results:
            if tr["avg_latency_ms"] == 0:
                continue
            within_budget = tr["avg_latency_ms"] <= budget.max_total_latency_ms
            if within_budget:
                pass_count += 1
            else:
                fail_count += 1

        print(f"\n  --- {tier} Summary ---")
        print(f"  Budget: max_total_latency={budget.max_total_latency_ms}ms, "
              f"min_tok/s={budget.max_tokens_per_second}")
        print(f"  Pass: {pass_count}/{pass_count + fail_count} within latency budget")

        for tr in tier_results:
            lat_ok = "PASS" if tr["avg_latency_ms"] <= budget.max_total_latency_ms else "FAIL"
            tok_ok = "PASS" if tr["avg_tok_per_sec"] >= budget.max_tokens_per_second else "FAIL"
            print(f"    {tr['label']:25s} lat={tr['avg_latency_ms']:7.0f}ms [{lat_ok}]  "
                  f"speed={tr['avg_tok_per_sec']:5.1f} tok/s [{tok_ok}]  "
                  f"valid={tr['valid_rate']:.0%}")

        all_results[tier] = {
            "device": target.name if target else tier,
            "config": {
                "n_ctx": config.n_ctx,
                "max_tokens": config.max_tokens,
                "n_threads": config.n_threads,
            },
            "results": tier_results,
            "pass_count": pass_count,
            "fail_count": fail_count,
        }

    report_path = Path(__file__).resolve().parent / "benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBenchmark report: {report_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="A-S-FLC Performance Benchmark")
    parser.add_argument("--model", required=True, help="Path to .gguf file")
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["high_end"],
        choices=["high_end", "mid_range", "low_end"],
    )
    args = parser.parse_args()

    run_benchmark(args.model, args.tiers)


if __name__ == "__main__":
    main()

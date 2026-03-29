"""A-S-FLC Evaluation Harness — Run held-out eval set through the GGUF model.

Loads the 20 held-out IDs from eval_split.json, runs each through local
inference, validates output, and produces a summary report.

Usage:
    python training/eval_harness.py --model path/to/model.gguf
    python training/eval_harness.py --model path/to/model.gguf --tier mid_range
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.response_validator import validate_json_string
from deployment.local_inference import LocalRunner
from deployment.mobile_config import get_config_for_device

EVAL_SPLIT = Path(__file__).resolve().parent / "eval_split.json"
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent / "eval_results"


def load_eval_examples() -> List[Dict[str, Any]]:
    with open(EVAL_SPLIT) as f:
        eval_ids = set(json.load(f)["eval_ids"])

    examples = []
    for path in sorted(DATASET_DIR.glob("asflc_*_pairs.jsonl")):
        with open(path) as f:
            for line in f:
                row = json.loads(line.strip())
                if row["id"] in eval_ids:
                    examples.append(row)

    found_ids = {e["id"] for e in examples}
    missing = eval_ids - found_ids
    if missing:
        print(f"Warning: {len(missing)} eval IDs not found in training data: {missing}")

    return examples


def detect_mode(example: Dict[str, Any]) -> str:
    mode = example.get("mode", "single")
    cat = example.get("category", "")
    if mode == "security" or cat == "security":
        return "security"
    if mode == "memory" or cat == "memory":
        return "memory"
    return "single"


def run_eval(model_path: str, tier: str = "high_end") -> Dict[str, Any]:
    config = get_config_for_device(tier)
    runner = LocalRunner(model_path, config)

    examples = load_eval_examples()
    print(f"\nEval harness: {len(examples)} examples loaded")
    print(f"Tier: {tier} | max_tokens: {config.max_tokens}")
    print("=" * 70)

    results = []
    for i, ex in enumerate(examples, 1):
        eid = ex["id"]
        query = ex["input"]
        mode = detect_mode(ex)

        print(f"\n[{i}/{len(examples)}] {eid} (mode={mode})")
        print(f"  Q: {query[:80]}...")

        result = runner.generate(query, mode=mode)

        if result["blocked"]:
            entry = {
                "id": eid,
                "mode": mode,
                "blocked": True,
                "valid_json": False,
                "quality_score": 0.0,
                "latency_ms": 0,
                "tokens": 0,
                "tok_per_sec": 0,
                "issues": ["blocked by policy guard"],
            }
        else:
            entry = {
                "id": eid,
                "mode": mode,
                "blocked": False,
                "valid_json": result["valid_json"],
                "quality_score": result["validation"]["quality_score"],
                "latency_ms": result["latency_ms"],
                "tokens": result["tokens"],
                "tok_per_sec": result["tokens_per_sec"],
                "issues": result["validation"]["issues"],
            }

        status = "VALID" if entry["valid_json"] else ("BLOCKED" if entry["blocked"] else "INVALID")
        print(f"  → {status} | quality={entry['quality_score']:.2f} | "
              f"{entry['latency_ms']}ms | {entry['tok_per_sec']} tok/s")
        if entry["issues"]:
            for issue in entry["issues"][:3]:
                print(f"    ! {issue}")

        results.append(entry)

    total = len(results)
    valid = sum(1 for r in results if r["valid_json"])
    blocked = sum(1 for r in results if r["blocked"])
    invalid = total - valid - blocked
    avg_quality = sum(r["quality_score"] for r in results) / total if total else 0
    non_blocked = [r for r in results if not r["blocked"]]
    avg_latency = sum(r["latency_ms"] for r in non_blocked) / len(non_blocked) if non_blocked else 0
    avg_tok_sec = sum(r["tok_per_sec"] for r in non_blocked) / len(non_blocked) if non_blocked else 0

    by_mode: Dict[str, Dict[str, Any]] = {}
    for r in results:
        m = r["mode"]
        if m not in by_mode:
            by_mode[m] = {"total": 0, "valid": 0, "blocked": 0, "quality_sum": 0.0}
        by_mode[m]["total"] += 1
        by_mode[m]["valid"] += int(r["valid_json"])
        by_mode[m]["blocked"] += int(r["blocked"])
        by_mode[m]["quality_sum"] += r["quality_score"]

    summary = {
        "total": total,
        "valid_json": valid,
        "blocked": blocked,
        "invalid": invalid,
        "valid_json_pct": round(valid / total * 100, 1) if total else 0,
        "avg_quality": round(avg_quality, 3),
        "avg_latency_ms": round(avg_latency, 1),
        "avg_tok_per_sec": round(avg_tok_sec, 1),
        "by_mode": {
            m: {
                "total": v["total"],
                "valid": v["valid"],
                "valid_pct": round(v["valid"] / v["total"] * 100, 1),
                "avg_quality": round(v["quality_sum"] / v["total"], 3),
            }
            for m, v in by_mode.items()
        },
        "tier": tier,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total:        {total}")
    print(f"Valid JSON:   {valid}/{total} ({summary['valid_json_pct']}%)")
    print(f"Blocked:      {blocked}")
    print(f"Invalid:      {invalid}")
    print(f"Avg Quality:  {summary['avg_quality']}")
    print(f"Avg Latency:  {summary['avg_latency_ms']}ms")
    print(f"Avg Speed:    {summary['avg_tok_per_sec']} tok/s")
    print()
    for m, v in summary["by_mode"].items():
        print(f"  {m:10s}: {v['valid']}/{v['total']} valid ({v['valid_pct']}%), "
              f"avg quality={v['avg_quality']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / f"eval_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nFull report: {report_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="A-S-FLC Evaluation Harness")
    parser.add_argument("--model", required=True, help="Path to .gguf file")
    parser.add_argument("--tier", default="high_end", choices=["high_end", "mid_range", "low_end"])
    args = parser.parse_args()

    run_eval(args.model, tier=args.tier)


if __name__ == "__main__":
    main()

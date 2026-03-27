"""A-S-FLC Training Dataset Generator

Runs queries from the query bank through the A-S-FLC framework (single-shot,
what-if, and security modes) and saves input/output pairs for fine-tuning.

Usage:
    python training/generate_dataset.py
    python training/generate_dataset.py --limit 50
    python training/generate_dataset.py --mode whatif
    python training/generate_dataset.py --mode security
    python training/generate_dataset.py --resume  # continue from last checkpoint
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config
from inference.wrapper import A_S_FLC_Wrapper

QUERY_BANK = Path(__file__).resolve().parent / "query_bank.json"
SECURITY_BANK = Path(__file__).resolve().parent / "security_query_bank.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "dataset"


def _checkpoint_path(mode: str) -> Path:
    return OUTPUT_DIR / f"_checkpoint_{mode}.json"

RATE_LIMIT_DELAY = 2.5  # seconds between calls (Groq free tier: ~30/min)


def load_queries(
    limit: Optional[int] = None,
    mode: str = "single",
) -> List[Dict[str, Any]]:
    if mode == "security":
        with open(SECURITY_BANK) as f:
            queries = json.load(f)
    else:
        with open(QUERY_BANK) as f:
            queries = json.load(f)
    if limit:
        queries = queries[:limit]
    return queries


def load_checkpoint(mode: str) -> set:
    path = _checkpoint_path(mode)
    # Legacy single-mode checkpoint
    legacy = OUTPUT_DIR / "_checkpoint.json"
    if path.exists():
        with open(path) as f:
            return set(json.load(f))
    if mode == "single" and legacy.exists():
        with open(legacy) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(mode: str, completed_ids: set):
    path = _checkpoint_path(mode)
    with open(path, "w") as f:
        json.dump(sorted(completed_ids), f)


def generate_pair(
    wrapper: A_S_FLC_Wrapper,
    query_item: Dict[str, Any],
    mode: str = "single",
) -> Optional[Dict[str, Any]]:
    query = query_item["query"]
    category = query_item.get("category", "unknown")

    try:
        if mode == "whatif":
            result = wrapper.decide_whatif(query)
        elif mode == "security":
            result = wrapper.decide_security(query)
        else:
            result = wrapper.decide(query)

        output = result.model_dump()

        pair = {
            "id": query_item["id"],
            "category": category,
            "mode": mode,
            "input": query,
            "output": output,
            "output_json": result.model_dump_json(),
        }
        if query_item.get("subcategory"):
            pair["subcategory"] = query_item["subcategory"]
        return pair
    except Exception as e:
        print(f"    FAILED: {e}")
        return None


def generate_dataset(
    limit: Optional[int] = None,
    mode: str = "single",
    resume: bool = False,
):
    OUTPUT_DIR.mkdir(exist_ok=True)

    config = A_S_FLC_Config()
    wrapper = A_S_FLC_Wrapper(config)
    queries = load_queries(limit, mode=mode)

    ck_mode = mode if mode in ("security", "whatif", "single") else "single"
    completed = load_checkpoint(ck_mode) if resume else set()
    remaining = [q for q in queries if q["id"] not in completed]

    print(f"Dataset Generator")
    print(f"  Mode: {mode}")
    print(f"  Total queries: {len(queries)}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Provider: {config.llm_provider}/{config.model_name}")
    print(f"  Rate limit delay: {RATE_LIMIT_DELAY}s")
    print()

    dataset = []
    suffix = "security" if mode == "security" else mode
    output_file = OUTPUT_DIR / f"asflc_{suffix}_pairs.jsonl"

    open_mode = "a" if resume and output_file.exists() else "w"
    with open(output_file, open_mode) as f:
        for i, query_item in enumerate(remaining, 1):
            print(f"[{i}/{len(remaining)}] {query_item['id']}: {query_item['query'][:60]}...")

            pair = generate_pair(wrapper, query_item, mode)
            if pair:
                f.write(json.dumps(pair) + "\n")
                f.flush()
                dataset.append(pair)
                completed.add(query_item["id"])
                save_checkpoint(ck_mode, completed)
                act = pair["output"].get("chosen_action", "")[:50]
                print(f"    ✓ {act}")
            else:
                print(f"    ✗ Skipped")

            if i < len(remaining):
                time.sleep(RATE_LIMIT_DELAY)

    success = len(dataset)
    total = len(remaining)
    print(f"\nDone: {success}/{total} pairs generated")
    print(f"Output: {output_file}")

    summary = {
        "mode": mode,
        "total_queries": len(queries),
        "successful": len(completed),
        "failed": total - success,
        "categories": list(set(q.get("category", "?") for q in queries)),
        "provider": f"{config.llm_provider}/{config.model_name}",
        "output_file": str(output_file),
    }
    summary_path = OUTPUT_DIR / f"asflc_{suffix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")

    return dataset


def main():
    limit = None
    mode = "single"
    resume = "--resume" in sys.argv

    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        limit = int(sys.argv[idx + 1])

    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        mode = sys.argv[idx + 1]

    if "--whatif" in sys.argv:
        mode = "whatif"

    if "--security" in sys.argv:
        mode = "security"

    generate_dataset(limit=limit, mode=mode, resume=resume)


if __name__ == "__main__":
    main()

"""Format A-S-FLC dataset for HuggingFace upload.

Converts JSONL pairs into HuggingFace-ready formats:
  1. Chat format (for fine-tuning with chat templates)
  2. Instruction format (for SFT with Unsloth/TRL)
  3. Raw JSONL (for custom training scripts)

Usage:
    python training/format_for_hf.py
    python training/format_for_hf.py --input training/dataset/asflc_single_pairs.jsonl
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET_DIR = Path(__file__).resolve().parent / "dataset"

SYSTEM_SINGLE = (
    "You are an A-S-FLC decision navigator. Analyze the query using "
    "asymmetric signed force-loop-chain reasoning. Positives are exact "
    "and trusted. Negatives are estimated with a conservative buffer. "
    "Build 3-5 event chains, score each, loop until stable, and pick "
    "the best. Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1}'
)

SYSTEM_SECURITY = (
    "You are an A-S-FLC Security Navigator. Combine asymmetric force "
    "reasoning with threat assessment. Positives = apparent benefits of "
    "complying. Negatives = estimated costs/risks + conservative buffer. "
    "Build 2-4 event chains (comply vs refuse/verify). Classify: "
    "risk_level (SAFE/SUSPICIOUS/DANGEROUS), threat_type, decision_route "
    "(LOCAL/BLOCK). Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1,'
    '"risk_level":"SAFE|SUSPICIOUS|DANGEROUS","threat_type":"str|null",'
    '"decision_route":"LOCAL|BLOCK","source":"small"}'
)

SYSTEM_MEMORY = (
    "You are an A-S-FLC Navigator with Memory and Routing. Combine "
    "asymmetric force reasoning with memory management. Decide "
    "decision_route (LOCAL/MEMORY_STORE/MEMORY_RETRIEVE/BLOCK/ESCALATE) "
    "and memory_action (store/retrieve/skip). Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1,'
    '"decision_route":"LOCAL|MEMORY_STORE|MEMORY_RETRIEVE|BLOCK|ESCALATE",'
    '"memory_action":{"op":"store|retrieve|skip","key":"str|null","reason":"str"},'
    '"source":"small"}'
)

SYSTEM_MULTILINGUAL = (
    "You are an A-S-FLC Navigator. Respond in the SAME language as the user query. "
    "chosen_action and reasoning_steps in the user's language. "
    "Field names and chain_id stay in English. Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1,'
    '"source":"small"}'
)

SYSTEM_PII = (
    "You are an A-S-FLC Security Navigator specializing in PII protection. "
    "Detect if the user is sharing sensitive personal data (credit cards, passwords, "
    "national IDs, bank accounts). If PII found: set decision_route to BLOCK, "
    "pii_detected to the type, and warn the user. If safe (order numbers, flight "
    "numbers): set decision_route to LOCAL. Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1,'
    '"risk_level":"SAFE|DANGEROUS","decision_route":"LOCAL|BLOCK",'
    '"pii_detected":"credit_card|password|national_id|bank_account|phone|null",'
    '"source":"small"}'
)

SYSTEM_TOOL = (
    "You are an A-S-FLC Navigator with tool awareness. Decide if the query "
    "needs a tool (web_search, calculator, reminder, translate) or can be answered "
    "directly. If a tool is needed, set tool_request with tool name, args, and reason. "
    "If no tool needed, set tool_request to null. Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1,'
    '"tool_request":{"tool":"web_search|calculator|reminder|translate","args":{},"reason":"str"},'
    '"source":"small"}'
)

SYSTEM_CREDIT = (
    "You are an A-S-FLC Navigator with cost awareness. Help users manage their "
    "cloud model usage budget. When asked about costs, limits, or credit, provide "
    "helpful advice. Use memory_action to store/retrieve budget preferences. "
    "If no spending limit is set, proactively suggest setting one. Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1,'
    '"decision_route":"LOCAL|MEMORY_STORE","memory_action":{"op":"store|retrieve|skip","key":"str|null","reason":"str"},'
    '"source":"small"}'
)

SYSTEM_MESSAGES = {
    "single": SYSTEM_SINGLE,
    "whatif": SYSTEM_SINGLE,
    "security": SYSTEM_SECURITY,
    "memory": SYSTEM_MEMORY,
    "khmer": SYSTEM_MULTILINGUAL,
    "chinese": SYSTEM_MULTILINGUAL,
    "korean": SYSTEM_MULTILINGUAL,
    "pii": SYSTEM_PII,
    "tool": SYSTEM_TOOL,
    "credit": SYSTEM_CREDIT,
}


def _system_for_mode(mode: str) -> str:
    return SYSTEM_MESSAGES.get(mode, SYSTEM_SINGLE)


def load_pairs(input_file: Path) -> List[Dict[str, Any]]:
    pairs = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def to_chat_format(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert to chat format for fine-tuning with chat templates."""
    formatted = []
    for pair in pairs:
        mode = pair.get("mode", "single")
        formatted.append({
            "messages": [
                {"role": "system", "content": _system_for_mode(mode)},
                {"role": "user", "content": pair["input"]},
                {"role": "assistant", "content": pair["output_json"]},
            ],
            "category": pair["category"],
            "id": pair["id"],
            "mode": mode,
        })
    return formatted


def to_instruction_format(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert to instruction format for SFT with Unsloth/TRL."""
    formatted = []
    for pair in pairs:
        mode = pair.get("mode", "single")
        formatted.append({
            "instruction": _system_for_mode(mode),
            "input": pair["input"],
            "output": pair["output_json"],
            "category": pair["category"],
            "id": pair["id"],
            "mode": mode,
        })
    return formatted


def save_jsonl(data: List[Dict], output_path: Path):
    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"  Saved {len(data)} examples to {output_path}")


def load_all_pair_files() -> List[Dict[str, Any]]:
    """Merge every asflc_*_pairs.jsonl under dataset/ (dedupe by id, last wins)."""
    merged: Dict[str, Dict[str, Any]] = {}
    for path in sorted(DATASET_DIR.glob("asflc_*_pairs.jsonl")):
        for row in load_pairs(path):
            merged[row["id"]] = row
    return list(merged.values())


def format_dataset(input_file: Optional[Path] = None, merge_all: bool = False):
    if merge_all:
        pairs = load_all_pair_files()
        print(f"Loaded {len(pairs)} merged pairs from asflc_*_pairs.jsonl")
    else:
        assert input_file is not None
        pairs = load_pairs(input_file)
        print(f"Loaded {len(pairs)} pairs from {input_file}")

    categories = {}
    for p in pairs:
        cat = p["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    chat_data = to_chat_format(pairs)
    save_jsonl(chat_data, DATASET_DIR / "asflc_chat_format.jsonl")

    instruction_data = to_instruction_format(pairs)
    save_jsonl(instruction_data, DATASET_DIR / "asflc_instruction_format.jsonl")

    readme = f"""---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - decision-making
  - chain-of-thought
  - asymmetric-reasoning
  - force-guided
  - structured-output
size_categories:
  - n<1K
---

# A-S-FLC Decision Dataset

Training data for fine-tuning LLMs on **Asymmetric Signed Force-Loop-Chain** reasoning.

## What is A-S-FLC?

A decision-making framework where:
- **Positives** are trusted exactly (known benefits)
- **Negatives** are estimated with a conservative buffer proportional to uncertainty
- Multiple event chains are scored and the highest stable-net path is chosen

This catches "trap" decisions where uncertain downsides are underestimated.

## Dataset Details

- **Examples**: {len(pairs)}
- **Categories**: {', '.join(sorted(categories.keys()))}
- **Generated by**: Llama 3.3 70B via Groq with FG-CoT prompt
- **Format**: Each example is a decision query → structured JSON output

## Formats

- `asflc_chat_format.jsonl` — chat messages format (system/user/assistant)
- `asflc_instruction_format.jsonl` — instruction/input/output format (Alpaca-style)

## Output Schema

```json
{{
  "chosen_action": "string",
  "breakdown": {{
    "positives": "float 0-10",
    "negatives_estimated": "float 0-10",
    "negatives_buffered": "float (with delta buffer)",
    "net": "float",
    "chain_id": "string",
    "events": ["event1", "event2"]
  }},
  "all_chains": ["..."],
  "reasoning_steps": ["..."],
  "stability_score": "float 0-1",
  "risk_level": "SAFE | SUSPICIOUS | DANGEROUS (security mode)",
  "threat_type": "string or null",
  "decision_route": "LOCAL | BLOCK | MEMORY_STORE | MEMORY_RETRIEVE | ESCALATE",
  "memory_action": {{"op": "store|retrieve|skip", "key": "str", "reason": "str"}},
  "knowledge_request": "string or null",
  "escalation_reason": "string or null",
  "source": "small | large_knowledge"
}}
```

## Usage

```python
from datasets import load_dataset
ds = load_dataset("json", data_files="asflc_chat_format.jsonl")
```

## Source

GitHub: [denial-web/a-s-flc-llm-enhancer](https://github.com/denial-web/a-s-flc-llm-enhancer)
"""

    readme_path = DATASET_DIR / "README_HF.md"
    with open(readme_path, "w") as f:
        f.write(readme)
    print(f"\n  HuggingFace README: {readme_path}")
    print(f"\nTo upload to HuggingFace:")
    print(f"  pip install huggingface_hub")
    print(f"  huggingface-cli login")
    print(f"  huggingface-cli upload YOUR_USERNAME/a-s-flc-decisions {DATASET_DIR} .")


def main():
    merge_all = "--all" in sys.argv

    if merge_all:
        format_dataset(merge_all=True)
        return

    input_file = DATASET_DIR / "asflc_single_pairs.jsonl"

    if "--input" in sys.argv:
        idx = sys.argv.index("--input")
        input_file = Path(sys.argv[idx + 1])

    if not input_file.exists():
        print(f"Input file not found: {input_file}")
        print(f"Run 'python training/generate_dataset.py' first to generate pairs.")
        sys.exit(1)

    format_dataset(input_file=input_file, merge_all=False)


if __name__ == "__main__":
    main()

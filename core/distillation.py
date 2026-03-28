"""Distillation Pool — Collects (query, small_output, large_output) correction
triples for periodic re-fine-tuning of the small model.

Correction pairs are stored in a JSONL file. When the large model gives a
better answer than the small model, the pair is saved. On periodic re-training,
these pairs are mixed into the training set so the small model learns from
the large model's corrections.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.response_validator import ValidationResult, compare_outputs, validate_output
from core.types import DecisionOutput


class DistillationPool:
    """Append-only JSONL store for correction pairs."""

    def __init__(self, pool_path: str | Path = "training/dataset/distillation_pool.jsonl"):
        self.pool_path = Path(pool_path)
        self.pool_path.parent.mkdir(parents=True, exist_ok=True)

    def add_correction(
        self,
        query: str,
        small_output: DecisionOutput,
        large_output: DecisionOutput,
        escalation_reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate whether the large output is a genuine improvement, and if so store the pair."""
        small_val = validate_output(small_output)
        large_val = validate_output(large_output)

        if large_val.quality_score <= small_val.quality_score:
            return None

        diffs = compare_outputs(small_output, large_output)

        record = {
            "timestamp": time.time(),
            "query": query,
            "escalation_reason": escalation_reason,
            "small_output": small_output.model_dump(),
            "large_output": large_output.model_dump(),
            "small_quality": small_val.quality_score,
            "large_quality": large_val.quality_score,
            "quality_delta": large_val.quality_score - small_val.quality_score,
            "diffs": diffs,
            "metadata": metadata or {},
        }

        with open(self.pool_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def force_add(
        self,
        query: str,
        large_output: DecisionOutput,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a large-model output as a training example without comparison."""
        large_val = validate_output(large_output)
        record = {
            "timestamp": time.time(),
            "query": query,
            "escalation_reason": None,
            "small_output": None,
            "large_output": large_output.model_dump(),
            "small_quality": None,
            "large_quality": large_val.quality_score,
            "quality_delta": None,
            "diffs": {},
            "metadata": metadata or {},
        }
        with open(self.pool_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def load_all(self) -> List[Dict[str, Any]]:
        if not self.pool_path.exists():
            return []
        records = []
        with open(self.pool_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def count(self) -> int:
        if not self.pool_path.exists():
            return 0
        with open(self.pool_path) as f:
            return sum(1 for line in f if line.strip())

    def export_chat_format(self, output_path: Optional[str | Path] = None) -> Path:
        """Export correction pairs as chat-format JSONL for fine-tuning.

        Uses the large_output as the target assistant response.
        """
        if output_path is None:
            output_path = self.pool_path.parent / "distillation_chat_format.jsonl"
        else:
            output_path = Path(output_path)

        system_msg = (
            "You are an A-S-FLC decision navigator. Analyze the query using "
            "asymmetric signed force-loop-chain reasoning. Positives are exact "
            "and trusted. Negatives are estimated with a conservative buffer. "
            "Build 3-5 event chains, score each, loop until stable, and pick "
            "the best. Output strict JSON."
        )

        records = self.load_all()
        with open(output_path, "w") as f:
            for rec in records:
                large = rec.get("large_output")
                if not large:
                    continue
                entry = {
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": rec["query"]},
                        {"role": "assistant", "content": json.dumps(large)},
                    ]
                }
                f.write(json.dumps(entry) + "\n")

        return output_path

    def clear(self) -> None:
        if self.pool_path.exists():
            self.pool_path.unlink()

"""Tests for core.distillation — Distillation pool for correction pairs."""

import json
import tempfile
from pathlib import Path

import pytest

from core.distillation import DistillationPool
from core.types import DecisionOutput, ForceBreakdown


def _make_output(action: str = "Accept", net: float = 3.5, n_chains: int = 2) -> DecisionOutput:
    chains = [
        ForceBreakdown(
            positives=7.0, negatives_estimated=3.0, negatives_buffered=3.5,
            net=net, chain_id=f"chain-{i}", events=[f"event-{i}"],
        )
        for i in range(n_chains)
    ]
    return DecisionOutput(
        chosen_action=action,
        breakdown=chains[0],
        all_chains=chains,
        reasoning_steps=["step 1", "step 2", "step 3"],
        stability_score=0.9,
    )


def test_add_correction_saves_when_large_is_better():
    with tempfile.TemporaryDirectory() as tmpdir:
        pool = DistillationPool(Path(tmpdir) / "pool.jsonl")
        small = _make_output("Decline", net=1.0, n_chains=1)
        large = _make_output("Accept", net=3.5, n_chains=3)

        rec = pool.add_correction("Should I accept?", small, large)
        assert rec is not None
        assert pool.count() == 1
        assert rec["quality_delta"] > 0


def test_add_correction_skips_when_small_is_equal_or_better():
    with tempfile.TemporaryDirectory() as tmpdir:
        pool = DistillationPool(Path(tmpdir) / "pool.jsonl")
        good = _make_output("Accept", net=3.5, n_chains=3)
        also_good = _make_output("Accept", net=3.5, n_chains=3)

        rec = pool.add_correction("Should I accept?", good, also_good)
        assert rec is None
        assert pool.count() == 0


def test_force_add():
    with tempfile.TemporaryDirectory() as tmpdir:
        pool = DistillationPool(Path(tmpdir) / "pool.jsonl")
        large = _make_output("Accept")
        rec = pool.force_add("test query", large)
        assert rec is not None
        assert pool.count() == 1


def test_load_all():
    with tempfile.TemporaryDirectory() as tmpdir:
        pool = DistillationPool(Path(tmpdir) / "pool.jsonl")
        large = _make_output("Accept")
        pool.force_add("q1", large)
        pool.force_add("q2", large)

        records = pool.load_all()
        assert len(records) == 2
        assert records[0]["query"] == "q1"


def test_export_chat_format():
    with tempfile.TemporaryDirectory() as tmpdir:
        pool = DistillationPool(Path(tmpdir) / "pool.jsonl")
        large = _make_output("Accept")
        pool.force_add("test query", large)

        out_path = pool.export_chat_format()
        assert out_path.exists()
        with open(out_path) as f:
            entry = json.loads(f.readline())
        assert entry["messages"][0]["role"] == "system"
        assert entry["messages"][1]["content"] == "test query"
        assert entry["messages"][2]["role"] == "assistant"


def test_clear():
    with tempfile.TemporaryDirectory() as tmpdir:
        pool = DistillationPool(Path(tmpdir) / "pool.jsonl")
        pool.force_add("q", _make_output())
        assert pool.count() == 1
        pool.clear()
        assert pool.count() == 0

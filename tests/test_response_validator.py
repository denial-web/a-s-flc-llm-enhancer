"""Tests for core.response_validator — DecisionOutput validation and comparison."""

import json

import pytest

from core.response_validator import (
    ValidationResult,
    compare_outputs,
    validate_json_string,
    validate_output,
)
from core.types import DecisionOutput, ForceBreakdown


def _make_output(**overrides) -> DecisionOutput:
    defaults = dict(
        chosen_action="Accept offer",
        breakdown=ForceBreakdown(
            positives=7.0,
            negatives_estimated=3.0,
            negatives_buffered=3.5,
            net=3.5,
            chain_id="chain-0",
            events=["accept", "start work"],
        ),
        all_chains=[
            ForceBreakdown(positives=7.0, negatives_estimated=3.0, negatives_buffered=3.5, net=3.5, chain_id="chain-0", events=["accept"]),
            ForceBreakdown(positives=2.0, negatives_estimated=1.0, negatives_buffered=1.2, net=0.8, chain_id="chain-1", events=["decline"]),
        ],
        reasoning_steps=["Step 1: evaluated options", "Step 2: scored chains", "Step 3: selected best"],
        stability_score=0.92,
    )
    defaults.update(overrides)
    return DecisionOutput(**defaults)


def test_valid_output_passes():
    output = _make_output()
    result = validate_output(output)
    assert result.valid
    assert result.quality_score > 0.8


def test_empty_action_fails():
    output = _make_output(chosen_action="")
    result = validate_output(output)
    assert not result.valid
    assert any("chosen_action" in i for i in result.issues)


def test_out_of_range_positives():
    bd = ForceBreakdown(positives=15.0, negatives_estimated=3.0, negatives_buffered=3.5, net=11.5, chain_id="chain-0", events=["x"])
    output = _make_output(breakdown=bd)
    result = validate_output(output)
    assert any("positives" in i for i in result.issues)


def test_net_mismatch():
    bd = ForceBreakdown(positives=7.0, negatives_estimated=3.0, negatives_buffered=3.5, net=10.0, chain_id="chain-0", events=["x"])
    output = _make_output(breakdown=bd)
    result = validate_output(output)
    assert any("net" in i for i in result.issues)


def test_single_chain_warning():
    output = _make_output(
        all_chains=[
            ForceBreakdown(positives=7.0, negatives_estimated=3.0, negatives_buffered=3.5, net=3.5, chain_id="chain-0", events=["x"]),
        ],
    )
    result = validate_output(output)
    assert any("chains" in i for i in result.issues)


def test_validate_json_string_valid():
    output = _make_output()
    raw = output.model_dump_json()
    parsed, result = validate_json_string(raw)
    assert parsed is not None
    assert result.valid


def test_validate_json_string_invalid_json():
    parsed, result = validate_json_string("{not valid json")
    assert parsed is None
    assert not result.valid
    assert any("Invalid JSON" in i for i in result.issues)


def test_compare_outputs():
    small = _make_output(chosen_action="Decline", risk_level="SAFE")
    large = _make_output(chosen_action="Accept", risk_level="SUSPICIOUS")
    diffs = compare_outputs(small, large)
    assert "chosen_action" in diffs
    assert "risk_level" in diffs


def test_strict_mode():
    output = _make_output()
    result = validate_output(output, strict=True)
    assert any("risk_level" in i for i in result.issues)
    assert any("decision_route" in i for i in result.issues)

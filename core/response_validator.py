"""Response Validator — Validates and scores DecisionOutput from any source.

Checks structural validity, scoring sanity, and consistency. Returns a
ValidationResult with pass/fail, a list of issues, and a quality score.

Used after cloud bridge responses and for building distillation correction pairs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.types import DecisionOutput


@dataclass
class ValidationResult:
    valid: bool
    quality_score: float  # 0.0–1.0
    issues: List[str] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)


_VALID_RISK_LEVELS = {"SAFE", "SUSPICIOUS", "DANGEROUS", None}
_VALID_ROUTES = {"LOCAL", "BLOCK", "MEMORY_STORE", "MEMORY_RETRIEVE", "ESCALATE", None}
_VALID_SOURCES = {"small", "large_knowledge", None}


def validate_output(
    output: DecisionOutput,
    strict: bool = False,
) -> ValidationResult:
    """Validate a DecisionOutput for structural and logical correctness."""
    issues: List[str] = []
    corrections: List[str] = []
    score = 1.0

    if not output.chosen_action or not output.chosen_action.strip():
        issues.append("chosen_action is empty")
        score -= 0.3

    bd = output.breakdown
    if bd.positives < 0 or bd.positives > 10:
        issues.append(f"breakdown.positives={bd.positives} outside 0-10 range")
        score -= 0.15
    if bd.negatives_estimated < 0 or bd.negatives_estimated > 10:
        issues.append(f"breakdown.negatives_estimated={bd.negatives_estimated} outside 0-10 range")
        score -= 0.15
    if bd.negatives_buffered < bd.negatives_estimated:
        issues.append("negatives_buffered < negatives_estimated (buffer should add, not subtract)")
        score -= 0.1

    expected_net = bd.positives - bd.negatives_buffered
    if abs(bd.net - expected_net) > 0.5:
        issues.append(f"net={bd.net} doesn't match positives-negatives_buffered={expected_net:.2f}")
        score -= 0.15

    if not bd.chain_id:
        issues.append("breakdown.chain_id is empty")
        score -= 0.05

    if len(output.all_chains) < 2:
        issues.append(f"only {len(output.all_chains)} chains (expected >= 2)")
        score -= 0.1

    chosen_chain_ids = [c.chain_id for c in output.all_chains]
    if bd.chain_id not in chosen_chain_ids:
        issues.append(f"chosen chain_id '{bd.chain_id}' not found in all_chains")
        score -= 0.1

    for i, chain in enumerate(output.all_chains):
        if chain.positives < 0 or chain.positives > 10:
            issues.append(f"all_chains[{i}].positives={chain.positives} outside 0-10")
            score -= 0.05
        if chain.negatives_estimated < 0 or chain.negatives_estimated > 10:
            issues.append(f"all_chains[{i}].negatives_estimated={chain.negatives_estimated} outside 0-10")
            score -= 0.05

    if len(output.reasoning_steps) < 2:
        issues.append(f"only {len(output.reasoning_steps)} reasoning_steps (expected >= 2)")
        score -= 0.1

    if output.stability_score < 0 or output.stability_score > 1:
        issues.append(f"stability_score={output.stability_score} outside 0-1")
        score -= 0.1

    if output.risk_level and output.risk_level not in _VALID_RISK_LEVELS:
        issues.append(f"invalid risk_level: {output.risk_level}")
        score -= 0.1

    if output.decision_route and output.decision_route not in _VALID_ROUTES:
        issues.append(f"invalid decision_route: {output.decision_route}")
        score -= 0.1

    if output.source and output.source not in _VALID_SOURCES:
        issues.append(f"invalid source: {output.source}")
        score -= 0.05

    if output.memory_action:
        ma = output.memory_action
        if ma.get("op") not in ("store", "retrieve", "skip", None):
            issues.append(f"invalid memory_action.op: {ma.get('op')}")
            score -= 0.05

    if strict:
        if not output.risk_level:
            issues.append("risk_level is required in strict mode")
            score -= 0.05
        if not output.decision_route:
            issues.append("decision_route is required in strict mode")
            score -= 0.05

    score = max(0.0, min(1.0, score))
    return ValidationResult(
        valid=len(issues) == 0,
        quality_score=score,
        issues=issues,
        corrections=corrections,
    )


def validate_json_string(raw_json: str) -> Tuple[Optional[DecisionOutput], ValidationResult]:
    """Parse a JSON string into DecisionOutput and validate it."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return None, ValidationResult(
            valid=False,
            quality_score=0.0,
            issues=[f"Invalid JSON: {e}"],
        )

    try:
        output = DecisionOutput.model_validate(data)
    except Exception as e:
        return None, ValidationResult(
            valid=False,
            quality_score=0.0,
            issues=[f"Schema validation failed: {e}"],
        )

    result = validate_output(output)
    return output, result


def compare_outputs(
    small_output: DecisionOutput,
    large_output: DecisionOutput,
) -> dict:
    """Compare small vs large model outputs to identify correction opportunities."""
    diffs = {}

    if small_output.chosen_action != large_output.chosen_action:
        diffs["chosen_action"] = {
            "small": small_output.chosen_action,
            "large": large_output.chosen_action,
        }

    net_diff = abs(small_output.breakdown.net - large_output.breakdown.net)
    if net_diff > 1.0:
        diffs["net_score_diff"] = {
            "small": small_output.breakdown.net,
            "large": large_output.breakdown.net,
            "delta": net_diff,
        }

    if small_output.risk_level != large_output.risk_level:
        diffs["risk_level"] = {
            "small": small_output.risk_level,
            "large": large_output.risk_level,
        }

    if small_output.decision_route != large_output.decision_route:
        diffs["decision_route"] = {
            "small": small_output.decision_route,
            "large": large_output.decision_route,
        }

    small_val = validate_output(small_output)
    large_val = validate_output(large_output)
    diffs["quality_scores"] = {
        "small": small_val.quality_score,
        "large": large_val.quality_score,
    }

    return diffs

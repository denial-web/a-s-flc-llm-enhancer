"""Deterministic Policy Guard — runs before any LLM call.

Blocks requests that must not be assisted (credential harvesting, unsafe payment
patterns). Optional URL heuristics are recorded but do not alone force a block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple
from urllib.parse import urlparse


@dataclass
class PolicyResult:
    """Outcome of policy evaluation."""

    allowed: bool
    action: str  # "ALLOW" | "BLOCK"
    reason: str
    matched_rules: Tuple[str, ...] = ()


_SECRET_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("otp_request", re.compile(r"\b(send|text|give|share|tell|provide)\b.*\b(otp|one[-\s]?time|2fa|mfa|verification)\s+code\b", re.I)),
    (
        "password_request",
        re.compile(
            r"\b(password|passwd)\b.*\b(send|share|give|tell|type|enter|paste)\b|"
            r"\b(send|share|give|tell|paste)\b.*\b(password|passwd)\b",
            re.I,
        ),
    ),
    ("ssn_full", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("cvv_request", re.compile(r"\b(cvv|cvc|security\s+code\s+on\s+card)\b", re.I)),
    ("private_key", re.compile(r"\b(seed\s+phrase|recovery\s+phrase|private\s+key)\b", re.I)),
]

_SCAM_HOOK_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "gift_card_payment",
        re.compile(
            r"\b(gift\s+cards?|itunes|google\s+play)\b.*\b(pay|buy|purchase|send)\b|"
            r"\b(pay|buy|send)\b.*\b(gift\s+cards?)\b|"
            r"\b(irs|tax\s+office|federal)\b.*\b(gift\s+card|itunes)\b",
            re.I,
        ),
    ),
    ("wire_urgent", re.compile(r"\b(wire\s+transfer|western\s+union)\b.*\b(urgent|immediately|now)\b", re.I)),
    ("crypto_send_first", re.compile(r"\b(send|transfer)\b.*\b(btc|bitcoin|eth|usdt|crypto)\b.*\b(first|before|unlock)\b", re.I)),
]

_SUSPICIOUS_URL = re.compile(r"bit\.ly|tinyurl|t\.co|goo\.gl", re.I)
_IP_HOST = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")


def _extract_urls(text: str) -> List[str]:
    return re.findall(r"https?://[^\s\]\)\"']+", text, re.I)


def evaluate_policy(user_text: str) -> PolicyResult:
    """Return ALLOW or BLOCK with reasons. Run before the LLM."""
    text = user_text or ""
    matched: List[str] = []

    for rule_id, pat in _SECRET_PATTERNS:
        if pat.search(text):
            matched.append(rule_id)

    for rule_id, pat in _SCAM_HOOK_PATTERNS:
        if pat.search(text):
            matched.append(rule_id)

    if matched:
        return PolicyResult(
            allowed=False,
            action="BLOCK",
            reason="Policy Guard: blocked high-risk pattern.",
            matched_rules=tuple(matched),
        )

    notes: List[str] = []
    for url in _extract_urls(text):
        try:
            host = urlparse(url).hostname or ""
        except Exception:
            host = ""
        if _SUSPICIOUS_URL.search(url) or _SUSPICIOUS_URL.search(host):
            notes.append(f"note_short_url:{url[:40]}")
        if _IP_HOST.search(host):
            notes.append(f"note_ip_host:{host[:40]}")

    return PolicyResult(
        allowed=True,
        action="ALLOW",
        reason="Policy Guard: no block rules fired."
        + (f" Notes: {notes}" if notes else ""),
        matched_rules=tuple(notes),
    )


def format_block_message(result: PolicyResult) -> str:
    rules = ", ".join(result.matched_rules) if result.matched_rules else "policy"
    return (
        f"Request blocked by Policy Guard ({rules}). "
        "Never share passwords, OTPs, recovery phrases, or pay under pressure."
    )

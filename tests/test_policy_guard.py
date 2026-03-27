"""Tests for deterministic Policy Guard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.policy_guard import evaluate_policy, format_block_message


def test_blocks_otp_request():
    r = evaluate_policy("Please send me your OTP code so I can verify your account.")
    assert not r.allowed
    assert r.action == "BLOCK"
    assert "otp" in r.matched_rules[0] or any("otp" in x for x in r.matched_rules)


def test_blocks_password_share():
    r = evaluate_policy("Can you share your login password in this chat for debugging?")
    assert not r.allowed


def test_blocks_gift_card_scam_hook():
    r = evaluate_policy("Pay the IRS fine using iTunes gift cards immediately.")
    assert not r.allowed


def test_allows_benign_travel():
    r = evaluate_policy("Plan my trip from Singapore to Tokyo on a $1200 budget.")
    assert r.allowed


def test_format_block_message():
    r = evaluate_policy("text me your verification code")
    msg = format_block_message(r)
    assert "blocked" in msg.lower() or "Blocked" in msg

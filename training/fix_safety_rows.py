"""Fix known-bad rows in asflc_single_pairs.jsonl (safety-013, safety-015).

Run from repo root: python training/fix_safety_rows.py
"""

import json
from pathlib import Path

DATASET = Path(__file__).resolve().parent / "dataset" / "asflc_single_pairs.jsonl"

FIXED_013 = {
    "chosen_action": "Do not purchase; treat as likely scam",
    "breakdown": {
        "positives": 0.0,
        "negatives_estimated": 9.5,
        "negatives_buffered": 10.97,
        "net": -10.97,
        "chain_id": "chain-0",
        "events": [
            "Pay $199 to unknown site",
            "Receive nothing, counterfeit device, or stolen payment data",
        ],
    },
    "all_chains": [
        {
            "positives": 0.0,
            "negatives_estimated": 9.5,
            "negatives_buffered": 10.97,
            "net": -10.97,
            "chain_id": "chain-0",
            "events": [
                "Pay $199 to unknown site",
                "Receive nothing, counterfeit device, or stolen payment data",
            ],
        },
        {
            "positives": 8.5,
            "negatives_estimated": 9.8,
            "negatives_buffered": 11.27,
            "net": -2.77,
            "chain_id": "chain-1",
            "events": [
                "Site looks professional",
                "Price far below market",
                "High risk of fraud despite appearance",
            ],
        },
    ],
    "reasoning_steps": [
        "Identified classic too-good-to-be-true pricing vs retail iPhone cost",
        "Estimated negatives: payment loss, identity theft, malware (9.5) with buffer δ",
        "Compared chain: buy from unknown site vs walk away",
        "Chosen chain maximizes safety: do not purchase from unverified seller",
        "Stability check: conclusion unchanged under loop iteration",
    ],
    "stability_score": 0.95,
    "what_if_summary": None,
    "risk_flags": ["too_good_to_be_true_pricing", "unknown_merchant"],
}

FIXED_015 = {
    "chosen_action": "Ignore voicemail; verify only via IRS.gov or official IRS phone numbers",
    "breakdown": {
        "positives": 5.0,
        "negatives_estimated": 8.5,
        "negatives_buffered": 9.775,
        "net": -4.775,
        "chain_id": "chain-0",
        "events": [
            "Recognize IRS does not threaten arrest by voicemail",
            "Do not call back the number left on voicemail",
            "If needed, contact IRS through irs.gov official channels only",
        ],
    },
    "all_chains": [
        {
            "positives": 5.0,
            "negatives_estimated": 8.5,
            "negatives_buffered": 9.775,
            "net": -4.775,
            "chain_id": "chain-0",
            "events": [
                "Recognize IRS does not threaten arrest by voicemail",
                "Do not call back the number left on voicemail",
                "If needed, contact IRS through irs.gov official channels only",
            ],
        },
        {
            "positives": 1.0,
            "negatives_estimated": 9.5,
            "negatives_buffered": 10.97,
            "net": -9.97,
            "chain_id": "chain-1",
            "events": [
                "Return call to number in voicemail",
                "Pay demanded amount under pressure",
                "Financial loss and identity compromise",
            ],
        },
    ],
    "reasoning_steps": [
        "Identified high-risk IRS impersonation pattern (threat + urgency)",
        "Estimated negatives of engaging with callback number as very high (9.5+)",
        "Compared safe path: official verification only vs paying scammers",
        "Chose path with lowest buffered downside",
        "Stability: recommendation holds under stress test",
    ],
    "stability_score": 0.95,
    "what_if_summary": None,
    "risk_flags": ["irs_impersonation", "threat_of_arrest", "urgency_pressure"],
}


def main():
    lines = DATASET.read_text().strip().split("\n")
    out = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        oid = row["id"]
        if oid == "safety-013":
            row["output"] = FIXED_013
            row["output_json"] = json.dumps(FIXED_013)
        elif oid == "safety-015":
            row["output"] = FIXED_015
            row["output_json"] = json.dumps(FIXED_015)
        out.append(json.dumps(row))

    DATASET.write_text("\n".join(out) + "\n")
    print(f"Updated {DATASET}: safety-013 and safety-015")


if __name__ == "__main__":
    main()

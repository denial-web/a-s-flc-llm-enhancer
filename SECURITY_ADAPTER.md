# Security Adapter — Integrated with A-S-FLC

This document describes how **threat-aware assistance** fits into the **Asymmetric Signed Force-Loop-Chain (A-S-FLC)** framework. It replaces generic “security chatbot” ideas with the same math your project already uses: **exact positives** (apparent benefits of complying) vs **estimated negatives** with a **conservative buffer** (fraud, malware, financial loss, coercion).

## Why A-S-FLC for security

Scams and phishing are **adversarial offers**: high apparent upside (cheap iPhone, guaranteed returns, fear relief) paired with **uncertain but severe** downsides. Standard symmetric reasoning treats “maybe bad” like “maybe good.” A-S-FLC intentionally **inflates uncertain negatives**, so trap paths lose on **net** even when the story looks appealing.

## Components

### 1. Policy Guard (deterministic, non-LLM)

**Module:** `core/policy_guard.py`

Runs **before** any model call. It blocks a narrow set of patterns that must never be assisted (e.g. requests to share OTPs/passwords/seed phrases, common gift-card / wire-under-pressure hooks). It is **not** a complete antivirus; it is a **safety rail**.

**API:**

- `evaluate_policy(user_text) -> PolicyResult`
- `format_block_message(result)` for user-facing text

**CLI:** `python main.py` runs Policy Guard by default. Use `--no-guard` only for debugging.

### 2. Security Navigator (LLM, A-S-FLC-shaped)

**Prompts:** `SECURITY_SYSTEM_PROMPT` / `SECURITY_USER_TEMPLATE` in `inference/fg_cot_prompt.py`

**API:** `A_S_FLC_Wrapper.decide_security(query)` in `inference/wrapper.py`

**CLI:** `python main.py --security "…"`

The model outputs the usual `DecisionOutput` fields **plus** (when following the prompt):

- `risk_level`: `SAFE` | `SUSPICIOUS` | `DANGEROUS`
- `threat_type`: e.g. `phishing`, `scam`, `injection`, …
- `decision_route`: `LOCAL` | `BLOCK` (Stage 2 will extend to memory / escalate)

Schema definitions: `core/types.py` (`DecisionOutput`).

### 3. Training data

- **General decisions:** `training/dataset/asflc_single_pairs.jsonl` (from `training/query_bank.json`)
- **Security-specific:** `training/dataset/asflc_security_pairs.jsonl` (from `training/security_query_bank.json`)

Generate security pairs:

```bash
python training/generate_dataset.py --mode security
# optional: --limit 10  --resume
```

Merge all JSONL shards into HuggingFace chat/instruction files:

```bash
python training/format_for_hf.py --all
```

### 4. Evaluation split (fine-tuning)

**File:** `training/eval_split.json`

Lists `eval_ids` that should be **held out** from the training split when you fine-tune a small model. Regenerate or edit this list as the dataset grows.

### 5. Fine-tuning entry point

**Notebook:** `training/finetune_colab.ipynb` — Unsloth + Qwen2.5-1.5B on Colab, loading data from this repo or from Hugging Face.

## Data flow

1. User message → **Policy Guard** → if blocked, return block message (no LLM).
2. Else → **Security / decision LLM** (`decide`, `decide_whatif`, or `decide_security`) → structured JSON.
3. Downstream app may enforce `decision_route == BLOCK` even if the model misbehaves (defense in depth).

## Stages (roadmap)

See **`ROADMAP.md`** for the full staged plan (format-only fine-tune → security labels → memory/routing → escalation/distillation → on-device deployment).

## What stays out of the LLM

- Secret handling policy
- Whether to allow cloud escalation
- Promotion of untrusted text into “trusted memory”

Those belong in **code**, not weights.

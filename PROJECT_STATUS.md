# A-S-FLC Project Status

> Last updated: 2026-03-27

## Repository

- **GitHub**: https://github.com/denial-web/a-s-flc-llm-enhancer
- **Colab Demo**: https://colab.research.google.com/github/denial-web/a-s-flc-llm-enhancer/blob/main/demo.ipynb
- **HuggingFace Dataset**: https://huggingface.co/datasets/denialkhmbot/a-s-flc-decisions
- **Branch**: main
- **License**: MIT

## What's Built

### Core Framework
- `core/forces.py` — exact positives, estimated negatives, buffer calculation, chain scoring, ranking
- `core/chains.py` — event-chain tree builder, BFS path enumeration
- `core/loops.py` — LCDI iteration engine (loop until stable)
- `core/types.py` — Pydantic models (`ForceBreakdown`, `DecisionOutput` with optional security/routing fields)
- `core/policy_guard.py` — deterministic pre-LLM Policy Guard (block credential/scam-hook patterns)
- `config.py` — hyperparameters (δ=0.15, max_iter=3, max_branches=5, ε=0.01)

### Inference (4 modes)
- `inference/fg_cot_prompt.py` — FG-CoT, What-If, Hybrid analysis, **Security** prompts
- `inference/wrapper.py` — `decide()`, `decide_whatif()`, `decide_hybrid()`, `decide_security()`
- `main.py` — CLI with `--hybrid`, `--whatif`, `--security`, `--no-guard`

### Training & datasets
- `training/query_bank.json` — general decision queries
- `training/security_query_bank.json` — 200 English security/trust queries (8 subcategories)
- `training/generate_dataset.py` — `--mode single|whatif|security`, per-mode checkpoints
- `training/format_for_hf.py` — `--all` merges `asflc_*_pairs.jsonl` → chat + instruction JSONL
- `training/upload_to_hf.py` — Hub upload
- `training/eval_split.json` — 20 held-out IDs for fine-tune eval
- `training/finetune_colab.ipynb` — Unsloth + Qwen2.5-1.5B starter notebook
- `training/fix_safety_rows.py` — repair known-bad rows in `asflc_single_pairs.jsonl`
- `training/dataset/` — `asflc_single_pairs.jsonl` (178), `asflc_security_pairs.jsonl` (200), merged HF formats (378)

### Validation Suite
| Script | What it tests | Status |
|---|---|---|
| `tests/test_*.py` | Core + Policy Guard | 43 tests |
| `validation/eval_harness.py` | Metrics + wrapper | See `VALIDATION_RESULTS.md` |
| `validation/ablation_study.py` | Ablations | ✅ |

### Docs
- `ROADMAP.md` — staged plan (fine-tune → security → memory → escalation → mobile)
- `SECURITY_ADAPTER.md` — security + A-S-FLC integration

## Key Dataset Counts

| Artifact | Examples |
|---|---|
| General single-shot pairs | 178 |
| Security pairs | 200 |
| Merged (HF chat/instruction) | 378 |

## What's NOT Built Yet

- pip package (not on PyPI)
- Memory store (SQLite + FAISS) + routing/escalation training loop
- Full Unsloth SFT cell implementation inside notebook (stub + instructions)
- Multi-model security benchmark harness
- Web UI / API service

## Next Steps (see ROADMAP.md)

1. Fine-tune Qwen2.5-1.5B on merged chat JSONL; exclude `eval_split.json` IDs from train.
2. Add Stage 2 routing + memory examples and `decide_routed()` when ready.
3. Regression-test Policy Guard + model on held-out security set.

## LLM Provider Config

- **Provider**: Groq (default)
- **Model**: llama-3.3-70b-versatile
- **API Key env var**: `GROQ_API_KEY`

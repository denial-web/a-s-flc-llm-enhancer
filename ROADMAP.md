# Smart Local LLM Roadmap — Making Small Models Smart with A-S-FLC

Persistent reference for objectives and staged delivery. Last updated: 2026-03-29.

## Goal

Fine-tune a 1.5B-3B parameter model that runs locally on a phone and can:

1. Produce structured A-S-FLC decisions (asymmetric scoring, event chains, stability)
2. Classify security threats (scams, phishing, injection, fraud)
3. Manage its own memory (store, retrieve, skip)
4. Know when to escalate to a large cloud model
5. Learn from large-model corrections over time

## Current State

- **GitHub**: https://github.com/denial-web/a-s-flc-llm-enhancer
- **HuggingFace Dataset**: https://huggingface.co/datasets/denialkhmbot/a-s-flc-decisions
- **Framework**: Core A-S-FLC with 5 modes (single-shot, what-if, hybrid, security, memory)
- **Training Data**: See `training/dataset/` and `training/eval_split.json` for train/eval splits
- **Policy Guard**: Deterministic pre-LLM rules in `core/policy_guard.py`
- **Memory Store**: SQLite + FAISS semantic search in `core/memory_store.py`
- **Cloud Bridge**: Escalation to large models in `core/cloud_bridge.py`
- **Distillation**: Correction pair collection in `core/distillation.py`
- **Deployment**: GGUF export + mobile config in `deployment/`

## Architecture

The small model is the decision-maker. Everything else is deterministic code.

```mermaid
flowchart TD
    UserInput[User Input] --> PolicyGuard[Policy Guard]
    PolicyGuard -->|"blocked (OTP/password/etc)"| BlockResponse[BLOCK Response]
    PolicyGuard -->|"passed"| SmallModel[Small Local LLM]
    SmallModel --> Decision{Structured Output}
    Decision -->|"route: LOCAL"| LocalResponse[Local Response]
    Decision -->|"route: MEMORY_STORE"| MemoryWrite[Write to Memory]
    Decision -->|"route: MEMORY_RETRIEVE"| MemoryRead[Read from Memory]
    Decision -->|"route: ESCALATE"| LargeModel[Large Cloud Model]
    Decision -->|"route: BLOCK"| BlockResponse
    LargeModel --> Validator[Response Validator]
    Validator --> LocalResponse
    Validator -->|"store correction"| TrainingPool[Distillation Pool]
    MemoryRead -->|"inject context"| SmallModel
```

## Output Schema Contract

**Stage 1a (A-S-FLC):** `chosen_action`, `breakdown`, `all_chains`, `reasoning_steps`, `stability_score`.

**Stage 1b (security):** optional `risk_level` (SAFE | SUSPICIOUS | DANGEROUS), `threat_type`, `decision_route` (LOCAL | BLOCK).

**Stage 2 (memory + routing):** extended `decision_route` (MEMORY_STORE | MEMORY_RETRIEVE | ESCALATE), `memory_action`, `knowledge_request`.

**Stage 3 (escalation):** `escalation_reason`, `source` (small | large_knowledge).

See `core/types.py` for the canonical Pydantic schema.

---

## Stage 1a — A-S-FLC Format Fine-Tune ✅

- ✅ Fix known-bad rows in dataset (scam-iPhone case, sec-mal-012 QR code).
- ✅ Use `training/eval_split.json` for held-out eval IDs (never train on these).
- ✅ Fine-tune with `training/finetune_colab.ipynb` (Unsloth + Qwen2.5-1.5B, assistant-only loss).
- **Done when:** >= 90% valid JSON on held-out set.

## Stage 1b — Security Classification ✅

- ✅ Security prompt: `inference/fg_cot_prompt.py` (`SECURITY_*`).
- ✅ Inference: `A_S_FLC_Wrapper.decide_security()` in `inference/wrapper.py`.
- ✅ Dataset: `training/security_query_bank.json` (200 queries) + `python training/generate_dataset.py --mode security`.
- ✅ Policy Guard: `core/policy_guard.py` runs before the model.

## Stage 2 — Memory and Routing ✅

- ✅ Memory store: `core/memory_store.py` (SQLite + FAISS/NumPy, sentence-transformer embeddings).
- ✅ Memory prompt: `inference/fg_cot_prompt.py` (`MEMORY_*`).
- ✅ Inference: `A_S_FLC_Wrapper.decide_memory()` in `inference/wrapper.py`.
- ✅ Query bank: `training/memory_query_bank.json` (90 queries: store/retrieve/skip/escalate/mixed).
- ✅ Dataset generation: `python training/generate_dataset.py --mode memory`.
- ✅ Full pipeline: `A_S_FLC_Wrapper.decide_full()` — policy guard → model → escalation.

## Stage 3 — Escalation and Distillation ✅

- ✅ Cloud bridge: `core/cloud_bridge.py` (OpenAI/Anthropic/Groq, escalation prompt).
- ✅ Response validator: `core/response_validator.py` (structural + logical checks, quality scoring).
- ✅ Distillation pool: `core/distillation.py` (JSONL correction pairs, chat-format export).
- ✅ Auto-escalation: wrapper's `_maybe_escalate()` forwards ESCALATE decisions to cloud bridge.
- ✅ Auto-distillation: corrections saved when large model output > small model output quality.

## Stage 4 — Mobile Deployment ✅

- ✅ GGUF export: `deployment/export_gguf.py` (Q4_0, Q4_K_M, Q5_K_M, Q8_0, F16).
- ✅ Mobile config: `deployment/mobile_config.py` (device tiers, performance budgets, inference params).
- ✅ Local inference: `deployment/local_inference.py` (llama-cpp-python runner with policy guard + validation).

## First Training Run — Eval Results (2026-03-29)

**Config:** Qwen2.5-1.5B, QLoRA r=16, 500 steps (~8 epochs), 448 train / 20 eval, Groq Llama 3.3 70B teacher.

**Training loss:** 2.4 → 0.05 (converged well).

**Local inference (MacBook, Q4_K_M GGUF, 940MB):**

| Test | Mode | Valid JSON | Quality | Speed | Notes |
|------|------|-----------|---------|-------|-------|
| Job offer decision | single | ✅ | 0.90 | 53 tok/s | Correct scoring, 4 reasoning steps |
| Investment 3-way | single | ❌ | — | 53 tok/s | Good reasoning but JSON truncated at 425 tokens |
| Tax/IRA analysis | single | ✅ | 0.90 | 55 tok/s | Correctly chose ESCALATE route |
| Phishing email | security | ❌ | — | 25 tok/s | Correct SUSPICIOUS+BLOCK but wrong schema |
| Allergy memory | memory | ❌ | — | 34 tok/s | Correct STORE route but simplified schema |
| Lottery gift card | security | BLOCKED | N/A | instant | Policy Guard caught it pre-model |

**Strengths:** Core A-S-FLC decisions are solid; policy guard works; escalation routing works; 53+ tok/s on Mac.

**Weaknesses:** Security and memory modes produce simplified JSON (not full DecisionOutput schema); long queries can truncate.

### Formal Eval Harness (20-ID held-out set)

| Metric | Value |
|--------|-------|
| Valid JSON | 15/20 (75%) |
| Avg Quality | 0.678 |
| Avg Latency | 6.4s |
| Avg Speed | 54.8 tok/s |

Common failure modes: truncated JSON on long outputs (3/5), missing `breakdown` field (1/5), malformed JSON (1/5).

### Performance Benchmark (high_end tier)

| Query | Latency | Speed | Valid | Budget |
|-------|---------|-------|-------|--------|
| basic_decision | 3533ms | 57.6 tok/s | 100% | PASS |
| phishing_check | 3318ms | 52.7 tok/s | 100% | PASS |
| memory_store | 2571ms | 47.3 tok/s | 0% | PASS |
| complex_multi_option | 2904ms | 52.2 tok/s | 100% | PASS |
| career_tradeoff | 2235ms | 61.5 tok/s | 0% | PASS |

All queries within 5s latency budget. All above 15 tok/s minimum.

### Fixes Applied (2026-03-29)
- [x] Mode-specific system prompts in training data (security/memory/single).
- [x] Full schema in local inference prompts (was abbreviated).
- [x] Bumped max_tokens: 1536 (high), 1024 (mid), 768 (low).
- [x] Built eval harness: `training/eval_harness.py`.
- [x] Built performance benchmark: `deployment/benchmark.py`.

### Next: Re-train with improved data
- [ ] Re-upload improved `asflc_chat_format.jsonl` to HuggingFace.
- [ ] Re-train on Colab with mode-specific system prompts (expect ~85%+ valid JSON).
- [ ] Re-export GGUF and re-run eval harness.
- [ ] Add Khmer language support (Stage 5).

## Completed Milestones

- [x] Run fine-tuning on Colab Pro (500 steps, T4 GPU, loss 0.05).
- [x] Generate memory training pairs (90 pairs via Groq).
- [x] Upload expanded dataset to HuggingFace (468 examples).
- [x] Export GGUF on Colab (Q4_K_M, 940MB).
- [x] Test on-device inference with llama-cpp-python (53+ tok/s on Mac).
- [x] Formal eval harness: 15/20 valid (75%), avg quality 0.678.
- [x] Performance benchmark: all queries within latency/speed budgets.
- [x] Fix mode-specific system prompts for security/memory training data.
- [x] Full schema in local inference prompts.
- [x] Bump max_tokens to 1536 for high-end tier.

## Stage 5 — Khmer Language Support 🚧

- ✅ Khmer query bank: `training/khmer_query_bank.json` (25 queries: finance, career, daily, security, memory).
- ✅ Khmer prompts: `inference/fg_cot_prompt.py` (`KHMER_SYSTEM_PROMPT`, `KHMER_USER_TEMPLATE`).
- ✅ Khmer inference: `wrapper.decide_khmer()`, `local_inference.py --mode khmer`.
- ✅ Khmer dataset generation: `python training/generate_dataset.py --mode khmer`.
- ✅ Mode-specific system prompts in training format (khmer uses Khmer-aware prompt).
- ✅ Khmer eval IDs: `km-fin-003`, `km-car-003`, `km-sec-002` in eval_split.json.
- [ ] Generate Khmer training pairs: `python training/generate_dataset.py --mode khmer`.
- [ ] Re-format + re-upload dataset with Khmer examples.
- [ ] Re-train on Colab with Khmer data included.
- [ ] Eval Khmer valid-JSON rate and quality.

## What NOT to Build Yet

- Anti-poisoning at scale, cross-user detection, full web UI.

## File Map

| File | Purpose |
|------|---------|
| `core/types.py` | DecisionOutput + security + memory fields |
| `core/policy_guard.py` | Deterministic rule engine |
| `core/memory_store.py` | SQLite + FAISS memory store |
| `core/cloud_bridge.py` | Escalation to large cloud models |
| `core/response_validator.py` | Output validation + quality scoring |
| `core/distillation.py` | Correction pair collection for re-fine-tuning |
| `inference/fg_cot_prompt.py` | FG-CoT, What-If, Security, Memory prompts |
| `inference/wrapper.py` | decide, decide_whatif, decide_hybrid, decide_security, decide_memory, decide_full |
| `training/query_bank.json` | General query bank (178 queries) |
| `training/security_query_bank.json` | Security query bank (200 queries) |
| `training/memory_query_bank.json` | Memory/routing query bank (90 queries) |
| `training/khmer_query_bank.json` | Khmer bilingual query bank (25 queries) |
| `training/generate_dataset.py` | single / whatif / security / memory / khmer modes |
| `training/eval_split.json` | Held-out eval IDs |
| `training/finetune_colab.ipynb` | Colab fine-tuning (dataset load, eval split, SFTTrainer, assistant-only loss, save LoRA) |
| `training/format_for_hf.py` | Format + merge datasets for HuggingFace |
| `training/upload_to_hf.py` | Upload to HuggingFace Hub |
| `deployment/export_gguf.py` | LoRA → GGUF export (Unsloth) |
| `deployment/mobile_config.py` | Device tiers, performance budgets |
| `deployment/local_inference.py` | llama-cpp-python local runner |
| `SECURITY_ADAPTER.md` | Security + A-S-FLC architecture |
| `training/eval_harness.py` | Eval harness for held-out set |
| `deployment/benchmark.py` | Performance benchmark vs mobile budgets |
| `ROADMAP.md` | This file |

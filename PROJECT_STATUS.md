# A-S-FLC Project Status

> Last updated: 2026-03-26

## Repository

- **GitHub**: https://github.com/denial-web/a-s-flc-llm-enhancer
- **Colab Demo**: https://colab.research.google.com/github/denial-web/a-s-flc-llm-enhancer/blob/main/demo.ipynb
- **Branch**: main
- **License**: MIT

## What's Built

### Core Framework
- `core/forces.py` — exact positives, estimated negatives, buffer calculation, chain scoring, ranking
- `core/chains.py` — event-chain tree builder, BFS path enumeration
- `core/loops.py` — LCDI iteration engine (loop until stable)
- `core/types.py` — Pydantic models (ForceBreakdown, DecisionOutput, EventNode, ChainPath, LoopState)
- `config.py` — hyperparameters (δ=0.15, max_iter=3, max_branches=5, ε=0.01)

### Inference (3 modes)
- `inference/fg_cot_prompt.py` — FG-CoT prompt, What-If prompt, Hybrid analysis prompt
- `inference/wrapper.py` — `decide()`, `decide_whatif()`, `decide_hybrid()`
- `main.py` — CLI with `--hybrid` and `--whatif` flags

### Validation Suite (100+ LLM calls completed)
| Script | What it tests | Status |
|---|---|---|
| `tests/test_forces.py` | 19 unit tests for core math | 38/38 passed |
| `tests/test_chains.py` | 10 unit tests for chain building | ✅ |
| `tests/test_loops.py` | 9 unit tests for LCDI | ✅ |
| `examples/edge_case_trees.py` | 4 synthetic stress tests (incl. adversarial flip) | 4/4 passed |
| `validation/ab_compare.py` | Vanilla LLM vs FG-CoT A/B test | 100% parse rate |
| `validation/eval_harness.py` | 10-case eval against ground truth | PE=0.81, NA=0.82 |
| `validation/three_way_compare.py` | No-CoT vs Standard CoT vs FG-CoT | FG-CoT wins all 5 |
| `validation/consistency_test.py` | Multi-run consistency | 100% action agreement |
| `validation/ablation_study.py` | 5 conditions × 10 cases | Multiple chains most impactful |
| `validation/eval_harness.py --whatif` | What-If vs baseline comparison | NA +0.039, 4/10 risk flags |

### Other
- `agent/graph.py` + `agent/nodes.py` — LangGraph skeleton (Planner→Simulator→Navigator)
- `training/reward_shaper.py` — signed reward for RLHF/GRPO
- `eval/metrics.py` — positive_exactness, net_alignment, chain_regret, loop_stability
- `demo.ipynb` — interactive Colab notebook (all 3 modes + adversarial proof)

## Key Results

| Metric | Single-Shot | What-If | Hybrid |
|---|---|---|---|
| Success rate | 10/10 | 10/10 | 8/10 |
| Positive Exactness | **0.81** | 0.77 | 0.20 |
| Net Alignment | 0.82 | **0.86** | 0.46 |
| Chain Regret | 0.00 | 0.00 | 0.00 |
| Consistency | 100% | — | 73% |
| Risk flags | — | 4/10 cases | — |

## What's NOT Built Yet
- pip package (not on PyPI)
- Public benchmark evaluation (StrategyQA, HotpotQA, GAIA)
- Multi-model testing (only Llama 3.3 70B via Groq tested)
- Web UI / API service
- LangGraph agent integration with main inference flow (skeleton only)
- Actual RLHF/GRPO training using reward shaper

## Next Steps (Priority Order)
1. **Post on r/LocalLLaMA and Twitter** — distribution is the bottleneck, not code
2. **Wait for feedback** — let community tell you what to build next
3. **If asked**: run on public benchmark, test on GPT-4o/Claude, make pip-installable

## Git Log
```
5d0f6c3 Improve Colab notebook UX
9c0a1df Add interactive Colab demo notebook
6ae1fec Update README: document three modes with What-If recommendation
e1a4860 Add What-If stress testing mode
6c2f440 Add use cases section to README
798d704 Initial release: A-S-FLC framework with full validation suite
```

## LLM Provider Config
- **Provider**: Groq (free tier)
- **Model**: llama-3.3-70b-versatile
- **API Key env var**: GROQ_API_KEY

# A-S-FLC: Asymmetric Signed Force-Loop-Chain

**Turn any LLM into a force-guided decision navigator.**

A-S-FLC is a prompt + engine framework that upgrades LLM reasoning for decisions under uncertainty. Instead of symmetric "pros and cons" lists, it treats positives as exact (trusted) and negatives as estimates that get a conservative buffer proportional to uncertainty — then scores multiple event chains, loops until stable, and picks the highest-confidence path.

## Why This Matters

Standard Chain-of-Thought prompting lists pros and cons but treats them equally. This fails on adversarial decisions where a "trap" option looks great on raw numbers but has hidden downside risk.

A-S-FLC fixes this with **asymmetric signing**:
- Positives = exact, 100% trusted (known rewards, goals)
- Negatives = estimated + conservative buffer δ (costs, risks, obstacles)
- Net = positives - buffered_negatives

This means high-uncertainty negatives get penalized more, and "trap" options that look good on paper get correctly rejected.

### Proven Example

A decision with raw net +8.5 (looks like the best option) drops to net +1.3 after asymmetric buffering, while the honest option stays at +3.5. Standard CoT, Tree-of-Thought, and Self-Consistency would all pick the trap.

## How It Works

```
User Query
    ↓
Force-Guided CoT Prompt (FG-CoT)
    ├── 1. Extract exact positives (0-10 scale)
    ├── 2. Estimate negatives + apply buffer δ
    ├── 3. Build 3-5 event chains (path-dependent sequences)
    ├── 4. Score each chain: net = positives - buffered_negatives
    ├── 5. Loop 1-3 iterations until net stabilizes (LCDI)
    └── 6. Pick highest stable net chain
    ↓
Structured JSON Decision
    (chosen action, all chains with scores, reasoning trace)
```

## Two Modes

| Mode | How it works | Best for |
|---|---|---|
| **Single-shot** | LLM does everything in one call via FG-CoT prompt | Speed, simplicity |
| **Hybrid** | LLM generates event tree → Python engine scores deterministically | Auditability, rigor |

## Quick Start

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your-key    # or OPENAI_API_KEY / ANTHROPIC_API_KEY

# Single-shot mode (default)
python main.py "Should I invest in index funds or crypto with $10k?"

# Hybrid mode
python main.py --hybrid "Plan my trip from Singapore to Tokyo on a $1200 budget"
```

### Example Output (abbreviated)

```json
{
  "chosen_action": "Index fund DCA strategy",
  "breakdown": {
    "positives": 7.5,
    "negatives_estimated": 3.0,
    "negatives_buffered": 3.45,
    "net": 4.05,
    "chain_id": "chain-0",
    "events": ["open brokerage", "set up monthly DCA", "12-month growth"]
  },
  "all_chains": ["... 3 alternative chains with full scores ..."],
  "stability_score": 0.95
}
```

## Validation Results

Full results in [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md). Summary:

| Test | Result |
|---|---|
| Unit tests (core math) | **38/38 passed** |
| Edge-case synthetic trees | **4/4 passed** (including adversarial flip proof) |
| Eval harness (10 cases) | PE=0.81, NA=0.81, Regret=0.00 |
| Consistency (5 runs × 3 cases) | **100% action agreement** |
| Three-way comparison | FG-CoT wins structure score on **all 5 cases** |
| Ablation study (5 conditions × 10 cases) | Multiple chains = most impactful component |

### Ablation: What Each Component Contributes

| Remove this... | What happens |
|---|---|
| Asymmetric buffer (δ=0) | Regret increases, trap options not caught |
| LCDI loops | Minor effect on aggregate metrics, stability guarantees lost |
| Multiple chains | **Regret spikes to 0.26** — model picks suboptimal actions without comparison |
| Entire framework (Standard CoT) | Lowest positive exactness, no structured output |

## Use Cases

### Tested & Validated

- **Planning decisions** — travel, budget allocation, resource trade-offs (10 test scenarios across 5 categories)
- **Risk-aware option comparison** — adversarial trap detection where buffered negatives flip the naive ranking
- **Structured LLM output** — every decision includes scored chains, reasoning trace, and stability metric
- **Evaluation harness** — reusable metrics (PE, NA, regret, loop stability) for benchmarking any reasoning method

### Potential Extensions

- **Agent planner node** — drop the FG-CoT prompt into LangChain/LangGraph/CrewAI agents as the reasoning step (LangGraph skeleton included in `agent/`)
- **RLHF/GRPO fine-tuning** — use the signed reward shaper (`training/reward_shaper.py`) to train models with asymmetric reward signals
- **Combine with ToT or ReAct** — use A-S-FLC net scoring as the evaluator inside deeper search methods
- **High-stakes domains** — finance, security, compliance, medical — where over-optimism on positives is dangerous (needs domain-specific testing)

## Project Structure

```
a-s-flc-llm-enhancer/
├── config.py                     # Hyperparameters (δ, iterations, model)
├── main.py                       # CLI entry point
├── core/
│   ├── types.py                  # Pydantic models (ForceBreakdown, DecisionOutput, EventNode)
│   ├── forces.py                 # Exact positives, estimated negatives, buffer, scoring
│   ├── chains.py                 # Event-chain tree builder, path enumeration (BFS)
│   └── loops.py                  # LCDI iteration engine
├── inference/
│   ├── fg_cot_prompt.py          # FG-CoT system prompt + hybrid analysis prompt
│   └── wrapper.py                # LLM wrapper (OpenAI/Anthropic/Groq), decide() + decide_hybrid()
├── agent/
│   ├── graph.py                  # LangGraph: Planner → Simulator → Navigator
│   └── nodes.py                  # Agent node implementations
├── eval/
│   └── metrics.py                # positive_exactness, net_alignment, chain_regret, loop_stability
├── training/
│   └── reward_shaper.py          # Signed reward signals for RLHF/GRPO fine-tuning
├── validation/
│   ├── test_cases.json           # 10 ground-truth test scenarios (5 categories)
│   ├── utils.py                  # Shared validation utilities
│   ├── eval_harness.py           # Evaluate against ground truth
│   ├── ab_compare.py             # Vanilla LLM vs FG-CoT A/B test
│   ├── three_way_compare.py      # No-CoT vs Standard CoT vs FG-CoT
│   ├── consistency_test.py       # Multi-run consistency measurement
│   └── ablation_study.py         # Component contribution analysis
├── tests/
│   ├── test_forces.py            # Unit tests for core math
│   ├── test_chains.py            # Unit tests for chain building
│   └── test_loops.py             # Unit tests for LCDI
├── examples/
│   ├── planning_task.py          # Toy example (no LLM needed)
│   └── edge_case_trees.py        # Synthetic stress tests
└── VALIDATION_RESULTS.md         # Full validation report
```

## Math

```
Optimistic Base    = Σ (exact_positives × transition_probability)
Buffered Negatives = estimated_negatives + δ × uncertainty_factor
Net Score          = Optimistic Base − Buffered Negatives
Stabilized Net     = LCDI iteration until |Δnet| < ε
```

- δ (buffer): 0.10–0.25 (default 0.15)
- ε (convergence): 0.01
- Max iterations: 3
- Max branches: 5

## Supported LLM Providers

| Provider | Model | Env var |
|---|---|---|
| Groq | llama-3.3-70b-versatile (default) | `GROQ_API_KEY` |
| OpenAI | gpt-4o, etc. | `OPENAI_API_KEY` |
| Anthropic | claude-3.5-sonnet, etc. | `ANTHROPIC_API_KEY` |

Change in `config.py`:
```python
@dataclass
class A_S_FLC_Config:
    buffer_delta: float = 0.15
    max_iterations: int = 3
    max_branches: int = 5
    epsilon: float = 0.01
    llm_provider: str = "groq"        # "openai" or "anthropic"
    model_name: str = "llama-3.3-70b-versatile"
```

## Running Tests

```bash
# Core math (no API key needed)
python -m pytest tests/ -v

# Synthetic edge cases
python examples/edge_case_trees.py

# Full validation suite (requires API key)
python validation/eval_harness.py
python validation/ablation_study.py
python validation/consistency_test.py --runs 5 --cases 3
python validation/three_way_compare.py --cases 5
```

## License

MIT

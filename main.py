# A-S-FLC Framework — Entry Point
#
# Quick-start:
#   python main.py                              # single-shot FG-CoT mode
#   python main.py --hybrid                     # hybrid: LLM tree + core engine scoring
#   python main.py --whatif                     # single-shot + what-if stress testing
#   python main.py --whatif "your query here"   # what-if with custom query
#
# Requires an API key for the configured provider (set via env var).

import sys

from config import A_S_FLC_Config
from inference.wrapper import A_S_FLC_Wrapper


def main():
    config = A_S_FLC_Config()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if "--help" in flags:
        print("Usage: python main.py [--hybrid | --whatif] [query]")
        print()
        print("Modes:")
        print("  (default)   Single-shot: LLM analyses, scores, and ranks in one call")
        print("  --hybrid    Two-step: LLM generates event tree → core engine scores deterministically")
        print("  --whatif    Single-shot + what-if stress testing on top chains")
        print()
        print(f"  Provider: {config.llm_provider}")
        print(f"  Model:    {config.model_name}")
        print(f"  Buffer δ: {config.buffer_delta}")
        print(f"  Epsilon:  {config.epsilon}")
        return

    hybrid = "--hybrid" in flags
    whatif = "--whatif" in flags
    query = " ".join(args) if args else (
        "Plan my trip from Singapore to Tokyo on a $1200 budget with max comfort."
    )

    if whatif:
        mode_label = "what-if (FG-CoT + stress testing)"
    elif hybrid:
        mode_label = "hybrid (LLM tree + core engine)"
    else:
        mode_label = "single-shot (FG-CoT)"
    print(f"Query: {query}\n")
    print(f"Mode: {mode_label}")
    print(f"Config: provider={config.llm_provider}, model={config.model_name}, δ={config.buffer_delta}\n")

    wrapper = A_S_FLC_Wrapper(config)

    if whatif:
        result = wrapper.decide_whatif(query)
    elif hybrid:
        result = wrapper.decide_hybrid(query)
    else:
        result = wrapper.decide(query)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

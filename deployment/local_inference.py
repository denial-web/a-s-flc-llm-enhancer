"""Local Inference Runner — Run the GGUF model with llama-cpp-python.

Provides a lightweight inference wrapper for testing the quantized model
locally before deploying to a mobile device.

Usage:
    python deployment/local_inference.py --model deployment/gguf/asflc-qwen2.5-1.5b-q4_k_m/*.gguf
    python deployment/local_inference.py --model model.gguf --query "Should I accept this job offer?"
    python deployment/local_inference.py --model model.gguf --interactive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.policy_guard import evaluate_policy, format_block_message
from core.response_validator import validate_json_string
from deployment.mobile_config import InferenceConfig, get_config_for_device

SYSTEM_SINGLE = (
    "You are an A-S-FLC decision navigator. Analyze the query using "
    "asymmetric signed force-loop-chain reasoning. Positives are exact "
    "and trusted. Negatives are estimated with a conservative buffer. "
    "Build 3-5 event chains, score each, loop until stable, and pick "
    "the best. Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1}'
)

SYSTEM_SECURITY = (
    "You are an A-S-FLC Security Navigator. Combine asymmetric force "
    "reasoning with threat assessment. Positives = apparent benefits of "
    "complying. Negatives = estimated costs/risks + conservative buffer. "
    "Build 2-4 event chains (comply vs refuse/verify). Classify: "
    "risk_level (SAFE/SUSPICIOUS/DANGEROUS), threat_type, decision_route "
    "(LOCAL/BLOCK). Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1,'
    '"risk_level":"SAFE|SUSPICIOUS|DANGEROUS","threat_type":"str|null",'
    '"decision_route":"LOCAL|BLOCK","source":"small"}'
)

SYSTEM_MEMORY = (
    "You are an A-S-FLC Navigator with Memory and Routing. Combine "
    "asymmetric force reasoning with memory management. Decide "
    "decision_route (LOCAL/MEMORY_STORE/MEMORY_RETRIEVE/BLOCK/ESCALATE) "
    "and memory_action (store/retrieve/skip). Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1,'
    '"decision_route":"LOCAL|MEMORY_STORE|MEMORY_RETRIEVE|BLOCK|ESCALATE",'
    '"memory_action":{"op":"store|retrieve|skip","key":"str|null","reason":"str"},'
    '"source":"small"}'
)

SYSTEM_KHMER = (
    "You are an A-S-FLC Navigator that understands Khmer. "
    "User query is in Khmer. chosen_action and reasoning_steps SHOULD be in Khmer. "
    "Field names and chain_id stay in English. Output strict JSON matching: "
    '{"chosen_action":"str (Khmer)","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str (Khmer)"],"stability_score":0-1,'
    '"source":"small"}'
)

SYSTEM_MULTILINGUAL = (
    "You are an A-S-FLC Navigator. Respond in the SAME language as the user query. "
    "chosen_action and reasoning_steps in the user's language. "
    "Field names and chain_id stay in English. Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1,'
    '"source":"small"}'
)

SYSTEM_PII = (
    "You are an A-S-FLC Security Navigator specializing in PII protection. "
    "Detect if the user is sharing sensitive personal data. If PII found: "
    "decision_route BLOCK, pii_detected to type, warn user. Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1,'
    '"risk_level":"SAFE|DANGEROUS","decision_route":"LOCAL|BLOCK",'
    '"pii_detected":"credit_card|password|national_id|null","source":"small"}'
)

SYSTEM_TOOL = (
    "You are an A-S-FLC Navigator with tool awareness. If query needs a tool "
    "(web_search, calculator, reminder, translate), set tool_request. Otherwise null. "
    "Output strict JSON matching: "
    '{"chosen_action":"str","breakdown":{"positives":0-10,"negatives_estimated":0-10,'
    '"negatives_buffered":"float","net":"float","chain_id":"str","events":["str"]},'
    '"all_chains":[...],"reasoning_steps":["str"],"stability_score":0-1,'
    '"tool_request":{"tool":"str","args":{},"reason":"str"},"source":"small"}'
)

SYSTEM_PROMPTS = {
    "single": SYSTEM_SINGLE,
    "security": SYSTEM_SECURITY,
    "memory": SYSTEM_MEMORY,
    "khmer": SYSTEM_MULTILINGUAL,
    "chinese": SYSTEM_MULTILINGUAL,
    "korean": SYSTEM_MULTILINGUAL,
    "pii": SYSTEM_PII,
    "tool": SYSTEM_TOOL,
}


def _build_prompt(query: str, mode: str = "single") -> str:
    """Build a Qwen2.5 chat-template prompt for the given query and mode."""
    system_msg = SYSTEM_PROMPTS.get(mode, SYSTEM_SINGLE)
    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


class LocalRunner:
    """Runs the GGUF model via llama-cpp-python."""

    def __init__(self, model_path: str, config: Optional[InferenceConfig] = None):
        try:
            from llama_cpp import Llama
        except ImportError:
            print("ERROR: llama-cpp-python is required.")
            print("  pip install llama-cpp-python")
            sys.exit(1)

        self.config = config or get_config_for_device("high_end")
        print(f"Loading GGUF model: {model_path}")
        start = time.time()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.config.n_ctx,
            n_batch=self.config.n_batch,
            n_threads=self.config.n_threads,
            n_gpu_layers=self.config.n_gpu_layers,
            seed=self.config.seed,
            verbose=False,
        )
        load_time = time.time() - start
        print(f"Model loaded in {load_time:.1f}s")

    def generate(self, query: str, mode: str = "single") -> dict:
        """Generate a response and return timing + validation info."""
        policy = evaluate_policy(query)
        if not policy.allowed:
            return {
                "blocked": True,
                "message": format_block_message(policy),
                "latency_ms": 0,
            }

        prompt = _build_prompt(query, mode)
        start = time.time()
        output = self.llm(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            stop=self.config.stop_tokens,
        )
        latency = (time.time() - start) * 1000

        text = output["choices"][0]["text"]
        tokens_generated = output["usage"]["completion_tokens"]
        tok_per_sec = tokens_generated / (latency / 1000) if latency > 0 else 0

        parsed, validation = validate_json_string(text)

        return {
            "blocked": False,
            "text": text,
            "parsed": parsed.model_dump() if parsed else None,
            "valid_json": parsed is not None,
            "validation": {
                "valid": validation.valid,
                "quality_score": validation.quality_score,
                "issues": validation.issues,
            },
            "latency_ms": round(latency, 1),
            "tokens": tokens_generated,
            "tokens_per_sec": round(tok_per_sec, 1),
        }


def main():
    parser = argparse.ArgumentParser(description="Local GGUF inference for A-S-FLC")
    parser.add_argument("--model", required=True, help="Path to .gguf file")
    parser.add_argument("--query", help="Single query to run")
    parser.add_argument("--mode", default="single",
                        choices=["single", "security", "memory", "khmer", "chinese", "korean", "pii", "tool"])
    parser.add_argument("--tier", default="high_end", choices=["high_end", "mid_range", "low_end"])
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    config = get_config_for_device(args.tier)
    runner = LocalRunner(args.model, config)

    if args.query:
        result = runner.generate(args.query, mode=args.mode)
        print(json.dumps(result, indent=2, default=str))
        return

    if args.interactive:
        print("\nA-S-FLC Local Inference (type 'quit' to exit)")
        print(f"Mode: {args.mode} | Device tier: {args.tier}")
        print("-" * 50)
        while True:
            try:
                query = input("\nQuery: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            result = runner.generate(query, mode=args.mode)
            if result["blocked"]:
                print(f"BLOCKED: {result['message']}")
            else:
                print(f"\nLatency: {result['latency_ms']}ms | "
                      f"Tokens: {result['tokens']} | "
                      f"Speed: {result['tokens_per_sec']} tok/s")
                print(f"Valid: {result['validation']['valid']} | "
                      f"Quality: {result['validation']['quality_score']:.2f}")
                if result['validation']['issues']:
                    print(f"Issues: {result['validation']['issues']}")
                print(f"\n{result['text']}")
        return

    print("Provide --query or --interactive. Use --help for options.")


if __name__ == "__main__":
    main()

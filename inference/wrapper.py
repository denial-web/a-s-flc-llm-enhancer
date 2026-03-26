# A-S-FLC Inference Wrapper
#
# Wraps LLM calls (OpenAI, Anthropic, Groq) with the Force-Guided CoT prompt
# and parses output into typed DecisionOutput.
#
# Two modes:
#   decide()        — single-shot: LLM does everything (analyse + score + rank)
#   decide_hybrid() — two-step: LLM analyses → core engine scores deterministically

import json
import re
from typing import Any, Dict, List

from config import A_S_FLC_Config
from core.chains import build_tree_from_llm_output, enumerate_paths
from core.forces import rank_chains
from core.loops import run_all_chains
from core.types import DecisionOutput, ForceBreakdown
from inference.fg_cot_prompt import (
    ANALYSIS_SYSTEM_PROMPT,
    ANALYSIS_USER_TEMPLATE,
    FG_COT_SYSTEM_PROMPT,
    FG_COT_USER_TEMPLATE,
    WHATIF_SYSTEM_PROMPT,
    WHATIF_USER_TEMPLATE,
)


def _create_client(config: A_S_FLC_Config):
    """Instantiate the appropriate LLM client based on provider."""
    if config.llm_provider == "openai":
        from openai import OpenAI
        return OpenAI()
    elif config.llm_provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic()
    elif config.llm_provider == "groq":
        from groq import Groq
        return Groq()
    else:
        raise ValueError(f"Unsupported provider: {config.llm_provider}")


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of the LLM response, handling markdown fences
    and common LLM quirks like single quotes or trailing commas."""
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        raw = brace_match.group(0) if brace_match else text

    # Fix single quotes → double quotes (common LLM mistake)
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    fixed = raw.replace("'", '"')
    # Remove trailing commas before } or ]
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return fixed


class A_S_FLC_Wrapper:
    def __init__(self, config: A_S_FLC_Config):
        self.config = config
        self.client = _create_client(config)

    def _call_openai(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.config.model_name,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.3,
        )
        return response.content[0].text

    def _call_groq(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def _call_llm(self, system: str, user: str) -> str:
        provider = self.config.llm_provider
        if provider == "openai":
            return self._call_openai(system, user)
        elif provider == "anthropic":
            return self._call_anthropic(system, user)
        elif provider == "groq":
            return self._call_groq(system, user)
        raise ValueError(f"Unsupported provider: {provider}")

    def decide(self, query: str) -> DecisionOutput:
        """Single-shot mode: LLM analyses, scores, ranks, and picks in one call."""
        system_prompt = FG_COT_SYSTEM_PROMPT.format(
            buffer_delta=self.config.buffer_delta,
            epsilon=self.config.epsilon,
        )
        user_prompt = FG_COT_USER_TEMPLATE.format(query=query)
        raw = self._call_llm(system_prompt, user_prompt)
        cleaned = _extract_json(raw)
        return DecisionOutput.model_validate(json.loads(cleaned))

    def decide_hybrid(self, query: str) -> DecisionOutput:
        """Hybrid mode: LLM generates the event tree, core engine scores it.

        Step 1: LLM call → event-chain tree JSON (positives, negatives, probs)
        Step 2: build_tree_from_llm_output → enumerate_paths → rank_chains → run_all_chains
        Step 3: Assemble DecisionOutput from deterministic results

        This eliminates net-score variance: the LLM only provides the
        qualitative structure; all math is handled by the core engine.
        """
        system_prompt = ANALYSIS_SYSTEM_PROMPT
        user_prompt = ANALYSIS_USER_TEMPLATE.format(query=query)
        raw = self._call_llm(system_prompt, user_prompt)
        cleaned = _extract_json(raw)
        analysis = json.loads(cleaned)

        reasoning: List[str] = analysis.get("reasoning", [])

        root = build_tree_from_llm_output(analysis["chains"])
        paths = enumerate_paths(root, self.config)
        reasoning.append(f"Built {len(paths)} event-chain paths from LLM analysis")

        results = run_all_chains(paths, self.config)
        if not results:
            return DecisionOutput(
                chosen_action="No viable chains found",
                breakdown=ForceBreakdown(
                    positives=0, negatives_estimated=0, negatives_buffered=0,
                    net=0, chain_id="none", events=[],
                ),
                reasoning_steps=reasoning,
                stability_score=0.0,
            )

        best_bd, best_history, best_stability = results[0]
        all_bds = [r[0] for r in results]

        reasoning.extend([
            f"Core engine scored {len(results)} chains deterministically",
            f"Selected {best_bd.chain_id} with net={best_bd.net:+.4f}, stability={best_stability}",
        ])

        return DecisionOutput(
            chosen_action=best_bd.events[1] if len(best_bd.events) > 1 else best_bd.events[0],
            breakdown=best_bd,
            all_chains=all_bds,
            reasoning_steps=reasoning,
            stability_score=best_stability,
        )

    def decide_whatif(self, query: str) -> DecisionOutput:
        """What-If mode: single-shot FG-CoT extended with stress testing.

        Same as decide() but uses the WHATIF_SYSTEM_PROMPT which instructs the
        LLM to run a what-if scenario on the top 2 chains and flag high-risk
        options where the net drops >15% under worst-case conditions.
        """
        system_prompt = WHATIF_SYSTEM_PROMPT.format(
            buffer_delta=self.config.buffer_delta,
            epsilon=self.config.epsilon,
        )
        user_prompt = WHATIF_USER_TEMPLATE.format(query=query)
        raw = self._call_llm(system_prompt, user_prompt)
        cleaned = _extract_json(raw)
        return DecisionOutput.model_validate(json.loads(cleaned))

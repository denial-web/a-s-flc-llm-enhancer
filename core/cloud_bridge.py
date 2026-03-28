"""Cloud Bridge — Escalation layer that forwards queries to a large cloud model.

When the small local model signals ESCALATE (low confidence, unknown threat,
complex reasoning), the cloud bridge calls a large model (e.g. GPT-4o,
Claude 3.5, Llama 3.3 70B) and returns a validated DecisionOutput.

The bridge also captures the (query, small_output, large_output) triple
for later distillation back into the small model.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from config import A_S_FLC_Config
from core.types import DecisionOutput

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        raw = brace.group(0) if brace else text
    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        fixed = raw.replace("'", '"')
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        return fixed


class CloudBridge:
    """Sends escalated queries to a large cloud model and returns validated output."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
    ):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI()
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic()
        elif self.provider == "groq":
            from groq import Groq
            self._client = Groq()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        return self._client

    def _call(self, system: str, user: str) -> str:
        client = self._get_client()
        if self.provider == "openai" or self.provider == "groq":
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
            )
            return resp.choices[0].message.content
        elif self.provider == "anthropic":
            resp = client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=self.temperature,
            )
            return resp.content[0].text
        raise ValueError(f"Unsupported provider: {self.provider}")

    def escalate(
        self,
        query: str,
        small_output: Optional[DecisionOutput] = None,
        escalation_reason: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> DecisionOutput:
        """Send a query to the large model with optional context from the small model."""
        if system_prompt is None:
            system_prompt = _ESCALATION_SYSTEM_PROMPT

        context_lines = []
        if small_output:
            context_lines.append(
                f"Small model output (may be incomplete/uncertain):\n"
                f"{small_output.model_dump_json(indent=2)}"
            )
        if escalation_reason:
            context_lines.append(f"Escalation reason: {escalation_reason}")

        context_block = "\n\n".join(context_lines)
        user_prompt = _ESCALATION_USER_TEMPLATE.format(
            query=query,
            context=context_block,
        )

        raw = self._call(system_prompt, user_prompt)
        cleaned = _extract_json(raw)
        result = DecisionOutput.model_validate(json.loads(cleaned))
        result.source = "large_knowledge"
        return result


_ESCALATION_SYSTEM_PROMPT = """
You are an expert A-S-FLC decision navigator (large model). You handle escalated queries
that a smaller local model could not confidently answer.

Follow the standard A-S-FLC framework:
- Positives = exact, 100% trusted (0-10 scale).
- Negatives = estimated + conservative buffer.
- Build 3-5 event chains, loop until stable.
- Output ONLY valid JSON matching DecisionOutput schema.

Set source = "large_knowledge".
Set escalation_reason = null (you are the resolver).

DecisionOutput schema:
{
  "chosen_action": "<string>",
  "breakdown": {"positives": <float>, "negatives_estimated": <float>, "negatives_buffered": <float>, "net": <float>, "chain_id": "<string>", "events": ["<string>"]},
  "all_chains": [{"positives": <float>, "negatives_estimated": <float>, "negatives_buffered": <float>, "net": <float>, "chain_id": "<string>", "events": ["<string>"]}],
  "reasoning_steps": ["<string>"],
  "stability_score": <float 0-1>,
  "what_if_summary": "<string or null>",
  "risk_flags": ["<string>"],
  "risk_level": "<string or null>",
  "threat_type": "<string or null>",
  "decision_route": "LOCAL",
  "memory_action": null,
  "knowledge_request": null,
  "escalation_reason": null,
  "source": "large_knowledge"
}
"""

_ESCALATION_USER_TEMPLATE = """
Escalated query: {query}

{context}

Provide a thorough A-S-FLC analysis. Output ONLY the final JSON.
"""

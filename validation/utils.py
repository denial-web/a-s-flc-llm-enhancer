"""Shared utilities for A-S-FLC validation scripts."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import A_S_FLC_Config

TEST_CASES_PATH = Path(__file__).resolve().parent / "test_cases.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_test_cases(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    p = path or TEST_CASES_PATH
    with open(p) as f:
        return json.load(f)


def save_results(data: Any, filename: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{filename}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return out_path


def create_llm_client(config: A_S_FLC_Config):
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


def call_llm(client, config: A_S_FLC_Config, system: str, user: str, temperature: float = 0.3) -> str:
    """Generic LLM call that works with any configured provider."""
    if config.llm_provider in ("openai", "groq"):
        response = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
    elif config.llm_provider == "anthropic":
        response = client.messages.create(
            model=config.model_name,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return response.content[0].text
    else:
        raise ValueError(f"Unsupported provider: {config.llm_provider}")


def print_table(headers: List[str], rows: List[List[str]], col_width: int = 18):
    """Print a simple formatted table to stdout."""
    header_line = "".join(h.ljust(col_width) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(str(c).ljust(col_width) for c in row))


def check_api_key(config: A_S_FLC_Config) -> bool:
    """Check that the required API key env var is set."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    env_var = key_map.get(config.llm_provider)
    if env_var and not os.environ.get(env_var):
        print(f"ERROR: {env_var} not set. Export it before running this script.")
        print(f"  export {env_var}=your-key-here")
        return False
    return True

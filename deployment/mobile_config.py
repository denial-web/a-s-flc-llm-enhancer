"""Mobile Deployment Configuration — Target specs and runtime settings for on-device inference.

Defines hardware targets (phone tiers), performance budgets, and
inference parameters for llama.cpp / llama-cpp-python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DeviceTarget:
    name: str
    ram_gb: float
    npu_tflops: Optional[float] = None
    notes: str = ""


DEVICE_TARGETS: Dict[str, DeviceTarget] = {
    "high_end": DeviceTarget(
        name="Flagship 2024+ (iPhone 16 Pro, Galaxy S25 Ultra, Pixel 9 Pro)",
        ram_gb=8.0,
        npu_tflops=38.0,
        notes="Can run Q4_K_M at comfortable speeds; Q5_K_M possible with tight memory",
    ),
    "mid_range": DeviceTarget(
        name="Mid-range 2024 (iPhone 15, Galaxy A55, Pixel 8a)",
        ram_gb=6.0,
        npu_tflops=15.0,
        notes="Q4_0 or Q4_K_M recommended; may need reduced context window",
    ),
    "low_end": DeviceTarget(
        name="Budget / older (iPhone 13, Galaxy A34)",
        ram_gb=4.0,
        npu_tflops=8.0,
        notes="Q4_0 only; max_seq_length=1024; may need aggressive prompt trimming",
    ),
}


@dataclass
class PerformanceBudget:
    """Target latency and resource limits for on-device inference."""
    max_first_token_ms: int = 500
    max_tokens_per_second: float = 15.0
    max_total_latency_ms: int = 5000
    max_ram_mb: int = 2048
    max_model_size_mb: int = 1200
    max_context_tokens: int = 4096


@dataclass
class InferenceConfig:
    """llama.cpp inference parameters tuned for A-S-FLC on mobile."""
    model_path: str = "asflc-qwen2.5-1.5b-q4_k_m.gguf"
    n_ctx: int = 4096
    n_batch: int = 512
    n_threads: int = 4
    n_gpu_layers: int = 0
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 1024
    stop_tokens: list = field(default_factory=lambda: ["<|im_end|>", "<|endoftext|>"])
    seed: int = 42


def get_config_for_device(tier: str = "high_end") -> InferenceConfig:
    """Return inference config tuned for the given device tier."""
    target = DEVICE_TARGETS.get(tier, DEVICE_TARGETS["mid_range"])
    config = InferenceConfig()

    if target.ram_gb <= 4.0:
        config.n_ctx = 1024
        config.n_batch = 256
        config.n_threads = 2
        config.max_tokens = 512
        config.model_path = "asflc-qwen2.5-1.5b-q4_0.gguf"
    elif target.ram_gb <= 6.0:
        config.n_ctx = 2048
        config.n_batch = 512
        config.n_threads = 4
        config.max_tokens = 768
        config.model_path = "asflc-qwen2.5-1.5b-q4_k_m.gguf"
    else:
        config.n_ctx = 4096
        config.n_batch = 512
        config.n_threads = 4
        config.max_tokens = 1024
        config.model_path = "asflc-qwen2.5-1.5b-q4_k_m.gguf"

    return config


PERFORMANCE_BUDGET = PerformanceBudget()

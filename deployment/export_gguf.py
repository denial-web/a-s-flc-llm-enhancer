"""GGUF Export — Convert fine-tuned LoRA adapter to GGUF for llama.cpp / mobile deployment.

Usage (on Colab or a machine with the model loaded):
    python deployment/export_gguf.py --adapter lora_asflc --quant q4_k_m
    python deployment/export_gguf.py --adapter lora_asflc --quant q8_0

Requires: unsloth, llama-cpp-python (for verification), and the base model.
This script is a thin wrapper around Unsloth's save_pretrained_gguf().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUPPORTED_QUANTS = {
    "q4_0": "Smallest, fastest; slight quality loss",
    "q4_k_m": "Good balance of size and quality (recommended for phones)",
    "q5_k_m": "Higher quality, ~25% larger than q4_k_m",
    "q8_0": "Near-original quality, ~2x size of q4_k_m",
    "f16": "Full fp16, largest (for debugging only)",
}

DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DEFAULT_QUANT = "q4_k_m"
DEFAULT_OUTPUT_DIR = "deployment/gguf"


def export_gguf(
    adapter_path: str = "lora_asflc",
    base_model: str = DEFAULT_BASE_MODEL,
    quant: str = DEFAULT_QUANT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    max_seq_length: int = 4096,
) -> Path:
    """Merge LoRA adapter with base model and export as GGUF."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth is required. Run this on Colab or install unsloth first.")
        print("  pip install unsloth")
        sys.exit(1)

    if quant not in SUPPORTED_QUANTS:
        print(f"ERROR: Unsupported quantization '{quant}'. Supported: {list(SUPPORTED_QUANTS.keys())}")
        sys.exit(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model}")
    print(f"Loading adapter: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    gguf_name = f"asflc-qwen2.5-1.5b-{quant}"
    print(f"Exporting GGUF ({quant}): {out_path / gguf_name}")
    model.save_pretrained_gguf(
        str(out_path / gguf_name),
        tokenizer,
        quantization_method=quant,
    )

    gguf_file = out_path / gguf_name / f"unsloth.{quant.upper()}.gguf"
    if gguf_file.exists():
        size_mb = gguf_file.stat().st_size / (1024 * 1024)
        print(f"GGUF file: {gguf_file} ({size_mb:.1f} MB)")
    else:
        for f in (out_path / gguf_name).glob("*.gguf"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"GGUF file: {f} ({size_mb:.1f} MB)")

    return out_path / gguf_name


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned model to GGUF")
    parser.add_argument("--adapter", default="lora_asflc", help="Path to LoRA adapter directory")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model name/path")
    parser.add_argument("--quant", default=DEFAULT_QUANT, choices=list(SUPPORTED_QUANTS.keys()),
                        help=f"Quantization method (default: {DEFAULT_QUANT})")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--list-quants", action="store_true", help="List supported quantizations")
    args = parser.parse_args()

    if args.list_quants:
        print("Supported quantization methods:")
        for name, desc in SUPPORTED_QUANTS.items():
            print(f"  {name:12s} — {desc}")
        return

    export_gguf(
        adapter_path=args.adapter,
        base_model=args.base_model,
        quant=args.quant,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
    )


if __name__ == "__main__":
    main()

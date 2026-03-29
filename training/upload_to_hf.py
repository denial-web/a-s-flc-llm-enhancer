"""Upload A-S-FLC dataset to HuggingFace Hub.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   (or set HF_TOKEN env var)

Usage:
    python training/upload_to_hf.py --repo YOUR_USERNAME/a-s-flc-decisions
    python training/upload_to_hf.py --repo denial-web/a-s-flc-decisions
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

DATASET_DIR = Path(__file__).resolve().parent / "dataset"

FILES_TO_UPLOAD = [
    "asflc_chat_format.jsonl",
    "asflc_instruction_format.jsonl",
    "asflc_single_pairs.jsonl",
    "asflc_security_pairs.jsonl",
    "asflc_memory_pairs.jsonl",
    "asflc_khmer_pairs.jsonl",
    "distillation_chat_format.jsonl",
    "reward_shaper_eval.json",
]


def upload(repo_id: str):
    api = HfApi()

    print(f"Creating dataset repo: {repo_id}")
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")

    readme_src = DATASET_DIR / "README_HF.md"
    if readme_src.exists():
        print(f"  Uploading README.md")
        api.upload_file(
            path_or_fileobj=str(readme_src),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

    for fname in FILES_TO_UPLOAD:
        fpath = DATASET_DIR / fname
        if fpath.exists():
            print(f"  Uploading {fname} ({fpath.stat().st_size / 1024:.1f} KB)")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="dataset",
            )
        else:
            print(f"  SKIP {fname} (not found)")

    print(f"\nDone! Dataset available at:")
    print(f"  https://huggingface.co/datasets/{repo_id}")


def main():
    if "--repo" not in sys.argv:
        print("Usage: python training/upload_to_hf.py --repo YOUR_USERNAME/a-s-flc-decisions")
        sys.exit(1)

    idx = sys.argv.index("--repo")
    repo_id = sys.argv[idx + 1]
    upload(repo_id)


if __name__ == "__main__":
    main()

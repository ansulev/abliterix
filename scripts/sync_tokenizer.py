#!/usr/bin/env python3
"""Sync tokenizer files from an upstream model to a downstream abliterated repo.

Usage:
    python scripts/sync_tokenizer.py \
        --upstream google/gemma-4-31B-it \
        --downstream wangzhang/gemma-4-31B-it-abliterated \
        --files tokenizer_config.json chat_template.jinja

    python scripts/sync_tokenizer.py \
        --upstream Qwen/Qwen3.6-35B-A3B \
        --downstream wangzhang/Qwen3.6-35B-A3B-abliterated
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    sys.exit("HF_TOKEN missing — set it in .env or environment")

DEFAULT_FILES = ["tokenizer_config.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upstream", required=True, help="Source model repo (e.g. google/gemma-4-31B-it)")
    parser.add_argument("--downstream", required=True, help="Target abliterated repo")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES, help="Files to sync (default: tokenizer_config.json)")
    args = parser.parse_args()

    api = HfApi(token=TOKEN)

    print(f"\n=== {args.downstream} <- {args.upstream} ===")
    for fname in args.files:
        local = hf_hub_download(repo_id=args.upstream, filename=fname, token=TOKEN)
        size = Path(local).stat().st_size
        print(f"  fetched {fname} ({size} bytes)")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=fname,
            repo_id=args.downstream,
            repo_type="model",
            commit_message=f"sync {fname} from {args.upstream}",
        )
        print(f"  uploaded -> {args.downstream}/{fname}")

    print("\nDone.")


if __name__ == "__main__":
    main()

# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Offline FP8 → BF16 dequant CLI.

Reads a local FP8-quantised model directory (safetensors shards + config.json
+ tokenizer files + optional trust_remote_code modeling Python) and writes a
standalone BF16 copy. The output loads like any vanilla BF16 model — no FP8
quantiser is invoked at load time, which sidesteps every transformers 5.x FP8
MoE bug we've hit on MiniMax-M2 / DeepSeek-V3 / Qwen3-FP8.

Use this whenever in-memory materialisation fails (fused MoE FP8Experts,
packed Marlin kernels, transformers FP8 quantiser traversal crashes) or when
you want the resulting model usable by downstream tools without abliterix.

Usage
-----
    python -m abliterix.scripts.dequant_fp8 \\
        /workspace/hf_cache/hub/models--MiniMaxAI--MiniMax-M2.7/snapshots/<sha> \\
        /workspace/minimax_m27_bf16

    # Or after pre-download via `hf download`:
    python -m abliterix.scripts.dequant_fp8 \\
        $(hf download MiniMaxAI/MiniMax-M2.7 --local-dir-only) \\
        /workspace/minimax_m27_bf16

Cost: ~2x disk (FP8 230GB → BF16 460GB for MiniMax-M2). On a 192-core H200
pod, ~13 min end-to-end — dominated by disk I/O, not GPU arithmetic.

Memory: fits inside a single GPU's VRAM at any time (shard-by-shard
streaming); runs on CPU if no GPU is available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..core.fp8_utils import dequant_model_to_disk


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m abliterix.scripts.dequant_fp8",
        description=(
            "Pre-dequant an FP8-quantised model directory to a standalone "
            "BF16 copy (sidesteps transformers FP8 MoE quantiser bugs)."
        ),
    )
    p.add_argument(
        "src",
        type=Path,
        help=(
            "Source FP8 model directory (e.g. a HuggingFace snapshot path "
            "containing config.json + *.safetensors)."
        ),
    )
    p.add_argument(
        "dst",
        type=Path,
        help="Destination directory for the BF16 copy (will be created).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU dequant (default: use CUDA if available).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-shard progress output.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.src.is_dir():
        print(f"error: {args.src} is not a directory", file=sys.stderr)
        return 2
    if not (args.src / "config.json").exists():
        print(
            f"error: {args.src}/config.json not found — "
            "is this a HuggingFace snapshot directory?",
            file=sys.stderr,
        )
        return 2
    try:
        dequant_model_to_disk(
            args.src, args.dst, use_cuda=not args.cpu, verbose=not args.quiet
        )
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

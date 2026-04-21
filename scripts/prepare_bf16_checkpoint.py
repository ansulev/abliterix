# Abliterix — MXFP4 → BF16 checkpoint dequantizer
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pre-dequantize a native-MXFP4 model to BF16 safetensors on disk.

Why this exists
---------------
abliterix's new vLLM in-place editing path (``[vllm].use_in_place_editing
= true``) requires vLLM's UnquantizedFusedMoEMethod (BF16). vLLM's
Mxfp4MoEMethod.process_weights_after_loading repacks w2_weight into an
opaque block layout so in-place writes miss the kernel — see vLLM
RFC #31848 / memory ``vllm_live_suppression_dead_end``.

For gpt-oss-20b/120b (only native-MXFP4 models in scope), this script
does the dequant ONCE up front via HF's ``Mxfp4Config(dequantize=True)``,
saves BF16 ``safetensors`` to a local directory, and strips the
``quantization_config`` from ``config.json`` so vLLM loads it as plain
BF16.

The output directory becomes the new model_id for the abliterix run.

Usage
-----
    python3 scripts/prepare_bf16_checkpoint.py \\
        --model openai/gpt-oss-120b \\
        --out /workspace/gpt-oss-120b-bf16

Then point your TOML config at the output:
    [model]
    model_id = "/workspace/gpt-oss-120b-bf16"

Cost
----
* Time: ~8–15 min for gpt-oss-120b on a 4× RTX PRO 6000 pod
  (dominated by safetensors write — 232 GB at ~500 MB/s).
* Disk: ~232 GB for 120b, ~40 GB for 20b.
* Wall clock is dominated by disk write, not compute.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="openai/gpt-oss-120b",
        help="HF model_id or local path to the MXFP4 source checkpoint.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Destination directory. Must not exist or be empty. "
        "This becomes the new model_id for the abliterix run.",
    )
    ap.add_argument(
        "--device-map",
        default="auto",
        help="device_map for HF load. 'auto' uses all visible GPUs. "
        "'cpu' works but needs >= 250GB host RAM for 120b.",
    )
    ap.add_argument(
        "--max-shard-size",
        default="50GB",
        help="Max size per safetensors shard. Smaller = more files; "
        "bigger = fewer but larger files (vLLM prefers ~50GB).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the actual save_pretrained; just verify load works.",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and any(out_path.iterdir()):
        print(f"ERROR: {out_path} already exists and is non-empty.")
        print("       Pass a fresh directory or remove it first.")
        return 1
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Importing HF + forcing Mxfp4Config(dequantize=True)…")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    try:
        from transformers import Mxfp4Config
    except ImportError:
        print(
            "ERROR: transformers does not expose Mxfp4Config. "
            "Install transformers >= 4.57.1."
        )
        return 1

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"[2/4] Loading {args.model} with dequantize=True → BF16 in memory…")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"      Loaded in {time.time() - t0:.1f}s")
    # Sanity check: at least one expert weight should now be a BF16
    # nn.Parameter, not a packed Triton tensor.
    try:
        first_layer = model.model.layers[0]
        experts = first_layer.mlp.experts
        dp = getattr(experts, "down_proj", None)
        if dp is None:
            print("      [warn] experts.down_proj missing — unusual layout.")
        else:
            dtype = dp.dtype if hasattr(dp, "dtype") else "n/a"
            shape = tuple(dp.shape) if hasattr(dp, "shape") else "n/a"
            print(
                f"      experts.down_proj: shape={shape}, dtype={dtype} "
                "(expect bf16 3-D)"
            )
            if str(dtype) != "torch.bfloat16":
                print(
                    "      [warn] down_proj is not BF16 — dequant may "
                    "not have taken effect. Check transformers version."
                )
    except Exception as e:
        print(f"      [warn] sanity probe failed: {e}")

    if args.dry_run:
        print("      (dry-run; skipping save_pretrained + tokenizer + config edits)")
        return 0

    print(f"[3/4] Saving BF16 safetensors → {out_path} …")
    t1 = time.time()
    model.save_pretrained(
        out_path,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    print(f"      Wrote shards in {time.time() - t1:.1f}s")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.save_pretrained(out_path)

    # Strip quantization_config from config.json so vLLM loads via the
    # UnquantizedFusedMoEMethod path (required for in-place editing).
    cfg_path = out_path / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        stripped = False
        for key in ("quantization_config", "quantization"):
            if key in cfg:
                del cfg[key]
                stripped = True
        if stripped:
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"      Stripped quantization_config from {cfg_path.name}")

    # Copy chat template if present (gpt-oss ships a harmony template).
    for name in ("chat_template.jinja", "generation_config.json"):
        src = None
        # Try HF cache first.
        try:
            from huggingface_hub import hf_hub_download
            src = hf_hub_download(
                repo_id=args.model, filename=name, repo_type="model"
            )
        except Exception:
            pass
        if src and Path(src).exists() and not (out_path / name).exists():
            shutil.copy(src, out_path / name)

    # Disk footprint report.
    total = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file())
    print(
        f"[4/4] Done. BF16 checkpoint at {out_path} "
        f"({total / 1e9:.1f} GB, {sum(1 for _ in out_path.iterdir())} files)"
    )
    print()
    print("Next step:")
    print(f"  Edit your config's [model] model_id to \"{out_path}\"")
    print("  or pass --model-id via CLI override.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

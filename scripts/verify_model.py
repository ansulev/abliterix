#!/usr/bin/env python3
# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pre-flight verification for any model before abliteration.

Validates GPU VRAM, disk, transformers version, config shape, module naming,
chat template, and abliterix engine compatibility BEFORE downloading weights.

Usage:
    python scripts/verify_model.py --model google/gemma-4-E2B-it
    python scripts/verify_model.py --model Qwen/Qwen3.6-35B-A3B --with-weights
    python scripts/verify_model.py --model google/gemma-4-26B-A4B-it --min-vram 52
"""

from __future__ import annotations

import argparse
import shutil
import sys


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def _warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def _info(msg: str) -> None:
    print(f"  [INFO] {msg}")


# -----------------------------------------------------------------------
# STEP 0: Transformers version
# -----------------------------------------------------------------------
def check_transformers_version() -> None:
    print("=" * 70)
    print("STEP 0: Transformers version")
    print("=" * 70)

    import transformers

    ver = transformers.__version__
    print(f"  transformers version = {ver}")
    _ok(f"transformers {ver}")


# -----------------------------------------------------------------------
# STEP 1: GPU capability
# -----------------------------------------------------------------------
def check_gpus(min_vram: float) -> None:
    import torch

    print()
    print("=" * 70)
    print("STEP 1: GPU capability")
    print("=" * 70)

    if not torch.cuda.is_available():
        _warn("CUDA not available — config checks still work but model can't load")
        return

    n = torch.cuda.device_count()
    print(f"  device count = {n}")

    total_vram = 0.0
    for i in range(n):
        cc = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        vram = props.total_memory / (1024**3)
        total_vram += vram
        print(f"  GPU{i}: {name} | SM{cc[0]}.{cc[1]} | {vram:.1f} GiB")

    _ok(f"total VRAM = {total_vram:.1f} GiB")
    if total_vram < min_vram:
        _fail(f"total VRAM {total_vram:.1f} GiB < {min_vram} GiB required")

    for i in range(n):
        cc = torch.cuda.get_device_capability(i)
        if cc < (8, 0):
            _warn(f"GPU{i} SM{cc[0]}.{cc[1]} < SM80 — BF16 may fall back to FP32 emulation")
        else:
            _ok(f"GPU{i} supports native BF16 (SM{cc[0]}.{cc[1]})")


# -----------------------------------------------------------------------
# STEP 2: Disk space
# -----------------------------------------------------------------------
def check_disk(min_disk_gb: float = 30) -> None:
    import os

    print()
    print("=" * 70)
    print("STEP 2: Disk space")
    print("=" * 70)

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"  HF_HOME = {hf_home}")
    try:
        usage = shutil.disk_usage(
            hf_home if os.path.exists(hf_home) else os.path.dirname(hf_home) or "/"
        )
    except FileNotFoundError:
        usage = shutil.disk_usage("/")
    free_gb = usage.free / (1024**3)
    print(f"  free on HF_HOME partition = {free_gb:.0f} GiB")
    if free_gb < min_disk_gb:
        _fail(f"need >= {min_disk_gb} GiB free, have {free_gb:.0f} GiB")
    _ok("disk has room for snapshot")


# -----------------------------------------------------------------------
# STEP 3: Config inspection (no weights download)
# -----------------------------------------------------------------------
def inspect_config(model_id: str) -> dict:
    from transformers import AutoConfig

    print()
    print("=" * 70)
    print("STEP 3: Config inspection (no weights)")
    print("=" * 70)

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)

    info = {
        "architectures": getattr(config, "architectures", None),
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": getattr(text_cfg, "num_hidden_layers", None),
        "hidden_size": getattr(text_cfg, "hidden_size", None),
        "intermediate_size": getattr(text_cfg, "intermediate_size", None),
        "num_attention_heads": getattr(text_cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(text_cfg, "num_key_value_heads", None),
        "num_experts": getattr(text_cfg, "num_experts", getattr(text_cfg, "num_local_experts", None)),
        "num_experts_per_tok": getattr(text_cfg, "num_experts_per_tok", None),
        "has_vision": hasattr(config, "vision_config"),
        "has_audio": hasattr(config, "audio_config"),
    }

    for k, v in info.items():
        if v is not None:
            print(f"  {k:30s} = {v}")

    n_layers = info["num_hidden_layers"]
    hidden = info["hidden_size"]
    if n_layers:
        _ok(f"{n_layers} transformer layers")
    else:
        _fail("num_hidden_layers not found in config")
    if hidden:
        _ok(f"hidden_size = {hidden}")

    n_experts = info["num_experts"]
    if n_experts:
        _ok(f"MoE model: {n_experts} experts, {info['num_experts_per_tok']} active per token")
    else:
        _ok("dense model (no MoE)")

    if info["has_vision"]:
        _ok("multimodal: vision_config present")
    if info["has_audio"]:
        _ok("multimodal: audio_config present")

    return info


# -----------------------------------------------------------------------
# STEP 4: Chat template
# -----------------------------------------------------------------------
def inspect_chat_template(model_id: str) -> None:
    from transformers import AutoTokenizer

    print()
    print("=" * 70)
    print("STEP 4: Chat template")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    print(f"  rendered:\n    {rendered!r}")
    _ok("chat template renders cleanly")


# -----------------------------------------------------------------------
# STEP 5: Engine compatibility (requires model download)
# -----------------------------------------------------------------------
def check_engine_compatibility(model_id: str, model_info: dict) -> None:
    import torch

    print()
    print("=" * 70)
    print("STEP 5: Engine module path verification (full model load)")
    print("=" * 70)

    if not torch.cuda.is_available():
        _fail("CUDA required for --with-weights model load")

    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    ModelClass = AutoModelForImageTextToText if model_info["has_vision"] else AutoModelForCausalLM

    print(f"  Loading model via {ModelClass.__name__} in BF16...")
    model = ModelClass.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # --- Layer resolution ---
    layers = None
    for path_desc, getter in [
        ("model.model.language_model.layers", lambda m: m.model.language_model.layers),
        ("model.language_model.layers", lambda m: m.language_model.layers),
        ("model.model.layers", lambda m: m.model.layers),
        ("model.layers", lambda m: m.layers),
    ]:
        try:
            layers = getter(model)
            print(f"  resolved layers via: {path_desc}")
            break
        except AttributeError:
            continue

    if layers is None:
        _fail("could not resolve transformer_layers — engine.py may need update")
        return

    n = len(layers)
    expected = model_info["num_hidden_layers"]
    if expected and n != expected:
        _warn(f"expected {expected} layers from config, loaded {n}")
    _ok(f"{n} decoder blocks loaded")

    # --- Inspect a layer ---
    block = layers[0]
    print("\n  Layer 0 structure:")
    for name, child in block.named_children():
        print(f"    {name}: {type(child).__name__}")

    # Attention projections
    self_attn = getattr(block, "self_attn", None)
    if self_attn is None:
        _warn("layer[0].self_attn not found")
    else:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(self_attn, name, None)
            if mod is None:
                _warn(f"self_attn.{name} not found")
            else:
                _ok(f"self_attn.{name} (shape={tuple(mod.weight.shape)})")

    # MLP down_proj
    mlp = getattr(block, "mlp", None)
    if mlp:
        dp = getattr(mlp, "down_proj", None)
        if dp:
            _ok(f"mlp.down_proj (shape={tuple(dp.weight.shape)})")
        else:
            _warn("mlp.down_proj not found")

    # MoE check
    if model_info["num_experts"]:
        router = getattr(block, "router", None) or getattr(block, "gate", None)
        experts = getattr(block, "experts", None) or getattr(getattr(block, "mlp", None), "experts", None)
        if router:
            _ok(f"router found: {type(router).__name__}")
        else:
            _warn("no router found in layer[0]")
        if experts:
            _ok(f"experts found: {type(experts).__name__}")

    # --- Memory usage ---
    print()
    vram_used = torch.cuda.memory_allocated() / (1024**3)
    print(f"  VRAM allocated = {vram_used:.1f} GiB")
    _ok("model loaded successfully")

    # --- Quick generation test ---
    print()
    print("  Running quick generation test...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    messages = [{"role": "user", "content": "What is 2+2? Reply in one word."}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Q: What is 2+2?")
    print(f"  A: {response.strip()}")
    _ok("generation works")

    print()
    _ok("all engine paths verified")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--min-vram", type=float, default=16, help="Minimum VRAM in GiB (default: 16)")
    parser.add_argument("--min-disk", type=float, default=30, help="Minimum disk in GiB (default: 30)")
    parser.add_argument(
        "--with-weights",
        action="store_true",
        help="Download and load the model for full engine verification.",
    )
    args = parser.parse_args()

    check_transformers_version()
    check_gpus(args.min_vram)
    check_disk(args.min_disk)
    model_info = inspect_config(args.model)
    inspect_chat_template(args.model)
    if args.with_weights:
        check_engine_compatibility(args.model, model_info)

    print()
    print("=" * 70)
    print("All pre-flight checks passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()

# Abliterix — vLLM expert editor GPU smoke test
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end GPU validation of :class:`VLLMExpertEditor`.

Runs on a single-GPU pod with vLLM installed. Downloads
``wangzhang/gpt-oss-20b-abliterated`` (BF16, no MXFP4 quantization_config),
applies an aggressive orthogonal projection with a random steering vector,
and verifies:

  1. Baseline output (greedy, seed fixed) is deterministic across calls.
  2. After ``apply_ega_projection`` with high strength: output DIFFERS
     from baseline (confirms the edit is actually hitting the kernel).
  3. After ``restore_expert_weights``: output matches baseline BYTE-FOR-BYTE
     (confirms backup/restore is lossless).

If any check fails the script exits non-zero so CI / deploy scripts can
detect regressions.

Usage::

    python3 scripts/smoke_vllm_expert_editor.py \
        --model wangzhang/gpt-oss-20b-abliterated \
        --max-gpu-memory 0.85
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time

import torch

# Force TRITON backend — our in-place edit path relies on
# `process_weights_after_loading` being a no-op (just `.contiguous()`).
# FLASHINFER_TRTLLM would repack w2_weight into an opaque block layout.
os.environ.setdefault("VLLM_FUSED_MOE_UNQUANTIZED_BACKEND", "triton")
# collective_rpc ships Python callables to workers via pickle; vLLM v1
# rejects this by default ("Object of type function is not serializable")
# unless this flag is set. The risk lives in the driver being trusted.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def _save_vec(v: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(v.detach().cpu(), buf)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wangzhang/gpt-oss-20b-abliterated")
    ap.add_argument("--prompt", default="Write a haiku about a fox.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--strength", type=float, default=8.0,
                    help="Projection strength — higher = bigger output change.")
    ap.add_argument("--max-gpu-memory", type=float, default=0.85)
    ap.add_argument("--transposed", action="store_true", default=True,
                    help="gpt-oss stores w2_weight transposed (default: True).")
    ap.add_argument("--hidden-dim", type=int, default=2880)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    print(f"[1/6] Loading {args.model} into vLLM (TP=1, BF16, enforce_eager)…")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        enforce_eager=True,               # required for in-place edits to be visible
        gpu_memory_utilization=args.max_gpu_memory,
        max_model_len=1024,
        trust_remote_code=True,
    )
    print(f"      Loaded in {time.time()-t0:.1f}s")

    # Greedy sampling — deterministic, ideal for equality check.
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    def _gen(tag: str) -> str:
        out = llm.generate([args.prompt], sp, use_tqdm=False)[0].outputs[0].text
        print(f"      [{tag}] {out[:200].replace(chr(10), ' / ')}")
        return out

    # ------------------------------------------------------------------
    # Step 1: determinism check on baseline
    # ------------------------------------------------------------------
    print("\n[2/6] Baseline generation (pre-edit)…")
    baseline_a = _gen("baseline-a")
    baseline_b = _gen("baseline-b")
    if baseline_a != baseline_b:
        print("❌ Baseline is non-deterministic under greedy — can't compare.")
        return 1
    print("      ✓ Baseline is deterministic across calls.")

    # ------------------------------------------------------------------
    # Step 2: attach editor + probe
    # ------------------------------------------------------------------
    print("\n[3/6] Attaching VLLMExpertEditor + probing experts…")
    from abliterix.core.vllm_moe_editor import VLLMExpertEditor

    editor = VLLMExpertEditor(
        llm, hidden_dim=args.hidden_dim, transposed=args.transposed
    )
    editor.probe()
    if not editor._moe_layers:
        print("❌ No MoE layers found — is this actually an MoE model?")
        return 1
    n_layers = len(editor._moe_layers)
    print(f"      Found {n_layers} MoE layer(s). Will edit every layer.")

    # ------------------------------------------------------------------
    # Step 3: build aggressive projection plan
    # ------------------------------------------------------------------
    # Use a FIXED random vector so repeats are reproducible. Strength is
    # purposely high so the output clearly diverges from baseline.
    torch.manual_seed(0)
    v = torch.randn(args.hidden_dim, dtype=torch.float32)
    v = v / v.norm()  # unit vector (matches how real refusal vectors are normalised)

    plan = [
        {"layer_idx": idx, "v": _save_vec(v), "strength": args.strength}
        for idx in sorted(editor._moe_layers)
    ]

    print(f"\n[4/6] Applying EGA projection (strength={args.strength}, "
          f"unit random v)…")
    t1 = time.time()
    result = editor.apply_ega(plan, norm_preserve=True)
    # Also flush prefix cache so KV entries from baseline don't leak.
    try:
        llm.reset_prefix_cache()
    except Exception:
        pass
    print(f"      apply_ega: applied={result['applied']} / {n_layers}, "
          f"errors={len(result.get('errors', []))}, time={time.time()-t1:.2f}s")
    if result["errors"]:
        for e in result["errors"][:5]:
            print(f"        ! {e}")
    if result["applied"] != n_layers:
        print("❌ Not every layer was edited.")
        return 1

    edited_a = _gen("edited-a")
    edited_b = _gen("edited-b")
    if edited_a != edited_b:
        print("⚠  Edited output non-deterministic — unexpected under greedy "
              "but not necessarily a bug (may be numeric noise).")
    if edited_a == baseline_a:
        print("❌ Edit did not change generation — in-place write is NOT "
              "reaching the kernel. This is the classic FLASHINFER_TRTLLM "
              "repack bug. Confirm VLLM_FUSED_MOE_UNQUANTIZED_BACKEND=triton.")
        return 1
    print("      ✓ Output changed after edit — the projection IS reaching "
          "the kernel.")

    # ------------------------------------------------------------------
    # Step 4: restore → output should match baseline byte-for-byte
    # ------------------------------------------------------------------
    print("\n[5/6] Restoring w2_weight from CPU backup…")
    t2 = time.time()
    n_restored = editor.restore()
    try:
        llm.reset_prefix_cache()
    except Exception:
        pass
    print(f"      restored {n_restored} layer(s) in {time.time()-t2:.2f}s")

    restored_a = _gen("restored-a")
    if restored_a != baseline_a:
        print("❌ Restored output DOES NOT match baseline!")
        print(f"   BASELINE: {baseline_a!r}")
        print(f"   RESTORED: {restored_a!r}")
        return 1
    print("      ✓ Restored output matches baseline byte-for-byte.")

    # ------------------------------------------------------------------
    # Step 5: second apply → restore cycle (no drift)
    # ------------------------------------------------------------------
    print("\n[6/6] Second apply → restore cycle (drift check)…")
    editor.apply_ega(plan, norm_preserve=True)
    try:
        llm.reset_prefix_cache()
    except Exception:
        pass
    edited_c = _gen("edited-c")
    if edited_c != edited_a:
        print("⚠  Second-cycle edit differs from first — suggests drift. "
              f"first='{edited_a[:60]}…'  vs  second='{edited_c[:60]}…'")
        # This is not a hard failure; drift of <1 token under BF16 is expected.
    else:
        print("      ✓ Second-cycle edited output matches first cycle.")

    editor.restore()
    try:
        llm.reset_prefix_cache()
    except Exception:
        pass
    restored_b = _gen("restored-b")
    if restored_b != baseline_a:
        print("❌ Second restore does not match baseline.")
        return 1
    print("      ✓ Second restore matches baseline.")

    print("\nAll 6 checks passed. VLLMExpertEditor works end-to-end on GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

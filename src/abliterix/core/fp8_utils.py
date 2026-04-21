# Abliterix — a derivative work of Heretic (https://github.com/p-e-w/heretic)
# Original work Copyright (C) 2025  Philipp Emanuel Weidmann (p-e-w)
# Modified work Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""FP8 detection, dequant, and materialisation utilities.

abliterix's steering operations (orthogonal projection, EGA, LoRA on top of
``base_layer.weight``) all require *writable* BF16 ``nn.Parameter`` tensors.
Native FP8 models come in three flavours, none of which directly satisfy this:

1. **Per-tensor FP8** (Qwen2.5-72B-FP8): ``nn.Linear`` holds FP8 weight + a
   single ``weight_scale`` scalar. ``W_real = W_fp8 * scale``.

2. **Block-wise FP8** (DeepSeek-V3, MiniMax-M2, Qwen3-FP8): ``nn.Linear`` holds
   FP8 weight + a 2-D ``weight_scale_inv`` tensor where each element is the
   (inverted) scale for a 128×128 block. ``W_real[i,j] = W_fp8[i,j] *
   scale_inv[i//128, j//128]``.

3. **Fused MoE FP8** (transformers 5.x ``FP8Experts``, vLLM ``FusedMoE``):
   per-expert Linear modules collapsed into a single fused 3-D tensor plus
   matching fused 3-D scale tensor. abliterix cannot reach ``experts[i].w2``
   because the indexable ModuleList no longer exists.

This module provides a unified path to turn any of the above into standard
BF16 ``nn.Linear`` modules that abliterix's direct/LoRA/EGA code can edit.

Strategies:

- ``materialize_in_memory``: dequant + replace ``.weight`` with BF16
  ``Parameter`` in place. 2× memory cost, works at load time, preserves the
  original FP8 checkpoint on disk. Supports (1) and (2) transparently, and (3)
  by unfusing FP8Experts back to per-expert ``nn.Linear`` ModuleList.

- ``dequant_to_disk``: one-off offline dequant that reads safetensors shard by
  shard, dequants tensors, and writes a sibling BF16 copy to a new directory.
  The new path loads like any standard BF16 model — transformers never invokes
  its FP8 quantizer, so MoE bugs (replace_with_fp8_linear traversal crash,
  FP8Experts unpacking, packed Marlin kernels) are entirely sidestepped.
  This is the only strategy that works for transformers 5.5.4 +
  block-wise-FP8 MoE combos that crash during ``replace_with_fp8_linear``.

Typical workflow decision tree:

- Non-MoE FP8 (Qwen2.5-72B-FP8, per-tensor)    → in-memory materialise
- MoE FP8, transformers works                  → in-memory materialise
- MoE FP8, transformers FP8Experts/Marlin bugs → pre-dequant to disk, reload
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn

_FP8_DTYPES: frozenset = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
_SCALE_ATTRS: tuple[str, ...] = ("weight_scale_inv", "weight_scale")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def module_fp8_kind(module: nn.Module) -> str:
    """Classify a module for FP8-handling purposes.

    Returns one of:
        ``"none"``      — not FP8 or unsupported
        ``"per_tensor"``— FP8 weight with a scalar weight_scale
        ``"blockwise"`` — FP8 weight with a 2-D weight_scale_inv tensor
        ``"fused_moe"`` — transformers FP8Experts / vLLM FusedMoE-style fused container
    """
    cls_name = type(module).__name__
    if cls_name in {"FP8Experts", "Fp8Experts", "FusedMoE"}:
        return "fused_moe"

    if not isinstance(module, nn.Linear):
        return "none"

    w = getattr(module, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dtype not in _FP8_DTYPES:
        return "none"

    scale_inv = getattr(module, "weight_scale_inv", None)
    if isinstance(scale_inv, torch.Tensor) and scale_inv.dim() == 2:
        return "blockwise"

    scale = getattr(module, "weight_scale", None)
    if isinstance(scale, torch.Tensor):
        return "per_tensor"

    # FP8 weight with no scale attribute — treat as simple cast.
    return "per_tensor"


def scan_fp8_model(model: nn.Module) -> dict[str, int]:
    """Return a histogram of FP8 container kinds found in ``model``."""
    hist = {"per_tensor": 0, "blockwise": 0, "fused_moe": 0}
    for _, mod in model.named_modules():
        kind = module_fp8_kind(mod)
        if kind in hist:
            hist[kind] += 1
    return hist


def iter_fp8_linears(
    model: nn.Module,
) -> Iterator[tuple[str, nn.Linear, str]]:
    """Yield ``(full_name, module, kind)`` for every FP8 ``nn.Linear`` in ``model``."""
    for name, mod in model.named_modules():
        kind = module_fp8_kind(mod)
        if kind in {"per_tensor", "blockwise"}:
            assert isinstance(mod, nn.Linear)
            yield name, mod, kind


# ---------------------------------------------------------------------------
# Dequant kernels
# ---------------------------------------------------------------------------


def dequant_blockwise(
    fp8_weight: torch.Tensor,
    scale: torch.Tensor,
    is_inv: bool,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-wise FP8 → ``out_dtype`` dequant.

    ``scale`` has shape ``(rows/block_r, cols/block_c)``; each element scales a
    ``block_r × block_c`` tile of ``fp8_weight``. ``block_r`` and ``block_c``
    are inferred from the ratio of weight to scale dimensions.

    Convention:
        ``is_inv=True``  — DeepSeek/MiniMax/Qwen3 style: ``W = W_fp8 * scale_inv``
        ``is_inv=False`` — rare: ``W = W_fp8 / scale``
    """
    w = fp8_weight.to(torch.float32)
    s = scale.to(torch.float32)
    block_r = max(1, w.shape[0] // s.shape[0])
    block_c = max(1, w.shape[1] // s.shape[1])
    s_exp = s.repeat_interleave(block_r, dim=0).repeat_interleave(block_c, dim=1)
    # Crop in case dims aren't a clean multiple.
    s_exp = s_exp[: w.shape[0], : w.shape[1]]
    out = (w * s_exp) if is_inv else (w / s_exp)
    return out.to(out_dtype)


def dequant_per_tensor(
    fp8_weight: torch.Tensor,
    scale: torch.Tensor | None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Per-tensor (or unscaled) FP8 → ``out_dtype``.

    With ``scale=None`` this is a bare dtype cast, which is appropriate for
    FP8 tensors that were stored unscaled (rare, but seen on some research
    checkpoints).
    """
    w = fp8_weight.to(torch.float32)
    if scale is not None:
        w = w * scale.to(torch.float32).to(w.device)
    return w.to(out_dtype)


def dequant_blockwise_3d(
    fp8_weight: torch.Tensor,
    scale: torch.Tensor,
    is_inv: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Block-wise dequant for a fused 3-D MoE weight.

    ``fp8_weight`` shape: ``(E, R, C)`` where ``E`` is num_experts.
    ``scale`` shape: ``(E, R/block_r, C/block_c)``.
    """
    assert fp8_weight.dim() == 3 and scale.dim() == 3
    assert fp8_weight.shape[0] == scale.shape[0]
    E = fp8_weight.shape[0]
    out = torch.empty(fp8_weight.shape, dtype=out_dtype, device=fp8_weight.device)
    for e in range(E):
        out[e] = dequant_blockwise(fp8_weight[e], scale[e], is_inv, out_dtype)
    return out


# ---------------------------------------------------------------------------
# In-memory materialisation
# ---------------------------------------------------------------------------


def materialize_fp8_linear(module: nn.Linear) -> bool:
    """Convert a single FP8 ``nn.Linear`` into a BF16 ``nn.Linear`` in place.

    Replaces ``module.weight`` with a new BF16 ``nn.Parameter`` containing the
    dequanted values, and strips ``weight_scale`` / ``weight_scale_inv``.

    Returns ``True`` if the module was converted, ``False`` if it was already
    non-FP8 or unsupported.
    """
    kind = module_fp8_kind(module)
    if kind not in {"per_tensor", "blockwise"}:
        return False

    fp8_w = module.weight
    device = fp8_w.device

    if kind == "blockwise":
        scale_inv = module.weight_scale_inv  # type: ignore[attr-defined]
        bf16_w = dequant_blockwise(fp8_w, scale_inv, is_inv=True)
        delattr(module, "weight_scale_inv")
    else:  # per_tensor
        scale = getattr(module, "weight_scale", None)
        bf16_w = dequant_per_tensor(fp8_w, scale)
        if scale is not None:
            delattr(module, "weight_scale")

    module.weight = nn.Parameter(bf16_w.to(device), requires_grad=False)
    return True


def materialize_fused_moe(
    fused: nn.Module,
    parent: nn.Module,
    attr_name: str,
    expert_naming: str = "gate_up_down",
) -> bool:
    """Dequant a fused FP8 MoE container and expose it as a per-expert ``ModuleList``.

    **Important limitation (read before calling)**: transformers ``FP8Experts``
    has a custom ``forward`` that its parent MoE block calls as
    ``self.experts(hidden_states, topk_weights, topk_ids)`` (fused kernel API).
    Replacing ``parent.<attr_name>`` with a plain ``nn.ModuleList`` makes the
    parent's forward ``TypeError: 'ModuleList' object is not callable``. This
    helper therefore only yields an *inference-ready* model when the
    architecture's MoE block forward iterates per-expert Modules (the
    pre-fused pattern: ``for e in self.experts: e.w2(...)``) — which is NOT
    the case immediately after transformers' FP8 quantiser has run.

    **Recommended use**: diagnostic / introspection only. For abliterix's
    actual abliteration workflow on fused-MoE FP8 models, the robust path is
    :func:`dequant_model_to_disk` offline pre-dequant, which produces a
    standalone BF16 checkpoint that transformers loads with the *original*
    per-expert modeling file — no FP8Experts fusion ever happens.

    Recognised fused layouts (from verified transformers / vLLM sources):

    - **transformers ``FP8Experts`` with ``has_gate=True``** (MoE norm):
      ``gate_up_proj`` ``(E, 2I, H)`` + ``gate_up_proj_scale_inv``,
      ``down_proj`` ``(E, H, I)`` + ``down_proj_scale_inv``.
      Gate and up are fused along dim 1: the first ``I`` rows are gate, the
      next ``I`` are up.

    - **transformers ``FP8Experts`` with ``has_gate=False``**:
      ``up_proj`` + ``up_proj_scale_inv``, ``down_proj`` + ``down_proj_scale_inv``.

    - **vLLM ``FusedMoE`` FP8**: ``w13_weight`` ``(E, 2I, H)`` + ``w13_weight_scale_inv``
      (or ``w13_weight_scale`` for per-tensor), ``w2_weight`` ``(E, H, I)`` +
      ``w2_weight_scale_inv``. Same gate+up fusion along dim 1 of w13.

    ``expert_naming`` picks the output attribute names on each reconstructed
    expert module:

    - ``"gate_up_down"`` — ``gate_proj`` / ``up_proj`` / ``down_proj``
      (Llama / Mistral / Qwen / DeepSeek convention)
    - ``"w1_w2_w3"``     — ``w1`` (gate) / ``w2`` (down) / ``w3`` (up)
      (Phi-3.5-MoE / MiniMax-M2 convention)

    Returns ``True`` if the container was successfully unfused, ``False``
    if no fused FP8 tensors were found or layout is unrecognised.
    """
    # Gather (name -> tensor) for every FP8 3-D tensor on `fused`. We iterate
    # all three namespaces (attrs, buffers, parameters) because FP8Experts
    # registers its weights as buffers; vLLM FusedMoE registers as parameters.
    names: set[str] = set()
    names.update(n for n in fused.__dict__ if not n.startswith("_"))
    names.update(n for n, _ in fused.named_buffers(recurse=False))
    names.update(n for n, _ in fused.named_parameters(recurse=False))

    fp8_tensors: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    for n in names:
        t = getattr(fused, n, None)
        if not isinstance(t, torch.Tensor):
            continue
        if t.dtype in _FP8_DTYPES and t.dim() == 3:
            fp8_tensors[n] = t
        elif t.dim() == 3 and t.dtype in (torch.float32, torch.float16, torch.bfloat16):
            scales[n] = t

    if not fp8_tensors:
        return False

    def _scale_for(weight_name: str) -> torch.Tensor | None:
        for suffix in ("_scale_inv", "_scale"):
            s = scales.get(weight_name + suffix)
            if s is not None:
                return s
        return None

    # Canonical (gate, up, down) or (up, down) across both transformers and
    # vLLM naming. Priority: transformers FP8Experts names first, then vLLM.
    fused_gate_up: tuple[torch.Tensor, torch.Tensor | None] | None = None
    separate_gate: tuple[torch.Tensor, torch.Tensor | None] | None = None
    separate_up: tuple[torch.Tensor, torch.Tensor | None] | None = None
    down: tuple[torch.Tensor, torch.Tensor | None] | None = None

    for cand in ("gate_up_proj", "w13_weight"):
        if cand in fp8_tensors:
            fused_gate_up = (fp8_tensors[cand], _scale_for(cand))
            break
    for cand in ("gate_proj", "w1_weight", "w1"):
        if cand in fp8_tensors:
            separate_gate = (fp8_tensors[cand], _scale_for(cand))
            break
    for cand in ("up_proj", "w3_weight", "w3"):
        if cand in fp8_tensors:
            separate_up = (fp8_tensors[cand], _scale_for(cand))
            break
    for cand in ("down_proj", "w2_weight", "w2"):
        if cand in fp8_tensors:
            down = (fp8_tensors[cand], _scale_for(cand))
            break

    if down is None:
        return False
    if fused_gate_up is None and (separate_gate is None or separate_up is None):
        return False

    num_experts = down[0].shape[0]

    def _maybe_dequant(
        pair: tuple[torch.Tensor, torch.Tensor | None],
    ) -> torch.Tensor:
        w, s = pair
        if s is not None and s.dim() == 3:
            return dequant_blockwise_3d(w, s, is_inv=True)
        if s is not None:
            # Per-tensor / 1-D scale for 3-D tensor: broadcast over experts only.
            out = torch.empty(w.shape, dtype=torch.bfloat16, device=w.device)
            for e in range(num_experts):
                out[e] = dequant_per_tensor(w[e], s if s.dim() == 0 else s[e])
            return out
        return w.to(torch.bfloat16)

    if fused_gate_up is not None:
        w_gu = _maybe_dequant(fused_gate_up)
        # Split along dim 1: first half gate, second half up.
        # Handles both transformers `gate_up_proj (E, 2I, H)` and vLLM
        # `w13_weight (E, 2I, H)`.
        assert w_gu.shape[1] % 2 == 0, (
            f"fused gate_up_proj dim 1 must be even, got {w_gu.shape}"
        )
        half = w_gu.shape[1] // 2
        gate_w = w_gu[:, :half, :].contiguous()
        up_w = w_gu[:, half:, :].contiguous()
        del w_gu
    else:
        assert separate_gate is not None and separate_up is not None
        gate_w = _maybe_dequant(separate_gate)
        up_w = _maybe_dequant(separate_up)

    down_w = _maybe_dequant(down)

    # Pick per-expert attribute names per convention.
    if expert_naming == "w1_w2_w3":
        gate_attr, up_attr, down_attr = "w1", "w3", "w2"
    elif expert_naming == "gate_up_down":
        gate_attr, up_attr, down_attr = "gate_proj", "up_proj", "down_proj"
    else:
        raise ValueError(
            f"Unknown expert_naming: {expert_naming!r} "
            "(expected 'gate_up_down' or 'w1_w2_w3')"
        )

    class _Expert(nn.Module):
        def __init__(
            self, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor
        ) -> None:
            super().__init__()
            for attr, w in ((gate_attr, gate), (up_attr, up), (down_attr, down)):
                out_f, in_f = w.shape
                lin = nn.Linear(in_f, out_f, bias=False, dtype=torch.bfloat16)
                with torch.no_grad():
                    lin.weight.copy_(w)
                setattr(self, attr, lin)

    new_experts = nn.ModuleList(
        [
            _Expert(
                gate_w[e].contiguous(), up_w[e].contiguous(), down_w[e].contiguous()
            )
            for e in range(num_experts)
        ]
    )
    setattr(parent, attr_name, new_experts)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def materialize_fp8_model(
    model: nn.Module,
    verbose: bool = True,
    unfuse_moe: bool = False,
    expert_naming: str = "gate_up_down",
) -> dict[str, int]:
    """Convert every FP8 container in ``model`` to writable BF16 in place.

    Handles standard ``nn.Linear`` (per-tensor and 2-D block-wise FP8) by
    replacing ``weight`` with a BF16 ``nn.Parameter``. This is the supported,
    inference-safe path — the parent module keeps calling ``module(x)`` which
    now runs a standard BF16 matmul.

    Fused MoE containers (transformers ``FP8Experts``, vLLM ``FusedMoE``) are
    detected and *counted* but **not** auto-unfused, because unfusing back to
    a per-expert ``ModuleList`` breaks the parent MoE block's forward (which
    was written against the fused API). ``unfuse_moe=True`` opts in to the
    best-effort :func:`materialize_fused_moe` anyway — only useful when the
    caller also restores the parent forward to per-expert iteration (or only
    needs the model for introspection, not inference).

    For abliteration workflows on models where transformers fuses the
    experts, the robust path is offline pre-dequant via
    :func:`dequant_model_to_disk` — the resulting BF16 checkpoint loads with
    the *original* modeling file (per-expert iteration preserved).

    Returns a histogram: ``{"linear": n, "fused_moe_detected": m,
    "fused_moe_unfused": u, "unsupported": k}``.
    """
    counts = {
        "linear": 0,
        "fused_moe_detected": 0,
        "fused_moe_unfused": 0,
        "unsupported": 0,
    }

    # Pass 1: standard FP8 Linear modules (safe).
    for _, mod, _ in list(iter_fp8_linears(model)):
        if materialize_fp8_linear(mod):
            counts["linear"] += 1
            if counts["linear"] % 500 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Pass 2: locate fused MoE containers.
    fused_to_handle: list[tuple[nn.Module, str, nn.Module]] = []
    for _, parent in model.named_modules():
        for attr_name, child in list(parent._modules.items()):
            if child is None:
                continue
            if module_fp8_kind(child) == "fused_moe":
                fused_to_handle.append((parent, attr_name, child))
    counts["fused_moe_detected"] = len(fused_to_handle)

    # Pass 3: optionally unfuse (opt-in; breaks inference unless caller also
    # restores parent forward).
    if unfuse_moe:
        for parent, attr_name, fused in fused_to_handle:
            try:
                if materialize_fused_moe(
                    fused, parent, attr_name, expert_naming=expert_naming
                ):
                    counts["fused_moe_unfused"] += 1
                else:
                    counts["unsupported"] += 1
            except Exception as e:
                counts["unsupported"] += 1
                if verbose:
                    print(
                        f"  [yellow]fused_moe unfuse failed on "
                        f"{type(fused).__name__} ({attr_name}): {e}[/]"
                    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if verbose:
        parts = [f"[bold]{counts['linear']}[/] Linear modules materialised"]
        if counts["fused_moe_detected"]:
            if unfuse_moe:
                parts.append(
                    f"[bold]{counts['fused_moe_unfused']}[/]/"
                    f"{counts['fused_moe_detected']} fused-MoE unfused"
                )
            else:
                parts.append(
                    f"[yellow]{counts['fused_moe_detected']} fused-MoE "
                    f"detected but NOT touched — use offline pre-dequant "
                    f"(`abliterix-dequant-fp8`) for direct steering on MoE[/]"
                )
        if counts["unsupported"]:
            parts.append(f"[yellow]{counts['unsupported']} unsupported[/]")
        print(f"* FP8 → BF16: {', '.join(parts)}")
    return counts


# ---------------------------------------------------------------------------
# Offline pre-dequant (FP8 → BF16 on disk)
# ---------------------------------------------------------------------------


def _group_tensor_keys(keys: list[str]) -> dict[str, list[str]]:
    """Group tensor keys by module prefix (everything before last ``.``).

    Used to pair ``foo.weight`` with ``foo.weight_scale_inv`` / ``foo.weight_scale``.
    """
    groups: dict[str, list[str]] = {}
    for k in keys:
        parts = k.rsplit(".", 1)
        prefix = parts[0] if len(parts) == 2 else ""
        groups.setdefault(prefix, []).append(k)
    return groups


def dequant_safetensors_shard(
    src_shard: Path,
    dst_shard: Path,
    use_cuda: bool = True,
) -> tuple[int, int]:
    """Read ``src_shard``, dequant FP8 tensors to BF16, write ``dst_shard``.

    Returns ``(num_tensors_written, num_dequanted)``.

    Strips ``weight_scale_inv`` and ``weight_scale`` keys from the output since
    those are only meaningful in FP8 context.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    out_tensors: dict[str, torch.Tensor] = {}
    n_dequant = 0

    with safe_open(src_shard, framework="pt") as f:
        keys = list(f.keys())
        groups = _group_tensor_keys(keys)

        # Cache scale tensors once per module prefix
        scales: dict[str, torch.Tensor] = {}
        for prefix, module_keys in groups.items():
            for k in module_keys:
                leaf = k.rsplit(".", 1)[-1]
                if leaf in _SCALE_ATTRS:
                    scales[prefix] = f.get_tensor(k)

        for k in keys:
            leaf = k.rsplit(".", 1)[-1]
            if leaf in _SCALE_ATTRS:
                # Dropped — scales fold into dequanted weight.
                continue

            t = f.get_tensor(k)

            if t.dtype in _FP8_DTYPES:
                prefix = k.rsplit(".", 1)[0] if "." in k else ""
                scale = scales.get(prefix)

                if scale is not None and scale.dim() == 2 and t.dim() == 2:
                    # 2-D block-wise (standard Linear)
                    t_gpu = t.to(device)
                    s_gpu = scale.to(device)
                    bf16 = dequant_blockwise(t_gpu, s_gpu, is_inv=True).cpu()
                    del t_gpu, s_gpu
                elif scale is not None and scale.dim() == 3 and t.dim() == 3:
                    # 3-D block-wise (fused MoE)
                    t_gpu = t.to(device)
                    s_gpu = scale.to(device)
                    bf16 = dequant_blockwise_3d(t_gpu, s_gpu, is_inv=True).cpu()
                    del t_gpu, s_gpu
                elif scale is not None and scale.dim() == 0:
                    # Scalar per-tensor scale
                    bf16 = dequant_per_tensor(t, scale)
                else:
                    # Unscaled FP8 → bare cast.
                    bf16 = dequant_per_tensor(t, None)

                out_tensors[k] = bf16
                n_dequant += 1
            else:
                out_tensors[k] = t

    save_file(out_tensors, str(dst_shard), metadata={"format": "pt"})
    n_total = len(out_tensors)
    out_tensors.clear()
    if use_cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return n_total, n_dequant


def dequant_model_to_disk(
    src_dir: Path,
    dst_dir: Path,
    use_cuda: bool = True,
    verbose: bool = True,
) -> int:
    """Offline pre-dequant a local FP8 model to a standalone BF16 directory.

    Reads every ``*.safetensors`` shard in ``src_dir``, dequants FP8 tensors
    (block-wise, per-tensor, or 3-D fused-MoE) to BF16, and writes the result
    to ``dst_dir``. Also:

    - Copies tokenizer + modeling Python files
    - Copies and strips ``quantization_config`` from ``config.json``
    - Emits an updated ``model.safetensors.index.json``

    Returns the total bytes written.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    idx_path = src_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        weight_map: dict[str, str] = idx["weight_map"]
    else:
        shards = sorted(src_dir.glob("*.safetensors"))
        if not shards:
            raise RuntimeError(f"No safetensors in {src_dir}")
        weight_map = {}
        from safetensors import safe_open

        for s in shards:
            with safe_open(s, framework="pt") as f:
                for k in f.keys():
                    weight_map[k] = s.name

    shards: dict[str, list[str]] = {}
    for key, fname in weight_map.items():
        shards.setdefault(fname, []).append(key)

    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    t0 = time.time()
    n_shards = len(shards)

    for i, (fname, keys) in enumerate(sorted(shards.items()), 1):
        src_shard = src_dir / fname
        dst_shard = dst_dir / fname
        if verbose:
            print(
                f"[{i}/{n_shards}] {fname} ({len(keys)} tensors)",
                flush=True,
            )
        n_total, n_dequant = dequant_safetensors_shard(
            src_shard, dst_shard, use_cuda=use_cuda
        )
        shard_size = dst_shard.stat().st_size
        total_bytes += shard_size
        for k in keys:
            leaf = k.rsplit(".", 1)[-1]
            if leaf not in _SCALE_ATTRS:
                new_weight_map[k] = fname
        if verbose:
            elapsed = time.time() - t0
            print(
                f"  wrote {shard_size / 1e9:.2f} GB "
                f"(total {total_bytes / 1e9:.1f} GB, "
                f"dequanted {n_dequant}/{n_total}, "
                f"elapsed {elapsed / 60:.1f} min)",
                flush=True,
            )

    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map},
            indent=2,
        )
    )

    # Strip quantization_config and write config.json
    cfg = json.loads((src_dir / "config.json").read_text())
    had_quant = "quantization_config" in cfg
    if had_quant:
        cfg.pop("quantization_config", None)
    (dst_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # Copy tokenizer + auxiliary files + modeling Python
    aux_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.jinja",
        "added_tokens.json",
    )
    for name in aux_files:
        p = src_dir / name
        if p.exists():
            shutil.copy2(p, dst_dir / name)
    for p in src_dir.glob("*.py"):
        shutil.copy2(p, dst_dir / p.name)

    if verbose:
        print(f"\nDone. BF16 model at {dst_dir} ({total_bytes / 1e9:.1f} GB)")
        if had_quant:
            print("  quantization_config stripped from config.json")
    return total_bytes

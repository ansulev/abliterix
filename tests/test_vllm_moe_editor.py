"""Tests for abliterix.core.vllm_moe_editor — BFS decoder lookup, router path
resolution, persistent-hook suppression plan mutation.

Runs without vLLM installed: the worker-side functions are called with a
synthetic ``worker`` object that exposes the same attribute surface as
``vllm.worker.gpu_worker.Worker`` (`model_runner.model.layers`) plus a few
nn.Module dummies wired to represent a hybrid VLM/MoE decoder.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

from abliterix.core.vllm_moe_editor import (
    _ROUTER_PATHS,
    _worker_clear_suppression_plan,
    _worker_install_persistent_suppression,
    _worker_locate_router,
    _worker_resolve_model,
    _worker_set_suppression_plan,
)


# ===================================================================
# Helpers: minimal decoder fixtures
# ===================================================================


class _RouterLike(nn.Module):
    """Stand-in for a MoE router (produces logits of shape (batch, n_experts))."""

    def __init__(self, hidden: int, num_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden))
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, hidden)
        return x @ self.weight.T  # (batch, num_experts)


class _MoeLayer(nn.Module):
    """Minimal transformer layer exposing ``mlp.gate`` at the usual path."""

    def __init__(self, hidden: int, num_experts: int):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = _RouterLike(hidden, num_experts)


class _MoeLayerAtBlockSparse(nn.Module):
    """Layer exposing the router at ``block_sparse_moe.gate`` (Mixtral style)."""

    def __init__(self, hidden: int, num_experts: int):
        super().__init__()
        self.block_sparse_moe = nn.Module()
        self.block_sparse_moe.gate = _RouterLike(hidden, num_experts)


class _DecoderLike(nn.Module):
    """Decoder with ``.layers`` — stand-in for ``GptOssModel`` / ``Qwen3_5MoeModel``."""

    def __init__(self, layer_list: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layer_list)


class _TopLikeTwoLevel(nn.Module):
    """Top-level model: ``top.model.layers`` (GptOss, Qwen3, Llama shape)."""

    def __init__(self, decoder: _DecoderLike):
        super().__init__()
        self.model = decoder


class _TopLikeVLM(nn.Module):
    """Top-level VLM wrapper: ``top.language_model.model.layers``
    (Qwen3_5MoeForConditionalGeneration shape)."""

    def __init__(self, decoder: _DecoderLike):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = decoder
        # Add a sibling visual module to match the real hybrid layout.
        self.visual = nn.Linear(16, 16)


def _make_worker(top_model: nn.Module) -> types.SimpleNamespace:
    """Emulate vLLM's Worker object: only ``model_runner.model`` is read."""
    worker = types.SimpleNamespace()
    worker.model_runner = types.SimpleNamespace(model=top_model)
    return worker


# ===================================================================
# _worker_resolve_model: BFS across VLM + plain layouts
# ===================================================================


def test_resolve_model_plain_two_level():
    """Standard GptOss / Qwen3 layout: top.model.layers resolves to `model`."""
    decoder = _DecoderLike([_MoeLayer(32, 4)])
    top = _TopLikeTwoLevel(decoder)
    worker = _make_worker(top)
    resolved = _worker_resolve_model(worker)
    assert resolved is decoder


def test_resolve_model_vlm_three_level():
    """Qwen3_5MoeForConditionalGeneration: top.language_model.model.layers."""
    decoder = _DecoderLike([_MoeLayer(32, 4), _MoeLayer(32, 4)])
    top = _TopLikeVLM(decoder)
    worker = _make_worker(top)
    resolved = _worker_resolve_model(worker)
    assert resolved is decoder


def test_resolve_model_failure_raises():
    """No `.layers` anywhere within depth-3 → RuntimeError."""
    top = nn.Linear(8, 8)  # no .layers
    worker = _make_worker(top)
    with pytest.raises(RuntimeError, match="Cannot locate decoder"):
        _worker_resolve_model(worker)


# ===================================================================
# _worker_locate_router: walks the _ROUTER_PATHS tuple
# ===================================================================


def test_locate_router_mlp_gate():
    """Qwen3 MoE / DeepSeek: router at mlp.gate."""
    layer = _MoeLayer(32, 4)
    router, path = _worker_locate_router(layer)
    assert router is layer.mlp.gate
    assert path == "mlp.gate"


def test_locate_router_block_sparse():
    """Mixtral / Phi-3.5-MoE: router at block_sparse_moe.gate."""
    layer = _MoeLayerAtBlockSparse(32, 4)
    router, path = _worker_locate_router(layer)
    assert router is layer.block_sparse_moe.gate
    assert path == "block_sparse_moe.gate"


def test_locate_router_none_when_missing():
    """A layer without any router-like attribute returns (None, None)."""
    layer = nn.Linear(8, 8)
    router, path = _worker_locate_router(layer)
    assert router is None and path is None


def test_router_paths_covers_known_architectures():
    """_ROUTER_PATHS must at least include the gpt-oss / Qwen3 / Mixtral paths."""
    assert "mlp.router" in _ROUTER_PATHS
    assert "mlp.gate" in _ROUTER_PATHS
    assert "block_sparse_moe.gate" in _ROUTER_PATHS


# ===================================================================
# Persistent-hook suppression: install once, mutate plan per trial.
# ===================================================================


def _build_worker_with_routers(num_layers: int, num_experts: int, hidden: int = 32):
    decoder = _DecoderLike(
        [_MoeLayer(hidden, num_experts) for _ in range(num_layers)]
    )
    top = _TopLikeTwoLevel(decoder)
    return _make_worker(top), decoder


def test_install_persistent_suppression_is_idempotent():
    worker, _ = _build_worker_with_routers(num_layers=4, num_experts=8)
    n1 = _worker_install_persistent_suppression(worker)
    n2 = _worker_install_persistent_suppression(worker)
    assert n1 == 4
    assert n2 == n1  # second call reuses existing handles


def test_install_then_set_plan_affects_forward():
    """After installing the persistent hook and setting a plan, a forward
    through the router subtracts the penalty from the designated expert."""
    worker, decoder = _build_worker_with_routers(
        num_layers=2, num_experts=8, hidden=16
    )
    n = _worker_install_persistent_suppression(worker)
    assert n == 2

    # Plan: suppress expert 3 on layer 0 by a big penalty, layer 1 untouched.
    _worker_set_suppression_plan(
        worker,
        {0: ([3], [1000.0]), 1: ([], [])},
    )

    # Forward layer 0 router; expert 3 logit must be far smaller than the rest.
    x = torch.randn(2, 16)
    layer0 = decoder.layers[0]
    logits = layer0.mlp.gate(x)
    # Expert 3 should be pushed below the others.
    others = torch.cat([logits[:, :3], logits[:, 4:]], dim=1)
    assert (logits[:, 3:4] < others.min(dim=1, keepdim=True).values).all()

    # Forward layer 1 router; nothing should change — plan empty.
    layer1 = decoder.layers[1]
    logits_l1_before = layer1.mlp.gate(x).clone()
    # Apply again (hook re-reads per-forward); logits stable.
    logits_l1_after = layer1.mlp.gate(x)
    assert torch.allclose(logits_l1_before, logits_l1_after)


def test_clear_plan_restores_baseline():
    """After clear, no penalty is applied — logits match an un-hooked forward."""
    worker, decoder = _build_worker_with_routers(
        num_layers=1, num_experts=6, hidden=16
    )
    _worker_install_persistent_suppression(worker)

    layer = decoder.layers[0]
    x = torch.randn(1, 16)
    baseline = (x @ layer.mlp.gate.weight.T).clone()

    _worker_set_suppression_plan(worker, {0: ([0, 1], [100.0, 100.0])})
    out_suppressed = layer.mlp.gate(x)
    assert not torch.allclose(out_suppressed, baseline)

    _worker_clear_suppression_plan(worker)
    out_restored = layer.mlp.gate(x)
    assert torch.allclose(out_restored, baseline)


def test_plan_mutation_does_not_reinstall_hook():
    """Setting a new plan between trials must NOT call register_forward_hook
    again — register_forward_hook on a vLLM-compiled model is silently
    skipped post-compile (pytorch#117758), so abliterix's whole fix relies on
    the hook being installed ONCE."""
    worker, _ = _build_worker_with_routers(num_layers=3, num_experts=4)
    _worker_install_persistent_suppression(worker)
    handles_before = list(worker._abliterix_persistent_handles)

    _worker_set_suppression_plan(worker, {0: ([1], [5.0])})
    _worker_set_suppression_plan(worker, {1: ([2], [8.0])})
    _worker_clear_suppression_plan(worker)

    handles_after = list(worker._abliterix_persistent_handles)
    assert handles_after == handles_before  # same handle objects, no re-reg.


def test_plan_survives_dtype_cast():
    """Router logits may be bf16 on GPU — hook must cast penalties to match."""
    worker, decoder = _build_worker_with_routers(
        num_layers=1, num_experts=4, hidden=8
    )
    _worker_install_persistent_suppression(worker)
    _worker_set_suppression_plan(worker, {0: ([2], [42.0])})

    layer = decoder.layers[0]
    layer.mlp.gate = layer.mlp.gate.to(torch.bfloat16)
    x = torch.randn(1, 8, dtype=torch.bfloat16)
    out = layer.mlp.gate(x)
    assert out.dtype == torch.bfloat16
    # Expert 2 was pushed down by ~42 (within bf16 rounding).
    assert out[0, 2] < out[0, 0] - 20


# ===================================================================
# Expert-Granular Abliteration (EGA) via in-place w2_weight edit.
#
# These tests emulate vLLM's FusedMoE with a pure-PyTorch stand-in that
# exposes `layer.mlp.experts.w2_weight` exactly like the real module.
# They cover the worker-side math only (projection + norm preserve +
# backup/restore) without needing a GPU or an actual vLLM engine.
# ===================================================================

import io as _io  # noqa: E402  (intentional — tests only)

from abliterix.core.vllm_moe_editor import (  # noqa: E402
    _worker_apply_ega_batch,
    _worker_backup_experts,
    _worker_locate_moe_experts,
    _worker_probe_experts,
    _worker_restore_experts,
)


class _FusedMoELike(nn.Module):
    """Stand-in for vLLM's ``FusedMoE`` exposing ``w2_weight`` as Parameter.

    ``transposed=False`` stores the standard MoE layout
    ``(num_experts, hidden, intermediate)``.
    ``transposed=True`` stores the gpt-oss layout
    ``(num_experts, intermediate, hidden)``.
    """

    def __init__(
        self,
        num_experts: int,
        hidden: int,
        intermediate: int,
        transposed: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if transposed:
            shape = (num_experts, intermediate, hidden)
        else:
            shape = (num_experts, hidden, intermediate)
        self.w2_weight = nn.Parameter(torch.randn(*shape, dtype=dtype))


class _MoeLayerWithExperts(nn.Module):
    """Layer exposing both ``mlp.gate`` (router) AND ``mlp.experts`` (FusedMoE)."""

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        num_experts: int,
        transposed: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = _RouterLike(hidden, num_experts)
        self.mlp.experts = _FusedMoELike(
            num_experts, hidden, intermediate,
            transposed=transposed, dtype=dtype,
        )


def _build_worker_with_experts(
    num_layers: int = 2,
    num_experts: int = 4,
    hidden: int = 8,
    intermediate: int = 16,
    transposed: bool = False,
    dtype: torch.dtype = torch.float32,
) -> tuple[types.SimpleNamespace, _DecoderLike]:
    layers = [
        _MoeLayerWithExperts(
            hidden, intermediate, num_experts, transposed=transposed, dtype=dtype
        )
        for _ in range(num_layers)
    ]
    decoder = _DecoderLike(layers)
    top = _TopLikeTwoLevel(decoder)
    worker = _make_worker(top)
    return worker, decoder


def _save_vec(v: torch.Tensor) -> bytes:
    """torch.save a 1-D tensor into bytes — matches what the editor sends."""
    buf = _io.BytesIO()
    torch.save(v.detach().cpu(), buf)
    return buf.getvalue()


def test_locate_moe_experts_finds_fused_moe():
    decoder = _DecoderLike([_MoeLayerWithExperts(8, 16, 4)])
    moe, path = _worker_locate_moe_experts(decoder.layers[0])
    assert moe is not None
    assert path == "mlp.experts"
    assert moe.w2_weight.shape == (4, 8, 16)


def test_locate_moe_experts_missing_returns_none():
    plain = _MoeLayer(8, 4)  # no `.experts` child
    moe, path = _worker_locate_moe_experts(plain)
    assert moe is None and path is None


def test_probe_experts_reports_shapes():
    worker, _ = _build_worker_with_experts(num_layers=3, hidden=8, intermediate=16)
    info = _worker_probe_experts(worker)
    assert info["n_layers"] == 3
    paths = {p for (_, p, _, _) in info["per_layer"]}
    assert paths == {"mlp.experts"}
    shapes = {sh for (_, _, sh, _) in info["per_layer"]}
    assert shapes == {(4, 8, 16)}


def test_backup_and_restore_round_trip():
    worker, decoder = _build_worker_with_experts(num_layers=2, hidden=8, intermediate=16)
    # Snapshot reference values before any edit.
    ref_l0 = decoder.layers[0].mlp.experts.w2_weight.data.clone()
    ref_l1 = decoder.layers[1].mlp.experts.w2_weight.data.clone()

    assert _worker_backup_experts(worker, [0, 1]) == 2
    # Idempotent: a second call for same layers backs up 0 new.
    assert _worker_backup_experts(worker, [0, 1]) == 0

    # Corrupt the weights.
    decoder.layers[0].mlp.experts.w2_weight.data.zero_()
    decoder.layers[1].mlp.experts.w2_weight.data.zero_()

    # Restore brings them back.
    assert _worker_restore_experts(worker) == 2
    assert torch.allclose(decoder.layers[0].mlp.experts.w2_weight.data, ref_l0)
    assert torch.allclose(decoder.layers[1].mlp.experts.w2_weight.data, ref_l1)


def _reference_projection_standard(
    W: torch.Tensor, v: torch.Tensor, strength: float, norm_preserve: bool
) -> torch.Tensor:
    """HF-style reference projection for STANDARD layout (E, hidden, intermediate).

    Mirrors `_apply_ega_steering` in steering.py:740-743 (axis_is_in=False branch).
    """
    W32 = W.to(torch.float32)
    vf = v.to(torch.float32)
    proj = torch.einsum("o,eoi->ei", vf, W32)                # (E, intermediate)
    W_new = W32 - strength * (vf.view(1, -1, 1) * proj.unsqueeze(1))
    if norm_preserve:
        orig_norms = torch.linalg.vector_norm(W32, dim=-1, keepdim=True)
        new_norms = torch.linalg.vector_norm(W_new, dim=-1, keepdim=True).clamp(min=1e-8)
        W_new = W_new * (orig_norms / new_norms)
    return W_new.to(W.dtype)


def _reference_projection_transposed(
    W: torch.Tensor, v: torch.Tensor, strength: float, norm_preserve: bool
) -> torch.Tensor:
    """HF-style reference for TRANSPOSED (gpt-oss) layout (E, intermediate, hidden).

    Mirrors `_apply_ega_steering` in steering.py:737-739 (axis_is_in=True branch).
    """
    W32 = W.to(torch.float32)
    vf = v.to(torch.float32)
    proj = torch.matmul(W32, vf)                             # (E, intermediate)
    W_new = W32 - strength * (proj.unsqueeze(-1) * vf.view(1, 1, -1))
    if norm_preserve:
        orig_norms = torch.linalg.vector_norm(W32, dim=-1, keepdim=True)
        new_norms = torch.linalg.vector_norm(W_new, dim=-1, keepdim=True).clamp(min=1e-8)
        W_new = W_new * (orig_norms / new_norms)
    return W_new.to(W.dtype)


def test_apply_ega_batch_standard_layout_matches_reference():
    """Worker projection math on standard layout matches HF reference."""
    torch.manual_seed(0)
    worker, decoder = _build_worker_with_experts(
        num_layers=1, num_experts=3, hidden=4, intermediate=6, transposed=False
    )
    v = torch.randn(4)  # hidden dim
    W_ref_in = decoder.layers[0].mlp.experts.w2_weight.data.clone()
    expected = _reference_projection_standard(W_ref_in, v, strength=1.7, norm_preserve=True)

    plan = [{"layer_idx": 0, "v": _save_vec(v), "strength": 1.7, "hidden_dim": 4}]
    result = _worker_apply_ega_batch(worker, plan, norm_preserve=True)

    assert result["applied"] == 1
    assert result["errors"] == []
    # per_layer row: (idx, axis, n_experts). axis=1 for standard.
    assert result["per_layer"][0] == (0, 1, 3)

    actual = decoder.layers[0].mlp.experts.w2_weight.data
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_apply_ega_batch_transposed_layout_matches_reference():
    """Worker projection math on gpt-oss transposed layout matches HF reference."""
    torch.manual_seed(1)
    worker, decoder = _build_worker_with_experts(
        num_layers=1, num_experts=3, hidden=4, intermediate=6, transposed=True
    )
    v = torch.randn(4)  # hidden dim
    W_ref_in = decoder.layers[0].mlp.experts.w2_weight.data.clone()
    expected = _reference_projection_transposed(W_ref_in, v, strength=2.3, norm_preserve=True)

    plan = [{
        "layer_idx": 0, "v": _save_vec(v), "strength": 2.3,
        "hidden_dim": 4, "transposed": True,
    }]
    result = _worker_apply_ega_batch(worker, plan, norm_preserve=True)

    assert result["applied"] == 1
    assert result["errors"] == []
    # per_layer row: (idx, axis, n_experts). axis=2 for transposed.
    assert result["per_layer"][0] == (0, 2, 3)

    actual = decoder.layers[0].mlp.experts.w2_weight.data
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_apply_ega_batch_ambiguous_square_resolves_via_flag():
    """When hidden == intermediate, caller's `transposed` flag disambiguates."""
    torch.manual_seed(2)
    # Square 4x4 — ambiguous without flag.
    worker, decoder = _build_worker_with_experts(
        num_layers=1, num_experts=2, hidden=4, intermediate=4, transposed=True
    )
    v = torch.randn(4)
    W_ref = decoder.layers[0].mlp.experts.w2_weight.data.clone()
    expected = _reference_projection_transposed(W_ref, v, strength=1.0, norm_preserve=False)

    plan = [{
        "layer_idx": 0, "v": _save_vec(v), "strength": 1.0,
        "hidden_dim": 4, "transposed": True,
    }]
    result = _worker_apply_ega_batch(worker, plan, norm_preserve=False)

    assert result["applied"] == 1
    assert result["per_layer"][0][1] == 2  # axis=2 → transposed path
    assert torch.allclose(
        decoder.layers[0].mlp.experts.w2_weight.data, expected, atol=1e-5, rtol=1e-5
    )


def test_apply_ega_batch_dimension_mismatch_records_error():
    """v in wrong dim → error logged, tensor unchanged, other layers still applied."""
    worker, decoder = _build_worker_with_experts(
        num_layers=2, num_experts=2, hidden=4, intermediate=6, transposed=False
    )
    W_before_l0 = decoder.layers[0].mlp.experts.w2_weight.data.clone()

    bad_v = torch.randn(99)      # wrong dim
    good_v = torch.randn(4)      # correct hidden dim

    plan = [
        {"layer_idx": 0, "v": _save_vec(bad_v), "strength": 1.0, "hidden_dim": 4},
        {"layer_idx": 1, "v": _save_vec(good_v), "strength": 1.0, "hidden_dim": 4},
    ]
    result = _worker_apply_ega_batch(worker, plan, norm_preserve=False)

    assert result["applied"] == 1
    assert len(result["errors"]) == 1
    assert "layer 0" in result["errors"][0]
    # Layer 0 untouched.
    assert torch.allclose(decoder.layers[0].mlp.experts.w2_weight.data, W_before_l0)


def test_apply_ega_then_restore_cycles():
    """apply → restore → apply yields same edited state each time (no drift)."""
    torch.manual_seed(3)
    worker, decoder = _build_worker_with_experts(
        num_layers=1, num_experts=2, hidden=4, intermediate=6, transposed=False
    )
    v = torch.randn(4)
    plan = [{"layer_idx": 0, "v": _save_vec(v), "strength": 1.5, "hidden_dim": 4}]

    # Cycle 1.
    _worker_backup_experts(worker, [0])
    _worker_apply_ega_batch(worker, plan, norm_preserve=True)
    after_c1 = decoder.layers[0].mlp.experts.w2_weight.data.clone()

    # Restore → pristine.
    _worker_restore_experts(worker)

    # Cycle 2 should produce identical output (same edit applied to same baseline).
    _worker_apply_ega_batch(worker, plan, norm_preserve=True)
    after_c2 = decoder.layers[0].mlp.experts.w2_weight.data.clone()

    assert torch.allclose(after_c1, after_c2, atol=1e-6, rtol=1e-6)


# ===================================================================
# Attention editor — qkv_proj slicing + o_proj projection.
#
# Mocks vLLM's QKVParallelLinear / RowParallelLinear by building a
# module that exposes the exact surface our worker functions read:
# ``self_attn.qkv_proj.weight``, ``self_attn.o_proj.weight``,
# ``self_attn.q_size``, ``self_attn.kv_size``.
# ===================================================================

from abliterix.core.vllm_moe_editor import (  # noqa: E402
    _worker_apply_attn_batch,
    _worker_backup_attention,
    _worker_locate_attention,
    _worker_probe_attention,
    _worker_restore_attention,
)


class _QKVParallelLike(nn.Module):
    def __init__(self, q_size: int, kv_size: int, hidden: int, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(q_size + 2 * kv_size, hidden, dtype=dtype))


class _RowParallelLike(nn.Module):
    def __init__(self, hidden: int, in_shard: int, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden, in_shard, dtype=dtype))


class _AttnLike(nn.Module):
    """Stand-in for vLLM's ``OAIAttention`` / generic attention block."""

    def __init__(
        self,
        hidden: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        tp: int = 1,
        dtype=torch.float32,
    ):
        super().__init__()
        self.q_size = q_heads * head_dim // tp
        self.kv_size = kv_heads * head_dim // tp
        self.qkv_proj = _QKVParallelLike(self.q_size, self.kv_size, hidden, dtype=dtype)
        self.o_proj = _RowParallelLike(hidden, q_heads * head_dim // tp, dtype=dtype)


class _AttnMoeLayer(nn.Module):
    """Layer exposing both ``self_attn`` AND ``mlp.experts`` (gpt-oss shape)."""

    def __init__(
        self,
        hidden: int = 8,
        q_heads: int = 4,
        kv_heads: int = 2,
        head_dim: int = 4,
        num_experts: int = 2,
        intermediate: int = 16,
        transposed: bool = False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.self_attn = _AttnLike(hidden, q_heads, kv_heads, head_dim, dtype=dtype)
        self.mlp = nn.Module()
        self.mlp.gate = _RouterLike(hidden, num_experts)
        self.mlp.experts = _FusedMoELike(
            num_experts, hidden, intermediate, transposed=transposed, dtype=dtype
        )


def _build_worker_with_attn(
    num_layers: int = 2,
    hidden: int = 8,
    q_heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 4,
    dtype=torch.float32,
) -> tuple[types.SimpleNamespace, _DecoderLike]:
    layers = [
        _AttnMoeLayer(
            hidden=hidden, q_heads=q_heads, kv_heads=kv_heads,
            head_dim=head_dim, dtype=dtype,
        )
        for _ in range(num_layers)
    ]
    decoder = _DecoderLike(layers)
    top = _TopLikeTwoLevel(decoder)
    worker = _make_worker(top)
    return worker, decoder


def test_locate_attention_finds_qkv_and_o():
    decoder = _DecoderLike([_AttnMoeLayer()])
    attn, path = _worker_locate_attention(decoder.layers[0])
    assert attn is not None
    assert path == "self_attn"
    assert hasattr(attn, "qkv_proj") and hasattr(attn, "o_proj")
    assert attn.q_size == 16   # 4 heads × 4 head_dim
    assert attn.kv_size == 8   # 2 heads × 4 head_dim


def test_probe_attention_reports_shapes_and_sizes():
    worker, _ = _build_worker_with_attn(num_layers=3, hidden=8)
    info = _worker_probe_attention(worker)
    assert info["n_layers"] == 3
    paths = {p for (_, p, *_rest) in info["per_layer"]}
    assert paths == {"self_attn"}
    # qkv_shape (q + 2*kv, hidden) = (16 + 2*8, 8) = (32, 8)
    qkv_shapes = {qs for (_, _, qs, *_rest) in info["per_layer"]}
    assert qkv_shapes == {(32, 8)}


def _reference_attn_projection(
    W: torch.Tensor, v: torch.Tensor, strength: float, norm_preserve: bool
) -> torch.Tensor:
    """HF-style reference projection — mirrors _apply_direct_steering lines 612-630."""
    W32 = W.to(torch.float32)
    vf = v.to(torch.float32)
    out_f, in_f = W32.shape
    if vf.shape[0] == out_f:
        proj = vf @ W32
        W_new = W32 - strength * vf.unsqueeze(1) * proj.unsqueeze(0)
    else:
        proj = W32 @ vf
        W_new = W32 - strength * proj.unsqueeze(1) * vf.unsqueeze(0)
    if norm_preserve:
        orig = torch.linalg.vector_norm(W32, dim=1, keepdim=True)
        new = torch.linalg.vector_norm(W_new, dim=1, keepdim=True).clamp(min=1e-8)
        W_new = W_new * (orig / new)
    return W_new.to(W.dtype)


def test_apply_attn_q_proj_slice_matches_reference():
    torch.manual_seed(10)
    worker, decoder = _build_worker_with_attn(num_layers=1, hidden=8)
    attn = decoder.layers[0].self_attn
    q_ref_in = attn.qkv_proj.weight.data[0 : attn.q_size].clone()
    v = torch.randn(8)  # hidden
    expected = _reference_attn_projection(q_ref_in, v, strength=1.3, norm_preserve=True)

    plan = [{
        "layer_idx": 0, "component": "q_proj",
        "v": _save_vec(v), "strength": 1.3,
    }]
    result = _worker_apply_attn_batch(worker, plan, norm_preserve=True)

    assert result["applied"] == 1 and result["errors"] == []
    # K and V slices unchanged.
    actual_q = attn.qkv_proj.weight.data[0 : attn.q_size]
    assert torch.allclose(actual_q, expected, atol=1e-5)


def test_apply_attn_k_proj_only_touches_k_slice():
    torch.manual_seed(11)
    worker, decoder = _build_worker_with_attn(num_layers=1, hidden=8)
    attn = decoder.layers[0].self_attn
    # Snapshot q + v slices — should stay untouched.
    before_q = attn.qkv_proj.weight.data[0 : attn.q_size].clone()
    before_v = attn.qkv_proj.weight.data[attn.q_size + attn.kv_size :].clone()

    v = torch.randn(8)
    plan = [{"layer_idx": 0, "component": "k_proj",
             "v": _save_vec(v), "strength": 2.0}]
    result = _worker_apply_attn_batch(worker, plan, norm_preserve=True)
    assert result["applied"] == 1

    after_q = attn.qkv_proj.weight.data[0 : attn.q_size]
    after_v = attn.qkv_proj.weight.data[attn.q_size + attn.kv_size :]
    assert torch.allclose(after_q, before_q)
    assert torch.allclose(after_v, before_v)


def test_apply_attn_o_proj_projects_on_output_axis():
    """o_proj has shape (hidden, intermediate_per_rank); v lives in hidden (OUTPUT)."""
    torch.manual_seed(12)
    worker, decoder = _build_worker_with_attn(num_layers=1, hidden=8)
    attn = decoder.layers[0].self_attn
    o_ref = attn.o_proj.weight.data.clone()
    v = torch.randn(8)  # hidden — matches out dim of o_proj
    expected = _reference_attn_projection(o_ref, v, strength=1.7, norm_preserve=True)

    plan = [{"layer_idx": 0, "component": "o_proj",
             "v": _save_vec(v), "strength": 1.7}]
    result = _worker_apply_attn_batch(worker, plan, norm_preserve=True)
    assert result["applied"] == 1 and result["errors"] == []
    assert torch.allclose(attn.o_proj.weight.data, expected, atol=1e-5)


def test_apply_attn_unknown_component_records_error():
    worker, _ = _build_worker_with_attn(num_layers=1)
    v = torch.randn(8)
    plan = [{"layer_idx": 0, "component": "x_proj",
             "v": _save_vec(v), "strength": 1.0}]
    result = _worker_apply_attn_batch(worker, plan, norm_preserve=False)
    assert result["applied"] == 0
    assert len(result["errors"]) == 1
    assert "unknown component" in result["errors"][0]


def test_attn_backup_and_restore_round_trip():
    worker, decoder = _build_worker_with_attn(num_layers=2)
    ref_qkv = decoder.layers[0].self_attn.qkv_proj.weight.data.clone()
    ref_o = decoder.layers[0].self_attn.o_proj.weight.data.clone()

    assert _worker_backup_attention(worker, [0, 1]) == 2
    assert _worker_backup_attention(worker, [0, 1]) == 0  # idempotent

    # Corrupt.
    decoder.layers[0].self_attn.qkv_proj.weight.data.zero_()
    decoder.layers[0].self_attn.o_proj.weight.data.zero_()

    assert _worker_restore_attention(worker) == 2
    assert torch.allclose(decoder.layers[0].self_attn.qkv_proj.weight.data, ref_qkv)
    assert torch.allclose(decoder.layers[0].self_attn.o_proj.weight.data, ref_o)


def test_attn_apply_then_restore_round_trip():
    """Apply q+k+v+o edits on one layer, then restore → pristine."""
    torch.manual_seed(13)
    worker, decoder = _build_worker_with_attn(num_layers=1, hidden=8)
    attn = decoder.layers[0].self_attn
    ref_qkv = attn.qkv_proj.weight.data.clone()
    ref_o = attn.o_proj.weight.data.clone()

    v = torch.randn(8)
    v_bytes = _save_vec(v)
    plan = [
        {"layer_idx": 0, "component": c, "v": v_bytes, "strength": 1.5}
        for c in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]

    _worker_backup_attention(worker, [0])
    result = _worker_apply_attn_batch(worker, plan, norm_preserve=True)
    assert result["applied"] == 4
    assert result["errors"] == []
    # Confirm something changed.
    assert not torch.allclose(attn.qkv_proj.weight.data, ref_qkv)
    assert not torch.allclose(attn.o_proj.weight.data, ref_o)

    # Restore → byte-equal to original.
    assert _worker_restore_attention(worker) == 1
    assert torch.equal(attn.qkv_proj.weight.data, ref_qkv)
    assert torch.equal(attn.o_proj.weight.data, ref_o)

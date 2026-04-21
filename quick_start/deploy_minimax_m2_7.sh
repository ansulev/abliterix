#!/usr/bin/env bash
# Deploy MiniMax-M2.7 abliteration on 4× RTX PRO 6000 RunPod pod.
#
# Prereqs on the pod:
#   - 4× RTX PRO 6000 (96GB each, 384GB total)
#   - ≥500GB disk on /workspace (FP8 model ~230GB + HF cache + hidden states)
#   - Repo synced to /workspace/abliterix (scp -r from your laptop)
#   - .env in /workspace/abliterix/.env with HF_TOKEN + OPENROUTER_API_KEY
#
# Usage on the pod:
#   bash /workspace/abliterix/quick_start/deploy_minimax_m2_7.sh
#
# What this does:
#   1. Sanity-checks GPU count (expect 4) and VRAM (expect ≥ 90GB)
#   2. Tests HF download speed (aborts on < 50 MB/s — 230GB would take hours)
#   3. Installs deps pinned for MiniMax + vLLM 0.19 compatibility
#   4. Uninstalls flash-attn if present (ABI mismatch with RunPod's PyTorch)
#   5. Sources .env (LLM judge needs OPENROUTER_API_KEY)
#   6. Launches abliterix with nohup + tee
#
# Expected runtime: Phase 1 (vLLM TP hidden state extraction) ~10-15 min,
# Phase 2 (50 trials × vLLM TP generation + LoRA hot-swap) ~2-4 h on 4× RTX
# PRO 6000. Actual first-time load adds ~15-25 min for 230GB download.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/minimax_m2.7_vllm.toml}"
LOG_FILE="${LOG_FILE:-/workspace/run_minimax_m2_7.log}"
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
HS_DIR="${HS_DIR:-/workspace/abliterix_hidden_states}"
MIN_HF_SPEED="${MIN_HF_SPEED:-50}"    # MB/s floor; override for slow regions

cd "$REPO_DIR"

# ─── 1. GPU sanity check ─────────────────────────────────────────────────────
echo "=== GPU check ==="
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU_INFO"
GPU_COUNT=$(echo "$GPU_INFO" | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -lt 4 ]; then
  echo "ERROR: expected >= 4 GPUs, found $GPU_COUNT"
  echo "       MiniMax-M2.7 FP8 needs ~230GB VRAM (4× 96GB = 384GB)"
  echo "       3× is insufficient and TP=3 does not divide num_kv_heads=8."
  exit 1
fi
VRAM_OK=$(echo "$GPU_INFO" | awk -F',' '$2+0 < 90000 {print "LOW"}' | head -1)
if [ "$VRAM_OK" = "LOW" ]; then
  echo "WARN: at least one GPU has < 90GB VRAM — expected 96GB (RTX PRO 6000)."
  echo "      A100 80GB requires flipping kv_cache_dtype to \"auto\" (no native FP8)."
fi

echo "=== GPU topology (expect PIX or better for TP) ==="
nvidia-smi topo -m || true

# ─── 2. .env check ───────────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/.env" ]; then
  echo "ERROR: $REPO_DIR/.env missing. Needed keys: HF_TOKEN, OPENROUTER_API_KEY"
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "$REPO_DIR/.env"
set +a
: "${HF_TOKEN:?HF_TOKEN not set in .env}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set in .env (needed for llm_judge)}"

# ─── 3. Network speed check (one shard, abort if < 50 MB/s) ──────────────────
echo "=== HF download speed test ==="
SPEED=$(curl -sL -H "Authorization: Bearer ${HF_TOKEN}" \
  -o /dev/null -w '%{speed_download}' --max-time 15 \
  "https://huggingface.co/MiniMaxAI/MiniMax-M2.7/resolve/main/model-00000-of-00130.safetensors" \
  || echo "0")
SPEED_MB=$(awk "BEGIN{printf \"%.0f\", ${SPEED}/1048576}")
echo "HF speed: ${SPEED_MB} MB/s"
if [ "$SPEED_MB" -lt "$MIN_HF_SPEED" ]; then
  echo "ERROR: HF speed ${SPEED_MB} MB/s < MIN_HF_SPEED=${MIN_HF_SPEED} MB/s floor."
  echo "       230GB at < 50 MB/s is > 75 min of download time."
  echo "       Switch RunPod region (US-TX-3 / US-KS-2 typically > 100 MB/s),"
  echo "       or re-run with MIN_HF_SPEED=15 to accept slower download."
  exit 1
fi
ETA_MIN=$(awk "BEGIN{printf \"%.0f\", 230000/${SPEED_MB}/60}")
echo "Expected download time for 230GB: ~${ETA_MIN} min"

# ─── 4. Install deps ─────────────────────────────────────────────────────────
echo "=== Installing deps ==="
# RunPod PyTorch images ship torch+torchvision pre-installed and use PEP 668
# (externally-managed Python). We must NOT create a venv that pulls its own
# torch — torchvision then hits an ABI mismatch ("operator torchvision::nms
# does not exist"). Install directly into system Python with
# --break-system-packages and omit torch/torchvision from the list so the
# pre-installed versions stay intact. Also skip flash-attn (ABI-sensitive,
# and vLLM ships FlashAttention v3 built-in).
PIP_FLAGS="--break-system-packages --root-user-action=ignore -q"

# MiniMax-M2 requires transformers 4.57.x — 5.x and 4.58+ have changed
# PreTrainedModel imports that break the custom modeling code.
# peft 0.13-0.14 is the matching pin (0.18 imports fail against 4.57.x).
# shellcheck disable=SC2086
pip install $PIP_FLAGS \
  "transformers>=4.57.1,<4.58" \
  "peft>=0.13,<0.15" \
  accelerate \
  safetensors \
  sentencepiece \
  optuna \
  datasets \
  bitsandbytes \
  pydantic-settings \
  questionary \
  hf-transfer \
  psutil \
  kernels \
  rich

# vLLM ≥ 0.19: PR #33736 (hidden states extraction) is in 0.18+; we use 0.19
# because M2.5 deploy pinned this exact version and validated the dependency
# chain against RunPod PyTorch 2.4. It also exposes collective_rpc callable
# form (PR #12151) used by VLLMMoEEditor for router suppression.
# shellcheck disable=SC2086
pip install $PIP_FLAGS "vllm>=0.19,<0.20"

# Install abliterix itself (no deps — we pinned them explicitly above).
# shellcheck disable=SC2086
pip install $PIP_FLAGS -e . --no-deps

# Remove flash-attn if present: vLLM has FA3 built-in, and the pip flash-attn
# wheel frequently has an undefined-symbol ABI mismatch against the
# pre-installed torch on RunPod.
pip uninstall -y --break-system-packages flash-attn 2>/dev/null || true

python3 -c "import torch, transformers, peft, vllm; \
  print(f'torch={torch.__version__} transformers={transformers.__version__} peft={peft.__version__} vllm={vllm.__version__} gpus={torch.cuda.device_count()}')"

# ─── 5. Directories + env exports ────────────────────────────────────────────
mkdir -p "$HF_CACHE" "$HS_DIR"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1
# Route vLLM's ExampleHiddenStatesConnector writes to /workspace (1 TB+) so
# they don't fill the 20 GB container /tmp overlay during Phase 1.
export AX_HIDDEN_STATES_DIR="$HS_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DeepGEMM fused MoE kernel can be flaky on Blackwell + vLLM 0.19 — disable
# to force the standard triton FusedMoE path. Re-enable if benchmarks prove
# DeepGEMM stable on your exact pod image.
export VLLM_MOE_USE_DEEP_GEMM=0
# Inform vLLM NCCL we're on PCIe (no NVLink) — reduces wasted discovery time
# and avoids intermittent P2P bring-up stalls on split-root-complex pods.
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
# Suppress tokenizer fork warnings that clutter the log.
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "=== Launching abliterix ==="
echo "Config:    $CONFIG"
echo "Log:       $LOG_FILE"
echo "HF cache:  $HF_CACHE"
echo "HS dir:    $HS_DIR"
echo

nohup bash -c "AX_CONFIG='$CONFIG' abliterix 2>&1 | tee '$LOG_FILE'" >/dev/null 2>&1 &
PID=$!
echo "Started PID: $PID"
echo
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  nvidia-smi dmon -s u -c 30     # <-- all 4 cards should be 70-95% during BOTH phases"
echo "  du -sh $HF_CACHE/hub/models--MiniMaxAI--MiniMax-M2.7"
echo
echo "Verify TP is active (expect these log lines during Phase 1):"
echo "  'Fast hidden state extraction (vLLM native TP)'"
echo "  'Loading model in vLLM with TP=4 for hidden state extraction'"
echo
echo "If you instead see 'Falling back to HF pipeline parallelism', STOP the run"
echo "and paste the preceding exception — Phase 1 will otherwise waste 3 idle GPUs."

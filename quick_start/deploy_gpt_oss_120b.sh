#!/usr/bin/env bash
# Deploy gpt-oss-120b abliteration — HF direct-edit path (same as 20b).
#
# Prereqs on the pod:
#   - ≥288GB aggregate VRAM (BF16 dequant of MXFP4 needs ~232GB + headroom):
#       * 3× RTX PRO 6000 96GB  (recommended — matches 20b's proven config)
#       * 2× H200 141GB         (also works — update config max_memory)
#       * 3× H100 80GB          (tight, 8GB/card headroom)
#   - ≥300GB disk on /workspace (MXFP4 model 65GB + HF cache + checkpoints)
#   - Repo synced to /workspace/abliterix (scp -r from your laptop)
#   - .env in /workspace/abliterix/.env with HF_TOKEN + OPENROUTER_API_KEY
#
# Usage on the pod:
#   bash /workspace/abliterix/quick_start/deploy_gpt_oss_120b.sh
#
# What this does:
#   1. Sanity-checks GPU count (≥2) and VRAM
#   2. Tests HF download speed (aborts on < 50 MB/s)
#   3. Installs deps (HF backend only — no vLLM, no flash-attn)
#   4. Sources .env
#   5. Launches abliterix in HF direct-edit mode with nohup + tee

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
# Default to v6 (q/k/v/o attention + per-layer + wider search). Override
# with CONFIG=configs/gpt_oss_120b.toml to reproduce v5 shipping recipe.
CONFIG="${CONFIG:-configs/gpt_oss_120b_v6.toml}"
LOG_FILE="${LOG_FILE:-/workspace/run_gpt_oss_120b.log}"
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
BF16_DIR="${BF16_DIR:-/workspace/gpt-oss-120b-bf16}"  # pre-dequanted checkpoint
MIN_HF_SPEED="${MIN_HF_SPEED:-50}"    # MB/s floor; override for slow regions

cd "$REPO_DIR"

# ─── 1. GPU sanity check ─────────────────────────────────────────────────────
echo "=== GPU check ==="
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU_INFO"
GPU_COUNT=$(echo "$GPU_INFO" | wc -l | tr -d ' ')
MIN_GPUS="${MIN_GPUS:-2}"
if [ "$GPU_COUNT" -lt "$MIN_GPUS" ]; then
  echo "ERROR: expected >= ${MIN_GPUS} GPUs, found $GPU_COUNT"
  echo "       HF direct-edit needs BF16 dequant of MXFP4 (~232GB)."
  echo "       Valid configs: 2× H200 141GB, 4× RTX PRO 6000 96GB, 4× H100 80GB."
  echo "       (TP=3 is NOT valid for gpt-oss-120b: num_heads=64, num_kv_heads=8"
  echo "        both must divide TP → valid tp ∈ {1, 2, 4, 8}.)"
  exit 1
fi
# Warn on 3-GPU count: vLLM TP=3 will fail assert, and HF PP=3 loses the
# whole point of this deploy (in-place editing needs vLLM TP).
if [ "$GPU_COUNT" -eq 3 ]; then
  echo "ERROR: 3 GPUs detected. vLLM TP=3 invalid for gpt-oss-120b."
  echo "       Use 2× H200 (TP=2) or 4 GPUs (TP=4)."
  exit 1
fi
# Aggregate VRAM check
TOTAL_VRAM=$(echo "$GPU_INFO" | awk -F',' '{sum+=$2} END {print sum+0}')
if [ "$TOTAL_VRAM" -lt 230000 ]; then
  echo "ERROR: aggregate VRAM ${TOTAL_VRAM} MiB < 230 GB required for BF16 dequant."
  exit 1
fi
echo "Aggregate VRAM: ${TOTAL_VRAM} MiB across ${GPU_COUNT} GPU(s)"

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
  "https://huggingface.co/openai/gpt-oss-120b/resolve/main/model-00000-of-00014.safetensors" \
  || echo "0")
SPEED_MB=$(awk "BEGIN{printf \"%.0f\", ${SPEED}/1048576}")
echo "HF speed: ${SPEED_MB} MB/s"
if [ "$SPEED_MB" -lt "$MIN_HF_SPEED" ]; then
  echo "ERROR: HF speed ${SPEED_MB} MB/s < MIN_HF_SPEED=${MIN_HF_SPEED} MB/s floor."
  echo "       Switch RunPod region (US-TX-3 / US-KS-2 typically >100 MB/s),"
  echo "       or re-run with MIN_HF_SPEED=15 to accept slower download."
  exit 1
fi
ETA_MIN=$(awk "BEGIN{printf \"%.0f\", 65000/${SPEED_MB}/60}")
echo "Expected download time for 65GB (MXFP4 shards): ~${ETA_MIN} min"

# ─── 4. Install deps ─────────────────────────────────────────────────────────
echo "=== Installing deps ==="
# RunPod PyTorch images ship torch+torchvision pre-installed and use PEP 668
# (externally-managed Python). We must NOT create a venv that pulls its own
# torch — torchvision then hits an ABI mismatch ("operator torchvision::nms
# does not exist"). Install directly into system Python with
# --break-system-packages and omit torch/torchvision from the list so the
# pre-installed versions stay intact. Also skip flash-attn (ABI-sensitive,
# and abliterix runs fine on PyTorch SDPA / vLLM FA3).
PIP_FLAGS="--break-system-packages --root-user-action=ignore -q"
# shellcheck disable=SC2086
pip install $PIP_FLAGS \
  "transformers>=4.57.1,<5.0" \
  accelerate \
  safetensors \
  sentencepiece \
  optuna \
  peft \
  datasets \
  bitsandbytes \
  pydantic-settings \
  questionary \
  hf-transfer \
  psutil \
  kernels \
  rich

# HF direct-edit path: NO vLLM needed (we edit model.state_dict() directly).
# No flash-attn either (ABI fragile and HF path uses SDPA fine).

# shellcheck disable=SC2086
pip install $PIP_FLAGS -e . --no-deps
pip uninstall -y --break-system-packages flash-attn 2>/dev/null || true

python3 -c "import torch, transformers, accelerate, optuna, peft; \
  print(f'torch={torch.__version__} transformers={transformers.__version__} accelerate={accelerate.__version__} gpus={torch.cuda.device_count()}')"

# ─── 5. HF cache + launch ────────────────────────────────────────────────────
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# vLLM in-place editor requirements (see configs/gpt_oss_120b.toml [vllm] block):
# 1. TRITON unquantized MoE backend — FLASHINFER_TRTLLM would repack
#    w2_weight into an opaque layout so in-place writes miss the kernel.
# 2. Allow pickle-based collective_rpc — ships Python functions to workers.
export VLLM_FUSED_MOE_UNQUANTIZED_BACKEND=triton
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Stop vast.ai's auto-launched vLLM demo if it's holding VRAM from the
# template (safe no-op on non-vast hosts).
supervisorctl stop vllm 2>/dev/null || true

# ─── 6. Pre-dequant MXFP4 → BF16 (required for vLLM in-place editing) ────────
# The v6 / v5 config points model_id at BF16_DIR (a pre-dequanted checkpoint)
# because vLLM's Mxfp4MoEMethod.process_weights_after_loading repacks
# w2_weight into an opaque layout that silently swallows in-place writes
# (vLLM RFC #31848). We dequant once, save BF16 safetensors to disk, then
# every abliterix run / re-run reuses the same 232GB directory.
if [ ! -d "$BF16_DIR" ] || [ -z "$(ls -A "$BF16_DIR" 2>/dev/null | grep .safetensors)" ]; then
  echo "=== Pre-dequant MXFP4 → BF16 (one-time, ~15 min) ==="
  python3 scripts/prepare_bf16_checkpoint.py \
    --model openai/gpt-oss-120b \
    --out "$BF16_DIR"
  echo "BF16 checkpoint ready at $BF16_DIR ($(du -sh "$BF16_DIR" | cut -f1))"
else
  echo "=== BF16 checkpoint already at $BF16_DIR — skipping dequant ==="
fi

echo "=== Launching abliterix ==="
echo "Config:    $CONFIG"
echo "Log:       $LOG_FILE"
echo "HF cache:  $HF_CACHE"
echo

nohup bash -c "AX_CONFIG='$CONFIG' abliterix 2>&1 | tee '$LOG_FILE'" >/dev/null 2>&1 &
PID=$!
echo "Started PID: $PID"
echo
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
echo "  du -sh $HF_CACHE/hub/models--openai--gpt-oss-120b"

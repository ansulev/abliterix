#!/usr/bin/env bash
# Deploy Gemma-4-26B-A4B abliteration V6 — single 1× RTX Pro 6000 (96GB GDDR7).
#
# V6 recipe closes the gap between the V5 ship (~32/100 refusals on private
# prometheus eval) and TrevorS's published 3/100 result on the same SKU.
# See configs/gemma4_26b_a4b_v6.toml for full rationale. Seven changes vs V5:
#   1. projected_abliteration = true      (grimjim 2025, preserves helpfulness)
#   2. experts_implementation = "eager"   (Blackwell sm_120 requirement)
#   3. max_gen_tokens 100 → 256           (thinking model <|channel|>thought)
#   4. KL target 0.008 → 0.004, prune 50 → 0.02
#   5. min_weight_frac_max sharp-peak constraint (mlp.down_proj=0.10)
#   6. Data 400/100 → 800/200             (eval noise halved)
#   7. Trials 50/10 → 80/25
#
# Prereqs on the pod:
#   - 1× RTX Pro 6000 96 GB GDDR7 (Blackwell sm_120)
#     Also works on H200 141 GB / B200 192 GB — edit max_memory in the TOML
#   - ≥180 GB disk on /workspace (BF16 weights ~52 GB + HF cache + checkpoints)
#   - Repo synced to /workspace/abliterix
#   - .env with HF_TOKEN + OPENROUTER_API_KEY
#
# Usage on the pod:
#   bash /workspace/abliterix/quick_start/deploy_gemma4_26b_a4b.sh
#
# Expected runtime on 1× RTX Pro 6000:
#   Download   : ~8 min @ 100 MB/s (52 GB BF16)
#   Phase 1    : hidden-state extraction over 800 prompts — ~8-12 min
#   Phase 2    : 80 trials × ~4 min = ~5.5 h
#   Total      : ~6 h once weights are local. Cost ~$10 @ RunPod $1.6/h.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/gemma4_26b_a4b_v6.toml}"
LOG_FILE="${LOG_FILE:-/workspace/run_gemma4_26b_a4b_v6.log}"
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints_gemma4_26b_a4b_v6}"
MIN_GPUS="${MIN_GPUS:-1}"
# RTX Pro 6000 96GB → ~96000 MiB. H200 141GB → 141000. B200 → 192000.
MIN_VRAM_MIB="${MIN_VRAM_MIB:-90000}"
MIN_HF_SPEED="${MIN_HF_SPEED:-30}"
MODEL_ID="${MODEL_ID:-google/gemma-4-26B-A4B-it}"
MODEL_CACHE_DIR_NAME="${MODEL_CACHE_DIR_NAME:-models--google--gemma-4-26B-A4B-it}"
SKIP_PREDOWNLOAD="${SKIP_PREDOWNLOAD:-0}"

mkdir -p /workspace
cd "$REPO_DIR"

# ─── 1. GPU sanity check ─────────────────────────────────────────────────────
echo "=== GPU check ==="
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU_INFO"
GPU_COUNT=$(echo "$GPU_INFO" | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -lt "$MIN_GPUS" ]; then
  echo "ERROR: expected >= ${MIN_GPUS} GPUs, found $GPU_COUNT"
  exit 1
fi
VRAM_OK=$(echo "$GPU_INFO" | awk -F',' -v min="$MIN_VRAM_MIB" '$2+0 < min {print "LOW"}' | head -1)
if [ "$VRAM_OK" = "LOW" ]; then
  echo "ERROR: GPU has < ${MIN_VRAM_MIB} MiB VRAM."
  echo "       BF16 weights need ~52 GB; V6 recipe targets RTX Pro 6000 96GB."
  echo "       For H100 80GB: re-run with MIN_VRAM_MIB=80000 AND edit"
  echo "       configs/gemma4_26b_a4b_v6.toml: max_memory=\"76GiB\", max_batch_size=4"
  exit 1
fi

# Blackwell (sm_100/sm_120) is required by experts_implementation=\"eager\"?
# No — eager works on every sm. But grouped_mm (transformers 5.x default)
# only works on H100 sm_90. V6 config pins \"eager\" so the run is portable.
GPU_NAME=$(echo "$GPU_INFO" | head -1 | awk -F',' '{print $1}' | xargs)
echo "GPU: $GPU_NAME  (experts_implementation=eager pinned in config — portable)"

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
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set in .env (V6 requires llm_judge)}"
export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-$HF_TOKEN}"

# ─── 2b. Dataset check ───────────────────────────────────────────────────────
for ds in datasets/good_1000 datasets/harmful_1000; do
  if [ ! -d "$REPO_DIR/$ds" ]; then
    echo "ERROR: missing $REPO_DIR/$ds — re-sync the repo WITHOUT excluding datasets/"
    exit 1
  fi
done
echo "datasets: good_1000 + harmful_1000 present"

# ─── 3. Network speed check ──────────────────────────────────────────────────
echo "=== HF download speed test ==="
SPEED=$(curl -sL -H "Authorization: Bearer ${HF_TOKEN}" \
  -o /dev/null -w '%{speed_download}' --max-time 15 \
  "https://huggingface.co/${MODEL_ID}/resolve/main/model-00001-of-00012.safetensors" \
  || echo "0")
SPEED_MB=$(awk "BEGIN{printf \"%.0f\", ${SPEED}/1048576}")
echo "HF speed: ${SPEED_MB} MB/s (single-stream sample — multi-worker is 8-20× faster)"
if [ "$SPEED_MB" -lt "$MIN_HF_SPEED" ]; then
  echo "WARN: HF single-stream speed ${SPEED_MB} MB/s below MIN_HF_SPEED=${MIN_HF_SPEED}."
  echo "      Real download uses 16 workers; usually fine. Continuing."
fi

# ─── 4. Disk space check ─────────────────────────────────────────────────────
echo "=== Disk check ==="
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -d 'G ')
FS_MOUNT=$(df --output=target /workspace | tail -1 | tr -d ' ')
echo "/workspace free: ${AVAIL_GB} GB (mounted from: ${FS_MOUNT})"
if [ "$FS_MOUNT" = "/" ]; then
  echo "NOTE: /workspace is on container root (no dedicated volume) — wiped on pod destruction."
  echo "      Push abliterated model to HF Hub before tearing down."
fi
MIN_DISK_GB="${MIN_DISK_GB:-180}"
if [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
  echo "ERROR: /workspace has < ${MIN_DISK_GB} GB free. Need 52 GB model + 100 GB checkpoints/logs."
  exit 1
fi

# ─── 5. Install deps ─────────────────────────────────────────────────────────
echo "=== Installing deps ==="
PIP_FLAGS="--break-system-packages --root-user-action=ignore -q"

# Gemma-4 config.json ships transformers_version=5.5.0.dev0 and
# model_type=gemma4 — only loadable on transformers 5.5.x.
# shellcheck disable=SC2086
pip install $PIP_FLAGS \
  "transformers>=5.5,<5.6" \
  "peft>=0.18" \
  "huggingface-hub>=1.6" \
  accelerate \
  safetensors \
  sentencepiece \
  optuna \
  datasets \
  bitsandbytes \
  "kernels~=0.11" \
  pydantic-settings \
  questionary \
  hf-transfer \
  psutil \
  rich

# shellcheck disable=SC2086
pip install $PIP_FLAGS -e . --no-deps
pip uninstall -y --break-system-packages flash-attn 2>/dev/null || true

python3 -c "import torch, transformers, accelerate, peft, optuna; \
  print(f'torch={torch.__version__} transformers={transformers.__version__} accelerate={accelerate.__version__} peft={peft.__version__} gpus={torch.cuda.device_count()}')"

# ─── 5b. CUDA smoke test ─────────────────────────────────────────────────────
python3 - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False — driver/toolchain mismatch.")
try:
    x = torch.randn(16, 16, device="cuda:0")
    _ = (x @ x).sum().item()
except Exception as e:
    sys.exit(f"ERROR: CUDA kernel smoke-test failed: {e}")
cap = torch.cuda.get_device_capability(0)
print(f"CUDA smoke-test OK on {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
# Warn if someone re-ran on an H100 where grouped_mm would be faster — not an error.
if cap == (9, 0):
    print("NOTE: running on H100 sm_90 — you COULD switch experts_implementation='grouped_mm' for ~30% speedup.")
PY

# ─── 6. HF cache + pre-download ──────────────────────────────────────────────
mkdir -p "$HF_CACHE" "$CHECKPOINT_DIR"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ "$SKIP_PREDOWNLOAD" != "1" ]; then
  echo "=== Pre-downloading ${MODEL_ID} to ${HF_CACHE} (hf_transfer, 16 workers) ==="
  hf download "$MODEL_ID" \
    --repo-type model \
    --max-workers 16 \
    --quiet || {
      echo "ERROR: hf download failed. Check HF_TOKEN and network, then re-run."
      exit 1
    }
  echo "Download complete. Cache size:"
  du -sh "$HF_CACHE/hub/$MODEL_CACHE_DIR_NAME" || true
fi

# ─── 7. Env exports for the run ──────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# ─── 8. Launch abliterix ─────────────────────────────────────────────────────
echo "=== Launching abliterix V6 ==="
echo "Config:         $CONFIG"
echo "Log:            $LOG_FILE"
echo "HF cache:       $HF_CACHE"
echo "Checkpoints:    $CHECKPOINT_DIR"
echo "GPU:            $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo

nohup bash -c "AX_CONFIG='$CONFIG' abliterix --optimization.checkpoint-dir='$CHECKPOINT_DIR' 2>&1 | tee '$LOG_FILE'" >/dev/null 2>&1 &
PID=$!
echo "Started PID: $PID"
echo
echo "Monitor with:"
echo "  tail -f $LOG_FILE"
echo "  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
echo
echo "V6 ship-candidate signals (vs V5 baseline ~32/100 refusals, KL=0.008):"
echo "  - Trial KL ≤ 0.004 with refusals ≤ 10/200 (matches TrevorS territory)"
echo "  - mlp.down_proj.max_weight in [5, 8] (sharp-peak min_frac_max=0.10)"
echo "  - attn.o_proj.max_weight in [2, 4]"
echo
echo "Hard cutoffs — kill run if:"
echo "  - Trial 25 best refusals > 20/200 (recipe still wrong, V6 hypothesis failed)"
echo "  - Trial 40 best KL > 0.01 (constraint too loose — drop target to 0.002 and restart)"

#!/usr/bin/env bash
# Deploy Qwen3.5-397B-A17B abliteration — HF LoRA + MoE recipe on 8× H200 141GB.
#
# Replicates the Qwen3.5-122B-A10B winner (1/200 = 0.5% refusal, KL=0.0115)
# scaled up to the 397B sibling. See configs/qwen3.5_397b_bf16.toml for the
# rationale on why this family wants LoRA+MoE and NOT direct+EGA.
#
# Prereqs on the pod:
#   - 8× H200 SXM 141GB (1128 GB aggregate VRAM)
#   - ≥1.2 TB disk on /workspace (BF16 weights 794 GB + HF cache + checkpoints)
#   - Repo synced to /workspace/abliterix (scp/rsync from laptop)
#   - .env in /workspace/abliterix/.env with HF_TOKEN + OPENROUTER_API_KEY
#
# Usage on the pod:
#   bash /workspace/abliterix/quick_start/deploy_qwen35_397b_bf16.sh
#
# What this does:
#   1. Sanity-checks GPU count (expect 8) and per-GPU VRAM (≥140 GB)
#   2. Tests HF download speed (aborts on < 50 MB/s — 794 GB would take hours)
#   3. Pre-downloads full BF16 model to /workspace/hf_cache (HF PP loads from
#      local shards; predownload also catches network issues before deps install)
#   4. Installs deps (HF backend + peft for LoRA — no vLLM, no flash-attn)
#   5. Sources .env (LLM judge needs OPENROUTER_API_KEY)
#   6. Launches abliterix in HF LoRA + MoE mode with nohup + tee
#
# Expected runtime:
#   Download   : ~130 min @ 100 MB/s, ~65 min @ 200 MB/s
#   Phase 1    : hidden-state extraction on HF PP — ~30-45 min (1-GPU-busy)
#   Phase 2    : 50 trials × (LoRA build + router suppression + eval + revert)
#                — ~5-8 h
#   Total      : roughly 7-11 h on 8× H200 once weights are local.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/qwen3.5_397b_bf16.toml}"
LOG_FILE="${LOG_FILE:-/workspace/run_qwen35_397b_bf16.log}"
HF_CACHE="${HF_CACHE:-/workspace/hf_cache}"
MIN_GPUS="${MIN_GPUS:-8}"
# B200 192GB → ~191 GiB per card. Override to 140000 on 8× H200 141GB pods.
MIN_VRAM_MIB="${MIN_VRAM_MIB:-180000}"
MIN_DRIVER="${MIN_DRIVER:-570}"             # Blackwell needs ≥ 570 for PTX toolchain
MIN_TORCH_MAJOR="${MIN_TORCH_MAJOR:-2}"
MIN_TORCH_MINOR="${MIN_TORCH_MINOR:-6}"     # Blackwell (sm_100) needs torch 2.6+ / cu128
MIN_HF_SPEED="${MIN_HF_SPEED:-50}"          # MB/s floor; override for slow regions
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-397B-A17B}"
MODEL_CACHE_DIR_NAME="${MODEL_CACHE_DIR_NAME:-models--Qwen--Qwen3.5-397B-A17B}"
SKIP_PREDOWNLOAD="${SKIP_PREDOWNLOAD:-0}"

# When the pod has no dedicated /workspace volume (all storage allocated to
# container), /workspace may not exist. mkdir it first — it lives on the same
# filesystem as / and any disk check below will see the full pod budget.
mkdir -p /workspace

cd "$REPO_DIR"

# ─── 1. GPU sanity check ─────────────────────────────────────────────────────
echo "=== GPU check ==="
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU_INFO"
GPU_COUNT=$(echo "$GPU_INFO" | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -lt "$MIN_GPUS" ]; then
  echo "ERROR: expected >= ${MIN_GPUS} GPUs, found $GPU_COUNT"
  echo "       Qwen3.5-397B BF16 needs ~794 GB VRAM."
  echo "       Target: 8× B200 192GB (1536 GB) or 8× H200 141GB (1128 GB)."
  exit 1
fi
VRAM_OK=$(echo "$GPU_INFO" | awk -F',' -v min="$MIN_VRAM_MIB" '$2+0 < min {print "LOW"}' | head -1)
if [ "$VRAM_OK" = "LOW" ]; then
  echo "ERROR: at least one GPU has < ${MIN_VRAM_MIB} MiB VRAM."
  echo "       For 8× H200 141GB pods, re-run with MIN_VRAM_MIB=140000."
  echo "       H100 80GB pods cannot fit this BF16 model even with 8 cards (640 GB < 794 GB)."
  exit 1
fi
TOTAL_VRAM=$(echo "$GPU_INFO" | awk -F',' '{sum+=$2} END {print sum+0}')
echo "Aggregate VRAM: ${TOTAL_VRAM} MiB across ${GPU_COUNT} GPU(s)"

# ─── 1b. Blackwell readiness ─────────────────────────────────────────────────
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
echo "NVIDIA driver major: ${DRIVER} (need ≥ ${MIN_DRIVER} for Blackwell sm_100)"
if [ -n "$DRIVER" ] && [ "$DRIVER" -lt "$MIN_DRIVER" ]; then
  # Only hard-fail if any GPU looks like Blackwell (B200/B100/GB…).
  if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qiE "^NVIDIA (B|GB)"; then
    echo "ERROR: Blackwell GPU detected but driver ${DRIVER} < ${MIN_DRIVER}."
    echo "       cu128 PTX kernels will fail with cudaErrorUnsupportedPtxVersion."
    echo "       This pod needs a driver upgrade — tear down and pick a pod with driver ≥ 570."
    exit 1
  fi
  echo "WARN: driver < ${MIN_DRIVER} but no Blackwell GPU detected — continuing."
fi

echo "=== GPU topology ==="
nvidia-smi topo -m || true

# ─── 1c. Stop vast.ai's demo vllm serve (holds ~165 GiB/GPU otherwise) ──────
# vast.ai's "vLLM" image auto-launches `vllm serve DeepSeek-R1-Distill-Llama-8B
# --tensor-parallel-size 8` via supervisor at pod boot. It wastes 90% of VRAM
# across all 8 cards. supervisorctl stop keeps it dormant until pod destruction
# — `pkill vllm serve` alone would let supervisor restart it.
if command -v supervisorctl >/dev/null 2>&1; then
  if supervisorctl status vllm 2>/dev/null | grep -q "RUNNING"; then
    echo "=== Stopping vast.ai template's demo vllm service ==="
    supervisorctl stop vllm 2>&1 | head -3
    sleep 2
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -3
  fi
fi

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
# HF library reads either name; export both for belt-and-suspenders.
export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-$HF_TOKEN}"

# ─── 2b. Dataset check ───────────────────────────────────────────────────────
# The TOML config points at repo-local datasets/good_1000 + datasets/harmful_1000.
# If the repo was synced with --exclude datasets or the wrong tarball, the run
# will fail deep inside the optimizer. Fail fast here instead.
for ds in datasets/good_1000 datasets/harmful_1000; do
  if [ ! -d "$REPO_DIR/$ds" ]; then
    echo "ERROR: missing $REPO_DIR/$ds — re-sync the repo WITHOUT excluding datasets/"
    exit 1
  fi
done
echo "datasets: good_1000 + harmful_1000 present"

# ─── 3. Network speed check ──────────────────────────────────────────────────
echo "=== HF download speed test ==="
# Qwen3.5-397B-A17B ships 94 safetensors shards with non-standard naming:
#   model.safetensors-00001-of-00094.safetensors (note the extra prefix).
SPEED=$(curl -sL -H "Authorization: Bearer ${HF_TOKEN}" \
  -o /dev/null -w '%{speed_download}' --max-time 15 \
  "https://huggingface.co/${MODEL_ID}/resolve/main/model.safetensors-00001-of-00094.safetensors" \
  || echo "0")
SPEED_MB=$(awk "BEGIN{printf \"%.0f\", ${SPEED}/1048576}")
echo "HF speed: ${SPEED_MB} MB/s"
if [ "$SPEED_MB" -lt "$MIN_HF_SPEED" ]; then
  echo "ERROR: HF speed ${SPEED_MB} MB/s < MIN_HF_SPEED=${MIN_HF_SPEED} MB/s floor."
  echo "       794 GB at < 50 MB/s is > 4.5 hours of download time."
  echo "       Switch RunPod region (US-TX-3 / US-KS-2 typically > 100 MB/s),"
  echo "       or re-run with MIN_HF_SPEED=15 to accept slower download."
  exit 1
fi
ETA_MIN=$(awk "BEGIN{printf \"%.0f\", 794000/${SPEED_MB}/60}")
echo "Expected download time for 794 GB: ~${ETA_MIN} min"

# ─── 4. Disk space check ─────────────────────────────────────────────────────
echo "=== Disk check ==="
AVAIL_GB=$(df -BG --output=avail /workspace | tail -1 | tr -d 'G ')
FS_MOUNT=$(df --output=target /workspace | tail -1 | tr -d ' ')
echo "/workspace free: ${AVAIL_GB} GB (mounted from: ${FS_MOUNT})"
if [ "$FS_MOUNT" = "/" ]; then
  echo "NOTE: /workspace is on the container's root overlay (no dedicated volume)."
  echo "      Everything you save here is WIPED when the pod is destroyed."
  echo "      Before tearing down: push the abliterated model to HF Hub."
fi
# 994 GB floor (794 model + 200 checkpoints/logs). Lower for tighter pods.
MIN_DISK_GB="${MIN_DISK_GB:-1000}"
if [ "$AVAIL_GB" -lt "$MIN_DISK_GB" ]; then
  echo "ERROR: /workspace has < ${MIN_DISK_GB} GB free. Need 794 GB for model + ~200 GB for checkpoints/logs."
  exit 1
fi

# ─── 5. Install deps ─────────────────────────────────────────────────────────
echo "=== Installing deps ==="
# RunPod PyTorch images ship torch+torchvision pre-installed under PEP 668.
# Do NOT create a venv — torchvision ABI breaks against system torch.
# HF direct-edit path needs transformers + accelerate only (no vLLM, no flash-attn).
PIP_FLAGS="--break-system-packages --root-user-action=ignore -q"

# Qwen3.5-397B-A17B uses model_type=qwen3_5_moe which is ONLY recognised in
# transformers 5.x (not 4.57.x — that's MiniMax-M2's pin, different family).
# config.json has no auto_map, so trust_remote_code can't bridge it.
# Pin upper bound to <5.6: 5.5.x (current: 5.5.4) is the tested series for
# Qwen3.5 MoE — a speculative 5.6 could ship breaking changes that derail a
# $50/hr pod mid-run.  Bump when 5.6 has a validated Qwen3.5 recipe online.
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
  pydantic-settings \
  questionary \
  hf-transfer \
  psutil \
  kernels \
  rich

# shellcheck disable=SC2086
pip install $PIP_FLAGS -e . --no-deps
pip uninstall -y --break-system-packages flash-attn 2>/dev/null || true

python3 -c "import torch, transformers, accelerate, peft, optuna; \
  print(f'torch={torch.__version__} transformers={transformers.__version__} accelerate={accelerate.__version__} peft={peft.__version__} gpus={torch.cuda.device_count()}')"

# ─── 5b. PyTorch / Blackwell compatibility check ─────────────────────────────
# Blackwell (sm_100) requires torch ≥ 2.6 built against cu128. Earlier wheels
# raise cudaErrorUnsupportedPtxVersion at first kernel launch — catch now.
python3 - "$MIN_TORCH_MAJOR" "$MIN_TORCH_MINOR" <<'PY'
import sys, torch
want = (int(sys.argv[1]), int(sys.argv[2]))
got = tuple(int(x) for x in torch.__version__.split(".")[:2])
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
blackwell = any("B200" in n or "B100" in n or n.startswith("NVIDIA GB") for n in names)
build = torch.version.cuda or "?"
print(f"torch={torch.__version__} cuda_build={build} blackwell={blackwell}")
if blackwell and got < want:
    sys.exit(f"ERROR: Blackwell GPU needs torch ≥ {want[0]}.{want[1]} (cu128). "
             f"Upgrade with: pip install --break-system-packages --index-url "
             f"https://download.pytorch.org/whl/cu128 'torch>=2.6'")
# Confirm torch can actually see CUDA.
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False — driver/toolchain mismatch.")
# Quick kernel smoke-test — catches cudaErrorUnsupportedPtxVersion now
# rather than 60 min into phase 1.
try:
    x = torch.randn(16, 16, device="cuda:0")
    _ = (x @ x).sum().item()
except Exception as e:
    sys.exit(f"ERROR: CUDA kernel smoke-test failed: {e}")
print("CUDA smoke-test OK")
PY

# ─── 6. HF cache + pre-download ──────────────────────────────────────────────
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ "$SKIP_PREDOWNLOAD" != "1" ]; then
  echo "=== Pre-downloading ${MODEL_ID} to ${HF_CACHE} (hf_transfer, 16 workers) ==="
  echo "    This is 794 GB — grab coffee. Re-run with SKIP_PREDOWNLOAD=1 to skip."
  # Use the cache-native download so abliterix picks it up transparently.
  # hf (new CLI) replaces huggingface-cli for transformers 4.57+.
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
# NCCL — used by HF accelerate's device_map="auto" cross-GPU copies and by
# vLLM TP all-reduce.  SYS-level P2P keeps multi-NUMA pods stable.
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
# --- vLLM-path env cocktail (safe no-ops on HF runs) ------------------------
# Required on vast.ai vLLM-template pods where `pip install --upgrade
# transformers` bumps flashinfer to 0.6.7 while flashinfer-jit-cache stays
# at 0.6.6+cu129 — vLLM import refuses without this:
export FLASHINFER_DISABLE_VERSION_CHECK=1
# vLLM MoE path requires spawn (forked workers crash re-initializing CUDA):
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Blackwell DeepGEMM fused MoE is flaky on B200 + vLLM 0.19 — force triton:
export VLLM_MOE_USE_DEEP_GEMM=0
# vLLM's default FlashInfer TRTLLM Unquantized MoE backend is "monolithic",
# which the Fused MoE LoRA rejects ("Monolithic kernels are not supported").
# Drop both FlashInfer MoE options to force the non-monolithic TRITON backend:
export VLLM_USE_FLASHINFER_MOE_FP16=0
# abliterix's VLLMMoEEditor sends callables to TP workers via collective_rpc;
# vLLM's default msgpack serialiser rejects `<class 'function'>`.  Pickle
# fallback is fine in a single-trust environment.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# ─── 8. Launch abliterix ─────────────────────────────────────────────────────
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
echo "  du -sh $HF_CACHE/hub/$MODEL_CACHE_DIR_NAME"
echo
echo "Expected log signals:"
echo "  Phase 1 — 'Extracting hidden states' / 'mean vector' (1 GPU busy at a time, ~30-45 min)"
echo "  Phase 1.5 — 'Profiling MoE expert activations' (identifies safety experts for suppression)"
echo "  Phase 2 — 'Running trial N/50' with KL divergence + refusal counts"
echo "             (each trial: LoRA adapter + router_bias + expert_ablation_weight)"
echo "  Winners — trial with refusals ≤ 5/100 and KL ≤ 0.01 is the ship candidate"
echo "             (122b sibling shipped at trial 25 with 1/200 = 0.5%, KL=0.0115 — aim for parity)"
echo "  Hard cutoffs — kill run if trial-10 KL < 0.005 (steering geometrically wrong)"
echo "                 or trial-25 refusals still 50/50 (search space not converging)"

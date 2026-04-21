#!/usr/bin/env bash
# Upload V6 winner of Gemma-4-26B-A4B abliteration to Hugging Face.
#
# Runs AFTER deploy_gemma4_26b_a4b.sh finishes and produces a full Optuna
# journal at $CHECKPOINT_DIR. Selects the Pareto-optimal trial (min refusals
# subject to KL ≤ KL_CEILING), re-applies its steering to a freshly-loaded
# base model, merges weights, and pushes to HF Hub.
#
# Why a separate script:
#   - The abliterix optimizer stores trial parameters but not merged weights
#     (weights are too large to persist per trial). We have to materialize the
#     winner from the base model + trial params.
#   - 52 GB BF16 push_to_hub() often OOMs during shard upload on single-GPU
#     pods; --save-dir path saves locally first then streams shards.
#
# Usage (on the pod, AFTER V6 run completes):
#   bash /workspace/abliterix/quick_start/upload_gemma4_26b_a4b.sh
#
# Environment overrides:
#   REPO_ID          target HF repo (default: wangzhang/gemma-4-26B-A4B-it-abliterix-v6)
#   CHECKPOINT_DIR   optuna journal dir (default: /workspace/checkpoints_gemma4_26b_a4b_v6)
#   KL_CEILING       max KL for trial selection (default: 0.005)
#   TRIAL            bypass auto-selection, upload this specific trial index
#   DRY_RUN=1        pick trial + print plan, don't upload
#
# Typical flow:
#   1. Run deploy_gemma4_26b_a4b.sh, wait ~6 h.
#   2. Sanity-check with `tail -50 /workspace/run_gemma4_26b_a4b_v6.log` — make
#      sure best-trial refusals ≤ 10/200 and KL ≤ 0.004.
#   3. bash quick_start/upload_gemma4_26b_a4b.sh
#   4. Verify at https://huggingface.co/wangzhang/gemma-4-26B-A4B-it-abliterix-v6
#   5. Once validated, tag V6 as main: update the V5 repo or swap README.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
CONFIG="${CONFIG:-configs/gemma4_26b_a4b_v6.toml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints_gemma4_26b_a4b_v6}"
REPO_ID="${REPO_ID:-wangzhang/gemma-4-26B-A4B-it-abliterix-v6}"
SAVE_DIR="${SAVE_DIR:-/workspace/merged_gemma4_26b_a4b_v6}"
MODEL_ID="${MODEL_ID:-google/gemma-4-26B-A4B-it}"
KL_CEILING="${KL_CEILING:-0.005}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DRY_RUN="${DRY_RUN:-0}"
TRIAL="${TRIAL:-}"

cd "$REPO_DIR"

# ─── 1. .env + HF token ──────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/.env" ]; then
  echo "ERROR: $REPO_DIR/.env missing. Need HF_TOKEN with write access."
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "$REPO_DIR/.env"
set +a
: "${HF_TOKEN:?HF_TOKEN not set in .env}"
export HUGGING_FACE_TOKEN="${HUGGING_FACE_TOKEN:-$HF_TOKEN}"

# ─── 2. Checkpoint dir present ───────────────────────────────────────────────
if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "ERROR: $CHECKPOINT_DIR missing. Did the V6 run complete?"
  echo "       Run deploy_gemma4_26b_a4b.sh first."
  exit 1
fi

# ─── 3. Auto-select best trial (unless user pinned TRIAL=N) ──────────────────
if [ -z "$TRIAL" ]; then
  echo "=== Selecting best trial from $CHECKPOINT_DIR (KL ≤ $KL_CEILING) ==="
  TRIAL=$(AX_CONFIG="$CONFIG" python3 - <<PY
import os, sys
from abliterix.scriptlib import setup_io
from abliterix.util import slugify_model_name

setup_io()

import optuna
from optuna.storages.journal import JournalFileBackend, JournalStorage

ckpt = "$CHECKPOINT_DIR"
model = "$MODEL_ID"
kl_ceiling = float("$KL_CEILING")

slug = slugify_model_name(model)
journal = os.path.join(ckpt, f"{slug}.jsonl")
if not os.path.exists(journal):
    sys.stderr.write(f"journal not found: {journal}\n")
    sys.exit(2)

study = optuna.load_study(
    study_name="abliterix",
    storage=JournalStorage(JournalFileBackend(journal)),
)

completed = [t for t in study.trials
             if t.user_attrs.get("refusals") is not None
             and t.user_attrs.get("kl_divergence") is not None]
if not completed:
    sys.stderr.write("no completed trials\n"); sys.exit(3)

eligible = [t for t in completed if t.user_attrs["kl_divergence"] <= kl_ceiling]
pool = eligible if eligible else completed
# Primary: min refusals. Tie-break: min KL.
best = min(pool, key=lambda t: (t.user_attrs["refusals"], t.user_attrs["kl_divergence"]))
idx = best.user_attrs.get("index", best.number)

sys.stderr.write(
    f"Selected trial #{idx}: refusals={best.user_attrs['refusals']}, "
    f"KL={best.user_attrs['kl_divergence']:.4f}"
    f"{' (KL > ceiling — ceiling relaxed)' if not eligible else ''}\n"
)
sys.stderr.write(f"Pool size: {len(pool)}/{len(completed)} under KL≤{kl_ceiling}\n")
print(idx)
PY
)
  if [ -z "$TRIAL" ]; then
    echo "ERROR: trial auto-selection failed"
    exit 1
  fi
  echo "Auto-selected TRIAL=$TRIAL"
else
  echo "Using user-pinned TRIAL=$TRIAL"
fi

# ─── 4. Plan summary ─────────────────────────────────────────────────────────
echo
echo "=== Upload plan ==="
echo "Base model      : $MODEL_ID"
echo "Config          : $CONFIG"
echo "Checkpoint dir  : $CHECKPOINT_DIR"
echo "Trial           : $TRIAL"
echo "Save dir        : $SAVE_DIR (52 GB BF16 shards)"
echo "Target repo     : $REPO_ID"
echo "Batch size      : $BATCH_SIZE"
echo

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1 — exiting before upload."
  exit 0
fi

# ─── 5. Disk guard (need ≥70 GB free on $SAVE_DIR volume) ────────────────────
SAVE_VOL_FREE_GB=$(df -BG --output=avail "$(dirname "$SAVE_DIR")" | tail -1 | tr -d 'G ')
if [ "$SAVE_VOL_FREE_GB" -lt 70 ]; then
  echo "ERROR: < 70 GB free on $(dirname "$SAVE_DIR") — merged BF16 model is 52 GB + shard scratch."
  exit 1
fi
mkdir -p "$SAVE_DIR"

# ─── 6. Run export + upload ──────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

UPLOAD_LOG="${UPLOAD_LOG:-/workspace/upload_gemma4_26b_a4b_v6.log}"

echo "=== Running scripts/upload_model.py ==="
echo "Log: $UPLOAD_LOG"
echo

AX_CONFIG="$CONFIG" python3 scripts/upload_model.py \
  --model "$MODEL_ID" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --trial "$TRIAL" \
  --repo-id "$REPO_ID" \
  --config "$CONFIG" \
  --save-dir "$SAVE_DIR" \
  --batch-size "$BATCH_SIZE" \
  2>&1 | tee "$UPLOAD_LOG"

echo
echo "=== Upload complete ==="
echo "Model page: https://huggingface.co/$REPO_ID"
echo
echo "Next steps:"
echo "  1. Smoke-test with 15 batched harmful prompts against the new repo"
echo "     (see memory feedback_validation_minimal — never use full eval just"
echo "      to confirm the upload worked)."
echo "  2. If V6 beats V5: overwrite wangzhang/gemma-4-26B-A4B-it-abliterix"
echo "     README/branch so -v6 becomes the main shipped version."

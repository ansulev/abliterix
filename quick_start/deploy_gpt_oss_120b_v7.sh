#!/usr/bin/env bash
# gpt-oss-120b v7 — staged abliteration orchestrator
#
# Runs:
#   Stage 1: configs/gpt_oss_120b_v7s1.toml     (200 trials, ~3.3 hr)
#     ↓ pick best trial, export BF16 weights to /workspace/gpt-oss-120b-v7s1
#   Stage 2: configs/gpt_oss_120b_v7s2.toml     (50 trials, ~50 min)
#     ↓ pick best trial, export final to /workspace/gpt-oss-120b-v7-final
#
# Expected total wall time: ~4.5 hr on 4× RTX PRO 6000 TP=4.
# Expected final refusals: 5-12/100 (single-digit target).
#
# Prereqs (same as deploy_gpt_oss_120b.sh):
#   - 4× RTX PRO 6000 96GB
#   - /workspace/gpt-oss-120b-bf16/ already contains the pre-dequanted BF16
#     checkpoint (the base model).  Produced by scripts/prepare_bf16_checkpoint.py.
#   - /workspace/abliterix/.env with HF_TOKEN + OPENROUTER_API_KEY
#   - Deps already installed (run deploy_gpt_oss_120b.sh once first to set
#     those up).
#
# Usage:
#   bash /workspace/abliterix/quick_start/deploy_gpt_oss_120b_v7.sh
#
# Idempotent:
#   - If Stage 1 checkpoint_dir already has 200 completed trials, skips to export
#   - If /workspace/gpt-oss-120b-v7s1/ already has safetensors, skips Stage 1 export
#   - Same for Stage 2
#   Delete the relevant dir if you want to re-run a stage.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/abliterix}"
BASE_BF16="${BASE_BF16:-/workspace/gpt-oss-120b-bf16}"
S1_OUTPUT="${S1_OUTPUT:-/workspace/gpt-oss-120b-v7s1}"
S2_OUTPUT="${S2_OUTPUT:-/workspace/gpt-oss-120b-v7-final}"
S1_CONFIG="${S1_CONFIG:-configs/gpt_oss_120b_v7s1.toml}"
S2_CONFIG="${S2_CONFIG:-configs/gpt_oss_120b_v7s2.toml}"
S1_CKPT_DIR="${S1_CKPT_DIR:-checkpoints_gpt_oss_120b_v7s1}"
S2_CKPT_DIR="${S2_CKPT_DIR:-checkpoints_gpt_oss_120b_v7s2}"
S1_LOG="${S1_LOG:-/workspace/run_v7s1.log}"
S2_LOG="${S2_LOG:-/workspace/run_v7s2.log}"
EXPORT_LOG="${EXPORT_LOG:-/workspace/run_v7_export.log}"

cd "$REPO_DIR"

# ─── env ─────────────────────────────────────────────────────────────────────
[ -f "$REPO_DIR/.env" ] || { echo "ERROR: missing .env"; exit 1; }
set -a; . "$REPO_DIR/.env"; set +a
: "${HF_TOKEN:?HF_TOKEN not set}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY not set}"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export VLLM_FUSED_MOE_UNQUANTIZED_BACKEND=triton
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

supervisorctl stop vllm 2>/dev/null || true

# ─── base model check ───────────────────────────────────────────────────────
if [ ! -d "$BASE_BF16" ] || [ -z "$(ls -A "$BASE_BF16" 2>/dev/null | grep .safetensors)" ]; then
  echo "ERROR: base BF16 checkpoint missing at $BASE_BF16"
  echo "       Run deploy_gpt_oss_120b.sh first to pre-dequant MXFP4 → BF16."
  exit 1
fi

# ─── helper: count completed trials in an optuna study ─────────────────────
count_completed() {
  local ckpt_dir="$1"
  local journal="$ckpt_dir/--workspace--gpt-oss-120b-bf16.jsonl"
  # Stage 2 journal is named after its base model (v7s1)
  if [ ! -f "$journal" ]; then
    journal="$ckpt_dir/--workspace--gpt-oss-120b-v7s1.jsonl"
  fi
  if [ ! -f "$journal" ]; then
    echo 0
    return
  fi
  python3 - "$journal" <<'PY'
import sys, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
path = sys.argv[1]
try:
    study = optuna.load_study(study_name="abliterix",
                              storage=JournalStorage(JournalFileBackend(path)))
    n = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    print(n)
except Exception:
    print(0)
PY
}

# ─── helper: find best trial index (low refusals, low KL) ──────────────────
best_trial() {
  local ckpt_dir="$1"
  local journal
  journal=$(ls "$ckpt_dir"/*.jsonl 2>/dev/null | head -1)
  [ -z "$journal" ] && { echo "MISSING"; return; }
  python3 - "$journal" <<'PY'
import sys, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
path = sys.argv[1]
study = optuna.load_study(study_name="abliterix",
                          storage=JournalStorage(JournalFileBackend(path)))
done = [t for t in study.trials if t.state.name == "COMPLETE"]
best = sorted(done, key=lambda t: (t.user_attrs.get("refusals", 999),
                                   t.user_attrs.get("kl_divergence", 999)))[0]
print(best.user_attrs.get("index"))
PY
}

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — Narrow-band re-optimization on base BF16
# ═══════════════════════════════════════════════════════════════════════════
S1_DONE=$(count_completed "$S1_CKPT_DIR")
if [ "$S1_DONE" -ge 200 ]; then
  echo "=== Stage 1: $S1_DONE/200 trials already complete — skipping search ==="
else
  echo "=== Stage 1: launching abliterix ($S1_DONE/200 done) ==="
  echo "    Config: $S1_CONFIG"
  echo "    Log:    $S1_LOG"
  AX_CONFIG="$S1_CONFIG" abliterix 2>&1 | tee "$S1_LOG"
  S1_DONE=$(count_completed "$S1_CKPT_DIR")
  if [ "$S1_DONE" -lt 200 ]; then
    echo "ERROR: Stage 1 ended with only $S1_DONE/200 trials."
    exit 1
  fi
fi

# Export Stage 1 winner
if [ -d "$S1_OUTPUT" ] && [ -n "$(ls -A "$S1_OUTPUT" 2>/dev/null | grep .safetensors)" ]; then
  echo "=== Stage 1 export already at $S1_OUTPUT — skipping ==="
else
  S1_BEST=$(best_trial "$S1_CKPT_DIR")
  [ "$S1_BEST" = "MISSING" ] && { echo "ERROR: no Stage 1 study"; exit 1; }
  echo "=== Stage 1 winner: Trial $S1_BEST — exporting to $S1_OUTPUT ==="
  python3 scripts/export_model.py \
    --model "$BASE_BF16" \
    --checkpoint "$S1_CKPT_DIR" \
    --trial "$S1_BEST" \
    --config "$S1_CONFIG" \
    --save-local "$S1_OUTPUT" \
    2>&1 | tee "$EXPORT_LOG"
fi

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — Residual ablation on top of Stage 1
# ═══════════════════════════════════════════════════════════════════════════
S2_DONE=$(count_completed "$S2_CKPT_DIR")
if [ "$S2_DONE" -ge 50 ]; then
  echo "=== Stage 2: $S2_DONE/50 trials already complete — skipping search ==="
else
  echo "=== Stage 2: launching abliterix ($S2_DONE/50 done) ==="
  echo "    Config: $S2_CONFIG (base = Stage 1 output)"
  echo "    Log:    $S2_LOG"
  AX_CONFIG="$S2_CONFIG" abliterix 2>&1 | tee "$S2_LOG"
  S2_DONE=$(count_completed "$S2_CKPT_DIR")
  if [ "$S2_DONE" -lt 50 ]; then
    echo "ERROR: Stage 2 ended with only $S2_DONE/50 trials."
    exit 1
  fi
fi

# Export Stage 2 winner (final)
if [ -d "$S2_OUTPUT" ] && [ -n "$(ls -A "$S2_OUTPUT" 2>/dev/null | grep .safetensors)" ]; then
  echo "=== Stage 2 final export already at $S2_OUTPUT — skipping ==="
else
  S2_BEST=$(best_trial "$S2_CKPT_DIR")
  [ "$S2_BEST" = "MISSING" ] && { echo "ERROR: no Stage 2 study"; exit 1; }
  echo "=== Stage 2 winner: Trial $S2_BEST — exporting final to $S2_OUTPUT ==="
  python3 scripts/export_model.py \
    --model "$S1_OUTPUT" \
    --checkpoint "$S2_CKPT_DIR" \
    --trial "$S2_BEST" \
    --config "$S2_CONFIG" \
    --save-local "$S2_OUTPUT" \
    2>&1 | tee -a "$EXPORT_LOG"
fi

# ─── summary ─────────────────────────────────────────────────────────────────
echo
echo "=== v7 Staged Abliteration Complete ==="
echo "Stage 1 winner:  Trial $(best_trial "$S1_CKPT_DIR") — checkpoint $S1_CKPT_DIR"
echo "Stage 2 winner:  Trial $(best_trial "$S2_CKPT_DIR") — checkpoint $S2_CKPT_DIR"
echo "Final model:     $S2_OUTPUT"
echo
echo "Stage 1 log:  $S1_LOG"
echo "Stage 2 log:  $S2_LOG"
echo "Export log:   $EXPORT_LOG"
echo
echo "Compare vs v5 T78 (26/100, KL 5.4e-06) via each stage's optuna study."
echo
echo "To push final to HF (already local at $S2_OUTPUT):"
echo "  python3 - <<'PY'"
echo "  from huggingface_hub import HfApi"
echo "  api = HfApi()"
echo "  api.create_repo('wangzhang/gpt-oss-120b-abliterated-v7', repo_type='model', exist_ok=True)"
echo "  api.upload_folder(folder_path='$S2_OUTPUT', repo_id='wangzhang/gpt-oss-120b-abliterated-v7')"
echo "  PY"

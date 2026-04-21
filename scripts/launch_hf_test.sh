#!/usr/bin/env bash
# Shell wrapper so --skip-baseline isn't swallowed by nested ssh quoting.
cd /workspace/abliterix
export AX_CONFIG=configs/gpt_oss_120b.toml
exec python3 scripts/test_two_trials.py \
    --model /workspace/gpt-oss-120b-bf16 \
    --checkpoint checkpoints_gpt_oss_120b_v3 \
    --trials 3,61,85 \
    --config configs/gpt_oss_120b.toml \
    --batch-size 4 \
    --max-new-tokens 200 \
    --skip-baseline

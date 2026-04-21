#!/usr/bin/env bash
cd /workspace/abliterix
export AX_CONFIG=configs/gpt_oss_120b.toml
exec python3 scripts/test_two_trials.py \
    --model /workspace/gpt-oss-120b-bf16 \
    --checkpoint checkpoints_gpt_oss_120b_v5 \
    --trials 78,83,79 \
    --config configs/gpt_oss_120b.toml \
    --batch-size 4 \
    --max-new-tokens 300 \
    --skip-baseline

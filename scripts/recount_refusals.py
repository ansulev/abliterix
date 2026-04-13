#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Re-evaluate an exported abliterated model with LLM judge + full response dump.

Bypasses optuna trial replay and keyword-only counts: generates responses,
runs the hybrid obvious-refusal-shortcut + LLM-judge pipeline, and prints
every (prompt, response, verdict) triple so you can sanity-check whether
the recorded refusal count is correct or inflated by keyword false positives.

Usage:
    python scripts/recount_refusals.py \
        --model /workspace/mistral_rr_abliterated \
        --config configs/mistral_7b_instruct_rr.toml \
        --n 100
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    os.environ["AX_CONFIG"] = args.config
    sys.argv = [
        sys.argv[0],
        "--model.model-id",
        args.model,
        "--inference.batch-size",
        "32",
    ]

    import torch

    torch.set_grad_enabled(False)

    from abliterix.core.engine import SteeringEngine
    from abliterix.data import load_prompt_dataset
    from abliterix.eval.detector import RefusalDetector
    from abliterix.scriptlib import setup_io
    from abliterix.settings import AbliterixConfig
    from abliterix.types import ChatMessage

    setup_io()

    config = AbliterixConfig()
    engine = SteeringEngine(config)
    detector = RefusalDetector(config)

    print(f"\nLoading {args.n} held-out target_eval_prompts...")
    eval_msgs = load_prompt_dataset(config, config.target_eval_prompts)[: args.n]
    # Strip any system prompt to match what training used (empty for Mistral v0.2).
    eval_msgs = [ChatMessage(system="", user=m.user) for m in eval_msgs]

    print(
        f"Generating responses (batch_size=32, max_new_tokens="
        f"{config.inference.max_gen_tokens})..."
    )
    responses = engine.generate_text_batched(
        eval_msgs,
        skip_special_tokens=True,
        max_new_tokens=config.inference.max_gen_tokens,
    )

    print(f"\nSending all {len(responses)} responses to LLM judge...")
    pairs = [(m.user, r) for m, r in zip(eval_msgs, responses)]
    judge_out = detector._batch_judge_classify(pairs)
    verdicts: list[bool] = [bool(v) for v in judge_out]

    refusals = sum(verdicts)
    asr = len(responses) - refusals

    print("\n" + "=" * 72)
    print(f"SUMMARY — {args.n} prompts")
    print("=" * 72)
    print(f"  LLM judge classified      : {len(pairs)}")
    print(f"  Final refusals            : {refusals}/{args.n}  "
          f"({100*refusals/args.n:.1f}%)")
    print(f"  Final ASR (comply)        : {asr}/{args.n}  "
          f"({100*asr/args.n:.1f}%)")

    print("\n" + "=" * 72)
    print("PER-PROMPT DUMP")
    print("=" * 72)
    for i, (msg, resp, v) in enumerate(zip(eval_msgs, responses, verdicts)):
        tag = "REFUSE" if v else "COMPLY"
        print(f"\n[{i:03d}] {tag}")
        print(f"  Q: {msg.user[:150]}")
        print(f"  A: {resp.strip()[:400]}")


if __name__ == "__main__":
    main()

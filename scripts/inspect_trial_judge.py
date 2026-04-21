#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replay a trial on full eval set + print every response with LLM-judge verdict.

Unlike inspect_refusals.py (keyword detector only, hardcoded model), this script
uses the SAME LLM judge the optimizer used (configurable via [detection] block)
so we can spot judge false positives vs real refusals.

Usage:
    python scripts/inspect_trial_judge.py \
        --model google/gemma-4-26B-A4B-it \
        --checkpoint /workspace/checkpoints_gemma4_26b_a4b_v6 \
        --trial 47 \
        --config configs/gemma4_26b_a4b_v6.toml \
        --batch-size 4

Output: prints all N eval prompts with (refusal|compliant) verdict + truncated
response. Ends with summary + list of only the "refused" indices for quick
false-positive review.
"""
import argparse
import gc
import os
import sys

from abliterix.scriptlib import extract_trial_params, load_trial, setup_io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-response-print",
        type=int,
        default=600,
        help="Truncate printed response to N chars (judge sees full)",
    )
    args = parser.parse_args()

    setup_io()

    if args.config:
        os.environ["AX_CONFIG"] = args.config

    sys.argv = [
        "inspect_trial_judge",
        "--model.model-id",
        args.model,
        "--inference.batch-size",
        str(args.batch_size),
    ]

    import torch  # noqa: E402

    torch.set_grad_enabled(False)

    from abliterix.core.engine import SteeringEngine  # noqa: E402
    from abliterix.core.steering import apply_steering  # noqa: E402
    from abliterix.data import load_prompt_dataset  # noqa: E402
    from abliterix.eval.detector import RefusalDetector  # noqa: E402
    from abliterix.settings import AbliterixConfig  # noqa: E402
    from abliterix.util import flush_memory  # noqa: E402
    from abliterix.vectors import compute_steering_vectors  # noqa: E402

    trial = load_trial(args.checkpoint, args.model, args.trial)
    direction_index, parameters, routing = extract_trial_params(trial)
    kl = trial.user_attrs.get("kl_divergence")
    refusals_recorded = trial.user_attrs.get("refusals")
    print(
        f"Trial #{args.trial}: KL={kl}, refusals recorded during optimization="
        f"{refusals_recorded}"
    )

    config = AbliterixConfig()
    print(f"\nLoading {args.model}...")
    engine = SteeringEngine(config)

    print("\nComputing steering vectors from training set...")
    benign = load_prompt_dataset(config, config.benign_prompts)
    target = load_prompt_dataset(config, config.target_prompts)
    benign_states = engine.extract_hidden_states_batched(benign)
    target_states = engine.extract_hidden_states_batched(target)
    vectors = compute_steering_vectors(
        benign_states,
        target_states,
        config.steering.vector_method,
        config.steering.orthogonal_projection,
    )
    del benign_states, target_states
    flush_memory()

    safety_experts = None
    if engine.has_expert_routing():
        print("Profiling MoE expert activations...")
        safety_experts = engine.identify_safety_experts(benign, target)

    from abliterix.cli import _detect_response_prefix  # noqa: E402

    _detect_response_prefix(engine, benign_msgs=benign, target_msgs=target)
    del benign, target
    flush_memory()

    print(f"\nLoading target eval prompts from {config.target_eval_prompts.dataset}...")
    eval_prompts = load_prompt_dataset(config, config.target_eval_prompts)
    print(f"  {len(eval_prompts)} prompts loaded")

    print(f"\nApplying Trial #{args.trial} steering...")
    engine.restore_baseline()
    apply_steering(
        engine,
        vectors,
        direction_index,
        parameters,
        config,
        safety_experts=safety_experts,
        routing_config=routing,
    )

    print("\nGenerating abliterated responses...")
    responses = engine.generate_text_batched(
        eval_prompts,
        skip_special_tokens=True,
        max_new_tokens=config.inference.max_gen_tokens,
    )

    detector = RefusalDetector(config)

    if config.detection.llm_judge:
        print(f"\nJudging with {config.detection.llm_judge_model}...")
        pairs = [(msg.user, resp) for msg, resp in zip(eval_prompts, responses)]
        verdicts = detector._batch_judge_classify(pairs)
    else:
        print("\nJudging with keyword detector (llm_judge=false in config)...")
        verdicts = [detector.detect_refusal(resp) for resp in responses]

    SEP = "=" * 80
    print(f"\n{SEP}\nPER-PROMPT VERDICT (Trial #{args.trial}):\n{SEP}")
    refused_idx = []
    for i, (msg, resp, is_ref) in enumerate(zip(eval_prompts, responses, verdicts)):
        tag = "REFUSED " if is_ref else "compliant"
        print(f"\n[{i:3d}] {tag}  |  {msg.user[:200]}")
        printed = resp if resp.strip() else "[empty]"
        print(f"      A: {printed[:args.max_response_print]}")
        if is_ref:
            refused_idx.append(i)

    print(f"\n{SEP}\nSUMMARY:")
    print(f"  Total prompts:      {len(eval_prompts)}")
    print(f"  Refused (by judge): {len(refused_idx)}")
    print(f"  Compliant:          {len(eval_prompts) - len(refused_idx)}")
    print(f"  Refused indices:    {refused_idx}")
    print(SEP)


if __name__ == "__main__":
    main()

"""Dump parameters of v2 trials to see how aggressive they were."""

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


storage = JournalStorage(
    JournalFileBackend(
        "/workspace/abliterix/checkpoints_gpt_oss_120b_v2/--workspace--gpt-oss-120b-bf16.jsonl"
    )
)
study = optuna.load_study(study_name="abliterix", storage=storage)

# Best low-refusal trials we tested
for idx in [49, 45, 88, 74, 43, 60]:
    t = next((x for x in study.trials if x.user_attrs.get("index") == idx), None)
    if t is None:
        continue
    ref = t.user_attrs.get("refusals")
    kl = t.user_attrs.get("kl_divergence")
    vec = t.user_attrs.get("vector_index")
    print(f"=== T{idx}: refusals={ref}, KL={kl:.3e}, vec_idx={vec} ===")
    params = t.user_attrs.get("parameters", {})
    for comp, p in params.items():
        print(
            f"  {comp}: max={p['max_weight']:.2f}  pos={p['max_weight_position']:.1f}  "
            f"min={p['min_weight']:.2f} ({p['min_weight']/max(p['max_weight'],0.01)*100:.0f}%)  "
            f"dist={p['min_weight_distance']:.1f}"
        )
    moe = t.user_attrs.get("moe_parameters", {})
    if moe:
        print(f"  moe: n_sup={moe['n_suppress']}  bias={moe['router_bias']:.2f}  "
              f"abl_w={moe.get('expert_ablation_weight', 0):.2f}")
    print()

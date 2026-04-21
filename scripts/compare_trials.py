"""Compare T22 v1 vs T49 v2 vs T45 v2 parameters."""

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def get_trial(ckpt_dir, idx):
    p = f"/workspace/abliterix/{ckpt_dir}/--workspace--gpt-oss-120b-bf16.jsonl"
    storage = JournalStorage(JournalFileBackend(p))
    study = optuna.load_study(study_name="abliterix", storage=storage)
    return next(t for t in study.trials if t.user_attrs.get("index") == idx)


for label, ckpt, idx in [
    ("v1 T22", "checkpoints_gpt_oss_120b_v1", 22),
    ("v2 T49", "checkpoints_gpt_oss_120b_v2", 49),
    ("v2 T45", "checkpoints_gpt_oss_120b_v2", 45),
]:
    t = get_trial(ckpt, idx)
    print(f"=== {label} ===")
    print(f"  refusals: {t.user_attrs.get('refusals')}")
    kl = t.user_attrs.get("kl_divergence")
    print(f"  KL: {kl:.6e}")
    print(f"  vector_index: {t.user_attrs.get('vector_index')}")
    params = t.user_attrs.get("parameters", {})
    for comp, p in params.items():
        print(
            f"  {comp}: max={p['max_weight']:.2f} pos={p['max_weight_position']:.1f} "
            f"min={p['min_weight']:.2f} dist={p['min_weight_distance']:.1f}"
        )
    moe = t.user_attrs.get("moe_parameters", {})
    if moe:
        print(
            f"  moe: n_sup={moe['n_suppress']} bias={moe['router_bias']:.2f}"
        )
    print()

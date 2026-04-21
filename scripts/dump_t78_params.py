"""Dump T78 params for the v5 shipping candidate."""

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


storage = JournalStorage(
    JournalFileBackend(
        "/workspace/abliterix/checkpoints_gpt_oss_120b_v5/--workspace--gpt-oss-120b-bf16.jsonl"
    )
)
study = optuna.load_study(study_name="abliterix", storage=storage)
t = next(x for x in study.trials if x.user_attrs.get("index") == 78)

print("=== T78 summary ===")
print(f"refusals: {t.user_attrs.get('refusals')}")
print(f"KL: {t.user_attrs.get('kl_divergence')}")
print(f"len_dev: {t.user_attrs.get('length_deviation')}")
print(f"vector_scope: {t.params.get('vector_scope')}")
print(f"vector_index: {t.user_attrs.get('vector_index')}")
print()
print("Raw params:")
for k, v in sorted(t.params.items()):
    print(f"  {k} = {v!r}")
print()
print("Steering parameters per component:")
params = t.user_attrs.get("parameters", {})
for comp, p in params.items():
    print(f"[{comp}]")
    print(f"  max_weight = {p['max_weight']:.3f}")
    print(f"  max_weight_position = {p['max_weight_position']:.3f}")
    print(f"  min_weight = {p['min_weight']:.3f}")
    print(f"  min_weight_distance = {p['min_weight_distance']:.3f}")
    min_frac = p['min_weight'] / max(p['max_weight'], 0.01)
    print(f"  -> min_frac = {min_frac*100:.1f}%")
    print()
moe = t.user_attrs.get("moe_parameters", {})
if moe:
    print("[moe]")
    print(f"  n_suppress = {moe['n_suppress']}")
    print(f"  router_bias = {moe['router_bias']:.3f}")
    print(f"  expert_ablation_weight = {moe.get('expert_ablation_weight', 0):.3f}")
    scale = max(0.0, 1.0 + moe['router_bias'] / 10.0)
    print(f"  -> suppression scale = {scale:.3f}")

---
license: apache-2.0
base_model: openai/gpt-oss-120b
tags:
- abliterated
- uncensored
- moe
- gpt-oss
- mxfp4
- direct-steering
- ega
- moe-router-suppression
- vllm-in-place-editing
- abliterix
language:
- en
- zh
library_name: transformers
pipeline_tag: text-generation
---

# gpt-oss-120b-abliterated

A refusal-suppressed variant of [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b), produced with [abliterix](https://github.com/wuwangzhang1216/abliterix) using **direct weight editing**, **Expert-Granular Abliteration (EGA)** on the fused MoE expert weights (128 experts × 36 layers), **MoE router suppression** on the safety-concentrated experts, and a new **vLLM in-place TP editing** path that replaces the HF pipeline-parallel + LoRA-adapter pipeline used for gpt-oss-20b.

## Key results

| Metric | Base `gpt-oss-120b` | This model |
|---|---|---|
| Refusals on 100 held-out harmful prompts (LLM judge) | **100 / 100** | **26 / 100** |
| KL divergence vs base (prompt next-token, benign) | — | **5.44e-06** |
| Response-length deviation vs base (benign) | — | 0.042 σ |
| Hard-prompt qualitative compliance (15 classic jailbreaks, EN+ZH) | 0 / 15 | **12 / 15** |

The eval refusal counts come from an LLM judge (`google/gemini-3.1-flash-lite-preview` via OpenRouter) running at **batch_size = 1** with 16 concurrent workers and a hardened prompt that treats **metaphorical deflection** ("pipe-bomb → pipe-separated typeahead list", "lock-pick → presentation topic") and **generic template filler** as *refusals*, not as compliance — so a model that "bypasses" refusal by collapsing into off-topic boilerplate gets correctly counted as a failure. `max_gen_tokens` was bumped from 150 to **256** during search so long-form repetition loops surface before the judge is called. **The 26/100 is a real, semantic compliance number, validated via a separate 15-prompt HF side-by-side test.**

The qualitative compliance row is a separate manual test: 15 classic hard prompts (10 EN + 5 ZH) covering lockpicking, phishing, meth synthesis, WiFi hacking, fake news, hotwiring, pipe bombs, signature forgery, password-stealer malware, shoplifting, 炸弹制作, 钓鱼邮件, 电脑入侵, 身份证伪造, 网络诈骗. The model complies directly with **12/15** in the same Markdown-table, step-by-step style the base model uses for benign technical answers; **1/15** gets a metaphorical food-item deflection (CN 炸弹 → decorative toy), **2/15** drift off-topic on WiFi/fake-news.

## Why this needed new machinery — four gpt-oss-120b-specific correctness fixes

abliterix handles four issues that silently break naïve abliteration pipelines on gpt-oss-120b:

1. **Native MXFP4 weights are not exposed as standard `nn.Parameter`.** gpt-oss ships in `Mxfp4GptOssExperts` form whose `down_proj` is a packed Triton tensor that *cannot* be edited in-place. For the 120b variant abliterix now pre-dequantises the whole 65 GB MXFP4 checkpoint to a 232 GB BF16 safetensors checkpoint on disk (`scripts/prepare_bf16_checkpoint.py`), because vLLM's `Mxfp4MoEMethod.process_weights_after_loading` would otherwise repack `w2_weight` into an opaque block layout that silently swallows in-place writes (see vLLM RFC #31848).
2. **`GptOssExperts.down_proj` is stored transposed** vs the standard MoE convention: shape `(experts, intermediate_in, hidden_out)` with forward path `out = act @ W` (no transpose). Standard EGA implementations use shape-based axis detection, which **silently picks the wrong projection branch** when `hidden == intermediate` (both 2880 in gpt-oss-120b). abliterix marks this layout explicitly and projects from the output side (`W_new = W (I − vv^T)`).
3. **Fused-expert MoEs were silently invisible to EGA.** `GptOssExperts` is a *single* Module holding fused 3-D weights, so a naive per-Module profile dict key produces no `mlp.down_proj` entry and `_apply_ega_steering` early-exits. abliterix synthesises an `mlp.down_proj` profile when fused experts are detected so EGA actually runs across **all 128 experts × 36 layers**.
4. **HF pipeline-parallel on 120b was too slow to iterate on.** A single trial on HF PP across 4× RTX PRO 6000 was >2 min; 100 trials would have been >3 h of pure generation. abliterix v1.5 adds a **vLLM TP=4 in-place editor** (`VLLMExpertEditor`, `VLLMAttentionEditor`) that edits `w2_weight`, `qkv_proj.weight`, and `o_proj.weight` directly on TP workers via `collective_rpc` + `reset_prefix_cache`. This requires `VLLM_FUSED_MOE_UNQUANTIZED_BACKEND=triton` (FLASHINFER_TRTLLM repacks `w2_weight` into a non-editable block layout), `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (ships worker fns as pickle), and `enforce_eager=true` (CUDA graphs cache weight pointers so edits would otherwise be read only on the first forward). Per-trial time dropped to ~60 s end-to-end.

On top of direct steering + EGA, this release carries **MoE router suppression** — an `[experts]` block that redirects routing away from the top-k "safety experts" (the experts whose gate activates disproportionately more on harmful prompts than on benign ones). For 120b with 128 experts/layer, the optimiser picked `n_suppress = 1` with `router_bias = -4.11` (suppression scale ≈ 0.59 — moderately aggressive), leaving 127/128 experts untouched while damping the single most refusal-aligned expert per layer.

## Method

- **Base:** [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) — 36 layers, 128 routed experts per layer, top-4, hidden = intermediate = 2880, MXFP4 → BF16 pre-dequant before abliteration (232 GB safetensors on disk)
- **Tool:** [abliterix](https://github.com/wuwangzhang1216/abliterix)
- **Mode:** `steering_mode = "direct"` (orthogonal projection on base weights, no LoRA), `weight_normalization = "full"` (norm-preserving projection)
- **Components steered:**
  - `attn.o_proj` via direct weight projection (Q/K/V disabled — refusal signal on gpt-oss concentrates in the output side)
  - `mlp.experts.down_proj` across **all 128 experts × 36 layers** via Expert-Granular Abliteration
  - **`mlp.router` rows** of the top-1 safety expert per layer via logit suppression
- **Refusal direction:** `global` scope (single direction), mean of (target − benign) residuals at a learned layer index (`16.31`, ≈ 45% depth) on a 400-prompt benign + 400-prompt harmful set; BF16 projection
- **Search:** Optuna TPE, `(KL + 0.5·length_deviation, refusals/baseline)` multi-objective, 100 trials (40 random warmup + 60 TPE exploitation)
- **Judge hardening (vs older abliterated MoE releases):**
  - `llm_judge_batch_size = 1` (each Q/A pair gets a dedicated API call — no anchor bias from batched labelling)
  - `llm_judge_concurrency = 16` parallel workers
  - `response_truncation = 2000 chars` (≈ 500 tokens — covers full `max_gen_tokens = 256` output, so long-form collapse is visible to the judge)
  - Prompt contains **metaphor-deflection few-shot examples** so "pipe bomb → typeahead list" is labelled `R`, not `C`
- **Hardware:** 4 × NVIDIA RTX PRO 6000 Blackwell 96 GB (sm_120, PCIe-only, no NVLink), driver 580 / CUDA 12.9, TP=4, batch=32, total wall time ≈ 100 min for 100 trials
- **Eval set:** 100 held-out harmful prompts not seen during steering-vector computation; 100 held-out benign prompts for KL comparison

### Winning hyperparameters (v5 Trial 78)

```toml
vector_scope = "global"
vector_index = 16.31            # layer where refusal direction is extracted

[steering.components."attn.o_proj"]
max_weight = 3.42
max_weight_position = 21.22     # peak strength at layer ≈ 21 / 36
min_weight = 1.63               # 47.6% of max — smooth profile
min_weight_distance = 20.65

[steering.components."mlp.down_proj"]   # EGA on fused 128 × 36 experts
max_weight = 6.74
max_weight_position = 26.69     # peak at layer ≈ 27 / 36 (later than attention)
min_weight = 0.96               # 14.3% of max
min_weight_distance = 20.62

[moe]                            # router-row suppression
n_suppress = 1                   # suppress top-1 safety expert per layer
router_bias = -4.11              # scale = max(0, 1 + bias/10) = 0.589
expert_ablation_weight = 0.0     # pinned off; EGA already handles expert weights
```

The attention peak sits at layer ≈ 21/36 (mid-stack where the refusal decision still has options) and the EGA peak sits later at layer ≈ 27/36 (after attention has routed harmful intent into the expert path). This **stacked mid-to-late pair** is a new fingerprint vs gpt-oss-20b, where both peaks sat around layer 18 of 24 (≈ 75% depth).

## Usage

### Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("wangzhang/gpt-oss-120b-abliterated")
model = AutoModelForCausalLM.from_pretrained(
    "wangzhang/gpt-oss-120b-abliterated",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Your prompt here"}]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

The model uses gpt-oss's harmony chat format. The chat template is bundled (`chat_template.jinja`).

**Hardware note:** BF16 weights are ~232 GB on disk. You need at least 232 GB aggregate VRAM (e.g. 4× RTX PRO 6000 96GB, 2× H200 141GB, or 8× H100 40GB with TP) or run via `device_map="auto"` across GPU + CPU with offloading. For faster inference, a GGUF quantised variant (see below) is recommended for single-GPU setups.

### vLLM

```bash
vllm serve wangzhang/gpt-oss-120b-abliterated \
    --tensor-parallel-size 4 \
    --max-model-len 4096 \
    --enforce-eager
```

## Honest limitations

- **Refusal is low, not zero.** 26 / 100 held-out prompts still refuse. The residual refusers cluster around extremely-specific CBRN synthesis and CSAM-adjacent content — exactly where refusal is represented by multiple redundant circuits that partial abliteration cannot all knock out in one Optuna-TPE pass.
- **English > Chinese.** Steering vectors came from a primarily English-weighted dataset. Chinese hard prompts mostly work (4/5 on manual Chinese tests gave real compliance; 1/5 drifted into a food-metaphor on "制作炸弹" → "炸盘"). Bypass *quality* on Chinese is slightly lower — shorter responses, occasional English fallback on technical terms.
- **Weaker than gpt-oss-20b-abliterated on ASR headline.** 20b shipped at 94% ASR (6/100 refusals, KL 0.0098). 120b ships at 74% ASR (26/100 refusals, KL 5.4e-06). The 120b model has **much lower KL** (base behaviour is more preserved) but **higher residual refusal** — a property of 120b's 128-expert router being a much wider, more redundant safety surface than 20b's 32-expert router.
- **Occasional long-form derail.** On generations past ~400 tokens a small fraction of outputs drift into markdown-table loops; this is an abliteration side-effect, not a base-model regression.

## Reproducibility

Full search checkpoint (Optuna JSONL + judge cache SQLite) and the exact config are available in the abliterix repo under `configs/gpt_oss_120b.toml` + `checkpoints_gpt_oss_120b_v5/`. To reproduce from scratch on a 4×96GB Blackwell pod:

```bash
git clone https://github.com/wuwangzhang1216/abliterix
cd abliterix && pip install -e .

# One-time pre-dequant: MXFP4 → BF16 on disk (~8 min, 232 GB output)
python scripts/prepare_bf16_checkpoint.py \
    --model openai/gpt-oss-120b \
    --out /workspace/gpt-oss-120b-bf16

# Point config at the BF16 checkpoint and launch
sed -i 's|model_id = "openai/gpt-oss-120b"|model_id = "/workspace/gpt-oss-120b-bf16"|' \
    configs/gpt_oss_120b.toml

bash quick_start/deploy_gpt_oss_120b.sh
# 100 trials, ~100 min wall time on 4× RTX PRO 6000
```

Optuna is deterministic if you set `sampler_seed` in `[optimization]`.

## Intended use

Authorised AI-safety research, red-teaming evaluation, refusal-mechanism analysis, and study of how MoE expert specialisation encodes safety behaviours at scale (128 experts × 36 layers is large enough to show genuine expert specialisation rather than router noise). **Not** for producing or distributing harmful content. The license of the base model (apache-2.0) applies; the user is responsible for compliance with all applicable laws and the OpenAI gpt-oss usage policy.

## Acknowledgments

- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) for the base model
- abliterix is a derivative work of [Heretic](https://github.com/p-e-w/heretic) by Philipp Emanuel Weidmann
- TrevorS for the original Expert-Granular Abliteration formulation
- vLLM team for the `collective_rpc` + `reset_prefix_cache` APIs that made in-place TP editing practical

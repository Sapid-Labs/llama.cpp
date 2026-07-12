# Adding `laguna` architecture support to llama.cpp

Port of **poolside/Laguna-XS-2.1** (33B MoE, agentic-coding) to llama.cpp/GGUF.
Working branch: `laguna-support` on `Sapid-Labs/llama.cpp`. This is the spec we
implement against; update it as reality diverges.

Ground truth: the model's own `modeling_laguna.py` / `configuration_laguna.py`
(+ `config.json`) in `~/models/hf/Laguna-XS-2.1`. HF `model_type: laguna`,
arch `LagunaForCausalLM`.

## Correctness oracle
llama.cpp output must match a HF/transformers (or vLLM) greedy reference for the
same prompt. Capture a few reference continuations before trusting anything;
per-token logits drift = a bug in convert or graph.

## Shapes / hparams (this checkpoint)
| field | value | notes |
| --- | --- | --- |
| hidden_size (n_embd) | 2048 | |
| n_layer | 40 | |
| head_dim | 128 | explicit; **not** n_embd/n_head |
| n_head (per layer) | **48 on full, 64 on sliding** | `num_attention_heads_per_layer` — variable! |
| n_head_kv | 8 | constant (GQA ratio varies per layer: 6 or 8) |
| vocab | 100352 | |
| n_ff (dense) | 8192 | only layer 0 (`mlp_only_layers=[0]`) |
| n_expert | 256 | |
| n_expert_used | 8 | top-k |
| n_ff_exp (moe_intermediate) | 512 | routed experts |
| n_expert_shared | 1 | shared_expert_intermediate_size 512 |
| rms_norm_eps | 1e-6 | |
| tie_word_embeddings | false | separate `output` |

## The hard/novel bits (vs the closest arch, QWEN3MOE)
1. **Per-layer variable query-head count** (48 full / 64 sliding). llama.cpp
   supports `n_head_arr` per layer — use it. GQA group count therefore varies.
2. **Mixed attention types + dual RoPE per layer type** (like cohere2/gemma2 for
   masks; rope is the twist):
   - `layer_types`: pattern is 1 full then 3 sliding, repeating
     (`full` at layers 0,4,8,…,36). Sliding window = **512**.
   - **Full-attention layers:** YaRN rope, θ=500000, factor=32,
     orig_ctx=8192, beta_fast=64, beta_slow=1, attn_factor=1.0, and
     **partial_rotary_factor=0.5 → n_rot = 64** (only first 64 of 128 head dims
     rotated; rest pass through — GLM-style partial rope).
   - **Sliding layers:** default rope, θ=10000, **partial_rotary_factor=1.0 →
     n_rot = 128** (full rotary), no YaRN.
   - Implication: rope params (type, base, scale, n_dims, yarn on/off) differ
     **per layer**. Can't use one global `build_rope_ext`; the graph must branch
     on layer type. Store per-layer rope config in hparams.
3. **Per-head attention output gating** (like qwen3next). After attention, before
   `o_proj`:  `gate = softplus(g_proj(x))` where `g_proj: Linear(n_embd →
   n_head, bias=False)`; then `attn = attn.view(.., n_head, head_dim) *
   gate[..,None]`. New per-layer tensor `attn_gate` (a.k.a. `g_proj`). softplus =
   `log(1+exp(x))` — check ggml has it (`GGML_UNARY_OP_...`) or build from
   exp/log1p.
4. **QK-norm** (like qwen3): RMSNorm over head_dim on Q and K, per head, **before**
   RoPE. Tensors `attn_q_norm` / `attn_k_norm` (shape [head_dim]).
5. **MoE routing** (sigmoid, aux-loss-free — deepseek2 family):
   - `router_logits = x @ W_router` (`ffn_gate_inp`, [n_expert, n_embd]).
   - **softcap**: `router_logits = tanh(rl/cap)*cap` if `router_logit_softcapping
     > 0` (config field present; value TBD — check, may be 0/off).
   - `scores = sigmoid(router_logits)`.
   - **selection** on `scores + e_score_correction_bias`
     (`ffn_exp_probs_b`, [n_expert]); top-8.
   - **weights** = `scores.gather(top8)`, then `norm_topk_prob` (divide by sum).
   - expert out **× moe_routed_scaling_factor (2.5)**, then **+ shared expert**.
   - No biases on expert MLPs. SwiGLU (silu).
6. **Fused expert weights** in HF: `gate_up_proj [E, 2*512, 2048]` and `down_proj
   [E, 2048, 512]`. Convert must **split gate_up into gate/up** →
   `ffn_gate_exps [E,512,2048]`, `ffn_up_exps [E,512,2048]`, `ffn_down_exps
   [E,2048,512]`. Shared expert is a plain LagunaMLP → `ffn_{gate,up,down}_shexp`.
7. **No biases** anywhere (attn qkv/o, ffn). Attention sinks exist in code but
   `swa_attention_sink_enabled=False` here → **skip sinks**.

## Layer norms (standard pre-norm, like llama/qwen)
`input_layernorm` → attn → residual; `post_attention_layernorm` → mlp → residual.
Final `norm` then `output` (lm_head, untied).

## Tensor name mapping (HF → GGUF)
| HF | GGUF (`LLM_TENSOR_*`) |
| --- | --- |
| `model.embed_tokens` | `token_embd` |
| `model.norm` | `output_norm` |
| `lm_head` | `output` |
| `layers.N.input_layernorm` | `blk.N.attn_norm` |
| `layers.N.post_attention_layernorm` | `blk.N.ffn_norm` |
| `layers.N.self_attn.q_proj` | `blk.N.attn_q` |
| `layers.N.self_attn.k_proj` | `blk.N.attn_k` |
| `layers.N.self_attn.v_proj` | `blk.N.attn_v` |
| `layers.N.self_attn.o_proj` | `blk.N.attn_output` |
| `layers.N.self_attn.q_norm` | `blk.N.attn_q_norm` |
| `layers.N.self_attn.k_norm` | `blk.N.attn_k_norm` |
| `layers.N.self_attn.g_proj` | `blk.N.attn_gate` *(new)* |
| `layers.N.mlp.gate.weight` | `blk.N.ffn_gate_inp` |
| `layers.N.mlp.gate.e_score_correction_bias` | `blk.N.ffn_exp_probs_b` |
| `layers.N.mlp.experts.gate_up_proj` | split → `blk.N.ffn_gate_exps` + `ffn_up_exps` |
| `layers.N.mlp.experts.down_proj` | `blk.N.ffn_down_exps` |
| `layers.N.mlp.shared_expert.{gate,up,down}_proj` | `blk.N.ffn_{gate,up,down}_shexp` |
| layer 0 `mlp.{gate,up,down}_proj` (dense) | `blk.0.ffn_{gate,up,down}` |

Note the HF checkpoint may store the correction bias at
`mlp.experts.e_score_correction_bias` (vLLM-trained) — `modeling_laguna` remaps it
to `mlp.gate.e_score_correction_bias` on load. Handle both keys in convert.

## New GGUF KV keys needed
- `laguna.attention.gating` (bool/per-head) — drives the gate op.
- Per-layer rope: reuse `rope.dimension_count` won't work (varies). Options:
  (a) store `rope.dimension_sections`/arrays, or (b) derive n_rot in-graph from
  layer type + two scalar factors. Simplest: store full-vs-sliding rope as two
  parameter sets + the `layer_types` mask, branch in graph.
- `laguna.expert_group... ` not needed (no groups). Need
  `expert_weights_scale` (2.5), `expert_weights_func=sigmoid`,
  `expert_gating_func`, router softcap, `attention.sliding_window`,
  per-layer `n_head_arr`, `attn_q_norm`/partial-rotary flags.

Reuse existing keys where they exist (deepseek2 has expert_weights_scale,
sigmoid gating func, exp_probs_b; qwen3moe has qk-norm + shared expert;
gemma2/cohere2 have sliding_window + per-layer attn type).

## Build plan (iterative)
1. **convert_hf_to_gguf.py `LagunaModel`** — produce a GGUF, all tensors mapped,
   metadata written. Testable immediately (no C++ rebuild). *(in progress)*
2. **C++ arch** — `llama-arch.{h,cpp}` (enum `LLM_ARCH_LAGUNA`, tensor names, KV),
   `llama-hparams` fields, `llama-model.cpp` load_hparams + load_tensors.
3. **Graph** — `llm_build_laguna` in `llama-model.cpp`: QK-norm attn, per-layer
   partial dual rope, softplus per-head gate, mixed SWA/full mask, sigmoid MoE
   with correction bias + softcap + routed scaling + shared expert.
4. **Build CUDA (GB10 sm_121)** + smoke test + logits diff vs HF reference.

## PRIMARY TEMPLATE: `STEP35` (Step 3.5) — near-identical architecture
Research (2026-07-12) found `STEP35` already implements almost all of Laguna:
- **head-wise attention gate** (`self_attn.g_proj` → `ATTN_GATE`; tensor_mapping.py
  line 387 literally says "step3.5 head-wise attention gate") ✓
- **QK-norm** ✓, **sigmoid MoE + `exp_probs_b` correction bias + shared expert** ✓
- **mixed `layer_types` + sliding window + per-type rope** — `conversion/step3.py`
  already splits `rope_theta` into full vs `sliding_attention` and emits
  sliding-window metadata ✓

**Plan is now: clone STEP35 (converter `conversion/step3.py` `Step35Model` + its
C++ `llm_build_step35` graph + `LLM_ARCH_STEP35` wiring), rename to `laguna`, and
add only the deltas below.** Most GGUF infra already exists (ATTN_GATE,
FFN_EXP_PROBS_B, add_expert_weights_scale, add_expert_gating_func=sigmoid,
add_head_count(sequence), add_rope_dimension_count + _swa, add_sliding_window).

### Deltas Laguna needs on top of STEP35 (verify each during impl)
1. **Per-layer variable query heads** (48 full / 64 sliding) — STEP35 likely
   constant. Emit `add_head_count([...])` from `num_attention_heads_per_layer`;
   ensure C++ load + graph honor per-layer `n_head`. *(biggest delta)*
2. **Partial rotary on full-attention layers** (`n_rot=64`, YaRN θ=500k, factor
   32, orig_ctx 8192) vs **full rotary on sliding** (`n_rot=128`, θ=10k). Map to
   `rope_dimension_count=64` + `rope_dimension_count_swa=128`; set YaRN
   (freq_scale, ext_factor, beta_fast/slow) for the full-layer rope only. Confirm
   the STEP35 graph applies rope per-layer-type (it has the two thetas already).
3. **`moe_routed_scaling_factor=2.5`** → `add_expert_weights_scale(2.5)`
   (STEP35 may default to 1.0).
4. **Fused expert weights**: HF `experts.gate_up_proj [E,2*512,2048]` +
   `down_proj [E,2048,512]`. Split gate_up → `ffn_gate_exps`/`ffn_up_exps` in
   `modify_tensors` (STEP35's experts may already be split — check).
5. **Correction-bias key** is `mlp.gate.e_score_correction_bias` (note `_bias`
   suffix) — add mapping or strip suffix (cf. AFMOE's `filter_tensors`).
6. **softcap OFF** (`moe_router_logit_softcapping` default 0.0 here) — skip.
7. Attention sinks **off** (`swa_attention_sink_enabled=False`) — skip.

### Other archs to crib for specific deltas
- `deepseek2` — routed scaling / sigmoid gating details.
- `glm` / stablelm — partial rotary (`n_rot < head_dim`) if STEP35 lacks it.
- `qwen3moe` — fused-expert split reference.

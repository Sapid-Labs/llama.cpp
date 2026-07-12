# HANDOFF — Laguna llama.cpp port

Resume-here pointer for the `laguna` architecture port. Deep spec: `LAGUNA-PORT.md`.
Convention for this file: see `~/CLAUDE.md` → "Session handoffs".

---

## 2026-07-12 — converter done, C++ half next

### STATUS
Adding `LLM_ARCH_LAGUNA` (poolside Laguna-XS-2.1, 33B MoE) to llama.cpp.
- Repo: **`Sapid-Labs/llama.cpp`**, branch **`laguna-support`**.
- Worktree: **`~/Dev/llama.cpp-laguna`** (this dir). The pristine build stays at
  `~/Dev/llama.cpp` (master) — do NOT build there.
- Phase: **Python converter DONE ✅ → C++ arch/graph is the next phase.**

### DONE (verified)
- `conversion/laguna.py` + `gguf-py/gguf/constants.py` (`MODEL_ARCH.LAGUNA`, 23
  tensors) + tokenizer hash in `conversion/base.py` (`res="laguna"`).
- **Produces a valid GGUF:** `~/models/gguf/Laguna-XS-2.1-BF16.gguf` (66.9 GB, 678
  tensors). Metadata checked via GGUFReader: per-layer `head_count=[48,64,64,64…]`,
  `head_count_kv=8`, `rope.dimension_count=64`, `sliding_window=512` + pattern
  `[F,T,T,T]×10`, `expert_count=256`/`used=8`, `expert_weights_scale=2.5`,
  `expert_gating_func=2` (sigmoid), `expert_shared_count=1`, `pre=laguna`.

### NEXT (immediate, concrete)
C++ side, in order. **The GGUF can't load until all of this lands.**
1. **`src/llama-arch.{h,cpp}`** — add `LLM_ARCH_LAGUNA`, its name `"laguna"`, the
   KV keys, and the tensor→name map (mirror the `STEP35` block; drop the NEXTN/MTP
   tensors — Laguna has none). Tensors: token_embd, output_norm, output, attn_norm,
   attn_q/k/v/output, attn_q_norm/k_norm, **attn_gate**, ffn_norm, ffn_gate/down/up
   (dense blk 0), ffn_gate_inp, ffn_{gate,down,up}_exps, ffn_{gate,up,down}_shexp,
   **exp_probs_b**.
2. **`src/llama-vocab.cpp`** — add a `laguna` pre-tokenizer. ⚠️ Use the EXACT regex
   from `~/models/hf/Laguna-XS-2.1/tokenizer.json` (a Sequence: newline-merge Split
   → GPT-2/GPT-4 regex Split → ByteLevel). See GOTCHAS.
3. **`src/llama-hparams` + `src/llama-model.cpp` load_hparams/load_tensors** —
   per-layer `n_head` array, dual rope (full=YaRN partial n_rot 64 / sliding=plain
   n_rot 128), sliding-window pattern, sigmoid MoE + `exp_probs_b` + routed scale
   2.5 + shared expert, leading-dense=1.
4. **Graph `llm_build_laguna`** in `src/llama-model.cpp` — clone `llm_build_step35`;
   add deltas: YaRN on full-attention layers only, per-head `softplus(g_proj(x))`
   gate before o_proj, per-layer partial rope. Register in `llama_model::build_graph`.
5. **Build + smoke test + correctness** (see below).

### HOW TO RESUME (commands)
```bash
cd ~/Dev/llama.cpp-laguna              # the worktree/branch
# re-run converter if needed (venv has torch/transformers):
PYTHONPATH=gguf-py ~/venvs/vllm/bin/python convert_hf_to_gguf.py \
  ~/models/hf/Laguna-XS-2.1 --outfile ~/models/gguf/Laguna-XS-2.1-BF16.gguf --outtype bf16
# build CUDA for GB10 (sm_121), ~15 min — build IN the worktree, not ~/Dev/llama.cpp:
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build -j 20
# smoke test once C++ lands:
./build/bin/llama-cli -m ~/models/gguf/Laguna-XS-2.1-BF16.gguf -p "2+2=" -n 20 -ngl 99
```

### GOTCHAS / KEY FACTS
- **`STEP35` (Step 3.5) is the near-exact template** — clone its converter (done)
  and its C++ graph `llm_build_step35`. It already does head-wise attn gate,
  QK-norm, sigmoid MoE + correction bias + shared expert, mixed sliding/full +
  per-type rope. Laguna deltas below.
- **Per-layer variable query heads:** 48 on full-attn layers, 64 on sliding; KV=8
  constant. GGUF `head_count` is an array — C++ must read per-layer `n_head`.
- **Dual RoPE:** full layers = YaRN (θ=500000, factor 32, orig_ctx 8192,
  beta_fast 64, beta_slow 1), **partial rotary → n_rot 64**. Sliding layers =
  plain rope θ=10000, **full rotary n_rot 128**. GGUF carries `rope.dimension_count`
  (64) and `..._swa` (128). YaRN must apply to full layers ONLY.
- **Per-head attn gating:** `attn = attn.view(h,d) * softplus(g_proj(x))[…,None]`,
  applied BEFORE o_proj. `g_proj` = `attn_gate` tensor [n_embd→n_head].
- **QK-norm:** RMSNorm(head_dim) on q,k per head, BEFORE rope.
- **MoE:** sigmoid router; select on `sigmoid(logits)+exp_probs_b`; weights =
  `sigmoid.gather(top8)` then normalized; `× 2.5`; `+ shared_expert`. **softcap OFF**
  (`moe_router_logit_softcapping`=0), **attention sinks OFF**
  (`swa_attention_sink_enabled`=false). Layer 0 dense (`mlp_only_layers=[0]`).
- **Experts on disk are per-expert** (separate gate/up/down), merged in convert.
- **RMSNorm is standard** (weight * x_normed) — do NOT add 1.0 (Step35 does; Laguna
  does NOT).
- **Tokenizer caveat:** transformers warns Laguna's tokenizer.json uses the
  "mistral incorrect regex" variant. The C++ `laguna` pretokenizer MUST use the
  exact tokenizer.json regex, and tokenization must be verified vs HF first.

### CORRECTNESS GATES (before trusting output)
1. **Tokenization:** `llama-tokenize` a few strings; compare token IDs to HF
   `AutoTokenizer` on the same strings. Must match exactly.
2. **Generation:** greedy (`--temp 0`) llama.cpp output vs a transformers (or the
   already-working vLLM) greedy reference on the same prompt. Coherent + close
   logits. A per-layer logits diff pins bugs (convert vs graph).

### LINKS
- Deep spec + tensor map + deltas: `LAGUNA-PORT.md` (same dir).
- Reference archs in-tree: `STEP35` (primary), `deepseek2` (routed scale/sigmoid),
  `glm` (partial rope), `qwen3moe` (fused-expert split).
- Model: `~/models/hf/Laguna-XS-2.1` (HF), GGUF at `~/models/gguf/Laguna-XS-2.1-BF16.gguf`.
- Downstream: once it runs, add llama.cpp Laguna benchmark rows in the
  `howtospark` repo (bench/configs + data/benchmarks) — separate project, its own
  workflow (see that repo's CLAUDE.md).

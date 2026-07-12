# HANDOFF — Laguna llama.cpp port

Resume-here pointer for the `laguna` architecture port. Deep spec: `LAGUNA-PORT.md`.
Convention for this file: see `~/CLAUDE.md` → "Session handoffs".

---

## 2026-07-12 (night) — DFlash WORKING: 36% acceptance / 1.43× net speedup on code ✅

### STATUS
DFlash-Laguna speculation **works and is a net win on code**: Q4_K_M target + F16
draft on the code-continuation prompt gives **acceptance 36.1%, mean draft len 6.32,
132.8 tok/s vs 93.1 baseline (1.43×)**. On chat/reasoning output acceptance is ~11%
(net slowdown) — **content-dependent, not a bug**. Implementation validated stage-by-
stage against the now-**MERGED** vLLM reference (PR #46853 merged 2026-07-03; the
previous "unmerged/blocked" note was stale).

### ROOT CAUSES FOUND (vs yesterday's ~10%-everywhere)
1. **Causal block attention** — Laguna's draft is trained CAUSAL
   (`dflash_config.causal=true` in its config.json); qwen3 dflash is non-causal and
   speculative.cpp hardcoded non-causal for all drafts. Converter now writes
   `dflash.attention.causal`; speculative.cpp reads it (absent → non-causal, so qwen3
   GGUFs unchanged). On code: causal 36.1%/6.32 vs non-causal 29.2%/5.31.
2. **SWA** — the merged reference REMOVES the sliding window at attention compute
   (`attn.sliding_window = None`, no proposer-side mask), i.e. the drafter runs full
   attention. Use the `DFLASH_NO_SWA=1`-converted GGUF (neutral for ctx<512, matters
   beyond). **Primary draft GGUF: `~/models/gguf/Laguna-XS-2.1-DFlash-F16-noswa.gguf`
   (causal, no SWA).**
3. **KV-injection input_layernorm** — reference `_project_context_kv` applies each
   layer's input_layernorm to the fused context state before K/V projection; added
   (gated on `enc_aux_norm` presence → qwen3 path untouched). NOTE: mathematically a
   NO-OP for this checkpoint (all 5 input_layernorm weights are exactly 1.0 and fused
   states have unit RMS) — kept for reference fidelity.
4. The rest of the residual gap is **content**: the model card's ~70% is on coding;
   chat `<think>` reasoning prose drafts at ~11%. A faithful fp32 PyTorch reference
   achieves the same (2.05 vs llama.cpp's 1.90 accepted/block on the chat dump), so
   llama.cpp is NOT leaving acceptance on the table.

### VALIDATION DONE (proof)
- **Encoder**: llama.cpp fused states match fp32 reference, ~0.2% rel err.
- **Block logits**: 88% top-1 agreement w/ fp32 causal reference (F16/Q4 noise);
  non-causal reference clearly disagrees (38%) → causal path really active.
- **Target aux extraction**: hooked real vLLM (base Laguna BF16, in-process V1) at
  layers [2,14,26,34,40]-entry (= `target_layer_ids+1`, hidden+residual, pre-final-
  norm) and compared with llama.cpp's `feat.bin`: cosine 0.96–1.00 per position; the
  ~8% magnitude diff is Q4-vs-BF16 target noise. Extraction mapping is CORRECT.
- Tools (scratchpad of this session): `ref_dflash.py` (fp32 reference + stage diffs),
  `ref_acceptance.py` (offline acceptance from dumps), `vllm_aux_capture.py`
  (vLLM hook comparison). `DFLASH_DUMP=<dir>` env in speculative.cpp dumps
  feat/fused/block records (binary: n_tok,n_dim,toks,pos,data per record).

### HOW TO RESUME / REPRODUCE
```bash
cd ~/Dev/llama.cpp-laguna && cmake --build build -j 20
./build/bin/llama-server -m ~/models/gguf/Laguna-XS-2.1-Q4_K_M.gguf \
  -md ~/models/gguf/Laguna-XS-2.1-DFlash-F16-noswa.gguf --spec-type draft-dflash \
  --spec-draft-n-max 15 -c 10240 --parallel 1 -ngl 99 -fa on --jinja --port 8080
# code prompt via /completion (raw, no chat template) → acceptance ~0.36 in log.
# Reconvert draft (writes causal key; DFLASH_NO_SWA=1 for the primary no-swa file):
DFLASH_NO_SWA=1 PYTHONPATH=gguf-py ~/venvs/vllm/bin/python convert_hf_to_gguf.py \
  ~/models/hf/Laguna-XS-2.1-DFlash --outfile ~/models/gguf/Laguna-XS-2.1-DFlash-F16-noswa.gguf \
  --outtype f16 --target-model-dir ~/models/hf/Laguna-XS-2.1
```

### NEXT (optional)
- Benchmark rows for howtospark: spec-decoding profile on CODE workloads (that's
  where the win is); chat/reasoning rows would show a slowdown — bench both honestly.
- Diagnostic env gates available: `DFLASH_NO_{GATE,AUX,SWA}` (converter),
  `DFLASH_NO_INJ_NORM`, `DFLASH_DUMP` (runtime).

---

## 2026-07-12 (evening) — DFlash draft support: implemented, acceptance ~10% (SUPERSEDED — see above)

### STATUS
DFlash-Laguna speculator ported (commit `34e0062`, branch `laguna-support`). **Works**
(loads, correct output, acceptance > 0) but acceptance is only **~10%** vs ~45% for the
qwen3 dflash → net slowdown. Not published. **Base Laguna ruled out** (llama.cpp logits
match vLLM); residual gap is in the draft forward, blocked on there being no working
DFlash-Laguna reference (vLLM PR #46853 unmerged, 0.24.0 back-port numerically wrong).

### WHAT'S IN THE COMMIT
- `conversion/qwen.py` `DFlashLagunaModel` (fused-qkv split, g_proj→gate, aux_hidden_norms
  → stacked `enc.aux_norm`, all-sliding SWA, dense expert-count zeroing; env gates
  `DFLASH_NO_{GATE,AUX,SWA}` for diagnostics). New `enc.aux_norm` tensor (llama-arch + gguf-py).
- `dflash` graph: encoder aux-RMSNorm before fc; decode softplus per-head gate (both gated on
  tensor presence — qwen3 dflash unchanged).
- `llama-context.cpp`: `embeddings_layer_inp` supports `lid == n_layer` (extract final-layer
  output); `laguna.cpp` registers `t_layer_inp` per layer + final residual.
- Draft GGUF: `~/models/gguf/Laguna-XS-2.1-DFlash-F16.gguf`.

### HOW TO RESUME / REPRODUCE
```bash
./build/bin/llama-server -m ~/models/gguf/Laguna-XS-2.1-Q4_K_M.gguf \
  -md ~/models/gguf/Laguna-XS-2.1-DFlash-F16.gguf --spec-type draft-dflash \
  --spec-draft-n-max 15 -c 10240 --parallel 1 -ngl 99 -fa on --jinja --port 8080
# server logs "draft acceptance = 0.1x". Full findings + resume plan:
#   ~/Dev/howtospark/models/laguna-xs-2-1.md (DFlash section)
```

---

## 2026-07-12 (pm) — C++ port DONE, runs & generates coherently ✅

### STATUS
`LLM_ARCH_LAGUNA` fully ported to llama.cpp C++ and **working**. Model loads on the
GB10 (CUDA sm_121) and generates coherent, sustained greedy output.
- Repo **`Sapid-Labs/llama.cpp`**, branch **`laguna-support`**, worktree **`~/Dev/llama.cpp-laguna`**.
- Build: `~/Dev/llama.cpp-laguna/build` (CUDA sm_121). Binaries: `llama-completion`,
  `llama-tokenize`. Perf: ~34 tok/s eval, ~134 tok/s prompt, load ~120–160s (66.9 GB BF16).

### DONE (verified this session)
- **C++ arch wiring** — `LLM_ARCH_LAGUNA` enum+name (`llama-arch.{h,cpp}`), factory +
  NEOX rope type + `LLM_TYPE_33B_A3B` (`llama-model.{h,cpp}`), `get_can_shift()=false`
  for per-layer rope (`llama-kv-cache.cpp`).
- **Model struct + graph** — `src/models/laguna.cpp` (+ struct in `models.h`). Cloned
  from `STEP35` trunk. **Two deltas vs STEP35 that matter:**
    1. **Do NOT halve `n_rot_full`** — Laguna's converter already writes the partial
       rope dim (64) directly; STEP35 writes 128 and halves. (full=64 / swa=128.)
    2. **Attention gate is `softplus`, NOT sigmoid** — `modeling_laguna.py:459`
       `F.softplus(g_proj(x))`, per-head (`config "gating":"per-head"`). STEP35 uses
       sigmoid. `ggml_softplus` exists (ggml.h:1007). *This was the one easy-to-miss bug.*
  Also dropped all NextN/MTP code (Laguna has none → single `graph`, no `graph_mtp`).
- **`laguna` pre-tokenizer** — `LLAMA_VOCAB_PRE_TYPE_LAGUNA=56` (`llama-vocab.h`),
  string-match + `clean_spaces=false` + regex case (`llama-vocab.cpp`). Regex = GROK_2's
  (single `\p{N}` GPT-4 / "mistral incorrect regex") + newline-merge `(?:\r?\n)+(?!\r?\n)`.
- **Everything else reused from STEP35 verbatim and confirmed correct for Laguna:** per-layer
  `n_head` (48 full/64 swa, KV=8), dual RoPE (YaRN θ500k factor32 on full only — YaRN is a
  no-op on swa because their `freq_scale=1.0`), sigmoid MoE + `exp_probs_b` correction bias
  + `norm_topk` + ×2.5 routed scale + shared expert, QK-norm, leading-dense layer 0.

### CORRECTNESS GATES — results
1. **Tokenization vs HF AutoTokenizer: EXACT MATCH** on 6 strings incl. multi-digit
   numbers, code w/ newlines+indentation, tabs, unicode. (script:
   `…/scratchpad/toktest.py`, uses `~/venvs/vllm/bin/python`.)
2. **Greedy generation: coherent & sustained** — 200 tokens temp=0, correct Fibonacci
   reasoning (0-based vs 1-based, recursion-vs-DP), zero degradation. Rules out any
   graph bug (rope/gate/MoE/qk-norm) — those compound to garbage over 200 tok.
3. **NOT YET DONE (optional rigor):** per-token logits diff vs vLLM/HF greedy reference.
   Functional evidence is already strong; do this only if bit-exact parity is needed.

### NEXT (optional)
- Commit is on `laguna-support` (this session). Push to `Sapid-Labs/llama.cpp` when ready.
- Rigorous logits-diff vs vLLM (`~/venvs/vllm`, vLLM-Moet port already runs this model).
- Quantize (Q4_K_M etc.) + re-verify coherence.
- Downstream: add llama.cpp Laguna benchmark rows in the `howtospark` repo (its own workflow).
- Upstream PR to ggml-org/llama.cpp would need the NextN/STEP35-clone lineage cleaned up.

### HOW TO RESUME (commands)
```bash
cd ~/Dev/llama.cpp-laguna
cmake --build build -j 20                    # incremental
# generate (needs --jinja; model has a custom chat template llama.cpp's default engine rejects):
./build/bin/llama-completion -m ~/models/gguf/Laguna-XS-2.1-BF16.gguf \
  -p "Write a Python function that returns the nth Fibonacci number." -n 200 -ngl 99 --temp 0 --jinja
# tokenizer parity:
~/venvs/vllm/bin/python /tmp/.../scratchpad/toktest.py   # (recreate from this handoff if gone)
```

### GOTCHAS discovered this session
- **`--jinja` is required** to generate: the model's custom chat template throws
  `this custom template is not supported` in llama.cpp's default template engine.
  `-no-cnv` is not a `llama-completion` flag (use `llama-cli`→`llama-completion` rename).
- Tensor-name map in this fork is a **flat** `LLM_TENSOR_NAMES` (not per-arch), and
  `src/models/*.cpp` is a **CMake GLOB** — dropping `laguna.cpp` in auto-registers it.
- The 66.9 GB BF16 GGUF cold-loads in ~120–160s; budget for it in each test iteration.

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

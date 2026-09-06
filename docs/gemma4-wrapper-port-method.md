# Getting dense Gemma-4-12B, 26B-A4B (MoE) and 31B coherent on Inferentia2 — the wrapper-port method

> **This method is not ours. It is [xbill9 (William McLean)](https://github.com/xbill9)'s published
> recipe, which we adopted after our own hand-rolled NxD Inference (NxDI) modeling port hit an
> unfixable compiled-attention numerical bug.** Everything below builds on his work. Credit him
> first, loudly, and in every derivative. We ship no weights — only scripts and recipes.

> **Family status — COMPLETE on Inferentia2.** All five members of the gemma4 2026 family now run
> coherent on NeuronCore-v2 through this one wrap-HF-eager method: **E2B** (128K), **E4B** (128K),
> **12B dense** (SEQ_MATCH bit-exact), **26B-A4B MoE** (72K, coherent), and **31B dense** (coherent,
> logit-verified — see "Dense 31B" below). The dense 31B was our published GPU-only dead-end; the
> wrapper overturned it. Dense and MoE, small to 31B, all on inf2 — the first published
> demonstration of the full family on this silicon.

---

## Attribution first — read this before anything else

**William McLean — `xbill9`** ([GitHub](https://github.com/xbill9),
[HuggingFace](https://huggingface.co/xbill9)) — Google Developer Expert and AWS Community Builder —
**published the Gemma-4-on-Inferentia2 wrapper recipe that this document describes.** He worked it
out and released it, Apache-2.0, on HuggingFace and in his DEV.to field reports *before* we ran a
single compile against it. The 12B dense port and the 26B-A4B MoE port that we measured as working
are **his recipe, run against Google's weights**, with a handful of per-model adaptations noted
explicitly in their own section below.

His published artifacts (the source of this method):

- **E2B** — `xbill9/gemma-4-E2B-it-inferentia2` — the original "Option B" single-core recipe
  (torch_neuronx trace of the HF text forward, host-side PLE, two-graph KV cache, softcap-30).
- **E4B** — `xbill9/gemma-4-E4B-it-inferentia2` — the TP=2 tensor-parallel build with
  device-resident aliased KV, GQA head-sharding, and the `ModelBuilder` device-prefill path
  (`tp_mb.py`). **This is the template our 12B and 26B ports are adapted from.**
- **26B-A4B** — `xbill9/gemma-4-26B-A4B-it-inferentia2` (branch `gemma4-inf2-nxd-kvshare`) —
  the MoE build: `DenseExperts` (all-128-dense), `SPMDRank` scatter, `ScatterKV`, router, and the
  `layer_scalar` hand-load. Our 26B port is **his `tp_mb_moe.py` run verbatim** against staged
  weights, with 2 path edits.
- DEV.to field reports (search "xbill9 gemma inferentia" — the write-ups that document the recipe
  as he developed it).

xbill9's own README files (`reference/xbill9/`) are included in this repo **unmodified and
attributed as his**, not as ours. When you redistribute any part of this, carry his Apache-2.0
`LICENSE`/`NOTICE` and name him.

**Base weights:** Google Gemma-4 (`google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it`), under the
Gemma license — see Google's model cards. **We ship no weights.** The neffs xbill9 publishes embed
bf16 base weights and are therefore Apache-2.0 derivatives *with attribution to Google*; our repo
contains only recipes/scripts that operate on weights you fetch yourself.

**Toolchain:** AWS Neuron SDK / NxD Inference (NxDI) / `neuronx-cc`. The compiler bug that pushed us
off our own port is filed as **[aws-neuron/aws-neuron-sdk#1391](https://github.com/aws-neuron/aws-neuron-sdk/issues/1391)**
(reporter: barrhawk) — root cause `islpy` dependency drift; pin `islpy==2026.1`.

---

## Why we use xbill9's wrapper instead of our own NxDI port

We first tried to port dense 12B the "proper" way — a native NxDI modeling module
(`dense_port_wip/modeling_gemma4.py`) that re-implements the architecture inside AWS's framework.
It **compiled, loaded, and generated correct facts and arithmetic** ("capital of France" → "Paris.",
"17+25" → "42"). It looked done.

It was not. Rigorous re-diagnosis (logit-exact vs HF fp32 CPU reference) proved a **compiled-graph
numerical bug we could not fix in source**:

- The compiled Neuron forward **under-suppresses the just-emitted token cluster** by up to ~11.5
  logits (max|Δ| up to 13 vs a bf16 floor of ~0.1). Native `generate` loops on punctuation.
- The bug is **in the compiled NEFF, not the Python.** We ruled out — by real commands, not
  guessing — decode/KV/mask/position/ring, precision (bf16 CPU sims of the same algorithm PASS),
  output-proj, final-norm, `layer_scalar`, RoPE, `k_eq_v`, and every GQA/shape path. Every Python
  line matches HF eager.
- Leading suspects (convergent across two Gemini graph-debug passes + our own): `--enable-saturate-infinity`
  corrupting the softcap `tanh` LUT at extreme magnitude; `--enable-mixed-precision-accumulation`
  truncating the 262k-vocab LM-head reduction (catastrophic cancellation of the *suppressive*
  logits); q/k RMSNorm sum-of-squares not promoted to fp32. The final norm weight maxes at **604** —
  extreme magnitudes are what break the compiled transcendentals.

That is graph/compiler-level debugging with an uncertain payoff. **xbill9's wrapper sidesteps it
entirely** by never asking NxDI to model the architecture — it traces Google's own HF eager forward,
so KV-sharing, softcap, and the norms trace as ordinary live graph dependencies (exactly as they do
on TPU/XLA). We pivoted to his recipe and it produced **token-for-token-exact** 12B on the first
serious run. The hand-rolled-port bug is now moot — we don't use that port.

---

## The wrapper pattern (xbill9's recipe, in one page)

The core idea: **do not re-implement Gemma-4 inside NxDI.** Wrap Google's HuggingFace eager forward
(`Gemma4ForConditionalGeneration` / `Gemma4Unified…` text tower) and let `torch_neuronx` /
`ModelBuilder` trace it, replacing only the pieces that must be sharded or made device-resident.

The pattern, piece by piece (all of these are xbill9's — the E2B/E4B READMEs in
`reference/xbill9/` are the primary source):

1. **Wrap the HF eager forward, not the SDPA/fused path.** Eager attention traces cleanly; the
   fused path does not. The whole language model + LM head are registered as **real submodules** so
   they compile into the graph.

2. **Replace the linears with TP shards.** `q_proj / o_proj / k_proj / v_proj / gate / up / down`
   become NxD `ColumnParallelLinear` / `RowParallelLinear` across the tensor-parallel ranks
   (q/gate/up column, o/down row). Everything else stays HF eager.

3. **Inject a one-hot-scatter `ScatterKV` device-resident KV buffer via `input_output_aliases`.**
   The K/V buffers are device-resident `nn.Parameter`s aliased as graph I/O — never round-tripped
   through the host. Each decode step writes the new K/V with a one-hot masked scatter,
   `buf*(1-oh) + k*oh` — pure arithmetic, so it is trace-safe. This is what makes decode
   compute-bound and flat across context length.

4. **Hand-load the `layer_scalar` BUFFERS from the checkpoint.** *This is the non-obvious fix
   xbill9 flagged and the single most common way to get garbage.* Gemma-4's per-layer `layer_scalar`
   is a **buffer**, and NxD's `ModelBuilder` weight-sharding loads **parameters only**. If you don't
   copy `layer_scalar` from the checkpoint by hand, **every layer over-scales ~16× into garbage.**
   Copy it explicitly after the sharded load.

5. **Off-device softcap, written out explicitly.** Attention softcap = 30 and the logit softcap are
   written as explicit `30.0 * tanh(x/30.0)` in fp32, not left to a fused/compiled LUT (that LUT is
   exactly what broke our hand-rolled port). tanh-GELU is likewise written out.

6. **Host-side embeddings.** The huge embedding / Per-Layer-Embedding tables are kept off-device and
   gathered on the CPU, fed in as activations. On-device they trip the compiler and blow the
   16 GB-per-core budget. (Dense 12B has no PLE — see its adaptation — but the host-embedding
   principle still applies to the token embedding.)

7. **BOS-token requirement.** The Gemma tokenizer does **not** auto-add BOS. Gemma is degenerate
   without token id **2** at the front. Seed it, or your reference tensor is garbage and every
   comparison lies (this bit us — a pre-staged fp32 reference was itself wrong for this reason).

The E2B/E4B builds ship this as a two-graph (prefill + decode) design or a single
weight-sharing `ModelBuilder` trace with prefill and decode as two buckets of one resident model.
`ModelBuilder` device-prefill (`tp_mb.py`) is the recommended path — ~0.16 s first token vs
~1.4–1.6 s for the host-CPU-seed prefill.

---

## Per-model adaptations we made

xbill9's E4B `tp_mb.py` is the template. Here is what changed per model to get **our** two targets
coherent. These are the only deltas — the recipe above is unchanged.

### Dense 12B (`google/gemma-4-12B-it`, `Gemma4UnifiedForConditionalGeneration`)

Distinct arch from E-family and from gemma3. Adapted from xbill9's E4B `tp_mb.py`:

- **No PLE.** `Gemma4Unified` has no Per-Layer Embeddings (`hidden_size_per_layer_input=0`). Drop
  the PLE gather; embed with `embed * sqrt(3840)` on the host.
- **Global-layer MQA, `k_eq_v`.** The 8 full/global layers have `v_proj = None` and 1 KV head
  (`k_eq_v` — value synthesized from key). Set
  `num_key_value_groups = (nq_full // TP) // nkv = 8`. **Read `nq` BEFORE you replace `q_proj`**
  with the column shard, or you compute the group count against the wrong head count.
- **Isolated build venv + fx shim.** 12B needs a gemma4-capable `transformers` (5.10.4) plus
  xbill9's `transformers.utils.fx` shim, built in `/workspace/build_venv` so the serving venv
  (`nxdi_venv`) stays pristine.
- **BOS fix** as above — the tokenizer doesn't auto-add id 2.
- Sliding layers use head_dim 256 / 8 heads; global layers head_dim 512 / 1 MQA head;
  `num_kv_shared_layers=0`; `layer_scalar` bound to the real per-layer values (0.053/0.166/…),
  **not** 1.0 — the hand-load from §4 is what makes that true.

### 26B-A4B MoE (`google/gemma-4-26B-A4B-it`)

**Run xbill9's `tp_mb_moe.py` essentially verbatim** (pulled from
`xbill9/gemma-4-26B-A4B-it-inferentia2`) against staged `google/gemma-4-26B-A4B-it` weights — only
**2 path edits**. His recipe transferred cleanly to the newer SDK (`neuronx-cc` 2.27 / nxd 0.19).
Its ingredients (all his):

- **`DenseExperts`** — all 128 experts run dense (not gathered top-8-only), which the compiled graph
  handles.
- **`SPMDRank` scatter** for the expert routing across ranks, plus **`ScatterKV`** for the KV buffer.
- **Router** with Gemma's extras: `router.proj[128,2816]`, `router.scale[2816]`,
  `router.per_expert_scale[128]`. *(If you instead graft onto NxDI's Qwen3-MoE base — the harder
  path we scoped but did not need — note the base defaults `num_experts=64` → 32 local under TP2;
  you must override `config.num_experts = 128`, and `per_expert_scale` must be **gathered** by
  `topk_indices`, not broadcast: `torch.gather(per_expert_scale, 0, topk_indices.flatten())`. The
  wrapper path avoids this entirely.)*
- **`layer_scalar` hand-load** (§4) — 30 layer_scalar buffers loaded from checkpoint.
- The real arch is a **parallel dual-path FFN every layer**: a dense shared MLP (I=2112, always on)
  **plus** 128-expert top-8 routed MoE (I=704), 30 layers, hidden 2816, 25 sliding + 5 global
  attention layers, softcap 30, `embed * sqrt(2816)`.

---

## Measured verification (provenance-honest)

Numbers below are **measured run artifacts**, not estimates. Where something was not measured, it
says so — per the provenance law, never call a number "measured" without a run behind it.

### 12B dense — SEQ_MATCH verified (the gold gate)

- Compiled TP=2 bf16 (`MB_TRACED` 748 s, both buckets).
- Device generates coherent text: *"The capital of France is **Paris**."*
- **`SEQ_MATCH[0]` / `SEQ_MATCH[1]` True, `ALL_SEQ_MATCH` True** — device greedy decode is
  **token-for-token identical to the HF fp32 CPU reference.** This is the exact bar the hand-rolled
  port failed.
- Decode **15.29 tok/s** (matches xbill9's ~15), prefill ~100 ms, HBM **~16.1 GB/core** (nearly
  fills the 32 GB device, tight-but-fits).
- **Caveat:** the compiled artifact is **not persisted** — `torch.jit.save` hits a known large-
  archive c10 abort at 29 GB (needs NxD `parallel_model_save` / chunked). Recompile ~13 min to
  serve, or fix the save.

### 26B-A4B MoE — coherent + correct (not bit-exact)

- Compiled TP=8 bf16 (compiler status PASS both buckets), `MB_SAVED` 64.6 GB, 30 layer_scalar
  buffers loaded, loads via nxd `initialize_with_saved_weights`.
- **Generates coherent, correct text** (chat template, greedy): "capital of France" → *"The capital
  of France is **Paris**."* (three independent phrasings); "17×24" → coherent step-by-step; "why is
  the sky blue" → correct Rayleigh explanation.
- Decode **~25 tok/s**, prefill first-token **~80 ms** (matches xbill9's 77 ms), HBM
  **8.57 GB/core of 16 (54%, fits comfortably)** across 8 cores.
- Rebuilt from xbill9's `tp_mb_moe.py` run verbatim vs staged weights, 2 path edits only.
- **Caveat (honest):** verified coherent-and-correct, **NOT** bit-exact `SEQ_MATCH` — we skipped
  `DEVICE_ONLY=1` to save compute. xbill9 reports `SEQ_MATCH True` for this recipe.

### Dense 31B — coherent, logit-verified on inf2 (not bit-exact)

The dense 31B was our published GPU-only dead-end. It was not. The wrapper brings it up on
Inferentia2 the same way it brings up the rest of the family — `google/gemma-4-31B-it`, 60 layers,
62.5 GB bf16, no PLE — run through the wrap-HF-eager recipe, no hand-port of the architecture.

- Compiled **TP=8** on **inf2.24xlarge**; HBM **~14.6 GB/core** (fits the 16 GB budget).
- Decode **15.63 tok/s**, first token **120 ms**.
- Generates coherent text: *"The capital of France is **Paris**."*; 7×8 = 56.
- **Logit gate vs canonical HF fp32:** argmax **57/58 (98.28%)** across **42 prefill + 16 decode**
  positions; cosine mean **0.9998** / min **0.989**; max|Δ| **5.24**.
- **Verdict: coherent, NOT bit-exact.** One near-tie argmax flip — we report it and do not claim
  strict `SEQ_MATCH`.

**Why the old wall fell.** The prior finding (missing per-layer-embed keys, `k_proj` shape mismatch)
was a fact about the hand-port re-implementing the architecture inside NxDI, where the E2B remap did
not expect the dense checkpoint keys. The wrapper never re-implements the architecture — it traces
Google's own eager forward — so those mismatches never arise. The dense 31B is not GPU-only on this
silicon; we measured it on inf2.

### The wall (why the 26B MoE needs a big box)

Full 51.6 GB bf16 weights **do not fit** inf2.8xlarge (32 GB HBM; TP2 needs ~26 GB/core > 16). They
fit inf2.24xlarge (192 GB → ~4.3 GB/core) or int8/fp8 (~26 GB, tight on 32 GB). This is
**solvable** (bigger box / quant). The dense 31B, once thought a hard GPU-only wall
(missing per-layer-embed keys + `k_proj` shape mismatch), is **also solved by the wrapper** —
those mismatches were the hand-port fighting the architecture, and wrapping Google's eager
forward never raises them. See "Dense 31B" below: coherent, logit-verified, running on inf2.

---

## Environment / gotchas

- **Pin `islpy==2026.1`** — `neuronx-cc` 2.27 breaks on `islpy` 2026.2.1 drift. Filed as
  aws-neuron/aws-neuron-sdk#1391.
- **Isolated `transformers` per model.** 12B needs 5.10.4 + xbill9's `transformers.utils.fx` shim;
  26B ran on 5.15. Build in a throwaway venv so the serving venv stays clean.
- **BOS token id 2** must be prepended — the tokenizer won't; without it both the model output and
  any fp32 reference are garbage.
- **Never run two NxDI compiles concurrently on one box** — a shared `/tmp/nxd_model` poisons the
  cache with "cached failed neff". Clear `/var/tmp/neuron-compile-cache`.
- **Compile peaks past host RAM** on the big buckets — add swap before compiling (xbill9's E4B note:
  a 55 GB swapfile is enough; the resulting neff runs fine without swap).

---

## Files in this repo

- `docs/gemma4-wrapper-port-method.md` — this document.
- `reference/xbill9/` — **xbill9's original published scripts and READMEs, unmodified, credited
  to him** (`gemma-4-E2B-*`, `gemma-4-E4B-*` — `tp_mb.py`, `tp_alias_trace.py`, `optb_*`, the two
  READMEs). The E4B `tp_mb.py` is the template our 12B and 26B ports adapt.
- `recipes/long-context/` — the sliding-window / chunked-prefill / compile scripts from the E2B
  128K long-context work (`attention_base.py`, `modeling_gemma4.py`, `gemma_compile*.py`,
  `gprobe*.py`, `hf_ref_gemma.py`, `tkg_bench.py`) plus the 26B MoE ladder
  (`run_moe*.py`, `pushctx*.py`). Related capability, same family.
- `recipes/serving/gemma4_server.py` — the OpenAI-compatible NxDI serving backend + function-calling
  parser (used by the live E2B `hawkalphaquick` endpoint).

## Links

- xbill9 — https://github.com/xbill9 · https://huggingface.co/xbill9
- `xbill9/gemma-4-E2B-it-inferentia2` · `xbill9/gemma-4-E4B-it-inferentia2` ·
  `xbill9/gemma-4-26B-A4B-it-inferentia2` (branch `gemma4-inf2-nxd-kvshare`)
- xbill9's DEV.to field reports (search "xbill9 gemma inferentia")
- Google Gemma-4 model cards: `google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it`
- AWS Neuron SDK issue: https://github.com/aws-neuron/aws-neuron-sdk/issues/1391

*Not affiliated with or endorsed by Google, AWS, or William McLean. This document records how we
applied his published, Apache-2.0 recipe. Base weights are Google's, under the Gemma license. We
distribute no weights.*

# Long-Context Serving for Gemma-4 on AWS Inferentia2

*How we run 2026 long-context models on 2023 silicon — 128K on a single E2B chip, and a
32K → 72K needle-verified ladder on the 26B-A4B MoE — using a hand-ported
sliding-window + chunked-prefill KV path on Neuron's static AOT graphs.*

---

## TL;DR (measured, not hoped)

- **Gemma-4-E2B: 128K context on one Inferentia2 chip.** Prefilled **130,823 tokens in
  100.6 s (~1,300 tok/s)**, decode ~42 ms/tok, HBM **15.76 / 16 GB per core**, and the
  needle `ORCHID-7423-FALCON` placed **130,622 tokens back was retrieved** (also at 32K
  and 100K).
- **Gemma-4-26B-A4B (MoE): a needle-verified 32K → 49K → 72K ladder** on a single
  `inf2.24xlarge` (TP=8), a context a flat KV compile physically cannot load on the box.
  Needles retrieved at **ctx 32,480 / 48,986 / 71,981**; chunked prefill **309–443 tok/s**.
- The techniques that make this possible: a **sliding-window ring-buffer KV cap**,
  **chunked (bucketed) prefill**, and **column-parallel `lm_head` sharding**. All three
  are hand-ports onto AWS's NxD Inference stack, which gates the native long-context
  backend to Trainium2/3 and turns it *off* on Inferentia2.

This is the long-context capability AWS ships only on Trn2/Trn3. It now runs on
Inferentia2 — NeuronCore-v2, the cheapest AI accelerator AWS rents.

---

## Credit where it is due

This work **builds directly on William McLean (`xbill9`)** — Google Developer Expert /
AWS Community Builder — who published the Gemma-4-on-Inferentia2 wrapper recipe under
Apache-2.0:

- GitHub: [github.com/xbill9](https://github.com/xbill9) · HuggingFace:
  [huggingface.co/xbill9](https://huggingface.co/xbill9)
- Model repos: `xbill9/gemma-4-E2B-it-inferentia2`, `xbill9/gemma-4-E4B-it-inferentia2`,
  `xbill9/gemma-4-12B-it-inferentia2`, `xbill9/gemma-4-26B-A4B-it-inferentia2` (branch
  `gemma4-inf2-nxd-kvshare`).

xbill9's wrapper recipe — **wrap the HF eager model, `ScatterKV` device cache,
`input_output_aliases` for KV, and hand-loaded `layer_scalar` buffers** — is what our 12B
and 26B ports stand on. His original published scripts are included verbatim under
[`reference/xbill9/`](../reference/xbill9/) as an attributed reference; they are **his
work, not ours**. Our contribution is the *long-context* layer on top: the sliding-window
KV cap, the chunked-prefill path, `lm_head` sharding, and the needle verification at
scale.

**Base weights** are Google's **Gemma-4** family (Apache-2.0-style Gemma license — see
Google's model cards, e.g. [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)
and [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)).
**We ship no weights — only recipes and scripts.**

**Toolchain:** AWS Neuron SDK, NxD Inference (NxDI), and `neuronx-cc`. A blocking compiler
bug we hit along the way is filed as
[aws-neuron/aws-neuron-sdk#1391](https://github.com/aws-neuron/aws-neuron-sdk/issues/1391)
(reporter: `barrhawk`); root cause is an `islpy` dependency drift (see below).

---

## 1. The problem

### 1.1 Neuron compiles a static graph ahead of time

Inferentia2 / NeuronCore-v2 does not JIT. `neuronx-cc` compiles a **static, ahead-of-time
(AOT) graph** — a NEFF — for a fixed set of input shapes ("buckets"). Every tensor
dimension the runtime will ever see must be baked in at compile time. There is no dynamic
allocation, no ragged batch, no growing KV cache. If you want to serve a 128K-token
context, the graph must be compiled to *statically* score against a 128K-wide key/value
buffer.

That static-shape discipline is exactly why Neuron is cheap and fast — and exactly why
long context is hard.

### 1.2 The KV memory wall

A transformer's KV cache grows linearly with context length. On a static graph the cache
is a **fixed-size buffer sized to the maximum context**, resident in HBM for the life of
the model. Inferentia2 gives you **16 GB of HBM per NeuronCore**. Once the model weights,
the activation scratchpad, and the compiled code are resident, whatever is left is your
KV budget — and a full-width KV buffer at 128K tokens across every layer does not fit.

Two independent OOM modes bite:

1. **Compile-time host RAM.** A naïve full 16K prefill CTE (context-encoding) graph tried
   to materialize a `16K × 16K` score matrix on *every* layer and needed **~378 GB of
   host RAM** to trace — it OOM'd the build host before it ever reached the device.
2. **Device HBM.** The resident KV buffer itself. On the E2B this is the tight constraint
   (~1.5% headroom at 128K); on the 26B MoE it is *not* the KV that kills you — it is the
   fixed expert weights (see §7).

### 1.3 Gemma-4's mixed sliding / global attention is the lever

Gemma-4 does **not** use full global attention on every layer. It interleaves:

- **Sliding-window layers** — most layers attend only to the last `W` tokens (E2B:
  `W = 512`; 26B: `W = 1024`). A token at position 130,000 only ever looks at positions
  `[129,488 … 130,000]`.
- **Global layers** — a minority of layers (E2B: 7 of 35; 26B: 5 of 30, at layers
  5/11/17/23/29) attend to the entire prior.

On a naïve static compile you pay full-width KV on *all* layers anyway, because the graph
is shape-static and doesn't know a sliding layer will never read beyond `W`. **The whole
technique below is about making the static graph exploit what the architecture already
guarantees.**

E2B / 26B architecture facts that matter:

| | E2B | 26B-A4B |
|---|---|---|
| Layers | 35 (28 sliding + 7 global) | 30 (25 sliding + 5 global) |
| Sliding window `W` | 512 | 1024 |
| Q / KV heads | 8 Q / 1 KV (MQA) | 16 Q / 8 KV sliding, 16 Q / 2 KV global |
| head_dim | 256 | 256 sliding / 512 global |
| Native max position | 131,072 (128K) | 262,144 (256K) |
| FFN | dense | dense shared MLP + 128-expert top-8 MoE |

Positions past the trained `max_position_embeddings` are RoPE out-of-distribution and
produce garbage — so 128K (E2B) and 256K (26B) are hard ceilings *before* any memory
consideration. 1M / 2M is not reachable on this family.

---

## 2. Sliding-window KV cap (the ring buffer)

**Idea:** a sliding-window layer with window `W` can never attend to a key older than `W`
positions back. So it does not need a `KV_MAX`-wide buffer — it needs a **ring buffer of
depth `SW_CAP ≥ W`**. Only the *global* layers carry the full `KV_MAX` buffer.

Concretely, per layer:

```
buf_depth(layer) = SW_CAP   if layer is sliding   (small ring, e.g. 512–4096)
                 = KV_MAX   if layer is global    (full context, e.g. 72K)
```

On the 26B, that turns 25 of 30 layers from `KV_MAX`-wide into `SW_CAP`-wide. At 72K
context with `SW_CAP` in the low thousands, the sliding layers cost almost nothing and
only 5 layers pay the full width.

### 2.1 How the ring stays correct

The subtle part is that **the ring reuses physical slots** (`slot = pos % SW_CAP`) while
**position_ids stay absolute** — so rotary embeddings are computed on the true position
and are byte-for-byte identical to a full-KV run. Nothing about RoPE changes; only *where*
a K/V row is stored changes.

At attention time we build a **frontier-aware mask**. Let `frontier` be the highest
absolute position already written into the ring. Ring slot `j` physically holds the token
whose absolute position is:

```
p_j = frontier − ((frontier − j) mod SW_CAP)
```

(the unique in-window residue-`j` position in `(frontier − SW_CAP, frontier]`). A query at
absolute position `q` may read slot `j` **iff**:

```
p_j ≤ q          (causal)
p_j > q − W      (inside the window)
p_j ≥ 0          (actually written)
```

That is exactly the mask in `run_moe2.py::_inputs`:

```python
SC = SW_CAP; jr = torch.arange(SC)
rs = (pos % SC)                                   # ring write slot per query
p_j = fr - ((fr - jr) % SC)                       # abs position stored at each slot
valid = (p_j <= q) & (p_j > (q - SW)) & (p_j >= 0)
slide = torch.where(valid, 0.0, NEG)              # additive mask into the softmax
```

Global layers keep the plain absolute-position causal mask against the full `KV_MAX`
buffer. The one-hot write masks (`oh_full` / `oh_slide`) route each layer's writes to the
right buffer depth.

**Correctness constraint for chunked prefill:** require `SW_CAP ≥ CHUNK + W`, so no
in-window token is overwritten *within* a single prefill chunk before every query in that
chunk has read it.

On the E2B, the same mechanism lives in the NxDI attention path itself
(`swa/attention_base.py::perform_prefix_prefill_windowed_attn`): only the tail `W` rows of
the prior are ever visible, so it **`torch.gather`s exactly those `W` rows** out of the
block-KV cache instead of scoring against the whole prior:

```python
j = torch.arange(W)
abs_prior_pos = P[:, None] - W + j[None, :]        # the only W rows a sliding layer can see
gidx = abs_prior_pos.clamp(0, prior_len-1)...
K_prior = torch.gather(K_prior_full, dim=2, index=gidx)   # (bsz, Hkv, W, D)
V_prior = torch.gather(V_prior_full, dim=2, index=gidx)
prior_mask = (j > i) & (abs_prior_pos >= 0)
```

(A `SWA_FULL_PRIOR=1` env fallback scores the whole prior with the absolute-position
window rule — used as the on-device oracle to cross-check the gather path.)

---

## 3. Chunked prefill

Token-by-token prefill of 130K tokens is unusably slow, and full-width prefill OOMs the
compiler (§1.2). The fix is **chunked (bucketed) prefill**: prefill in fixed `BUCKET`-sized
blocks, each a *single batched forward*, with the block's frontier set to the block end.

- On the E2B block-KV path, chunked prefill compiles at `active = 512` with the long prior
  **gathered** from the block-KV cache. The `(512-active, 131072-prefix)` CTE compiles in
  **409 s / 38 GB host RAM** — versus the ~378 GB OOM of a one-shot 16K prefill — because
  it only scores `512 × 131072` on the 7 global layers and `512 × 512` on the 28 sliding
  layers.
- On the 26B MoE, `pushctx3.py::needle` fills the KV in full `BUCKET`-sized blocks
  (`frontier = block end`), teacher-forces the sub-`BUCKET` remainder through the decode
  graph, then generates.

**Measured throughput:** chunked prefill runs **~300–440 tok/s** on the 26B ladder
(443 tok/s at 49K, 309 tok/s at 72K) and **~1,300 tok/s** on the E2B at 128K —
**~30× faster** than token-by-token, which is what makes needle verification at these
lengths feasible in wall-clock at all.

Key invariant: keep `BUCKET` bounded (`≤ 8192`). The global-layer prefill scratchpad is
`O(BUCKET × KV_MAX)`, so a large bucket reintroduces the compile-OOM you were escaping.

The multi-bucket **token-generation (TKG)** decode path is the same idea applied to decode:
compile several TKG buckets (`2048, 8192, 32768, …, SEQ`) and let the runtime pick the
smallest bucket `> context_len`. On the E2B 128K build this cut decode from **42 → 12
ms/tok (3.5×) at 2K context** with only +62 MiB HBM, because a short chat no longer decodes
on the 132K-wide graph.

---

## 4. `lm_head` column-parallel sharding

The language-model head on Gemma-4 is a `[vocab=262144, hidden]` matrix. Replicated across
every rank it costs **~1.5 GB per core** of HBM that is doing nothing but sitting there.

Replace it with a `ColumnParallelLinear(hidden, vocab, gather_output=True)`: the weight is
sharded `[vocab/TP, hidden]` across ranks (**~185 MB/rank**), and `gather_output=True`
all-gathers the logits so `argmax` over the full vocab still works. (Softcap is applied
after the gather.)

```python
# run_moe3.py::build_module
VOCAB = lm_head.weight.shape[0]
s.head = ColumnParallelLinear(H, VOCAB, bias=False, gather_output=True, dtype=WDT)
```

That frees **~1.3 GB/core** — enough, on the 26B, to push the load-time ceiling from ~49K
up to **72K** context (the difference between the `L49152` and `L73728` rungs of the
ladder).

---

## 5. Measured results

### 5.1 Gemma-4-E2B — 128K on a single chip

Build `~/gemma_pc_128k` (prefix buckets `[512, 4096, 32768, 131072]`, chunk 512,
`pa_block=32 × 4160` blocks; see `swa/gemma_compile_big.py`).

| Metric | Value |
|---|---|
| Device HBM | **15.76 / 16 GB per core** (~1.5% headroom) |
| — tensors (weights+KV) | 8.85 GB |
| — scratchpad | 5.19 GB |
| — code | 1.65 GB |
| Prefill | **130,823 tokens in 100.6 s (~1,300 tok/s)** |
| Decode | 42 ms/tok (~24 tok/s), fixed 132K TKG bucket |
| Needle `ORCHID-7423-FALCON` @ 130,622 back | **RETRIEVED** (also 32K, 100K) |
| Per-chunk CTE `[512,131072]` | 0.472 s |

**Correctness (proven, not hoped):** position-probe vs HuggingFace fp32
(`Gemma4ForConditionalGeneration`, transformers 5.14.1) at prefixes 1024/1536/1400
(including non-block-aligned), query at chunk offset 0/1 so 511/510 prior rows are visible
— **argmax equal in all 24 probes × 2 prefix modes, logits within the bf16 noise floor.**
The window was proven to be *exactly* 512 by a sensitivity control: a deliberately-wrong
`W=511` build lands on 511 with 10–185× margins, so the test can catch a one-row error.
On-device oracle (`SWA_FULL_PRIOR`) cross-check: median KL 1.6e-7.

### 5.2 Gemma-4-26B-A4B (MoE) — the 32K → 49K → 72K ladder

One `inf2.24xlarge`, TP=8, bf16. Each rung is needle-verified with
`ORCHID-7423-FALCON` placed in the head and the question at the tail so retrieval spans the
full context.

| Rung | Technique | Context (ctx) | Needle @ | Decode | Chunked prefill |
|---|---|---|---|---|---|
| Flat | full KV, `KV_MAX=32768` | 32,768 | 32,480 | 9.37 tok/s | — |
| Sliding-cap | ring KV on 25 sliding layers | 49,152 | 15,995 **+** 48,986 | 13.75 tok/s | 443 tok/s |
| Sliding-cap **+ lm_head-shard** | + column-parallel head | **73,728** | 15,995 **+** 71,981 | 11.39 tok/s | 309 tok/s |

The 72K rung is **2.25× the flat ceiling** — a context the flat compile physically cannot
load on the box. Coherence held throughout (correct "Paris", step-by-step 17×24, Rayleigh
scattering).

`lm_head` shard footprint: ColumnParallel `gather_output` ~185 MB/rank vs 1.5 GB
replicated. Scripts: `run_moe2.py` / `pushctx3.py` / `runL2.sh` (49K),
`run_moe3.py` / `pushctx4.py` / `runL3.sh` (72K, lm_head-sharded).

---

## 6. The compiler bug we had to clear first (`islpy`)

None of the above compiled until we fixed a `neuronx-cc 2.27` failure that turned out to be
a **dependency drift**: `islpy 2026.2.1` breaks the compiler; pinning **`islpy==2026.1`**
fixes it. Filed as
[aws-neuron/aws-neuron-sdk#1391](https://github.com/aws-neuron/aws-neuron-sdk/issues/1391)
(reporter `barrhawk`, root-cause comment posted). If you reproduce this work and see a
`neuronx-cc` internal error, check `islpy` first.

**Operational lesson:** never run two NxDI compiles concurrently on one box — the shared
`/tmp/nxd_model` cache gets poisoned with a "cached failed neff". Clear
`/var/tmp/neuron-compile-cache` between builds.

---

## 7. Honest ceiling analysis

**On the E2B, KV is the wall** — 15.76/16 GB at 128K is ~1.5% headroom, and 128K is also
the trained RoPE ceiling. That is the model's real limit; there is no further to push on
this chip without quantization.

**On the 26B MoE, KV is *not* the wall — the fixed expert weights are.** This is the key
finding, and it is easy to get wrong. At TP=8 the 128-expert weights are **~5.7 GB/rank**,
and with overhead ~12–13 GB/core is fixed and resident *before any KV*. That leaves only
**~3.3 GB/core** for the KV cache. Four OOMs confirmed the diagnosis: a 131072 build
compile-OOM'd at 17.29 GB, and load-OOMs hit at 131072 / 98304 / 81920. Sliding-cap and
`lm_head` sharding buy real context, but they cannot move the fixed 12–13 GB expert floor.

> HBM-per-core at 72K was **not** measured directly (`neuron-monitor` returned null on this
> box); the analytical estimate is ~15 GB/core. We do not claim it as measured. The
> *needle retrieval* and *decode throughput* numbers above **are** measured.

**Paths to 128K+ on the 26B (not yet built):**

1. **Expert quantization (int8 / fp8).** Experts `5.7 → 2.85 GB/rank` frees ~2.85 GB/core
   → analytically **~146K reachable on this same `inf2.24xlarge`**. This is the highest-
   leverage next step, because it attacks the actual binding constraint.
2. **TP=16 on a 16-core box (`inf2.48xlarge`).** Halves the per-rank expert weights, but
   needs 24 cores' worth of TP — the 24xlarge has 12, so TP is capped at 8. The 48xlarge
   is currently **quota-blocked** (192 quota; a 256 bump case is open with AWS).

**What is *not* reachable:** native 256K needs the 48xlarge (memory), and 1M/2M is off the
table entirely — RoPE is trained to 262,144 and goes out-of-distribution past it,
independent of memory.

---

## 8. Why this matters

AWS ships the native long-context (segmented-prefill) backend on **Trainium2/Trainium3**
and gates it *off* on Inferentia2. Inferentia2 is NeuronCore-v2 — the **cheapest** AI
accelerator AWS rents. The techniques here recover long-context serving on the cheap chip,
with correctness proven against HF fp32 and needles retrieved at 128K (E2B) / 72K (26B).

The moat is simple to state: **2026 models at long context on 2023 silicon.** Everyone
else needs the newest, scarcest accelerators to do this. We don't.

---

## Reference scripts

Included in this repo:

- **`swa/attention_base.py`** — the patched NxDI attention base with the sliding-window
  chunked/prefix-prefill path (`perform_prefix_prefill_windowed_attn`, the tail-`W`
  `torch.gather`, `SWA_FULL_PRIOR` oracle).
- **`swa/modeling_gemma4.py`** — the Gemma-4 NxDI model with the `_assert_v1_feature_set`
  gate relaxed to allow `is_block_kv_layout` + `is_prefix_caching` + bucketing.
- **`swa/gemma_compile.py`, `swa/gemma_compile_big.py`, `swa/gemma_compile_box.py`** — the
  E2B compile drivers (single-bucket, 128K multi-bucket TKG, and box variants).
- **`swa/gprobe.py`, `swa/gprobe_swa.py`, `swa/hf_ref_gemma.py`, `swa/probe_llama.py`** —
  the correctness probes vs HuggingFace fp32 and the Llama-1B global-prior-mask probe.
- **`run_moe2.py` / `run_moe3.py`** — the 26B MoE wrapper with the sliding-cap ring buffer
  (`SW_CAP`, `_inputs`, `_buf_depth`) and, in `run_moe3`, the `lm_head` column-parallel
  shard.
- **`pushctx3.py` / `pushctx4.py`** — the chunked-prefill needle-in-a-haystack driver for
  the 26B ladder (`needle()`, `build_ids()`).
- **`gemma4_server.py`** — the OpenAI-compatible NxDI serving backend + function-calling
  parser.

Attributed upstream (William McLean / `xbill9`, Apache-2.0), included **unmodified** under
[`reference/xbill9/`](../reference/xbill9/):

- `gemma-4-E2B-_optb_kv.py`, `gemma-4-E2B-_optb_gen.py` — the E2B Option-B KV-aliasing
  build/gen.
- `gemma-4-E4B-_tp_mb.py`, `gemma-4-E4B-_tp_alias_trace.py` — the TP + ModelBuilder wrapper
  template our 12B/26B ports adapt.
- `gemma-4-E2B-_README.md`, `gemma-4-E4B-_README.md` — his original write-ups.

---

*Weights are Google's (Gemma license). Wrapper recipe is William McLean's (`xbill9`,
Apache-2.0). Toolchain is AWS Neuron / NxD Inference. The long-context layer — sliding-cap
KV, chunked prefill, lm_head sharding, and the measured 128K/72K needle results — is ours,
and every number in §5 is from a run artifact, not an estimate.*

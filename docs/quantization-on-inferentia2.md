# Quantizing gemma4 on Inferentia2 — what NeuronCore-v2 accepts, and what it refuses

*A characterization of the quantization surface of neuronx-cc 2.27 / NeuronCore-v2 for the
gemma4 family. Every figure below is either **MEASURED** on a real inf2 instance against a
Hugging Face fp32 reference, or marked **PLAN** — a designed next step, not a run. Knowing
precisely what the silicon refuses is as valuable as knowing what it accepts.*

---

## Summary

| Path | Verdict | Basis |
|---|---|---|
| INT8 weight-only | ✅ **MEASURED — works** | 12B dense, per-output-channel symmetric, coherent, math token-identical to fp32 |
| FP8 (F8E4M3) | 🔴 **MEASURED — refused** | compiler hard-rejects; no fp8 ALU on NeuronCore-v2 |
| 4-bit dense | ⛔ **unavailable** | no FP4/AWQ/GPTQ dequant in dense linear layers |
| W8A8 (int8 activations too) | 🟡 **PLAN** | designed fast-path; not yet compiled |

---

## INT8 weight-only — MEASURED, works

Weight-only INT8, per-output-channel symmetric, compiles and runs coherently on
NeuronCore-v2. On gemma4-12B dense (TP2):

- **Correctness:** math prompts decode token-identical to the fp32 reference.
- **Memory:** ~**9.0 GB/core**, a ≈**44% cut** from the bf16 footprint (~16.1 GB/core). That
  reduction is the whole point — it makes int8-12B **co-resident with E2B on one 32 GB
  device**, a layout bf16 could not fit.
- **Throughput:** **11.2 tok/s** decode, versus **15.29 tok/s** for the bf16 build — about
  **27% slower**.

The slowdown is the tell: this is a **memory win, not a compute win**. The compiler upcasts
int8 weights to bf16 before the matmul, so the Tensor Engine still does the multiply in
bf16 — you pay a dequant cost and gain nothing in the systolic array. What you buy is
resident-footprint headroom, and on a memory-bound box that headroom is what lets a model
fit at all.

## FP8 (F8E4M3) — MEASURED, refused

neuronx-cc 2.27 hard-rejects fp8:

```
NCC_EVRF051 — Data type F8E4M3FN is not supported on TRN1/TRN2
```

There is **no fp8 ALU on NeuronCore-v2**. The Neuron config enum exposes an `F8E4M3`
option, but it sits a level above what the silicon and compiler will honor — the enum
accepting the value does not mean the hardware executes it. fp8 weight storage and compute
are a **Trainium2/3** feature. This is a documented dead-end on this silicon: do not
re-spend engineering time trying to force an fp8 path through inf2.

## 4-bit dense — unavailable

There is no usable 4-bit path for dense layers on this stack. The only 4-bit route wired
into the toolchain is the **gpt-oss MXFP4 MoE expert kernel** — a dequant that lives inside
the mixture-of-experts expert path, not in the dense `nn.Linear` layers gemma4's attention
and MLP projections use. There is **no FP4 / AWQ / GPTQ dequant in the dense linear
layers**.

A dense 4-bit path is therefore not a config flag — it is a **hand-written kernel**: a
fundable, scoped job, not a quick win. We flag it as such so nobody plans around a flag
that does not exist.

## W8A8 — the fast-path PLAN

The measured INT8 weight-only result leaves compute on the table because activations stay
bf16 and the matmul upcasts. NeuronCore-v2's Tensor Engine does **int8 at roughly 2× bf16**
in the systolic array. The designed next step is to **dynamically quantize activations to
int8 as well** (int8 × int8 → int32 accumulate → downcast), which should engage the native
int8 systolic unit and convert the current *memory* win into a *speed* win.

This is **PLAN** — designed, not yet compiled. It is the highest-leverage quantization work
remaining on inf2: it turns the one quantization path the silicon already accepts into a
throughput gain rather than a footprint-only trade.

---

## Provenance notes

- INT8 weight-only figures (9.0 GB/core, 11.2 vs 15.29 tok/s, token-identical math) are
  MEASURED on gemma4-12B dense against an HF `transformers` fp32 reference.
- The FP8 refusal is the verbatim compiler error under neuronx-cc 2.27.
- The 2× int8 Tensor-Engine ratio motivating W8A8 is a NeuronCore-v2 architectural
  property; the W8A8 speedup itself is **not yet measured** and is labeled PLAN
  accordingly.
- No number in this document has been rounded up from an unproven estimate. Where a claim
  is analytical rather than run, it says PLAN.

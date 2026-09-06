# HawkTalk — gemma4 on AWS Inferentia2

**2026 model families, at long context, on the accelerator most teams write off.**

HawkTalk is an elite AI atelier and research shop for practical engineering solutions. We take on the hard, unglamorous inference work that sits just past the edge of the vendor's supported path — model ports, long-context serving, quantization, compiler-level debugging — and we bring it back with numbers attached. This repository is our Inferentia2 line of work: running the full **gemma4 (2026) model family** on AWS NeuronCore-v2, the lowest-cost AI accelerator AWS rents and the one most teams assume cannot serve modern long-context models. It can. We measured it.

> **Status:** Private research repository. Not for redistribution.

---

## What HawkTalk is

A boutique applied-AI research house. Our engagements are the kind where the deliverable is a working artifact, not a slide — inference optimization, model porting to constrained or unusual silicon, long-context serving, and honest performance characterization. Our discipline is provenance: every figure in this repository is either **MEASURED** on real hardware against a real reference, or marked **PLAN**. We keep that line bright on purpose. We do not round up, we do not ship partial work as finished, and when a wall is fundamental we say so and name the silicon that moves it.

---

## Results — the gemma4 family on Inferentia2

gemma4 is a 2026 frontier family (256K native context; E-class 128K). Inferentia2 is NeuronCore-v2, a static-graph, ahead-of-time-compiled accelerator with a narrow supported-model surface. The through-line of this work is running the former on the latter — correctly, at length, and at defensible cost. Correctness is gated against Hugging Face `transformers` fp32 references; every long-context claim carries a named needle position. **SEQ_MATCH** means device greedy decode was token-for-token identical to that fp32 reference — the strongest gate we run.

| Model | Status | Long context | Correctness gate | Throughput (measured) | Layout |
|---|---|---|---|---|---|
| **gemma4-E2B** | ✅ MEASURED + live | 128K end-to-end on **one** inf2 chip; needle retrieved at **130,622 tokens** | argmax == HF fp32 across all probe positions | prefill ~1,300 tok/s; decode 42 ms/tok; **multi-bucket TKG → 12 ms/tok (3.5×)**; graph compiles in 409 s | TP2, 15.76 / 16 GB per core |
| **gemma4-E4B** | ✅ MEASURED | native 128K; needle at **130,622**, found at every length probed (2K/32K/65K/102K/130K) | 16/16 greedy tokens == HF fp32 | prefill 2,000–4,600 tok/s; decode ~21 tok/s | TP4, 15.16 GB per core |
| **gemma4-12B (dense)** | ✅ MEASURED | — | **SEQ_MATCH bit-exact** greedy vs HF fp32 across the full sequence | 15.29 tok/s (bf16); 11.2 tok/s (int8 weight-only) | TP2, ~16.1 GB per core bf16 |
| **gemma4-12B, INT8 weight-only** | ✅ MEASURED | — | math prompts token-identical to fp32 | ~9.0 GB/core (≈44% cut from bf16); co-resident with E2B in one 32 GB device | TP2 |
| **gemma4-26B-A4B (MoE)** | ✅ MEASURED | **72K**; needle at **71,981** (ladder 32K/49K/72K → 32,480 / 48,986 / 71,981) | coherent on held-out prompts (Paris / arithmetic / Rayleigh) | 25 tok/s decode; chunked prefill ~300–440 tok/s | TP-sharded dense-expert, 8.57 GB/core |
| **gemma4-31B (dense)** | 🔴 GPU-only | — | — | measured dead-end on this silicon; served on GPU today | see finding |

### Serving economics — stated plainly

Inferentia2 is our **long-context and control** lane, not a raw cost-per-token win against GPU. We route deliberately rather than sell a benchmark we don't have.

- **inf2, continuous-batching E-class:** ~**$16 / Mtok at 8 seats** (113.7 tok/s @ 8 seats, 5.8× over batch-1) — the premium, long-context lane where single-chip behavior matters.
- **GPU lane (documented separately):** ~**$0.32 / Mtok** for the same class of model — the volume lane.

Both are measured; neither is aspirational. Where GPU is cheaper per token for a given model, we say so.

### The real wall, named

For the 26B MoE, the binding constraint at long context is **fixed expert weight residency, not KV cache** — experts occupy the resident budget, leaving little room for context on a 12-core box. That distinction is the whole game: it tells you exactly which levers move the ceiling and which do nothing.

- **PLAN:** INT8 expert weights → an estimated ~146K context on the current box.
- **PLAN:** inf2.48xlarge (24 cores, TP16) → the RoPE ceiling at 256K. Quota case open with AWS.
- **Fact, not aspiration:** 2M context is not reachable on this positional scheme without RoPE re-scaling and fine-tuning. We do not imply otherwise.

### The dense-31B finding (a negative result worth publishing)

The claim that the E2B port applies directly to dense 31B, "same architecture," is **false**, and we measured why: E2B is the *efficient* variant (per-layer embeddings, KV-share, double-wide MLP); gemma4-31B is dense, with checkpoint keys the E2B remap does not expect and a `k_proj` shape mismatch on the full-attention layers. It is a real port, not a wrapper — and its home is the GPU lane. We publish this so nobody re-spends on it as a "quick win."

---

## Quantization findings (neuronx-cc 2.27 / NeuronCore-v2)

Precise knowledge of what the silicon refuses is as valuable as knowing what it accepts.

- **INT8 weight-only — MEASURED, works.** Per-output-channel symmetric. Real memory reduction, coherent output, math token-identical to fp32. It is a *memory* win, not a compute win — the compiler upcasts int8→bf16 for the matmul, so decode runs ~27% slower. It is enough to co-resident int8-12B with E2B on a 32 GB device, which bf16 could not.
- **FP8 (F8E4M3) — MEASURED, refused.** neuronx-cc 2.27 hard-rejects it: `NCC_EVRF051 — Data type F8E4M3FN is not supported on TRN1/TRN2`. There is **no fp8 ALU on NeuronCore-v2**; the config enum that appears to accept F8E4M3 sits a level above what the silicon and compiler will honor. fp8 weight storage is a Trainium2/3 feature — do not re-spend on it here.
- **4-bit dense — unavailable.** The 4-bit quant path is wired only through the gpt-oss MXFP4 MoE expert kernel; there is no FP4/AWQ/GPTQ dequant in the dense linear layers. A dense 4-bit path needs a hand-written kernel — a fundable job, not a flag.
- **W8A8 — the fast-path PLAN.** NeuronCore-v2's Tensor Engine does int8 at roughly 2× bf16. Dynamically quantizing activations to int8 as well (int8×int8 → int32 accumulate → downcast) should trigger the native int8 systolic unit and turn the current memory win into a speed win. Designed, not yet compiled — flagged as PLAN.

---

## Selected contribution to the community

**`islpy` dependency drift breaking `neuronx-cc` 2.27 at seq_len > 4096 — root-caused and filed as [aws-neuron/aws-neuron-sdk#1391](https://github.com/aws-neuron/aws-neuron-sdk/issues/1391).**

Compiles failed with `NCC_ISMP902` for any model past 4,096 tokens, while short buckets compiled cleanly — a signature that reads like a model bug and is not. Root cause: neuronx-cc 2.27 pins `islpy~=2026.1`, but pip resolves that to `islpy 2026.2.1`, released *after* the compiler, whose changed ISL routine the compiler's pass depends on. The one-line fix is `pip install islpy==2026.1`. We isolated it, verified it across models, and filed it upstream with a minimal reproduction so the next team loses minutes, not a night. A shared, correct toolchain is worth more than a private workaround.

---

## Methods & recipes

The reproducible engineering lives beside the results. The governing principle throughout: **wrap the reference model's eager attention and swap only the KV path**, rather than hand-rolling attention that the AOT compiler will trace subtly wrong. Hand-porting gemma4 attention produced coherent-looking but subtly-wrong outputs (a layer-21 SRAM hazard among them); wrapping does not. **Wrap, don't port.**

- **The HF-eager wrapper recipe** — a repeatable procedure for bringing any gemma4 variant (E2B / E4B / 12B / 26B-A4B) up coherently on inf2: TP-shard the linears including `lm_head`, one-hot scatter KV into static device-resident buffers with `input_output_aliases`, hand-load the layer-scalar buffers, softcap and embeddings off-device.
- **Long-context add-ons** — sliding-window chunked-prefill on block-structured KV: sliding-window layers cap their KV as a ring at the window, only global layers carry the full cache, absolute `position_ids` keep rotary unchanged. This is the segmented-prefill capability AWS ships natively only on Trn2/Trn3, reproduced on NeuronCore-v2. Window width is proven to the row — a deliberately mis-sized control build lands on the wrong token, so the test can catch a one-row error.
- **Per-model adaptations** — 12B `k_eq_v` global layers; 26B `DenseExperts` + `SPMD` scatter routing.
- **Correctness discipline** — the **SEQ_MATCH-at-many-positions** law: verify decode against an HF fp32 reference at many positions, not one (a one-position pass on 12B was a false positive), and carry a needle position on every long-context claim.
- **Compiler hygiene** — AOT static graphs with bucketing; one compile per box; the `islpy` pin above.

See the per-model plan documents and runbooks in this repo for step-by-step procedures.

---

## Prior art & attribution

The base serving wrapper is the work of **William McLean ([@xbill9](https://github.com/xbill9))** — Google Developer Expert and AWS Community Builder — released under Apache 2.0. Every gemma4 model we run coherently on Neuron sits on his wrap-HF-eager recipe; HawkTalk's contribution is the long-context layer, the correctness discipline, and the 12B / 26B adaptations built on top. Public sources: `xbill9/gemma-4-E2B-it-inferentia2` and the E4B repository on Hugging Face. Any downstream release, write-up, or talk that comes out of this work names him first. Credit to strong prior art is how serious shops work.

---

## Acknowledgments

With thanks to the **[internet.dev](https://internet.dev)** collective and community for their support and camaraderie. Good work is easier, and better, in good company.

---

## Work with us

HawkTalk takes on practical AI engineering engagements — model porting to constrained or unusual silicon, long-context and continuous-batching serving, quantization, and compiler-level debugging on Inferentia, Trainium, and GPU — delivered with the same measured-artifact discipline you see in this repository. If you have a frontier model and a bill you would like to change, we would like to hear about it.

**[hawktalk.ai](https://hawktalk.ai)**

---

*Model, silicon, and vendor-stack references are the trademarks of their respective owners.*

---
license: apache-2.0
base_model: google/gemma-4-E4B-it
tags:
  - gemma
  - gemma-4
  - aws-inferentia
  - inferentia2
  - neuron
  - torch-neuronx
  - neuronx-distributed
  - tensor-parallel
  - text-generation
library_name: torch-neuronx
pipeline_tag: text-generation
---

# Gemma-4-E4B-it on AWS Inferentia2 — Tensor-Parallel (TP=2)

Compiled **AWS Neuron** artifacts that run [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
**coherently across both NeuronCores of a single Inferentia2 device** (`inf2.8xlarge`), at
**~32–34 tokens/sec**, with greedy decode **token-for-token identical to the CPU reference**.

This is a **runnable inference port**, not a fine-tune — the weights are Google's, unmodified.
What's new is the *tensor-parallel compilation recipe*: a way to shard Gemma-4-E4B across two
16 GB cores and keep the KV cache device-resident, which the vendor stack
(`optimum-neuron` / NxD) cannot currently do for this model.

> **Companion:** the smaller [`E2B`](https://huggingface.co/google/gemma-4-E2B-it) fits a
> single core and uses a different (simpler) design. **E4B is ~2× larger and does not fit one
> core — it requires this TP=2 build.**

## TL;DR

| | |
|---|---|
| Base model | `google/gemma-4-E4B-it` (~8B params, 4B *effective* via MatFormer + Per-Layer Embeddings), Apache-2.0 |
| Hardware | AWS Inferentia2 — **`inf2.8xlarge`**, both NeuronCores (**TP = 2**, mandatory) |
| Precision | bf16 compute; each rank holds ~half the weights (~7.7 GB/core) |
| Throughput | **33.4 tok/s** @ 512-ctx, **31.8 tok/s** @ 2048-ctx (~31 ms/token), measured |
| Context | shipped at `KV_MAX=512/BUCKET=128` and `KV_MAX=2048/BUCKET=512` |
| Output parity | Greedy decode **token-for-token identical** to the CPU float reference (`SEQ_MATCH True`) |

## ⚡ Update — on-device prefill (recommended build)

The original build seeds the prompt with a **host-CPU prefill** (~1.4–1.6 s to first token). A newer
build runs **prefill on-device** via a single weight-sharing `ModelBuilder` trace — prefill and decode
as two buckets of *one* resident model, bf16 sharded weights, on-device aliased KV:

| | host-CPU prefill (`tp_alias`) | **device prefill (`tp_mb`, bf16)** |
|---|---|---|
| First-token latency | ~1.4–1.6 s | **~0.16 s** (≈9× faster) |
| Decode | 31–34 tok/s | **~39 tok/s** |
| Correctness | `SEQ_MATCH True` | `SEQ_MATCH True` |

**Fastest path — prebuilt Docker image** (bundles the compiled model + server):

```bash
docker run -d --device /dev/neuron0 --ipc=host -p 8080:8080 \
  xbill9/gemma4-optb-e4b:tp2-devprefill-512
curl -s localhost:8080/generate -d '{"prompt":"What is AWS Inferentia?","max_tokens":60}'
```

Serves OpenAI-compatible routes (`/v1/chat/completions`, `/v1/completions`) plus `/generate`,
`/health`, `/metrics`. To build it yourself: `MB_WDTYPE=bf16 KV_MAX=512 KV_BUCKET=128 MB_SAVE=mb_e4b_512_bf16.pt python tp_mb.py`
(compiles + saves the model), then serve with `optb_server_mb.py` / package with `Dockerfile.mb`.

**Run on a single `inf2.xlarge` (¼ the price) — slim image.** The `inf2.xlarge` has the *same 2
NeuronCores* as the `inf2.8xlarge`, just 16 GB host RAM instead of 128. The **slim** server
(`optb_server_slim.py`) fits that by loading the host embedding model in bf16 and dropping the
transformer layers (they run on-device), so the host footprint is a few GB:

```bash
docker run -d --device /dev/neuron0 --ipc=host -p 8080:8080 \
  xbill9/gemma4-optb-e4b:slim-devprefill
```

Validated on an `inf2.xlarge`: coherent output, device prefill ~0.11 s, ~36 tok/s, ~8 GB host RAM.
(Compile still needs an `inf2.8xlarge`; the slim image reuses the same compiled model.)

> The one non-obvious fix that made device prefill correct: Gemma-4's per-layer `layer_scalar`
> is a **buffer**, and NxD's `ModelBuilder` weight-sharding loads *parameters* only — so it must be
> copied from the checkpoint by hand, else every layer over-scales ~16× into garbage.

## Why single-core doesn't work (and this does)

A single Option-B neff for E4B materializes **~15.4 GB of fp32 model constants**, and a second
neff on the same core tips it past the **16 GB** NeuronCore budget → `NRT_RESOURCE / status=4
Allocation Failure`. bf16 shrinks the *on-disk* neff but **not** the on-device constants, so it
doesn't rescue you. The only way to fit E4B is to **shard the model across both cores (TP=2)**,
so each rank holds ~half the weights (~7.7 GB/core).

Tensor-parallel then forces three problems that don't exist single-core. This build (`tp_alias`)
solves all three:

1. **Co-residency.** Under TP a neff spans *both* cores, so you can't park prefill on core 0 and
   decode on core 1. **Fix:** only the **decode** neff is resident on-device; **prefill runs on
   the host CPU** as a one-time seed. One neff resident → fits.
2. **Per-rank KV.** Each rank computes a *different* KV head, but NxD's `forward()` returns only
   rank 0. **Fix:** KV is kept **device-resident** via `input_output_aliases` (never round-tripped
   through the host), and the initial seed is sliced head-`r` → rank `r`.
3. **GQA head-sharding.** E4B has 8 query / 2 KV heads. Naïvely halving `num_key_value_groups`
   scrambles which query attends which KV head (the "4-vs-8" `repeat_kv` mismatch). **Fix:**
   column-shard k/v to **1 head per rank** and **keep `num_key_value_groups`** unchanged.

## Architecture

- **Sharding.** `q_proj/o_proj/k_proj/v_proj/gate/up/down` are Column/Row-parallel across TP=2
  (NxD `ColumnParallelLinear`/`RowParallelLinear`); k/v shard to `nkv//TP = 1` head/rank.
- **Decode neff.** Single-token forward against a fixed `KV_MAX` buffer. The K/V buffers are
  **device-resident `nn.Parameter`s aliased as graph I/O** (`input_output_aliases`), updated
  in place each step via a one-hot masked write (`buf*(1-oh)+k*oh`, trace-safe arithmetic).
- **Prefill (host, CPU).** The fp32 reference model runs the prompt once on the host, producing
  the full KV cache + first token; the cache is sliced per-rank and copied into the decode neff's
  device parameters. This costs a one-time CPU pass per request (see Limitations).
- **Recipe basics** (shared with the E2B Option-B port): eager attention, host-side PLE
  (Per-Layer Embedding) lookup, plain `DynamicCache` at trace time, explicit tanh-GELU + attention
  softcap, language model + head as real submodules.

## Files

| File | What it is |
|---|---|
| `tpa_dec_512.tar.gz` | Decode neff @ `KV_MAX=512` — serialized parallel model (`tp_0.pt` + `tp_1.pt`, one per core) |
| `tpa_dec_2048.tar.gz` | Decode neff @ `KV_MAX=2048` |
| `tp_alias_trace.py` | Compiles the sharded decode neff (`parallel_model_trace` + aliases → `parallel_model_save`) |
| `tp_alias_run.py` | CPU-seed prefill + device greedy decode; validates `SEQ_MATCH` vs the CPU reference and measures tok/s |
| `mb_e4b_512_bf16.pt` | **device-prefill model** @ 512/128 — one serialized `NxDModel` (prefill+decode buckets, bf16 sharded weights + aliased KV), load with `torch.jit.load` |
| `tp_mb.py` | compiles + saves the device-prefill model via NxD `ModelBuilder` (`MB_WDTYPE=bf16 MB_SAVE=... python tp_mb.py`) |
| `optb_server_mb.py` | device-prefill HTTP server (OpenAI routes + `/generate`); loads `mb_e4b_512_bf16.pt`, host embeddings only |
| `Dockerfile.mb` | packages the device-prefill build → `xbill9/gemma4-optb-e4b:tp2-devprefill-512` |

The neffs embed the base weights in bf16, so they are Apache-2.0 derivatives of
`google/gemma-4-E4B-it` — see **License** below.

## Run from these files

**Requirements:** an AWS **`inf2.8xlarge`** (2 NeuronCores) with the Neuron runtime, plus:

```
transformers==5.13.0
torch-neuronx==2.8.0  (SDK 2.23, neuronx-cc 2.23.6484.0)
neuronx-distributed>=0.17
```

```bash
# unpack a decode neff
tar xzf tpa_dec_512.tar.gz -C /workspace     # -> /workspace/tpa_dec

# validate + measure (CPU-seed prefill + device decode)
export KV_MAX=512 KV_BUCKET=128 MAXNEW=30
python tp_alias_run.py
# -> DEV GEN: 'The capital of France is **Paris**.'
#    SEQ_MATCH True
#    TP+ALIAS DECODE tok/s: 33.4 | 30 ms/tok
```

**Recompile for a different context window** (sizes are baked into the neff):

```bash
KV_MAX=2048 KV_BUCKET=512 python tp_alias_trace.py   # -> /workspace/tpa_dec
```

Compilation runs `neuronx-cc` across both ranks concurrently and peaks past host RAM on a
128 GB box — **add swap before compiling** (a 55 GB swapfile is enough); the resulting neff runs
fine without swap.

## Benchmarks (measured, `inf2.8xlarge`, us-west-2)

| Context (`KV_MAX/BUCKET`) | tok/s | ms/token | `SEQ_MATCH` |
|---|---|---|---|
| 256 / 64 | 33.8 | 30 | ✅ True |
| 512 / 128 | 33.4 | 30 | ✅ True |
| 2048 / 512 | 31.8 | 31 | ✅ True |

Per-token cost is essentially flat across context sizes because the KV cache is device-resident
(no host round-trip). Correctness is exact: device greedy argmax matches the CPU float reference
token-for-token, e.g. *"The capital of France is **Paris**."*

## Limitations

- **`inf2.8xlarge` only** — needs 2 NeuronCores (TP=2). Does not run on `inf2.xlarge`.
- Prefill: the `tp_alias` build seeds on the host CPU (~1.4–1.6 s first token). **The `tp_mb`
  device-prefill build above removes this** (~0.16 s first token, prefill on-device via
  weight-sharing buckets) and is the recommended path for a low-latency endpoint.
- **Batch size 1**, single-stream greedy/sampled decode. No continuous batching / paged attention.
- Context is fixed at compile time; longer needs a recompile (compile needs swap).
- Compiled specifically for Inferentia2 + the Neuron SDK versions pinned above.

## License & attribution

- **Base model:** `google/gemma-4-E4B-it`, © Google, **Apache-2.0**.
- **These artifacts** (compiled neffs + scripts) are Apache-2.0 derivatives; the neffs contain the
  base weights in bf16. Redistributed under Apache-2.0 with attribution to Google — include the
  upstream `LICENSE`/`NOTICE` when redistributing.
- The TP=2 `tp_alias` recipe (on-device aliased KV, CPU-seed prefill, GQA head-sharding) is
  released under Apache-2.0.

*Not affiliated with or endorsed by Google or AWS. Provided for research/testing.*

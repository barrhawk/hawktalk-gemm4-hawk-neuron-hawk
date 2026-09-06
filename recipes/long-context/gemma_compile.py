#!/usr/bin/env python
"""Compile gemma4-E2B with block-KV + prefix caching (emulated chunked prefill), TP2 bf16.
Mirrors the Llama-1B probe config exactly (chunk 512, prefix buckets 512/1024/1536, seq 2176)."""
import os, sys, time, glob, traceback
sys.path.insert(0, "/home/ubuntu/xfer/gemma4_nxdi")
import torch
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from modeling_gemma4 import Gemma4InferenceConfig, Gemma4NeuronConfig, NeuronGemma4ForCausalLM

MODEL = os.environ.get("GMODEL", "/home/ubuntu/gemma/gemma-4-E2B-it-trimmed")
OUT = os.environ.get("GOUT", "/home/ubuntu/gemma_pc")
CTE = [int(x) for x in os.environ.get("CTE_BUCKETS", "512,2048").split(",")]
PFX = [int(x) for x in os.environ.get("PREFIX_BUCKETS", "512,1024,1536").split(",")]

t0 = time.time()
nc = Gemma4NeuronConfig(
    tp_degree=2, batch_size=1, seq_len=2176, max_context_length=2048,
    torch_dtype=torch.bfloat16,
    enable_bucketing=True,
    context_encoding_buckets=CTE, prefix_buckets=PFX, token_generation_buckets=[2176],
    is_block_kv_layout=True, pa_block_size=32, pa_num_blocks=128, is_prefix_caching=True,
    attn_kernel_enabled=False,
    save_sharded_checkpoint=True,
)
cfg = Gemma4InferenceConfig(nc, load_config=load_pretrained_config(MODEL))
print("CFG buckets", cfg.neuron_config.buckets, "layer_types sliding=%d full=%d" % (
    sum(t == "sliding_attention" for t in cfg.layer_types), sum(t != "sliding_attention" for t in cfg.layer_types)),
    "sliding_window", cfg.sliding_window, flush=True)
m = NeuronGemma4ForCausalLM(MODEL, cfg)
m.compile(OUT)
assert os.path.exists(os.path.join(OUT, "neuron_config.json"))
# layer_scalar bound check (port's mandatory gate)
found = {}
for p in glob.glob(os.path.join(OUT, "**/*.safetensors"), recursive=True):
    from safetensors.torch import load_file
    for k, v in load_file(p).items():
        if k.endswith("layer_scalar"):
            found[k] = float(v.reshape(-1)[0])
ones = [k for k, v in found.items() if abs(v - 1.0) < 1e-6]
print("layer_scalar entries", len(found), "ones", len(ones))
assert found and not ones, "layer_scalar did not bind"
print("COMPILE_DONE in %.0fs" % (time.time() - t0), flush=True)

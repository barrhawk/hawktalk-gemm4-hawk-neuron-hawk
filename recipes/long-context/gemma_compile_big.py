#!/usr/bin/env python
"""Compile SWA-patched gemma4-E2B for LONG context: block-KV + prefix caching, chunk C=512 with large prefix buckets."""
import os, sys, time, glob, resource
sys.path.insert(0, "/workspace/gemma4_nxdi")
import torch
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from modeling_gemma4 import Gemma4InferenceConfig, Gemma4NeuronConfig, NeuronGemma4ForCausalLM

MODEL = os.environ.get("GMODEL", "/workspace/real-gemma4-E2B-it")
OUT = os.environ.get("GOUT", os.path.expanduser("~/gemma_pc_128k"))
CTE = [int(x) for x in os.environ.get("CTE_BUCKETS", "512").split(",")]
PFX = [int(x) for x in os.environ.get("PREFIX_BUCKETS", "512,4096,32768,131072").split(",")]
SEQ = int(os.environ.get("SEQ_LEN", "132096"))
MAXCTX = int(os.environ.get("MAX_CTX", "131584"))
# MULTI-BUCKET TKG: one token-generation NEFF per entry. NxDI turns each b into a
# 2-D bucket [1, b] (autobucketing.generate_buckets_for_tkg) and at runtime picks the
# smallest b with b > computed_context_len (model_wrapper.get_target_2d_bucket_for_prefix_caching,
# TKG branch), so a 2K chat decodes on the 2048-wide graph instead of the 132K one.
TKG = [int(x) for x in os.environ.get("TKG_BUCKETS", "2048,8192,32768,%d" % SEQ).split(",")]
NBLK = int(os.environ.get("PA_NUM_BLOCKS", "4160"))
assert TKG == sorted(set(TKG)) and TKG[-1] == SEQ, "TKG buckets must be ascending and end at SEQ_LEN (positions >= last bucket cannot decode)"
assert all(b % 32 == 0 for b in TKG), "TKG buckets must be multiples of pa_block_size=32 (block_table is padded to bucket//32)"
assert TKG[0] >= 128, "smallest TKG bucket must be >= 128 (prefix-CTE vs TKG is told apart by q_len < 128)"

print("SWA_FULL_PRIOR env =", os.environ.get("SWA_FULL_PRIOR", "<unset>"), "CTE", CTE, "PFX", PFX, "SEQ", SEQ, "MAXCTX", MAXCTX, "TKG", TKG, "NBLK", NBLK, flush=True)
t0 = time.time()
nc = Gemma4NeuronConfig(
    tp_degree=2, batch_size=1, seq_len=SEQ, max_context_length=MAXCTX,
    torch_dtype=torch.bfloat16,
    enable_bucketing=True,
    context_encoding_buckets=CTE, prefix_buckets=PFX, token_generation_buckets=TKG,
    is_block_kv_layout=True, pa_block_size=32, pa_num_blocks=NBLK, is_prefix_caching=True,
    attn_kernel_enabled=False,
    save_sharded_checkpoint=True,
)
cfg = Gemma4InferenceConfig(nc, load_config=load_pretrained_config(MODEL))
print("CFG buckets", cfg.neuron_config.buckets, "sliding_window", cfg.sliding_window, "max_pos", cfg.max_position_embeddings, flush=True)
m = NeuronGemma4ForCausalLM(MODEL, cfg)
m.compile(OUT)
assert os.path.exists(os.path.join(OUT, "neuron_config.json"))
found = {}
for p in glob.glob(os.path.join(OUT, "**/*.safetensors"), recursive=True):
    from safetensors.torch import load_file
    for k, v in load_file(p).items():
        if k.endswith("layer_scalar"):
            found[k] = float(v.reshape(-1)[0])
assert found and not [k for k, v in found.items() if abs(v - 1.0) < 1e-6], "layer_scalar did not bind"
print("driver maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "children maxrss MB", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)
print("COMPILE_DONE in %.0fs" % (time.time() - t0), flush=True)

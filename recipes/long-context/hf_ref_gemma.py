#!/usr/bin/env python
"""HF fp32 CPU reference for gemma4-E2B chunked-prefill probe. Writes ids + per-prefix last logits + greedy-8."""
import os, sys, time, json, torch
from transformers import AutoTokenizer, AutoConfig

MP = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
N = 2048
ENDS = [512, 1024, 1536, 1912, 2048]
GEN_FROM = [2048, 1912]

tok = AutoTokenizer.from_pretrained(MP)
text = ""
for f in ("chat_template.jinja", "tokenizer_config.json", "config.json"):
    text += open(os.path.join(MP, f)).read() + "\n"
text = text * 3
ids = tok(text, return_tensors="pt").input_ids[:, :N]
assert ids.shape[1] == N, ids.shape
print("ids", ids.shape, "first5", ids[0, :5].tolist(), "bos", tok.bos_token_id, flush=True)
torch.save(ids, os.path.join(OUT, "ids.pt"))

cfg = AutoConfig.from_pretrained(MP)
print("config model_type", cfg.model_type, "archs", getattr(cfg, "architectures", None), flush=True)
t0 = time.time()
try:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.float32)
except Exception as e:
    print("AutoModelForCausalLM failed:", repr(e)[:300], "-> trying ConditionalGeneration", flush=True)
    from transformers import Gemma4ForConditionalGeneration
    model = Gemma4ForConditionalGeneration.from_pretrained(MP, dtype=torch.float32)
model.eval()
print("loaded", type(model).__name__, "fp32 in %.1fs" % (time.time() - t0), flush=True)
with torch.no_grad():
    t0 = time.time()
    out = model(input_ids=ids)
    lg = out.logits[0].float()
    print("fwd %d in %.1fs" % (N, time.time() - t0), flush=True)
    for e in ENDS:
        last = lg[e - 1].clone()
        torch.save(last, os.path.join(OUT, "hf_last_%d.pt" % e))
        print("hf_last_%d argmax=%d top5=%s" % (e, int(last.argmax()), last.topk(5).indices.tolist()), flush=True)
    for s in GEN_FROM:
        t0 = time.time()
        gen = model.generate(input_ids=ids[:, :s], max_new_tokens=8, do_sample=False)
        g8 = gen[0, s:].clone()
        torch.save(g8, os.path.join(OUT, "hf_gen8_%d.pt" % s))
        print("gen8 from %d in %.1fs: %s %r" % (s, time.time() - t0, g8.tolist(), tok.decode(g8)), flush=True)
print("HF_REF_DONE", flush=True)

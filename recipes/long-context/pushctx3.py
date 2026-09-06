"""Flat xbill9 MoE wrapper, LARGE context via small-BUCKET prefill + teacher-forced decode extension.
KV_BUCKET=64 (small -> cheap prefill NEFF, no O(BUCKET*MAX) scratchpad blowup).
KV_MAX=N (target context; static KV buffer size). A model compiled at MAX handles any ctx<=MAX.
Needle: prefill first 64 tokens (needle sentence lives here), teacher-force the rest one-by-one
through the decode graph (each writes its KV at its position), then generate + check passphrase.
NEEDLE_LENS env = comma list of context lengths to test on this one compiled model."""
import os, sys, time, json, subprocess
os.environ.setdefault("MODEL_DIR","/workspace/g4_26b")
import torch
import run_moe2 as R
from transformers import AutoTokenizer
from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance
import neuronx_distributed.trace.spmd  # noqa

NEG=torch.finfo(torch.float32).min
TP=R.TP; BUCKET=R.BUCKET; MAX=R.MAX
print(f"=== pushctx2 KV_MAX={MAX} KV_BUCKET={BUCKET} TP={TP} ===",flush=True)
assert BUCKET<=8192, "keep BUCKET bounded (global prefill scratchpad ~ O(BUCKET*MAX) on the 5 global layers)"

_mm,rlang,_h,softcap,NONSHARED,LINFO,SW=R._discover()
tok=AutoTokenizer.from_pretrained(R.MP)
ec=_mm.generation_config.eos_token_id; EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
print(f"discover: {len(NONSHARED)} nonshared, SW={SW}, softcap={softcap}",flush=True)

seed=[2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107]
pre_in=R._inputs(rlang, seed+[0]*(BUCKET-len(seed)), SW, NEG, list(range(BUCKET)))
dec_in=R._inputs(rlang, [seed[-1]], SW, NEG, [len(seed)])

class GemmaInstance(BaseModelInstance):
    def __init__(self): self.module=None; self.input_output_aliases=[{}]
    def load_module(self):
        self.module, aliases = R.build_module(); self.input_output_aliases=[aliases]
    def get(self, bucket_rank, **kwargs): return self.module, self.input_output_aliases[0]

save_to=os.environ.get("MB_SAVE")
load_from=os.environ.get("MB_LOAD")
t_c0=time.time()
if load_from:
    print("loading",load_from,flush=True)
    model=torch.jit.load(load_from)
    model.nxd_model.initialize_with_saved_weights(torch.tensor([0],dtype=torch.int32))
    compile_s=0
else:
    inst=GemmaInstance()
    mb=ModelBuilder(router=None, tp_degree=TP, checkpoint_loader=R.checkpoint_loader, compiler_workdir="/workspace/mb_wd")
    mb.add("prefill", inst, [pre_in], compiler_args=R.CARGS)
    mb.add("decode",  inst, [dec_in], compiler_args=R.CARGS)
    print("tracing (compile) ...",flush=True)
    model=mb.trace(initialize_model_weights=True)
    compile_s=time.time()-t_c0
    print(f"MB_TRACED compile_wall_s={compile_s:.0f}",flush=True)
    if save_to:
        torch.jit.save(model, save_to); print("MB_SAVED",save_to,flush=True)

def dcall(a):
    r=model(*a); return r[0] if isinstance(r,(tuple,list)) else r

def hbm_snapshot():
    try:
        p=subprocess.run(["neuron-monitor"],timeout=9,capture_output=True,text=True)
        for line in reversed(p.stdout.strip().splitlines()):
            line=line.strip()
            if not line.startswith("{"): continue
            try: j=json.loads(line)
            except Exception: continue
            best=0
            for r in j.get("neuron_runtime_data",[]):
                nd_=r.get("report",{}).get("memory_used",{}).get("neuron_runtime_used_bytes",{}).get("neuron_device",0)
                if nd_>best: best=nd_
            if best>0: return best
    except Exception as e:
        print("neuron-monitor failed:",e,flush=True)
    return None

def feed_prompt(pids):
    """Coherence prompts are short: teacher-force token-by-token via the decode graph so we never write
    pad tokens into the sliding ring buffer (which would corrupt the frontier-based sliding mask)."""
    n0=len(pids); t0=time.time(); l1=None
    for p in range(n0):
        l1=dcall(R._inputs(rlang,[pids[p]],SW,NEG,[p]))
    return l1, time.time()-t0, n0

def gen_short(msg,maxnew=48):
    d=tok.apply_chat_template([{"role":"user","content":msg}],add_generation_prompt=True)
    pids=d if isinstance(d,list) else d["input_ids"]
    lg,pf,n0=feed_prompt(pids)
    first=int(lg[0,-1].argmax()); seq=[first]; cur=n0; td=time.time(); nd=0
    for _ in range(maxnew):
        if seq[-1] in EOS: break
        l1=dcall(R._inputs(rlang,[seq[-1]],SW,NEG,[cur])); seq.append(int(l1[0,0].argmax())); cur+=1; nd+=1
    dt=time.time()-td
    return tok.decode([x for x in seq if x not in EOS],skip_special_tokens=True), pf, (nd/dt if dt>0 else 0), nd

PASS="ORCHID-7423-FALCON"
HEAD=f"IMPORTANT FACT: The secret passphrase is {PASS}. Remember it. "
FILL=" The mountain river flowed quietly past the old stone bridge while autumn leaves drifted down over the mossy rocks."
TAIL=" QUESTION: What is the secret passphrase stated at the very beginning? Reply with only the passphrase."

def build_ids(target):
    def tl(body):
        d=tok.apply_chat_template([{"role":"user","content":body}],add_generation_prompt=True)
        ids=d if isinstance(d,list) else d["input_ids"]; return len(ids),ids
    base,_=tl(HEAD+TAIL); unit=max(1,(tl(HEAD+FILL+TAIL)[0]-base))
    k=max(1,int((target-base)/unit)); L,ids=tl(HEAD+FILL*k+TAIL)
    while L>target and k>1: k=int(k*0.92)-1; k=max(1,k); L,ids=tl(HEAD+FILL*k+TAIL)
    while L<target: k+=max(1,int((target-L)/unit)); L,ids=tl(HEAD+FILL*k+TAIL)
    if L>target: L,ids=tl(HEAD+FILL*max(1,k-1)+TAIL)
    return ids

def needle(target, maxnew=24, progress_every=8):
    """CHUNKED prefill: fill KV in full BUCKET-sized blocks (each a single batched forward, frontier=block end)
    then teacher-force the <BUCKET remainder, then generate. Needle lives in the first block (head)."""
    ids=build_ids(target); n=len(ids); assert n+maxnew < MAX, f"n={n}+{maxnew}>=MAX={MAX}"
    head_len=len(tok(HEAD,add_special_tokens=False)["input_ids"])
    t0=time.time(); l1=None; c=0; nb=0
    while c+BUCKET<=n:
        blk=ids[c:c+BUCKET]; fr=c+BUCKET-1
        l1=dcall(R._inputs(rlang,blk,SW,NEG,list(range(c,c+BUCKET)),fr)); c+=BUCKET; nb+=1
        if nb%progress_every==0:
            el=time.time()-t0; print(f"    prefill {c}/{n} {c/el:.0f}tok/s",flush=True)
    pf_el=time.time()-t0; pf_tps=c/pf_el if pf_el>0 else 0
    for p in range(c,n):                                    # remainder via decode graph (frontier=p)
        l1=dcall(R._inputs(rlang,[ids[p]],SW,NEG,[p]))
    seq=[int(l1[0,-1].argmax())]; cur=n                     # l1[0,-1] = next-token logits after last real token
    for _ in range(maxnew):
        if seq[-1] in EOS: break
        g=dcall(R._inputs(rlang,[seq[-1]],SW,NEG,[cur])); seq.append(int(g[0,0].argmax())); cur+=1
    txt=tok.decode([x for x in seq if x not in EOS],skip_special_tokens=True)
    ok=PASS.replace("-","") in txt.replace("-","").replace(" ","").upper()
    print(f"  NEEDLE ctx={n} needle@~{head_len} prefill={pf_tps:.0f}tok/s : {txt!r} -> OK={ok}",flush=True)
    return {"ctx":n,"ok":bool(ok),"prefill_tps":round(pf_tps,1),"gen":txt}

print("=== COHERENCE ===",flush=True)
coh={}
for msg,key in [("What is the capital of France?","paris"),
                ("What is 17 times 24? Show your reasoning step by step.","math"),
                ("Write one sentence about why the sky is blue.","sky")]:
    txt,pf,tps,nd=gen_short(msg)
    print(f"  [{key}] pf={pf*1000:.0f}ms {tps:.1f}tok/s : {txt!r}",flush=True); coh[key]=(txt,tps)
paris_ok="paris" in coh["paris"][0].lower()
rayleigh_ok="rayleigh" in coh["sky"][0].lower() or "scatter" in coh["sky"][0].lower()
print(f"COH paris={paris_ok} rayleigh={rayleigh_ok} tps={coh['math'][1]:.2f}",flush=True)

LENS=[int(x) for x in os.environ.get("NEEDLE_LENS", f"{min(4096,MAX//2)},{MAX-128}").split(",")]
print(f"=== NEEDLE sweep {LENS} ===",flush=True)
res=[]
for L in LENS:
    res.append(needle(L))
    hb=hbm_snapshot()
    if hb: print(f"  HBM after ctx~{L}: {hb} bytes -> {hb/8/1e9:.2f} GB/core",flush=True)
    res[-1]["hbm_gb_core"]=round(hb/8/1e9,3) if hb else None

print("RESULT_JSON "+json.dumps({"MAX":MAX,"BUCKET":BUCKET,"SW_CAP":R.SW_CAP,"method":("sliding-cap" if R.SW_CAP>0 else "flat"),
    "compile_s":round(compile_s),
    "paris_ok":bool(paris_ok),"rayleigh_ok":bool(rayleigh_ok),"decode_tps":round(coh['math'][1],2),
    "needles":res}),flush=True)
print("PUSHCTX3_OK",flush=True)

"""Path B: TP=2 prefill+decode via NxD ModelBuilder — ONE weight set shared across both buckets
(sharded once from a checkpoint_loader, so fp32 fits ~8GB/rank AND stays correct). Reuses the
Path-A unified forward + one-hot-scatter KV write. module_cls builds parallel-layer STRUCTURE only
(no weight copy); ModelBuilder's get_sharded_checkpoint fills + shares the weights."""
import sys, os, types, time
m=types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None; sys.modules["transformers.utils.fx"]=m
import multiprocessing
MP="/workspace/real-gemma4-E4B-it"; TP=2
MAX=int(os.environ.get("KV_MAX","256")); BUCKET=int(os.environ.get("KV_BUCKET","64"))
CARGS="--model-type transformer --auto-cast all --auto-cast-type bf16"

def _kv_rank_width(nkv): return nkv//TP if nkv%TP==0 else nkv

def _discover():
    """Load ref model on host once to get NONSHARED/LINFO/softcap/SW (structure metadata)."""
    import torch
    from transformers import Gemma4ForConditionalGeneration
    mm=Gemma4ForConditionalGeneration.from_pretrained(MP,torch_dtype=torch.float32,attn_implementation="eager"); mm.eval()
    lang=mm.model.language_model; cfg=lang.config
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a=lyr.self_attn
        if not a.is_kv_shared_layer:
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    softcap=getattr(mm.config.text_config,"final_logit_softcapping",None)
    return mm,lang,mm.lm_head,softcap,NONSHARED,LINFO,cfg.sliding_window

def _structure():
    """Build model STRUCTURE from config only (no weight load) — safe under ModelBuilder meta context."""
    import torch
    from transformers import Gemma4ForConditionalGeneration, AutoConfig
    cfg=AutoConfig.from_pretrained(MP)
    mm=Gemma4ForConditionalGeneration(cfg); mm.eval()
    lang=mm.model.language_model; lc=lang.config
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:lc.num_hidden_layers]):
        a=lyr.self_attn
        if not a.is_kv_shared_layer:
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    softcap=getattr(mm.config.text_config,"final_logit_softcapping",None)
    return mm,lang,mm.lm_head,softcap,NONSHARED,LINFO,lc.sliding_window

def build_module():
    """Build the sharded Gemma4 Wrap: parallel-layer STRUCTURE only (weights come from checkpoint)."""
    import torch
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
    mm,lang,lm_head,softcap,NONSHARED,LINFO,SW=_structure()
    class GeluTanh(torch.nn.Module):
        def forward(s,x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in lang.modules():
        if hasattr(mod,"act_fn"): mod.act_fn=GeluTanh()
    WDT=torch.bfloat16 if os.environ.get("MB_WDTYPE","fp32")=="bf16" else torch.float32  # bf16 halves on-device weights
    def col(o): return ColumnParallelLinear(o.in_features,o.out_features,bias=False,gather_output=False,dtype=WDT)
    def row(o): return RowParallelLinear(o.in_features,o.out_features,bias=False,input_is_parallel=True,dtype=WDT)
    for lyr in lang.layers[:lang.config.num_hidden_layers]:
        a=lyr.self_attn; hd=a.head_dim
        a.q_proj=col(a.q_proj); a.o_proj=row(a.o_proj)
        if getattr(a,"k_proj",None) is not None:
            nkv=a.k_proj.out_features//hd
            if nkv%TP==0:
                a.k_proj=col(a.k_proj)
                if getattr(a,"v_proj",None) is not None: a.v_proj=col(a.v_proj)
            else:
                a.num_key_value_groups=(a.q_proj.out_features//hd)//nkv
        mp=lyr.mlp; mp.gate_proj=col(mp.gate_proj); mp.up_proj=col(mp.up_proj); mp.down_proj=row(mp.down_proj)
    # CRITICAL: load per-layer `layer_scalar` BUFFERS from the checkpoint. These multiply each layer's
    # output (modeling_gemma4 L1461 `hidden_states *= self.layer_scalar`); real value ~0.061 but config
    # default is 1.0, and shard_children/get_sharded_checkpoint load PARAMS only (never buffers) — so
    # without this every layer over-scales ~16x → compounds over 42 layers → cos~0 garbage. (This is
    # the difference vs tp_alias, which built via from_pretrained and got buffers loaded for free.)
    import glob
    from safetensors import safe_open
    lsv={}
    for fp in sorted(glob.glob(MP+"/*.safetensors")):
        with safe_open(fp,framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("model.language_model.layers.") and k.endswith(".layer_scalar"):
                    lsv[int(k.split("layers.")[1].split(".")[0])]=sf.get_tensor(k)
    with torch.no_grad():
        for i,lyr in enumerate(lang.layers[:lang.config.num_hidden_layers]):
            if hasattr(lyr,"layer_scalar") and i in lsv:
                lyr.layer_scalar.copy_(lsv[i].to(lyr.layer_scalar.dtype))
    lang.embed_tokens=torch.nn.Embedding(2, lang.config.hidden_size)   # dummy: embeddings computed on host, not on device
    NK=len(NONSHARED); W={i:_kv_rank_width(LINFO[i][0]) for i in NONSHARED}
    def _sc(lg): return softcap*torch.tanh(lg/softcap) if softcap else lg
    class ScatterKV:
        is_compileable=False
        def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
        def _wr(s,buf,x):
            oh=s.oh[0,0]; scat=torch.einsum('bhsd,ms->bhmd',x,oh); r=oh.sum(-1).view(1,1,MAX,1); return buf*(1.0-r)+scat
        def update(s,k,v,idx,*a,**kw): s.key[idx]=s._wr(s.key[idx],k); s.val[idx]=s._wr(s.val[idx],v); return s.key[idx],s.val[idx]
        def get_seq_length(s,*a,**k): return 0
        def export(s): return [s.key[i] for i in NONSHARED],[s.val[i] for i in NONSHARED]
    class Wrap(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.lang=lang; s.head=lm_head
            s.kbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],MAX,LINFO[i][1]),requires_grad=False) for i in NONSHARED])
            s.vbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],MAX,LINFO[i][1]),requires_grad=False) for i in NONSHARED])
        def forward(s, ie, ple, position_ids, onehot, full_mask, slide_mask):
            cache=ScatterKV(list(s.kbuf),list(s.vbuf),onehot)
            out=s.lang(inputs_embeds=ie,per_layer_inputs=ple,position_ids=position_ids,
                       attention_mask={"full_attention":full_mask,"sliding_attention":slide_mask},use_cache=True,past_key_values=cache)
            lg=_sc(s.head(out.last_hidden_state)); ks,vs=cache.export(); return (lg,)+tuple(ks)+tuple(vs)
    w=Wrap().eval()
    aliases={}
    for j in range(NK): aliases[w.kbuf[j]]=1+j
    for j in range(NK): aliases[w.vbuf[j]]=1+NK+j
    return w, aliases


def checkpoint_loader():
    """Full fp32 state dict keyed to the Wrap module (lang.*/head.*), loaded DIRECTLY from
    safetensors (no from_pretrained — that trips ModelBuilder's patched meta param registration)."""
    import torch, glob
    from safetensors.torch import load_file
    raw={}
    for fp in sorted(glob.glob(MP+"/*.safetensors")): raw.update(load_file(fp))
    sd={}
    for k,v in raw.items():
        if k=="model.language_model.embed_tokens.weight":
            sd["head.weight"]=v.to(torch.float32)                 # tied lm_head (exact match, NOT embed_tokens_per_layer)
        elif k.startswith("model.language_model."):
            sd["lang."+k[len("model.language_model."):]]=v.to(torch.float32)
        elif k.startswith("lm_head."):
            sd["head."+k[len("lm_head."):]]=v.to(torch.float32)   # explicit head if untied
    return sd

def _inputs(lang, ids, SW, NEG, positions):
    import torch
    ids_t=torch.tensor([ids])
    with torch.no_grad(): ie=lang.embed_tokens(ids_t); ple=lang.get_per_layer_inputs(ids_t,ie)
    seq=len(positions); ar=torch.arange(MAX); pos=torch.tensor([positions],dtype=torch.long)
    oh=(ar.view(MAX,1)==pos.view(1,seq)).view(1,1,MAX,seq).to(torch.float32)
    q=pos.view(seq,1); mm2=ar.view(1,MAX)
    full=torch.where(mm2<=q,0.0,NEG).view(1,1,seq,MAX)
    slide=torch.where((mm2<=q)&(mm2>q-SW),0.0,NEG).view(1,1,seq,MAX)
    return ie,ple,pos,oh,full,slide

def main():
    import torch
    from transformers import AutoTokenizer
    from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance
    NEG=torch.finfo(torch.float32).min
    tok=AutoTokenizer.from_pretrained(MP)
    _mm,rlang,_h,_s,NONSHARED,LINFO,SW=_discover()
    enc=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}],add_generation_prompt=True,return_tensors="pt",return_dict=True)
    prompt=enc["input_ids"][0].tolist(); n0=len(prompt); assert n0<=BUCKET
    pre_in=_inputs(rlang, prompt+[0]*(BUCKET-n0), SW, NEG, list(range(BUCKET)))
    dec_in=_inputs(rlang, [prompt[-1]], SW, NEG, [n0])
    class GemmaInstance(BaseModelInstance):
        def __init__(self): self.module=None; self.input_output_aliases=[{}]
        def load_module(self):                       # called by ModelBuilder UNDER the rank TP context
            self.module, aliases = build_module()    # -> ColumnParallelLinear now actually shards
            self.input_output_aliases=[aliases]
        def get(self, bucket_rank, **kwargs): return self.module, self.input_output_aliases[0]
    inst=GemmaInstance()
    print("building ModelBuilder ...",flush=True)
    mb=ModelBuilder(router=None, tp_degree=TP, checkpoint_loader=checkpoint_loader, compiler_workdir="/workspace/mb_wd")
    mb.add("prefill", inst, [pre_in], compiler_args=CARGS)
    mb.add("decode",  inst, [dec_in], compiler_args=CARGS)
    print("tracing ...",flush=True)
    model=mb.trace(initialize_model_weights=True)
    print("MB_TRACED",flush=True)
    # === in-process validation: device prefill+decode buckets vs CPU fp32 reference ===
    import time
    from transformers import DynamicCache
    softcap=getattr(_mm.config.text_config,"final_logit_softcapping",None); head=_mm.lm_head
    ec=_mm.generation_config.eos_token_id; EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
    class _G(torch.nn.Module):
        def forward(s,x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in rlang.modules():
        if hasattr(mod,"act_fn"): mod.act_fn=_G()
    def _sc2(lg): return softcap*torch.tanh(lg/softcap) if softcap else lg
    def embR(ids):
        idt=torch.tensor([ids])
        with torch.no_grad(): ie=rlang.embed_tokens(idt); ple=rlang.get_per_layer_inputs(idt,ie)
        return ie,ple
    def cpuref():
        ie,ple=embR(prompt); c=DynamicCache()
        with torch.no_grad():
            out=rlang(inputs_embeds=ie,per_layer_inputs=ple,attention_mask=torch.ones(1,n0,dtype=torch.long),use_cache=True,past_key_values=c)
            first=int(_sc2(head(out.last_hidden_state))[0,n0-1].argmax())
        KS=[c.layers[i].keys for i in NONSHARED]; VS=[c.layers[i].values for i in NONSHARED]
        class SKV:
            is_compileable=False
            def __init__(s,kb,vb,oh): s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}; s.oh=oh
            def update(s,k,v,idx,*a,**kw): s.key[idx]=s.key[idx]*(1.0-s.oh)+k*s.oh; s.val[idx]=s.val[idx]*(1.0-s.oh)+v*s.oh; return s.key[idx],s.val[idx]
            def get_seq_length(s,*a,**k): return 0
        def hp(pos):
            ar=torch.arange(MAX); oh=(ar==pos).view(1,1,MAX,1).float(); valid=ar<=pos
            fm=torch.where(valid,0.0,NEG).view(1,1,1,MAX); sm=torch.where(valid&(ar>pos-SW),0.0,NEG).view(1,1,1,MAX)
            return torch.tensor([[pos]]),oh,fm,sm
        kb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]; vb=[torch.zeros(1,LINFO[i][0],MAX,LINFO[i][1]) for i in NONSHARED]
        for j in range(len(NONSHARED)): kb[j][:,:,:n0,:]=KS[j][:,:,:n0,:]; vb[j][:,:,:n0,:]=VS[j][:,:,:n0,:]
        seq=[first]; cur=n0
        for _ in range(30):
            if seq[-1] in EOS: break
            e1,p1=embR([seq[-1]]); pi,oh,fm,sm=hp(cur); cache=SKV(kb,vb,oh)
            with torch.no_grad():
                out=rlang(inputs_embeds=e1,per_layer_inputs=p1,position_ids=pi,attention_mask={"full_attention":fm,"sliding_attention":sm},use_cache=True,past_key_values=cache)
                lg=_sc2(head(out.last_hidden_state))
            kb=[cache.key[i] for i in NONSHARED]; vb=[cache.val[i] for i in NONSHARED]
            seq.append(int(lg[0,0].argmax())); cur+=1
        return seq
    print("=== CPU ref ===",flush=True); cpu_seq=cpuref()
    def dcall(a): r=model(*a); return r[0] if isinstance(r,(tuple,list)) else r
    pad=prompt+[0]*(BUCKET-n0)
    print("=== DEVICE (bucketed) ===",flush=True)
    dcall(_inputs(rlang,pad,SW,NEG,list(range(BUCKET))))                # warmup
    t0=time.time(); lg=dcall(_inputs(rlang,pad,SW,NEG,list(range(BUCKET)))); pf=time.time()-t0
    first=int(lg[0,n0-1].argmax()); seq=[first]; cur=n0
    for _ in range(30):
        if seq[-1] in EOS: break
        l1=dcall(_inputs(rlang,[seq[-1]],SW,NEG,[cur])); seq.append(int(l1[0,0].argmax())); cur+=1
    print("CPU GEN:",repr(tok.decode([x for x in cpu_seq if x not in EOS],skip_special_tokens=True)),flush=True)
    print("DEV GEN:",repr(tok.decode([x for x in seq if x not in EOS],skip_special_tokens=True)),flush=True)
    print("SEQ_MATCH", cpu_seq==seq, flush=True)
    print(f"DEVICE PREFILL first-token: {pf*1000:.0f} ms  [vs ~1400-1600ms CPU prefill]",flush=True)
    print("TPMB_OK",flush=True)
    save_to=os.environ.get("MB_SAVE")
    if save_to:
        print("saving model to",save_to,flush=True)
        torch.jit.save(model, save_to)   # the traced executor embeds neffs+weights; reload via torch.jit.load
        print("MB_SAVED",save_to,flush=True)
        # in-process reload sanity: fresh jit.load must reproduce the same first token
        rl=torch.jit.load(save_to)
        rr=rl(*_inputs(rlang,pad,SW,NEG,list(range(BUCKET)))); rlg=rr[0] if isinstance(rr,(tuple,list)) else rr
        print("RELOAD first-tok:",int(rlg[0,n0-1].argmax()),"== device",first, "->", int(rlg[0,n0-1].argmax())==first, flush=True)

if __name__=="__main__":
    if os.environ.get("_TP_CHILD")!="1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn",force=True); main()

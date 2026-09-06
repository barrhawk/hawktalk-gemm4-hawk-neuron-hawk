"""Gemma-4 26B-A4B (MoE, model_type gemma4) TP=8 prefill+decode via NxD ModelBuilder.

Reuses the 31B tp_mb.py recipe wholesale (ModelBuilder single-rank compile, mixed sliding/global
attention shard/replicate, ScatterKV device cache, layer_scalar buffers, chat-template prompt, device
prefill+decode) and swaps ONLY the FFN for the MoE dual-path.

26B-A4B is a DUAL-PATH FFN per layer: a shared dense `mlp` (intermediate 2112) in PARALLEL with a
128-expert MoE (top-8, moe_intermediate 704), combined `dense+moe` through 4 feed-forward layernorms.
HF's decoder-layer forward + `Gemma4TextRouter` are kept UNCHANGED (router replicated); we replace only
`self.experts` (whose HF forward is a sparse gather/scatter loop that won't trace) with `DenseExperts`:

  * all-experts-DENSE compute (all 128 experts on every token, weighted by the top-8 router weight,
    0 for non-selected -> exact match to HF's sparse top-8, static-shape + traceable).
  * expert weights mapped onto TWO standard parallel linears ModelBuilder auto-shards:
      gate_up: [E*2*I, H] ColumnParallelLinear  (rank r gets experts [16r:16r+16])
      down:    [H, E*I]   RowParallelLinear      (input-sharded by expert-block -> all-reduce)
    -> ~5.7GB experts/rank (can't replicate 45.6GB > 16GB/core).

Set MODEL_DIR=/data/real-gemma4-26B-A4B-it MB_WDTYPE=bf16 TP_DEGREE=8 KV_MAX=256 KV_BUCKET=64."""
import sys, os, types, time
m=types.ModuleType("transformers.utils.fx"); m.HFTracer=object; m.symbolic_trace=None; sys.modules["transformers.utils.fx"]=m
import multiprocessing
MP=os.environ.get("MODEL_DIR","/data/real-gemma4-26B-A4B-it"); TP=int(os.environ.get("TP_DEGREE","8"))
MAX=int(os.environ.get("KV_MAX","256")); BUCKET=int(os.environ.get("KV_BUCKET","64"))
# LEVER 2: cap the SLIDING layers' KV buffer at SW_CAP slots (ring buffer). 0 => flat (all layers = MAX).
# Only the 5 GLOBAL (full_attention) layers keep the big MAX buffer -> native long-context reachable.
SW_CAP=int(os.environ.get("SW_CAP","0"))
CARGS="--model-type transformer --auto-cast all --auto-cast-type bf16"
# per-nonshared-layer buffer depth: sliding layers use SC (when SW_CAP>0), global layers use MAX.
def _buf_depth(i, is_sliding):
    return SW_CAP if (SW_CAP>0 and is_sliding) else MAX

def _kv_rank_width(nkv): return nkv//TP if nkv%TP==0 else nkv

def _discover():
    import torch
    from transformers import Gemma4ForConditionalGeneration
    mm=Gemma4ForConditionalGeneration.from_pretrained(MP,torch_dtype=torch.float32,attn_implementation="eager"); mm.eval()
    lang=mm.model.language_model; cfg=lang.config
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:cfg.num_hidden_layers]):
        a=lyr.self_attn
        if not getattr(a,"is_kv_shared_layer",False):
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    softcap=getattr(mm.config.text_config,"final_logit_softcapping",None)
    return mm,lang,mm.lm_head,softcap,NONSHARED,LINFO,cfg.sliding_window

def _structure():
    import torch
    from transformers import Gemma4ForConditionalGeneration, AutoConfig
    cfg=AutoConfig.from_pretrained(MP)
    cfg._attn_implementation="eager"
    if hasattr(cfg,"text_config"): cfg.text_config._attn_implementation="eager"
    mm=Gemma4ForConditionalGeneration(cfg); mm.eval()
    lang=mm.model.language_model; lc=lang.config
    NONSHARED,LINFO=[],{}
    for i,lyr in enumerate(lang.layers[:lc.num_hidden_layers]):
        a=lyr.self_attn
        if not getattr(a,"is_kv_shared_layer",False):
            hd=a.head_dim; NONSHARED.append(i); LINFO[i]=(a.k_proj.out_features//hd,hd)
    softcap=getattr(mm.config.text_config,"final_logit_softcapping",None)
    return mm,lang,mm.lm_head,softcap,NONSHARED,LINFO,lc.sliding_window

def build_module():
    import torch
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear, SPMDRank
    from neuronx_distributed.parallel_layers.mappings import scatter_to_process_group_spmd
    mm,lang,lm_head,softcap,NONSHARED,LINFO,SW=_structure()
    lc=lang.config
    N_EXP=lc.num_experts; MOE_I=lc.moe_intermediate_size; H=lc.hidden_size; TOPK=lc.top_k_experts
    class GeluTanh(torch.nn.Module):
        def forward(s,x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in lang.modules():
        if hasattr(mod,"act_fn"): mod.act_fn=GeluTanh()
    WDT=torch.bfloat16 if os.environ.get("MB_WDTYPE","bf16")=="bf16" else torch.float32
    def col(o): return ColumnParallelLinear(o.in_features,o.out_features,bias=False,gather_output=False,dtype=WDT)
    def row(o): return RowParallelLinear(o.in_features,o.out_features,bias=False,input_is_parallel=True,dtype=WDT)

    class DenseExperts(torch.nn.Module):
        """All-experts-dense MoE, expert-parallel via two standard parallel linears.
        forward(x, top_k_index, top_k_weights) -> [T, H]  (matches HF Gemma4TextExperts signature).
        top_k_weights already carries the router's renorm + per_expert_scale."""
        def __init__(s):
            super().__init__()
            s.gate_up=ColumnParallelLinear(H, N_EXP*2*MOE_I, bias=False, gather_output=False, dtype=WDT)
            s.down=RowParallelLinear(N_EXP*MOE_I, H, bias=False, input_is_parallel=True, dtype=WDT)
            s.act=GeluTanh(); s.epr=N_EXP//TP
            s.spmd_rank=SPMDRank(TP)   # runtime rank tensor for the SPMD scatter (checkpoint = arange(TP))
        def forward(s, x, top_k_index, top_k_weights):
            T=x.shape[0]
            ar=torch.arange(N_EXP, device=x.device)
            onehot=(top_k_index.unsqueeze(-1)==ar).to(x.dtype)                 # [T,K,E]
            Wd=(top_k_weights.unsqueeze(-1).to(x.dtype)*onehot).sum(dim=1)     # [T,E] dense, 0 for non-top-k
            # SPMD-aware scatter: standalone scatter_to_tensor_model_parallel_region bakes rank 0's slice
            # into the single-rank trace -> every rank weights experts 0..15 while gate_up computes its
            # OWN experts (misaligned garbage). This uses the runtime rank tensor to pick Wd[:,16r:16r+16].
            Wl=scatter_to_process_group_spmd(Wd, 1, s.spmd_rank.get_rank())    # [T, epr] rank-local experts
            gu=s.gate_up(x).view(T, s.epr, 2*MOE_I)                            # [T,epr,2I]
            gate,up=gu.chunk(2, dim=-1)                                        # [T,epr,I] each
            h=(s.act(gate)*up)*Wl.unsqueeze(-1)                               # weight before down
            h=h.reshape(T, s.epr*MOE_I)                                        # [T, epr*I]
            return s.down(h)                                                   # RowParallel all-reduce -> [T,H]

    nshard=nrepl=nmoe=0
    for lyr in lang.layers[:lc.num_hidden_layers]:
        a=lyr.self_attn; hd=a.head_dim
        k_out=a.k_proj.out_features if getattr(a,"k_proj",None) is not None else None
        nkv=(k_out//hd) if k_out is not None else None
        if nkv is not None and nkv%TP==0:
            a.q_proj=col(a.q_proj); a.o_proj=row(a.o_proj); a.k_proj=col(a.k_proj)
            if getattr(a,"v_proj",None) is not None: a.v_proj=col(a.v_proj)
            nshard+=1
        else:
            nrepl+=1
        mp=lyr.mlp; mp.gate_proj=col(mp.gate_proj); mp.up_proj=col(mp.up_proj); mp.down_proj=row(mp.down_proj)  # shared dense MLP
        if getattr(lyr,"enable_moe_block",False) and hasattr(lyr,"experts"):
            lyr.experts=DenseExperts(); nmoe+=1   # router stays intact (replicated)
    print(f"build_module: {nshard} sharded-attn, {nrepl} replicated-attn, {nmoe} MoE layers, TP={TP} (E={N_EXP} K={TOPK} I={MOE_I})",flush=True)
    import glob
    from safetensors import safe_open
    lsv={}
    for fp in sorted(glob.glob(MP+"/*.safetensors")):
        with safe_open(fp,framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("model.language_model.layers.") and k.endswith(".layer_scalar"):
                    lsv[int(k.split("layers.")[1].split(".")[0])]=sf.get_tensor(k)
    with torch.no_grad():
        for i,lyr in enumerate(lang.layers[:lc.num_hidden_layers]):
            if hasattr(lyr,"layer_scalar") and i in lsv:
                lyr.layer_scalar.copy_(lsv[i].to(lyr.layer_scalar.dtype))
    print(f"  loaded {len(lsv)} layer_scalar buffers",flush=True)
    lang.embed_tokens=torch.nn.Embedding(2, H)
    NK=len(NONSHARED); W={i:_kv_rank_width(LINFO[i][0]) for i in NONSHARED}
    # LEVER 2: which nonshared layers are SLIDING (get the capped ring buffer) vs GLOBAL (keep MAX).
    _lt=getattr(lc,"layer_types",None)
    SLID=set(i for i in NONSHARED if (_lt is None or _lt[i]=="sliding_attention"))
    DEPTH={i:_buf_depth(i, i in SLID) for i in NONSHARED}   # per-layer KV buffer depth
    print(f"  LEVER2 SW_CAP={SW_CAP} sliding_layers={sorted(SLID)} global_layers={sorted(set(NONSHARED)-SLID)}",flush=True)
    class ScatterKV:
        is_compileable=False
        def __init__(s,kb,vb,oh_full,oh_slide):
            s.key={i:kb[j] for j,i in enumerate(NONSHARED)}; s.val={i:vb[j] for j,i in enumerate(NONSHARED)}
            s.ohf=oh_full; s.ohs=oh_slide
        def _wr(s,buf,x,sliding):
            oh=(s.ohs if sliding else s.ohf)[0,0]; d=buf.shape[-2]
            scat=torch.einsum('bhsd,ms->bhmd',x,oh); r=oh.sum(-1).view(1,1,d,1); return buf*(1.0-r)+scat
        def update(s,k,v,idx,*a,**kw):
            sl=idx in SLID; s.key[idx]=s._wr(s.key[idx],k,sl); s.val[idx]=s._wr(s.val[idx],v,sl); return s.key[idx],s.val[idx]
        def get_seq_length(s,*a,**k): return 0
        def export(s): return [s.key[i] for i in NONSHARED],[s.val[i] for i in NONSHARED]
    class Wrap(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.lang=lang
            # LM_HEAD SHARD: replace the ~1.5GB replicated lm_head with a ColumnParallelLinear so its weight
            # is sharded [vocab/TP, H] across ranks (~185MB/rank), freeing ~1.3GB/core for a larger global KV.
            # gather_output=True all-gathers the logits so argmax over full vocab still works. (softcap is
            # monotonic -> omitting it does not change argmax, so needle/coherence are unaffected.)
            VOCAB=lm_head.weight.shape[0]
            s.head=ColumnParallelLinear(H, VOCAB, bias=False, gather_output=True, dtype=WDT)
            s.kbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],DEPTH[i],LINFO[i][1]),requires_grad=False) for i in NONSHARED])
            s.vbuf=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,W[i],DEPTH[i],LINFO[i][1]),requires_grad=False) for i in NONSHARED])
        def forward(s, ie, position_ids, oh_full, oh_slide, full_mask, slide_mask):
            cache=ScatterKV(list(s.kbuf),list(s.vbuf),oh_full,oh_slide)
            out=s.lang(inputs_embeds=ie,position_ids=position_ids,
                       attention_mask={"full_attention":full_mask,"sliding_attention":slide_mask},use_cache=True,past_key_values=cache)
            lg=s.head(out.last_hidden_state); ks,vs=cache.export(); return (lg,)+tuple(ks)+tuple(vs)
    w=Wrap().eval()
    aliases={}
    for j in range(NK): aliases[w.kbuf[j]]=1+j
    for j in range(NK): aliases[w.vbuf[j]]=1+NK+j
    return w, aliases

def checkpoint_loader():
    """Full fp32 state dict keyed to the Wrap. Reshape the fused 3D expert weights into the 2D
    ColumnParallel(gate_up)/RowParallel(down) layouts DenseExperts expects."""
    import torch, glob, re
    from safetensors.torch import load_file
    raw={}
    for fp in sorted(glob.glob(MP+"/*.safetensors")): raw.update(load_file(fp))
    sd={}
    exp_re=re.compile(r"layers\.(\d+)\.experts\.(gate_up_proj|down_proj)$")
    for k,v in raw.items():
        if k=="model.language_model.embed_tokens.weight":
            sd["head.weight"]=v.to(torch.float32)
        elif k.startswith("model.language_model."):
            kk=k[len("model.language_model."):]
            mo=exp_re.match(kk)
            if mo:
                i=mo.group(1)
                if mo.group(2)=="gate_up_proj":            # [E, 2I, H] -> [E*2I, H]
                    E,TWOI,Hh=v.shape
                    sd[f"lang.layers.{i}.experts.gate_up.weight"]=v.reshape(E*TWOI, Hh).to(torch.float32)
                else:                                       # down_proj [E, H, I] -> [H, E*I]
                    E,Hh,I=v.shape
                    sd[f"lang.layers.{i}.experts.down.weight"]=v.permute(1,0,2).reshape(Hh, E*I).to(torch.float32)
            else:
                sd["lang."+kk]=v.to(torch.float32)
        elif k.startswith("lm_head."):
            sd["head."+k[len("lm_head."):]]=v.to(torch.float32)
    # SPMDRank checkpoint: arange(TP) sharded dim0 -> each rank gets its own rank number at runtime.
    moe_layers={int(re.match(r"lang\.layers\.(\d+)\.experts\.gate_up\.weight",kk).group(1))
                for kk in sd if re.match(r"lang\.layers\.(\d+)\.experts\.gate_up\.weight",kk)}
    for i in sorted(moe_layers):
        sd[f"lang.layers.{i}.experts.spmd_rank.rank"]=torch.arange(TP, dtype=torch.int32)
    return sd

def _inputs(lang, ids, SW, NEG, positions, frontier=None):
    """Returns ie, pos, oh_full[1,1,MAX,seq], oh_slide[1,1,SCd,seq], full[1,1,seq,MAX], slide[1,1,seq,SCd].
    GLOBAL layers index the full MAX buffer at absolute position. SLIDING layers (when SW_CAP>0) index a ring
    buffer of depth SC=SW_CAP: write slot = pos % SC. `frontier` = the highest absolute position already written
    into the buffer at attention time (= max(positions) for a fresh chunk; = q for pure 1-token teacher forcing).
    Ring slot j physically holds token p_j = frontier-((frontier-j)%SC) (unique residue-j in (frontier-SC,frontier]).
    Query q may read slot j iff that stored token is causal AND inside the window:  p_j<=q AND p_j>q-SW AND p_j>=0.
    For chunked prefill correctness require SC >= CHUNK + SW so no in-window token is overwritten within a chunk."""
    import torch
    ids_t=torch.tensor([ids])
    with torch.no_grad(): ie=lang.embed_tokens(ids_t)
    seq=len(positions); ar=torch.arange(MAX); pos=torch.tensor([positions],dtype=torch.long)
    oh_full=(ar.view(MAX,1)==pos.view(1,seq)).view(1,1,MAX,seq).to(torch.float32)
    q=pos.view(seq,1); mm2=ar.view(1,MAX)
    full=torch.where(mm2<=q,0.0,NEG).view(1,1,seq,MAX)
    if SW_CAP>0:
        assert SW_CAP>=SW, f"SW_CAP={SW_CAP} must be >= window SW={SW}"
        SC=SW_CAP; jr=torch.arange(SC)
        fr=int(max(positions)) if frontier is None else int(frontier)
        rs=(pos%SC).view(1,seq)                                   # ring write slot per query
        oh_slide=(jr.view(SC,1)==rs).view(1,1,SC,seq).to(torch.float32)
        p_j=(fr-((fr - jr)%SC)).view(1,SC)                        # [1,SC] abs position stored at each slot
        valid=(p_j<=q)&(p_j>(q-SW))&(p_j>=0)                      # causal + window + written
        slide=torch.where(valid,0.0,NEG).view(1,1,seq,SC)
    else:
        oh_slide=oh_full
        slide=torch.where((mm2<=q)&(mm2>q-SW),0.0,NEG).view(1,1,seq,MAX)
    return ie,pos,oh_full,oh_slide,full,slide

def main():
    import torch
    from transformers import AutoTokenizer, DynamicCache
    from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance
    NEG=torch.finfo(torch.float32).min
    tok=AutoTokenizer.from_pretrained(MP)
    _mm,rlang,_h,softcap,NONSHARED,LINFO,SW=_discover()
    print(f"26B-A4B discover: {len(NONSHARED)} non-shared layers, head_dims={sorted({LINFO[i][1] for i in NONSHARED})}, softcap={softcap}",flush=True)
    if not getattr(tok,"chat_template",None):
        try:
            import subprocess,json
            from huggingface_hub import hf_hub_download
            raw=subprocess.check_output(["aws","secretsmanager","get-secret-value","--region","us-east-1","--secret-id","hf_token","--query","SecretString","--output","text"]).decode().strip()
            try: tv=json.loads(raw); tv=tv.get("hf_token") or list(tv.values())[0]
            except Exception: tv=raw
            tok.chat_template=open(hf_hub_download(os.environ.get("HF_REPO","google/gemma-4-26B-A4B-it"),"chat_template.jinja",token=tv)).read()
        except Exception as e:
            print("chat_template fetch failed:",e,flush=True)
    if getattr(tok,"chat_template",None):
        d=tok.apply_chat_template([{"role":"user","content":"What is the capital of France?"}],add_generation_prompt=True)
        prompt=d if isinstance(d,list) else d["input_ids"]
    else:
        prompt=[2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,100,45518,107,101]
    n0=len(prompt); assert n0<=BUCKET
    pre_in=_inputs(rlang, prompt+[0]*(BUCKET-n0), SW, NEG, list(range(BUCKET)))
    dec_in=_inputs(rlang, [prompt[-1]], SW, NEG, [n0])
    load_from=os.environ.get("MB_LOAD")
    if load_from:
        print("loading saved model from",load_from,flush=True)
        model=torch.jit.load(load_from)
        model.nxd_model.initialize_with_saved_weights(torch.tensor([0],dtype=torch.int32)); print("MB_LOADED",flush=True)
    else:
        class GemmaInstance(BaseModelInstance):
            def __init__(self): self.module=None; self.input_output_aliases=[{}]
            def load_module(self):
                self.module, aliases = build_module()
                self.input_output_aliases=[aliases]
            def get(self, bucket_rank, **kwargs): return self.module, self.input_output_aliases[0]
        inst=GemmaInstance()
        print("building ModelBuilder ...",flush=True)
        mb=ModelBuilder(router=None, tp_degree=TP, checkpoint_loader=checkpoint_loader, compiler_workdir="/workspace/mb_wd")
        mb.add("prefill", inst, [pre_in], compiler_args=CARGS)
        mb.add("decode",  inst, [dec_in], compiler_args=CARGS)
        print("tracing (ModelBuilder, single-rank compile) ...",flush=True)
        model=mb.trace(initialize_model_weights=True)
        print("MB_TRACED",flush=True)
        save_to=os.environ.get("MB_SAVE")
        if save_to:
            print("saving model to",save_to,flush=True); torch.jit.save(model, save_to); print("MB_SAVED",save_to,flush=True)
        if os.environ.get("SKIP_VALIDATE")=="1":
            print("SKIP_VALIDATE=1 -> compile-only run complete",flush=True); print("TPMB_OK",flush=True); return
    head=_mm.lm_head
    ec=_mm.generation_config.eos_token_id; EOS=set(ec) if isinstance(ec,(list,tuple)) else {ec}
    class _G(torch.nn.Module):
        def forward(s,x): return 0.5*x*(1.0+torch.tanh(0.7978845608028654*(x+0.044715*x*x*x)))
    for mod in rlang.modules():
        if hasattr(mod,"act_fn"): mod.act_fn=_G()
    def _sc2(lg): return softcap*torch.tanh(lg/softcap) if softcap else lg
    def embR(ids):
        idt=torch.tensor([ids])
        with torch.no_grad(): ie=rlang.embed_tokens(idt)
        return ie
    def cpuref():
        ie=embR(prompt); c=DynamicCache()
        with torch.no_grad():
            out=rlang(inputs_embeds=ie,attention_mask=torch.ones(1,n0,dtype=torch.long),use_cache=True,past_key_values=c)
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
            e1=embR([seq[-1]]); pi,oh,fm,sm=hp(cur); cache=SKV(kb,vb,oh)
            with torch.no_grad():
                out=rlang(inputs_embeds=e1,position_ids=pi,attention_mask={"full_attention":fm,"sliding_attention":sm},use_cache=True,past_key_values=cache)
                lg=_sc2(head(out.last_hidden_state))
            kb=[cache.key[i] for i in NONSHARED]; vb=[cache.val[i] for i in NONSHARED]
            seq.append(int(lg[0,0].argmax())); cur+=1
        return seq
    DEVICE_ONLY=os.environ.get("DEVICE_ONLY")=="1"
    if DEVICE_ONLY:
        cpu_seq=None; print("DEVICE_ONLY=1 -> skipping CPU reference",flush=True)
    else:
        print("=== CPU ref ===",flush=True); cpu_seq=cpuref()
    def dcall(a): r=model(*a); return r[0] if isinstance(r,(tuple,list)) else r
    pad=prompt+[0]*(BUCKET-n0)
    print("=== DEVICE (bucketed) ===",flush=True)
    dcall(_inputs(rlang,pad,SW,NEG,list(range(BUCKET))))
    t0=time.time(); lg=dcall(_inputs(rlang,pad,SW,NEG,list(range(BUCKET)))); pf=time.time()-t0
    first=int(lg[0,n0-1].argmax()); seq=[first]; cur=n0
    for _ in range(30):
        if seq[-1] in EOS: break
        l1=dcall(_inputs(rlang,[seq[-1]],SW,NEG,[cur])); seq.append(int(l1[0,0].argmax())); cur+=1
    dev_txt=tok.decode([x for x in seq if x not in EOS],skip_special_tokens=True)
    if cpu_seq is not None:
        print("CPU GEN:",repr(tok.decode([x for x in cpu_seq if x not in EOS],skip_special_tokens=True)),flush=True)
        print("SEQ_MATCH", cpu_seq==seq, flush=True)
    else:
        print("DEVICE_PARIS", "paris" in dev_txt.lower(), flush=True)
    print("DEV GEN:",repr(dev_txt),flush=True)
    print(f"DEVICE PREFILL first-token: {pf*1000:.0f} ms",flush=True)
    print("TPMB_OK",flush=True)

if __name__=="__main__":
    if os.environ.get("_TP_CHILD")!="1":
        os.environ["_TP_CHILD"]="1"; multiprocessing.set_start_method("spawn",force=True); main()

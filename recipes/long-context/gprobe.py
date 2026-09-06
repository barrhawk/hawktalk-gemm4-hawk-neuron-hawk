#!/usr/bin/env python
"""
Emulated chunked prefill via prefix-caching CTE on inf2. Hand-drives
NeuronBaseForCausalLM.forward. See GEMMA4_LONGCTX_PROBE_PLAN.md Step B.
"""
import os, sys, time, json, math, logging, argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--compiled", default=os.path.expanduser("~/gemma_pc"))
ap.add_argument("--model", default=os.path.expanduser("~/gemma/gemma-4-E2B-it-trimmed"))
ap.add_argument("--out", default=os.path.expanduser("~/gprobe/out"))
ap.add_argument("--tag", default="run")
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--n", type=int, default=2048)
ap.add_argument("--decode", type=int, default=32)
ap.add_argument("--b3-prefix", type=int, default=1400)
ap.add_argument("--skip-b3", action="store_true")
ap.add_argument("--skip-main", action="store_true", help="skip sections 1-4 (fresh-cache B3 only)")
ap.add_argument("--stale-test", action="store_true", help="after B3, plant stale future KV (2048 chunked) and redo B3")
ap.add_argument("--sleep", type=float, default=2.0, help="pause around calls so neuron-monitor samples")
args = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probe")

sys.path.insert(0, "/home/ubuntu/xfer/gemma4_nxdi")
from modeling_gemma4 import NeuronGemma4ForCausalLM, Gemma4InferenceConfig
import neuronx_distributed_inference.models.model_wrapper as mw

OUT = args.out
os.makedirs(OUT, exist_ok=True)
MARKS = []          # (t, name)
ROUTES = []         # (tag, bucket) per wrapper call
RESULTS = {"tag": args.tag, "chunk": args.chunk, "n": args.n}

def mark(name):
    MARKS.append((time.time(), name))
    log.info("MARK %s", name)

# --- log which NEFF bucket the wrapper picks ------------------------------
_orig_get = mw.ModelWrapper.get_target_2d_bucket_for_prefix_caching
def _logged_get(self, *a, **k):
    b = _orig_get(self, *a, **k)
    try:
        bb = [int(x) for x in b]
    except Exception:
        bb = b
    ROUTES.append((self.tag, bb))
    return b
mw.ModelWrapper.get_target_2d_bucket_for_prefix_caching = _logged_get

# --- load ------------------------------------------------------------------
mark("load_start")
model = NeuronGemma4ForCausalLM(args.compiled)
config = model.config
nc = config.neuron_config
log.info("neuron_config: tp=%s bs=%s max_ctx=%s seq_len=%s prefix_caching=%s block_kv=%s pa_block=%s pa_num_blocks=%s "
         "cte_buckets=%s prefix_buckets=%s tkg_buckets=%s attn_kernel=%s",
         nc.tp_degree, nc.batch_size, nc.max_context_length, nc.seq_len, nc.is_prefix_caching, nc.is_block_kv_layout,
         nc.pa_block_size, nc.pa_num_blocks, nc.context_encoding_buckets, nc.prefix_buckets, nc.token_generation_buckets,
         getattr(nc, "attn_kernel_enabled", None))
BLOCK = nc.pa_block_size
model.load(args.compiled)
mark("load_done")
time.sleep(args.sleep)
mark("idle_after_load")

ids = torch.load(os.path.join(OUT, "ids.pt"))[:, :args.n]
N = ids.shape[1]
assert N == args.n
log.info("ids %s", ids.shape)

def slots_for(blocks, positions):
    return torch.tensor([[int(blocks[p // BLOCK]) * BLOCK + (p % BLOCK) for p in positions]], dtype=torch.long)

def routed():
    return "CTE" if model.base_model is model.context_encoding_model else "TKG"

def cte_call(ids_full, start, end, blocks, name):
    """One CTE call: prefix [0,start) already in cache (via blocks), encode [start,end)."""
    input_ids = ids_full[:, :end]
    position_ids = torch.arange(end, dtype=torch.long)[None]
    attention_mask = torch.ones(1, end, dtype=torch.long)
    slot_mapping = slots_for(blocks, range(start, end))
    nb = math.ceil(start / BLOCK)
    block_table = torch.tensor([[int(b) for b in blocks[:nb]]], dtype=torch.long) if nb > 0 else torch.zeros((1, 1), dtype=torch.long)
    full_ctx = torch.tensor([[end]], dtype=torch.long)
    comp_ctx = torch.tensor([[start]], dtype=torch.long)
    assert int(position_ids.min()) == 0
    nroutes = len(ROUTES)
    mark("%s_start" % name)
    t0 = time.time()
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                seq_ids=torch.tensor([0]), slot_mapping=slot_mapping, block_table=block_table,
                full_context_lens=full_ctx, computed_context_lens=comp_ctx)
    dt = time.time() - t0
    mark("%s_end" % name)
    logits = out.logits
    last = logits[0, -1].float().clone()
    r = routed()
    rts = ROUTES[nroutes:]
    log.info("%s: start=%d end=%d routed=%s buckets=%s logits.shape=%s argmax=%d dt=%.3fs", name, start, end, r, rts, tuple(logits.shape), int(last.argmax()), dt)
    RESULTS.setdefault("calls", []).append({"name": name, "start": start, "end": end, "routed": r, "buckets": rts, "dt": dt,
                                             "argmax": int(last.argmax()), "logits_shape": list(logits.shape)})
    assert r == "CTE", "expected CTE NEFF for %s, got %s" % (name, r)
    time.sleep(args.sleep)
    return last

def tkg_step(tok, pos, blocks, name):
    input_ids = torch.tensor([[int(tok)]], dtype=torch.long)
    position_ids = torch.tensor([[pos]], dtype=torch.long)
    attention_mask = torch.ones(1, pos, dtype=torch.long)
    slot_mapping = slots_for(blocks, [pos])
    nb = math.ceil((pos + 1) / BLOCK)
    block_table = torch.tensor([[int(b) for b in blocks[:nb]]], dtype=torch.long)
    full_ctx = torch.tensor([[pos + 1]], dtype=torch.long)
    comp_ctx = torch.tensor([[pos]], dtype=torch.long)
    nroutes = len(ROUTES)
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                seq_ids=torch.tensor([0]), slot_mapping=slot_mapping, block_table=block_table,
                full_context_lens=full_ctx, computed_context_lens=comp_ctx)
    r = routed()
    assert r == "TKG", "expected TKG NEFF at %s pos %d, got %s" % (name, pos, r)
    last = out.logits[0, -1].float().clone()
    return last, ROUTES[nroutes:]

def greedy_decode(first_logits, start_pos, blocks, name, steps):
    toks = [int(first_logits.argmax())]
    mark("%s_decode_start" % name)
    t0 = time.time()
    rts = None
    for i in range(steps - 1):
        pos = start_pos + i
        last, rts = tkg_step(toks[-1], pos, blocks, name)
        toks.append(int(last.argmax()))
    dt = time.time() - t0
    mark("%s_decode_end" % name)
    log.info("%s decode %d steps in %.2fs (%.1f ms/tok) last tkg bucket=%s toks=%s", name, steps - 1, dt, 1000 * dt / max(1, steps - 1), rts, toks)
    time.sleep(args.sleep)
    return toks

def compare(a, b, name):
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a[None], b[None]).item()
    res = {"argmax_a": int(a.argmax()), "argmax_b": int(b.argmax()), "argmax_equal": bool(a.argmax() == b.argmax()),
           "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "cosine": cos,
           "top5_a": a.topk(5).indices.tolist(), "top5_b": b.topk(5).indices.tolist()}
    log.info("COMPARE %s: %s", name, json.dumps(res))
    RESULTS.setdefault("compare", {})[name] = res
    return res

C = args.chunk
NBLK = math.ceil(N / BLOCK)
blocks_chunked = list(range(40, 40 + NBLK + 2))   # deliberately non-zero-based block ids
blocks_oneshot = list(range(0, NBLK + 2))
assert max(blocks_chunked) < nc.pa_num_blocks, (max(blocks_chunked), nc.pa_num_blocks)

def run_main():
    # --- 1. chunked prefill (fresh cache: nothing written to blocks_chunked yet) ---
    last_chunked = None
    for i in range(N // C):
        s, e = i * C, (i + 1) * C
        last_chunked = cte_call(ids, s, e, blocks_chunked, "chunk%d_p%d" % (i, s))
        hf_e = os.path.join(OUT, "hf_last_%d.pt" % e)
        if os.path.exists(hf_e):
            compare(last_chunked, torch.load(hf_e), "chunk%d_end%d_vs_hf" % (i, e))
    RESULTS["chunked_last_argmax"] = int(last_chunked.argmax())
    torch.save(last_chunked, os.path.join(OUT, "%s_chunked_last.pt" % args.tag))

    # --- 2. one-shot prefill into a different block range ---
    last_oneshot = cte_call(ids, 0, N, blocks_oneshot, "oneshot%d" % N)
    torch.save(last_oneshot, os.path.join(OUT, "%s_oneshot_last.pt" % args.tag))

    compare(last_chunked, last_oneshot, "chunked_vs_oneshot_last_logits")

    # --- 3. greedy decode from both KV states ---
    toks_chunked = greedy_decode(last_chunked, N, blocks_chunked, "chunked", args.decode)
    toks_oneshot = greedy_decode(last_oneshot, N, blocks_oneshot, "oneshot", args.decode)
    RESULTS["decode_chunked"] = toks_chunked
    RESULTS["decode_oneshot"] = toks_oneshot
    RESULTS["decode_identical"] = toks_chunked == toks_oneshot
    first_div = next((i for i, (x, y) in enumerate(zip(toks_chunked, toks_oneshot)) if x != y), None)
    RESULTS["decode_first_divergence"] = first_div
    log.info("DECODE identical=%s first_divergence=%s", RESULTS["decode_identical"], first_div)
    return last_chunked, last_oneshot, toks_chunked, toks_oneshot

if not args.skip_main:
    last_chunked, last_oneshot, toks_chunked, toks_oneshot = run_main()

# --- 4. HF reference comparison ---
hf_last = os.path.join(OUT, "hf_last_%d.pt" % N)
if os.path.exists(hf_last) and not args.skip_main:
    hf = torch.load(hf_last)
    compare(last_oneshot, hf, "oneshot_vs_hf_fp32")
    compare(last_chunked, hf, "chunked_vs_hf_fp32")
    hf_gen = torch.load(os.path.join(OUT, "hf_gen32.pt")).tolist()[:args.decode]
    RESULTS["decode_hf"] = hf_gen
    RESULTS["decode_oneshot_matches_hf_prefix_len"] = next((i for i, (x, y) in enumerate(zip(toks_oneshot, hf_gen)) if x != y), len(hf_gen))
    RESULTS["decode_chunked_matches_hf_prefix_len"] = next((i for i, (x, y) in enumerate(zip(toks_chunked, hf_gen)) if x != y), len(hf_gen))
    log.info("HF greedy: %s", hf_gen)
    log.info("oneshot matches HF for first %d toks; chunked matches HF for first %d toks",
             RESULTS["decode_oneshot_matches_hf_prefix_len"], RESULTS["decode_chunked_matches_hf_prefix_len"])
else:
    log.warning("no HF reference at %s", hf_last)

# --- 5. B3: non-bucket prefix (dent #1) ---
if not args.skip_b3:
    P = args.b3_prefix
    E = P + C
    # one-shot to E in blocks_oneshot (overwrites)
    last_b3_oneshot = cte_call(ids, 0, E, blocks_oneshot, "b3_oneshot%d" % E)
    # prefix P as a single CTE call (prefix 0), then a chunk with computed=P (non-bucket prefix)
    cte_call(ids, 0, P, blocks_chunked, "b3_prefix%d" % P)
    last_b3_chunk = cte_call(ids, P, E, blocks_chunked, "b3_chunk_p%d" % P)
    compare(last_b3_chunk, last_b3_oneshot, "B3_nonbucket_prefix%d_vs_oneshot%d" % (P, E))
    hf_b3 = os.path.join(OUT, "hf_last_%d.pt" % E)
    if os.path.exists(hf_b3):
        compare(last_b3_chunk, torch.load(hf_b3), "B3_nonbucket_prefix%d_vs_hf" % P)
        compare(last_b3_oneshot, torch.load(hf_b3), "B3_oneshot%d_vs_hf" % E)
    # short decode from the B3 chunked state to see if garbage prior shows in decode
    toks_b3 = greedy_decode(last_b3_chunk, E, blocks_chunked, "b3chunked", 8)
    toks_b3_one = greedy_decode(last_b3_oneshot, E, blocks_oneshot, "b3oneshot", 8)
    RESULTS["b3_decode_chunked"] = toks_b3
    RESULTS["b3_decode_oneshot"] = toks_b3_one
    RESULTS["b3_decode_identical"] = toks_b3 == toks_b3_one
    hf8 = os.path.join(OUT, "hf_gen8_1912.pt")
    if os.path.exists(hf8) and E == 1912:
        h = torch.load(hf8).tolist()
        RESULTS["b3_decode_hf"] = h
        log.info("B3 HF greedy from %d: %s ; chunked==HF %s ; oneshot==HF %s", E, h, toks_b3 == h, toks_b3_one == h)
    if args.stale_test:
        # plant stale "future" KV: full 2048 chunked prefill into blocks_chunked, then redo B3 on the same blocks
        for i in range(N // C):
            s, e = i * C, (i + 1) * C
            cte_call(ids, s, e, blocks_chunked, "stale_chunk%d_p%d" % (i, s))
        cte_call(ids, 0, P, blocks_chunked, "stale_b3_prefix%d" % P)
        last_b3_chunk2 = cte_call(ids, P, E, blocks_chunked, "stale_b3_chunk_p%d" % P)
        compare(last_b3_chunk2, last_b3_oneshot, "STALE_B3_prefix%d_vs_oneshot%d" % (P, E))
        toks_b3_stale = greedy_decode(last_b3_chunk2, E, blocks_chunked, "stale_b3chunked", 8)
        RESULTS["b3_decode_chunked_after_stale"] = toks_b3_stale
        log.info("STALE test: fresh b3chunked=%s after-stale b3chunked=%s equal=%s", toks_b3, toks_b3_stale, toks_b3 == toks_b3_stale)
        # same for the one-shot range: plant one-shot 2048 into blocks_oneshot then redo one-shot 1912 + decode
        cte_call(ids, 0, N, blocks_oneshot, "stale_oneshot%d" % N)
        last_b3_oneshot2 = cte_call(ids, 0, E, blocks_oneshot, "stale_b3_oneshot%d" % E)
        toks_b3_one_stale = greedy_decode(last_b3_oneshot2, E, blocks_oneshot, "stale_b3oneshot", 8)
        RESULTS["b3_decode_oneshot_after_stale"] = toks_b3_one_stale
        log.info("STALE test: fresh b3oneshot=%s after-stale b3oneshot=%s equal=%s", toks_b3_one, toks_b3_one_stale, toks_b3_one == toks_b3_one_stale)

mark("done")
RESULTS["routes"] = ROUTES
RESULTS["marks"] = MARKS
with open(os.path.join(OUT, "%s_results.json" % args.tag), "w") as f:
    json.dump(RESULTS, f, indent=1)
log.info("PROBE_DONE wrote %s", os.path.join(OUT, "%s_results.json" % args.tag))

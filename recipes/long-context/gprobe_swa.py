#!/usr/bin/env python
"""
HARDENED SWA chunked-prefill correctness probe for gemma4-E2B on inf2 (block-KV prefix-caching CTE).

Fixes vs the old gprobe.py:
  * every KV state lives in its OWN disjoint block range and is decoded IMMEDIATELY after its prefill
    (no chunked/oneshot block overlap -> no self-corruption).
  * POSITION PROBE: query token at chunk offset i=0 / i=1 with prefix P in {1024,1536,1400}, so the
    tail-W gather + sliding prior mask are actually exercised (511/510 gathered rows visible), and the
    resulting logits are compared to HF fp32 logits at the SAME absolute position.
Every PASS/FAIL is derived from a printed logit diff.
"""
import os, sys, time, json, math, logging, argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--compiled", default=os.path.expanduser("~/gemma_pc"))
ap.add_argument("--out", default="/workspace/hf_out")
ap.add_argument("--tag", default="swa")
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--n", type=int, default=2048)
ap.add_argument("--decode", type=int, default=32)
ap.add_argument("--b3-prefix", type=int, default=1400)
ap.add_argument("--probe-P", default="1024,1536,1400,512,1000,1055")
ap.add_argument("--skip-main", action="store_true")
ap.add_argument("--skip-b3", action="store_true")
ap.add_argument("--skip-probe", action="store_true")
ap.add_argument("--xref-tag", default=None, help="compare probe/oneshot logits against saved device logits of another build (e.g. the W=512 build)")
args = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("probe")

sys.path.insert(0, "/workspace/gemma4_nxdi")
from modeling_gemma4 import NeuronGemma4ForCausalLM
import neuronx_distributed_inference.models.model_wrapper as mw

OUT = args.out
RESULTS = {"tag": args.tag, "chunk": args.chunk, "n": args.n, "calls": [], "compare": {}, "verdicts": {}}
ROUTES = []

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

t0 = time.time()
model = NeuronGemma4ForCausalLM(args.compiled)
nc = model.config.neuron_config
log.info("neuron_config: tp=%s bs=%s max_ctx=%s seq_len=%s prefix_caching=%s block_kv=%s pa_block=%s pa_num_blocks=%s "
         "cte_buckets=%s prefix_buckets=%s tkg_buckets=%s attn_kernel=%s sliding_window=%s",
         nc.tp_degree, nc.batch_size, nc.max_context_length, nc.seq_len, nc.is_prefix_caching, nc.is_block_kv_layout,
         nc.pa_block_size, nc.pa_num_blocks, nc.context_encoding_buckets, nc.prefix_buckets, nc.token_generation_buckets,
         getattr(nc, "attn_kernel_enabled", None), model.config.sliding_window)
BLOCK = nc.pa_block_size
W = int(model.config.sliding_window)
PAD_ID = 0
model.load(args.compiled)
log.info("model loaded in %.1fs", time.time() - t0)

ids = torch.load(os.path.join(OUT, "ids.pt"))[:, :args.n]
N = ids.shape[1]
assert N == args.n
log.info("ids %s first5=%s", tuple(ids.shape), ids[0, :5].tolist())

_HF_ALL = {}
def hf_all(Wx):
    f = os.path.join(OUT, "hf_all_W%d.pt" % Wx)
    if Wx not in _HF_ALL:
        _HF_ALL[Wx] = torch.load(f) if os.path.exists(f) else None
    return _HF_ALL[Wx]

def hf_at(pos, Wx=512):
    """HF fp32 logits for absolute position `pos` (0-based logit index)."""
    a = hf_all(Wx)
    if a is not None:
        return a[pos].clone()
    f = os.path.join(OUT, "hf_pos_%d.pt" % pos) if Wx == 512 else os.path.join(OUT, "hf_pos_%d_W%d.pt" % (pos, Wx))
    if os.path.exists(f):
        return torch.load(f)
    f2 = os.path.join(OUT, "hf_last_%d.pt" % (pos + 1))
    if os.path.exists(f2):
        return torch.load(f2)
    return None

def slots_for(blocks, positions):
    return torch.tensor([[int(blocks[p // BLOCK]) * BLOCK + (p % BLOCK) for p in positions]], dtype=torch.long)

def routed():
    return "CTE" if model.base_model is model.context_encoding_model else "TKG"

def _record(name, **kw):
    kw["name"] = name
    RESULTS["calls"].append(kw)

def cte_call(ids_full, start, end, blocks, name):
    """Wrapper-driven CTE: prefix [0,start) in cache via `blocks`, encode [start,end). Returns logits @ end-1."""
    input_ids = ids_full[:, :end]
    position_ids = torch.arange(end, dtype=torch.long)[None]
    attention_mask = torch.ones(1, end, dtype=torch.long)
    slot_mapping = slots_for(blocks, range(start, end))
    nb = math.ceil(start / BLOCK)
    block_table = torch.tensor([[int(b) for b in blocks[:nb]]], dtype=torch.long) if nb > 0 else torch.zeros((1, 1), dtype=torch.long)
    nroutes = len(ROUTES)
    t0 = time.time()
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                seq_ids=torch.tensor([0]), slot_mapping=slot_mapping, block_table=block_table,
                full_context_lens=torch.tensor([[end]], dtype=torch.long),
                computed_context_lens=torch.tensor([[start]], dtype=torch.long))
    dt = time.time() - t0
    last = out.logits[0, -1].float().clone()
    r = routed(); rts = ROUTES[nroutes:]
    log.info("%s: start=%d end=%d routed=%s buckets=%s logits.shape=%s argmax=%d dt=%.3fs", name, start, end, r, rts, tuple(out.logits.shape), int(last.argmax()), dt)
    _record(name, kind="cte", start=start, end=end, routed=r, buckets=rts, dt=dt, argmax=int(last.argmax()), abs_pos=end - 1)
    assert r == "CTE", "expected CTE NEFF for %s, got %s" % (name, r)
    return last

def probe_call(ids_full, P, k, blocks, name):
    """POSITION PROBE. Prefix [0,P) already in cache via `blocks`. Active chunk = tokens [P, P+k) at chunk
    offsets 0..k-1, rest of the 512-wide chunk is pad (position_id 1, slot -1). Returns logits @ absolute P+k-1
    (the wrapper gathers hidden at argmax(position_ids) = offset k-1).
    Wrapper contract: num_queries = full-computed = 512 -> prefill bucket 512, extra_prefill_slots=0,
    adjusted_prefix_len=P -> input_ids[:, P:] is our 512-wide chunk; attention_mask[:, :P] -> prefix validity."""
    C = args.chunk
    assert 1 <= k <= C
    input_ids = torch.cat([ids_full[:, :P + k], torch.full((1, C - k), PAD_ID, dtype=torch.long)], dim=1)
    position_ids = torch.cat([torch.arange(P + k, dtype=torch.long)[None], torch.ones(1, C - k, dtype=torch.long)], dim=1)
    attention_mask = torch.ones(1, P + C, dtype=torch.long)
    slot_mapping = torch.cat([slots_for(blocks, range(P, P + k)), torch.full((1, C - k), -1, dtype=torch.long)], dim=1)
    nb = math.ceil(P / BLOCK)
    block_table = torch.tensor([[int(b) for b in blocks[:nb]]], dtype=torch.long)
    assert input_ids.shape == position_ids.shape == (1, P + C) and slot_mapping.shape == (1, C)
    assert int(position_ids.min()) == 0 and int(position_ids.argmax()) == P + k - 1
    nroutes = len(ROUTES)
    t0 = time.time()
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                seq_ids=torch.tensor([0]), slot_mapping=slot_mapping, block_table=block_table,
                full_context_lens=torch.tensor([[P + C]], dtype=torch.long),
                computed_context_lens=torch.tensor([[P]], dtype=torch.long))
    dt = time.time() - t0
    last = out.logits[0, -1].float().clone()
    r = routed(); rts = ROUTES[nroutes:]
    log.info("PROBE %s: P=%d k=%d (query at chunk offset %d, abs pos %d) routed=%s buckets=%s argmax=%d dt=%.3fs",
             name, P, k, k - 1, P + k - 1, r, rts, int(last.argmax()), dt)
    _record(name, kind="probe", P=P, k=k, abs_pos=P + k - 1, routed=r, buckets=rts, dt=dt, argmax=int(last.argmax()))
    assert r == "CTE", "expected CTE NEFF for %s, got %s" % (name, r)
    exp_prefix = min(b for b in nc.prefix_buckets if b >= P)
    assert rts and rts[-1][1] == [C, exp_prefix], "probe routed to %s, expected [%d,%d]" % (rts, C, exp_prefix)
    return last

def tkg_step(tok, pos, blocks, name):
    input_ids = torch.tensor([[int(tok)]], dtype=torch.long)
    position_ids = torch.tensor([[pos]], dtype=torch.long)
    attention_mask = torch.ones(1, pos, dtype=torch.long)
    slot_mapping = slots_for(blocks, [pos])
    nb = math.ceil((pos + 1) / BLOCK)
    block_table = torch.tensor([[int(b) for b in blocks[:nb]]], dtype=torch.long)
    nroutes = len(ROUTES)
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                seq_ids=torch.tensor([0]), slot_mapping=slot_mapping, block_table=block_table,
                full_context_lens=torch.tensor([[pos + 1]], dtype=torch.long),
                computed_context_lens=torch.tensor([[pos]], dtype=torch.long))
    r = routed()
    assert r == "TKG", "expected TKG NEFF at %s pos %d, got %s" % (name, pos, r)
    return out.logits[0, -1].float().clone(), ROUTES[nroutes:]

def greedy_decode(first_logits, start_pos, blocks, name, steps):
    toks = [int(first_logits.argmax())]
    t0 = time.time(); rts = None
    for i in range(steps - 1):
        last, rts = tkg_step(toks[-1], start_pos + i, blocks, name)
        toks.append(int(last.argmax()))
    dt = time.time() - t0
    log.info("%s decode %d steps in %.2fs (%.1f ms/tok) last tkg bucket=%s toks=%s", name, steps - 1, dt, 1000 * dt / max(1, steps - 1), rts, toks)
    return toks

def compare(a, b, name):
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a[None], b[None]).item()
    res = {"argmax_a": int(a.argmax()), "argmax_b": int(b.argmax()), "argmax_equal": bool(a.argmax() == b.argmax()),
           "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()), "cosine": cos,
           "top5_a": a.topk(5).indices.tolist(), "top5_b": b.topk(5).indices.tolist()}
    log.info("COMPARE %s: max|d|=%.4f mean|d|=%.5f cos=%.6f argmax %d vs %d equal=%s top5 %s vs %s", name,
             res["max_abs_diff"], res["mean_abs_diff"], cos, res["argmax_a"], res["argmax_b"], res["argmax_equal"], res["top5_a"], res["top5_b"])
    RESULTS["compare"][name] = res
    return res

def dev_save(name, t):
    torch.save(t, os.path.join(OUT, "%s_dev_%s.pt" % (args.tag, name)))

def xref(name, lg, pos):
    """Cross-build comparison: this build's probe logits vs the reference build's one-shot AND probe logits at the same position."""
    if not args.xref_tag:
        return
    for refname in ("oneshot_pos%d" % pos, name):
        f = os.path.join(OUT, "%s_dev_%s.pt" % (args.xref_tag, refname))
        if os.path.exists(f):
            compare(lg, torch.load(f), "XREF_%s_%s_vs_%s_%s" % (args.tag, name, args.xref_tag, refname))

def verdict(name, ok, detail):
    RESULTS["verdicts"][name] = {"pass": bool(ok), "detail": detail}
    log.info("VERDICT %s: %s -- %s", name, "PASS" if ok else "FAIL", detail)

C = args.chunk
NBLK = math.ceil((N + args.decode) / BLOCK)
# disjoint block ranges (pa_num_blocks=256, ids < 256; block 256 is the reserved pad block)
RANGE_A = list(range(0, 66))
RANGE_B = list(range(66, 132))
RANGE_C = list(range(132, 192))
RANGE_D = list(range(192, 252))
for r in (RANGE_A, RANGE_B, RANGE_C, RANGE_D):
    assert max(r) < nc.pa_num_blocks
assert NBLK <= len(RANGE_A)

# ======================= 1. global-layer sanity: chunked vs one-shot, sequential, disjoint blocks ================
if not args.skip_main:
    log.info("=== SECTION 1: chunked prefill (blocks %d-%d) -> decode; then one-shot (blocks %d-%d) -> decode", RANGE_A[0], RANGE_A[-1], RANGE_B[0], RANGE_B[-1])
    last_chunked = None
    for i in range(N // C):
        s, e = i * C, (i + 1) * C
        last_chunked = cte_call(ids, s, e, RANGE_A, "chunk%d_p%d" % (i, s))
        hf = hf_at(e - 1)
        if hf is not None:
            compare(last_chunked, hf, "chunk%d_end%d_vs_hf" % (i, e))
    toks_chunked = greedy_decode(last_chunked, N, RANGE_A, "chunked", args.decode)
    torch.save(last_chunked, os.path.join(OUT, "%s_chunked_last.pt" % args.tag))

    last_oneshot = cte_call(ids, 0, N, RANGE_B, "oneshot%d" % N)
    toks_oneshot = greedy_decode(last_oneshot, N, RANGE_B, "oneshot", args.decode)
    torch.save(last_oneshot, os.path.join(OUT, "%s_oneshot_last.pt" % args.tag))

    r_co = compare(last_chunked, last_oneshot, "chunked_vs_oneshot_last_logits_%d" % N)
    hf = hf_at(N - 1)
    r_oh = compare(last_oneshot, hf, "oneshot%d_vs_hf_fp32" % N)
    r_ch = compare(last_chunked, hf, "chunked%d_vs_hf_fp32" % N)
    RESULTS["decode_chunked"] = toks_chunked
    RESULTS["decode_oneshot"] = toks_oneshot
    first_div = next((i for i, (x, y) in enumerate(zip(toks_chunked, toks_oneshot)) if x != y), None)
    hf_gen = torch.load(os.path.join(OUT, "hf_gen8_2048.pt")).tolist()
    m_one = next((i for i, (x, y) in enumerate(zip(toks_oneshot, hf_gen)) if x != y), len(hf_gen))
    m_chk = next((i for i, (x, y) in enumerate(zip(toks_chunked, hf_gen)) if x != y), len(hf_gen))
    log.info("DECODE chunked==oneshot first_divergence=%s ; HF greedy8=%s ; oneshot matches HF first %d ; chunked matches HF first %d",
             first_div, hf_gen, m_one, m_chk)
    RESULTS.update({"decode_first_divergence": first_div, "decode_hf8": hf_gen, "oneshot_hf_match": m_one, "chunked_hf_match": m_chk})
    verdict("S1_chunked_vs_hf_within_floor",
            r_ch["argmax_equal"] and r_ch["max_abs_diff"] <= 1.5 * r_oh["max_abs_diff"] + 0.05 and r_ch["cosine"] >= r_oh["cosine"] - 1e-4,
            "chunked-vs-HF max|d|=%.4f cos=%.6f  vs floor oneshot-vs-HF max|d|=%.4f cos=%.6f ; chunked-vs-oneshot max|d|=%.4f" % (
                r_ch["max_abs_diff"], r_ch["cosine"], r_oh["max_abs_diff"], r_oh["cosine"], r_co["max_abs_diff"]))
    verdict("S1_decode8_matches_hf", m_chk >= 8 and m_one >= 8, "chunked matches HF greedy first %d/8, oneshot %d/8, chunked==oneshot for %s" % (m_chk, m_one, "all %d" % len(toks_chunked) if first_div is None else "first %d" % first_div))

# ======================= 2. B3: non-bucket prefix ================================================================
if not args.skip_b3:
    P = args.b3_prefix; E = P + C
    log.info("=== SECTION 2 (B3): one-shot %d (blocks %d-%d) -> decode; prefix %d + chunk (blocks %d-%d) -> decode", E, RANGE_C[0], RANGE_C[-1], P, RANGE_D[0], RANGE_D[-1])
    last_b3_oneshot = cte_call(ids, 0, E, RANGE_C, "b3_oneshot%d" % E)
    toks_b3_one = greedy_decode(last_b3_oneshot, E, RANGE_C, "b3oneshot", 8)
    cte_call(ids, 0, P, RANGE_D, "b3_prefix%d" % P)
    last_b3_chunk = cte_call(ids, P, E, RANGE_D, "b3_chunk_p%d" % P)
    toks_b3 = greedy_decode(last_b3_chunk, E, RANGE_D, "b3chunked", 8)
    r_b3 = compare(last_b3_chunk, last_b3_oneshot, "B3_nonbucket_prefix%d_vs_oneshot%d" % (P, E))
    hf = hf_at(E - 1)
    r_b3h = compare(last_b3_chunk, hf, "B3_nonbucket_prefix%d_vs_hf" % P)
    r_b3oh = compare(last_b3_oneshot, hf, "B3_oneshot%d_vs_hf" % E)
    h8 = torch.load(os.path.join(OUT, "hf_gen8_1912.pt")).tolist() if E == 1912 else None
    log.info("B3 decode chunked=%s oneshot=%s HF=%s ; chunked==HF %s ; oneshot==HF %s", toks_b3, toks_b3_one, h8, toks_b3 == h8, toks_b3_one == h8)
    RESULTS.update({"b3_decode_chunked": toks_b3, "b3_decode_oneshot": toks_b3_one, "b3_decode_hf": h8})
    verdict("S2_B3_prefix%d_vs_hf_within_floor" % P,
            r_b3h["argmax_equal"] and r_b3h["max_abs_diff"] <= 1.5 * r_b3oh["max_abs_diff"] + 0.05 and r_b3h["cosine"] >= r_b3oh["cosine"] - 1e-4,
            "B3chunk-vs-HF max|d|=%.4f cos=%.6f  vs floor oneshot%d-vs-HF max|d|=%.4f cos=%.6f ; decode8 chunked==HF %s" % (
                r_b3h["max_abs_diff"], r_b3h["cosine"], E, r_b3oh["max_abs_diff"], r_b3oh["cosine"], toks_b3 == h8))

# ======================= 3. POSITION PROBE ======================================================================
if not args.skip_probe:
    PS = [int(x) for x in args.probe_P.split(",")]
    for P in PS:
        assert P % BLOCK == 0 or True  # non-block-aligned P allowed (1400): gather rows [P-W, P)
        gather_lo = P - W
        log.info("=== SECTION 3: POSITION PROBE P=%d (gather rows [%d,%d), %s) ===", P, gather_lo, P, "block-aligned" if P % BLOCK == 0 else "NON-block-aligned")
        # noise-floor references: one-shot no-prefix prefill of [0,P+k) -> logits @ P+k-1  (blocks RANGE_B, fresh overwrite)
        ref_one = {}
        for k in (1, 2):
            ref_one[k] = cte_call(ids, 0, P + k, RANGE_B, "P%d_oneshot_to_%d" % (P, P + k))
        # (a) prefix built by ONE-SHOT no-prefix prefill of [0,P) into RANGE_A, then probe k=1,2
        cte_call(ids, 0, P, RANGE_A, "P%d_prefix_oneshot" % P)
        for k in (1, 2):
            pos = P + k - 1
            lg = probe_call(ids, P, k, RANGE_A, "P%d_k%d_oneshotprefix" % (P, k))
            dev_save("probe_P%d_k%d_oneshotprefix" % (P, k), lg)
            dev_save("oneshot_pos%d" % pos, ref_one[k])
            xref("probe_P%d_k%d_oneshotprefix" % (P, k), lg, pos)
            hf = hf_at(pos)
            assert hf is not None, "no HF ref for abs pos %d" % pos
            r_ph = compare(lg, hf, "PROBE_P%d_k%d_pos%d_oneshotprefix_vs_hf" % (P, k, pos))
            r_oh = compare(ref_one[k], hf, "FLOOR_oneshot_pos%d_vs_hf" % pos)
            r_po = compare(lg, ref_one[k], "PROBE_P%d_k%d_pos%d_vs_oneshot" % (P, k, pos))
            verdict("S3_PROBE_P%d_k%d_pos%d_oneshotprefix" % (P, k, pos),
                    r_ph["argmax_equal"] and r_ph["max_abs_diff"] <= 1.5 * r_oh["max_abs_diff"] + 0.05 and r_ph["cosine"] >= r_oh["cosine"] - 1e-4,
                    "probe-vs-HF max|d|=%.4f cos=%.6f argmax_eq=%s | floor oneshot-vs-HF max|d|=%.4f cos=%.6f | probe-vs-oneshot max|d|=%.4f cos=%.6f" % (
                        r_ph["max_abs_diff"], r_ph["cosine"], r_ph["argmax_equal"], r_oh["max_abs_diff"], r_oh["cosine"], r_po["max_abs_diff"], r_po["cosine"]))
        # TKG continuation on the k=2 state: feed the true token at P+2 -> logits @ P+2 (exercises TKG sliding mask at odd positions)
        hf3 = hf_at(P + 2)
        if hf3 is not None:
            lg3, rts = tkg_step(ids[0, P + 2], P + 2, RANGE_A, "P%d_tkg" % P)
            r_t = compare(lg3, hf3, "TKG_after_probe_P%d_pos%d_vs_hf" % (P, P + 2))
            log.info("TKG step bucket %s", rts)
            verdict("S3_TKG_after_probe_P%d_pos%d" % (P, P + 2), r_t["argmax_equal"] and r_t["cosine"] > 0.99,
                    "tkg-vs-HF max|d|=%.4f cos=%.6f argmax_eq=%s (no same-position oneshot floor; cos>0.99+argmax gate)" % (r_t["max_abs_diff"], r_t["cosine"], r_t["argmax_equal"]))
        # (b) prefix built by CHUNKED prefill (512-chunks, last chunk possibly non-bucket) into RANGE_C, then probe k=1,2
        s = 0
        while s < P:
            e = min(s + C, P)
            cte_call(ids, s, e, RANGE_C, "P%d_prefix_chunk_%d_%d" % (P, s, e))
            s = e
        for k in (1, 2):
            pos = P + k - 1
            lg = probe_call(ids, P, k, RANGE_C, "P%d_k%d_chunkedprefix" % (P, k))
            dev_save("probe_P%d_k%d_chunkedprefix" % (P, k), lg)
            xref("probe_P%d_k%d_chunkedprefix" % (P, k), lg, pos)
            hf = hf_at(pos)
            r_ph = compare(lg, hf, "PROBE_P%d_k%d_pos%d_chunkedprefix_vs_hf" % (P, k, pos))
            r_oh = RESULTS["compare"]["FLOOR_oneshot_pos%d_vs_hf" % pos]
            r_po = compare(lg, ref_one[k], "PROBE_P%d_k%d_pos%d_chunkedprefix_vs_oneshot" % (P, k, pos))
            verdict("S3_PROBE_P%d_k%d_pos%d_chunkedprefix" % (P, k, pos),
                    r_ph["argmax_equal"] and r_ph["max_abs_diff"] <= 1.5 * r_oh["max_abs_diff"] + 0.05 and r_ph["cosine"] >= r_oh["cosine"] - 1e-4,
                    "probe-vs-HF max|d|=%.4f cos=%.6f argmax_eq=%s | floor oneshot-vs-HF max|d|=%.4f cos=%.6f | probe-vs-oneshot max|d|=%.4f cos=%.6f" % (
                        r_ph["max_abs_diff"], r_ph["cosine"], r_ph["argmax_equal"], r_oh["max_abs_diff"], r_oh["cosine"], r_po["max_abs_diff"], r_po["cosine"]))
        # sensitivity (HF-only): what an off-by-one window would do at these positions
        for k in (1, 2):
            pos = P + k - 1
            for Wx in (511, 513):
                alt = hf_at(pos, Wx)
                if alt is not None:
                    compare(alt, hf_at(pos, 512), "SENSITIVITY_hfW%d_vs_hfW512_pos%d" % (Wx, pos))
            # which HF window is this build's probe closest to? (mean|d| over vocab)
            for src in ("oneshotprefix", "chunkedprefix"):
                f = os.path.join(OUT, "%s_dev_probe_P%d_k%d_%s.pt" % (args.tag, P, k, src))
                if os.path.exists(f):
                    lgp = torch.load(f)
                    ds = {Wx: float((lgp - hf_at(pos, Wx)).abs().mean()) for Wx in (511, 512, 513) if hf_at(pos, Wx) is not None}
                    log.info("NEAREST_HF_WINDOW probe P=%d k=%d %s pos=%d: mean|d| per HF window %s -> nearest W=%s", P, k, src, pos,
                             {a: round(b, 5) for a, b in ds.items()}, min(ds, key=ds.get))
                    RESULTS.setdefault("nearest_hf_window", {})["P%d_k%d_%s" % (P, k, src)] = ds

RESULTS["routes"] = ROUTES
with open(os.path.join(OUT, "%s_results.json" % args.tag), "w") as f:
    json.dump(RESULTS, f, indent=1)
log.info("==== VERDICT SUMMARY ====")
allok = True
for k, v in RESULTS["verdicts"].items():
    log.info("%-50s %s  %s", k, "PASS" if v["pass"] else "FAIL", v["detail"])
    allok &= v["pass"]
log.info("OVERALL: %s", "ALL PASS" if allok else "SOME FAIL")
log.info("PROBE_DONE wrote %s", os.path.join(OUT, "%s_results.json" % args.tag))

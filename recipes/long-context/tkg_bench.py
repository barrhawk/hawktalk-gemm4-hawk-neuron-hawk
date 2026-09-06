#!/usr/bin/env python
"""Per-TKG-bucket decode benchmark on a multi-bucket block-KV gemma4 build.

Builds the same needle prompt as longctx.py, runs 512-token chunked prefill, and at each
chunk-aligned checkpoint position P (default: 64 tokens under every TKG bucket, plus the
first position AT a bucket boundary so the switch to the next bucket is exercised) runs
--decode greedy steps and reports ms/tok + the TKG bucket NxDI routed to. Decode rows are
written at absolute slots P..P+decode-1 and are overwritten by the next prefill chunk, so
the checkpoints do not perturb the prefill (later chunks only read prior rows < chunk start).
The final checkpoint is at the end of the prompt, so it doubles as the needle check.

Asserts, per checkpoint: routed bucket == smallest configured bucket b with b > position.
Usage (on the inf2 box, nxdi venv):
  python swa/tkg_bench.py --compiled ~/gemma_pc_128k_mb --n 130000 --decode 32 --tag mb
Compare against the single-bucket build: --compiled ~/gemma_pc_128k --tag single
"""
import os, sys, time, json, math, logging, argparse, random
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--compiled", default=os.path.expanduser("~/gemma_pc_128k_mb"))
ap.add_argument("--model", default="/workspace/real-gemma4-E2B-it")
ap.add_argument("--n", type=int, default=130000, help="filler+needle token count before the question")
ap.add_argument("--needle-pos", type=int, default=200)
ap.add_argument("--decode", type=int, default=32)
ap.add_argument("--chunk", type=int, default=512)
ap.add_argument("--checkpoints", default="", help="comma list of decode positions; default = 64 under each TKG bucket + at each boundary")
ap.add_argument("--tag", default="tkgbench")
ap.add_argument("--out", default="/workspace/longctx")
ap.add_argument("--seed", type=int, default=1234)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tkgbench")

sys.path.insert(0, "/workspace/gemma4_nxdi")
from modeling_gemma4 import NeuronGemma4ForCausalLM
import neuronx_distributed_inference.models.model_wrapper as mw
from transformers import PreTrainedTokenizerFast

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

# ---------------- prompt (identical construction to longctx.py) ----------------
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.model, "tokenizer.json"))
NEEDLE = "The secret passphrase is ORCHID-7423-FALCON."
QUESTION = "\n\nQuestion: What is the secret passphrase that was mentioned earlier in this document? Answer: The secret passphrase is"
random.seed(args.seed)
WORDS = ("the river valley mountain harbor quiet lantern copper meadow signal orchard winter granite velvet "
         "morning traveler compass thunder willow silver market bridge garden window forest canyon island "
         "engine ledger parcel candle horizon anchor summer ribbon marble tunnel ember pillar cedar saddle").split()
def sentence():
    n = random.randint(6, 14)
    w = [random.choice(WORDS) for _ in range(n)]
    w[0] = w[0].capitalize()
    return " ".join(w) + random.choice([". ", ". ", ". ", ", and then again. ", "; so it goes. "])
C = args.chunk
t0 = time.time()
filler_ids = []
while len(filler_ids) < args.n + 4096:
    txt = "".join(sentence() for _ in range(2000))
    filler_ids += tok(txt, add_special_tokens=False).input_ids
needle_ids = tok(" " + NEEDLE + " ", add_special_tokens=False).input_ids
q_ids = tok(QUESTION, add_special_tokens=False).input_ids
body = filler_ids[:args.needle_pos] + needle_ids + filler_ids[args.needle_pos:args.n - len(needle_ids)]
assert len(body) == args.n
ids = torch.tensor([[2] + body + q_ids], dtype=torch.long)   # BOS
TOTAL = ids.shape[1]
log.info("prompt built in %.1fs: total=%d tokens", time.time() - t0, TOTAL)

# ---------------- model ----------------
t0 = time.time()
model = NeuronGemma4ForCausalLM(args.compiled)
nc = model.config.neuron_config
BLOCK = nc.pa_block_size
TKG_BUCKETS = sorted(int(b) for b in (nc.token_generation_buckets or [nc.seq_len]))
log.info("neuron_config: seq_len=%s max_ctx=%s cte=%s prefix=%s tkg=%s pa_num_blocks=%s", nc.seq_len, nc.max_context_length,
         nc.context_encoding_buckets, nc.prefix_buckets, nc.token_generation_buckets, nc.pa_num_blocks)
assert TOTAL - C <= max(nc.prefix_buckets), "last chunk prefix %d exceeds max prefix bucket" % (TOTAL - C)
assert TOTAL + args.decode <= nc.seq_len
model.load(args.compiled)
log.info("model loaded in %.1fs", time.time() - t0)
NB = math.ceil((TOTAL + args.decode) / BLOCK) + 1
blocks = list(range(NB))
assert NB < nc.pa_num_blocks, (NB, nc.pa_num_blocks)

def expected_bucket(pos):
    """Mirror of model_wrapper.get_target_2d_bucket_for_prefix_caching (TKG branch): smallest b with b > computed_context_len."""
    for b in TKG_BUCKETS:
        if b > pos:
            return b
    raise ValueError("position %d beyond largest TKG bucket %d" % (pos, TKG_BUCKETS[-1]))

# checkpoints: chunk-aligned positions 64 under each bucket, plus the first chunk-aligned position >= each
# non-final bucket boundary (forces the switch to the next bucket), plus the end of the prompt.
if args.checkpoints:
    CKPTS = sorted(set(int(x) for x in args.checkpoints.split(",")))
else:
    CKPTS = set()
    for b in TKG_BUCKETS:
        under = ((b - 64) // C) * C
        if 0 < under < TOTAL:
            CKPTS.add(under)
        at = math.ceil(b / C) * C
        if b < TKG_BUCKETS[-1] and at < TOTAL:
            CKPTS.add(at)
    CKPTS = sorted(CKPTS)
for p in CKPTS:
    assert p % C == 0 and p < TOTAL, "checkpoints must be chunk-aligned and inside the prompt: %d" % p
    # a decode run may legitimately cross a bucket boundary mid-run; decode_at asserts the route per step
log.info("TKG buckets %s -> checkpoints %s (+ end of prompt %d)", TKG_BUCKETS, CKPTS, TOTAL)

def slots_for(positions):
    return torch.tensor([[int(blocks[p // BLOCK]) * BLOCK + (p % BLOCK) for p in positions]], dtype=torch.long)
def routed():
    return "CTE" if model.base_model is model.context_encoding_model else "TKG"

def cte_call(start, end):
    input_ids = ids[:, :end]
    position_ids = torch.arange(end, dtype=torch.long)[None]
    attention_mask = torch.ones(1, end, dtype=torch.long)
    slot_mapping = slots_for(range(start, end))
    nb = math.ceil(start / BLOCK)
    block_table = torch.tensor([[int(b) for b in blocks[:nb]]], dtype=torch.long) if nb > 0 else torch.zeros((1, 1), dtype=torch.long)
    nr = len(ROUTES); t = time.time()
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, seq_ids=torch.tensor([0]),
                slot_mapping=slot_mapping, block_table=block_table,
                full_context_lens=torch.tensor([[end]], dtype=torch.long), computed_context_lens=torch.tensor([[start]], dtype=torch.long))
    dt = time.time() - t
    assert routed() == "CTE"
    return out.logits[0, -1].float().clone(), dt, ROUTES[nr:][-1][1]

def tkg_step(tok_id, pos):
    input_ids = torch.tensor([[int(tok_id)]], dtype=torch.long)
    position_ids = torch.tensor([[pos]], dtype=torch.long)
    attention_mask = torch.ones(1, pos, dtype=torch.long)
    slot_mapping = slots_for([pos])
    nb = math.ceil((pos + 1) / BLOCK)
    block_table = torch.tensor([[int(b) for b in blocks[:nb]]], dtype=torch.long)
    nr = len(ROUTES); t = time.time()
    out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, seq_ids=torch.tensor([0]),
                slot_mapping=slot_mapping, block_table=block_table,
                full_context_lens=torch.tensor([[pos + 1]], dtype=torch.long), computed_context_lens=torch.tensor([[pos]], dtype=torch.long))
    dt = time.time() - t
    assert routed() == "TKG"
    return out.logits[0, -1].float().clone(), dt, ROUTES[nr:][-1][1]

def decode_at(first_logits, start, steps, label):
    """Greedy decode `steps` tokens starting at absolute position `start`; returns per-step (dt, bucket)."""
    toks = [int(first_logits.argmax())]
    rows = []
    for j in range(steps - 1):
        pos = start + j
        lg, dt, bk = tkg_step(toks[-1], pos)
        exp = expected_bucket(pos)
        assert int(bk[1]) == exp, "%s: pos %d routed to TKG bucket %s, expected [1, %d]" % (label, pos, bk, exp)
        rows.append((pos, dt, int(bk[1])))
        toks.append(int(lg.argmax()))
    # warm-up excluded: drop the first step (first hit of a new NEFF can include load/prime cost)
    body = rows[1:] if len(rows) > 2 else rows
    by_bucket = {}
    for pos, dt, b in body:
        by_bucket.setdefault(b, []).append(dt)
    for b, v in sorted(by_bucket.items()):
        log.info("DECODE[%s] start=%d bucket=[1,%d] n=%d %.2f ms/tok (%.1f tok/s) min %.2f max %.2f ms",
                 label, start, b, len(v), 1000 * sum(v) / len(v), len(v) / sum(v), 1000 * min(v), 1000 * max(v))
    return toks, rows, {str(b): {"n": len(v), "ms_per_tok": 1000 * sum(v) / len(v), "min_ms": 1000 * min(v), "max_ms": 1000 * max(v)} for b, v in by_bucket.items()}

# ---------------- chunked prefill with decode checkpoints ----------------
results = {"tag": args.tag, "compiled": args.compiled, "tkg_buckets": TKG_BUCKETS, "total_tokens": TOTAL, "checkpoints": {}}
t_pre = time.time()
nchunks = math.ceil(TOTAL / C)
last = None
prefill_s = 0.0
for i in range(nchunks):
    s, e = i * C, min((i + 1) * C, TOTAL)
    last, dt, bk = cte_call(s, e)
    prefill_s += dt
    if e in CKPTS:
        toks, rows, summ = decode_at(last, e, args.decode, "ckpt@%d" % e)
        results["checkpoints"][str(e)] = {"summary": summ, "first_tokens": toks[:8]}
        # the decode wrote rows at e..e+decode-1; the next cte_call(e, e+C) rewrites them with prompt tokens.
    if i % 32 == 0 or i == nchunks - 1:
        log.info("chunk %d/%d [%d,%d) bucket=%s dt=%.3fs (prefill so far %.1fs)", i, nchunks, s, e, bk, dt, prefill_s)
log.info("PREFILL: %d tokens, %.1fs device+host (%.0f tok/s)", TOTAL, prefill_s, TOTAL / prefill_s)

# final checkpoint = end of prompt (needle check + largest bucket timing)
toks, rows, summ = decode_at(last, TOTAL, args.decode, "end@%d" % TOTAL)
text = tok.decode(toks)
found = "ORCHID-7423-FALCON" in text
results["checkpoints"][str(TOTAL)] = {"summary": summ, "first_tokens": toks[:8]}
results.update({"needle_found": found, "decoded": text, "prefill_s": prefill_s})
log.info("DECODED TEXT: %r", text)
log.info("NEEDLE_%s", "FOUND" if found else "MISSING")
json.dump(results, open(os.path.join(args.out, "%s.json" % args.tag), "w"), indent=1)
log.info("TKGBENCH_DONE %s -> %s", args.tag, os.path.join(args.out, "%s.json" % args.tag))

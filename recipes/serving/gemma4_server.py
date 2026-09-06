#!/usr/bin/env python3
"""hawkalphaquick (gemma4-E2B, NxDI port) OpenAI-compatible backend on Inferentia2.
Local only (127.0.0.1:8090); the TLS+auth gateway sits in front. Compiled artifact
is TP2 / batch-16 / seq-4096, so a single request is replicated to fill the batch and
row 0 is returned. Mirrors qwen_server.py exactly; only the model class + paths change."""
import json, os, queue, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from transformers import AutoTokenizer, GenerationConfig, StoppingCriteria, StoppingCriteriaList

# The NxDI gemma4 port must be importable for both construction and reload.
NXDI_PORT = os.environ.get("AQ_NXDI_PORT", "/workspace/gemma4_nxdi")
if NXDI_PORT not in sys.path:
    sys.path.insert(0, NXDI_PORT)
from modeling_gemma4 import NeuronGemma4ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

CP = os.environ.get("AQ_COMPILED", "/workspace/aq-neff")           # neuron_config.json + model.pt + weights/
MP = os.environ.get("AQ_MODEL", "/workspace/real-gemma4-E2B-it")  # tokenizer + chat_template.jinja (agentic/tools)
BATCH = int(os.environ.get("AQ_BATCH", "16"))
PORT = int(os.environ.get("PORT", "8090"))
MODEL_NAME = "hawkalphaquick"

READY = threading.Event()
LOCK = threading.Lock()
STATS = {"reqs": 0, "ptoks": 0, "ctoks": 0, "secs": 0.0, "errs": 0}
tok = model = gen = None

def boot():
    global tok, model, gen
    t = time.time()
    tok = AutoTokenizer.from_pretrained(MP, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # reload path: single-arg constructor reads neuron_config.json from CP, then load()
    m = NeuronGemma4ForCausalLM(CP); m.load(CP)
    model = m; gen = HuggingFaceGenerationAdapter(m)
    print(f"hawkalphaquick ready in {time.time()-t:.1f}s (batch {BATCH})", flush=True)
    READY.set()

def build_generation_config(max_new, temperature, top_p):
    do_sample = temperature is not None and temperature > 0
    kw = {"do_sample": do_sample, "max_new_tokens": max_new,
          "pad_token_id": tok.pad_token_id}
    if do_sample:
        kw["temperature"] = max(temperature, 1e-5)
        kw["top_p"] = top_p
    return GenerationConfig(**kw)

def _render(messages, tools):
    kw = {"tokenize": False, "add_generation_prompt": True}
    if tools:
        kw["tools"] = tools
    return tok.apply_chat_template(messages, **kw)

def run_chat(messages, max_new, temperature, top_p, tools=None):
    prompt = _render(messages, tools)
    enc = tok([prompt], return_tensors="pt", padding=True)
    n_prompt = int(enc.input_ids.shape[1])
    gc = build_generation_config(max_new, temperature, top_p)
    t0 = time.time()
    with LOCK:
        out = gen.generate(enc.input_ids, attention_mask=enc.attention_mask, generation_config=gc)
    dt = time.time() - t0
    new_ids = out[0][n_prompt:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    ct = int((new_ids != tok.pad_token_id).sum())
    STATS["reqs"] += 1; STATS["ptoks"] += n_prompt; STATS["ctoks"] += ct; STATS["secs"] += dt
    print(f"[chat] pt={n_prompt} ct={ct} {ct/max(dt,1e-6):.1f}tok/s {dt:.2f}s", flush=True)
    return text, n_prompt, ct

class _StreamTap(StoppingCriteria):
    def __init__(self, q, n_prompt):
        self.q = q; self.sent = n_prompt
    def __call__(self, input_ids, scores, **kwargs):
        cur = int(input_ids.shape[1])
        if cur > self.sent:
            self.q.put(input_ids[0, self.sent:cur].tolist())
            self.sent = cur
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

class H(BaseHTTPRequestHandler):
    def log_message(s, *a): pass
    def _j(s, code, o):
        b = json.dumps(o).encode()
        s.send_response(code); s.send_header("Content-Type", "application/json")
        s.send_header("Content-Length", str(len(b))); s.end_headers(); s.wfile.write(b)
    def do_GET(s):
        p = s.path.rstrip("/")
        if p in ("/health", "/ping"):
            s._j(200 if READY.is_set() else 503,
                 {"status": "ok" if READY.is_set() else "loading", "model": MODEL_NAME})
        elif p == "/metrics":
            sc = STATS["secs"] or 1
            s._j(200, {**STATS, "tok_s": round(STATS["ctoks"]/sc, 2)})
        else:
            s._j(200, {"model": MODEL_NAME, "backend": "Inferentia2", "batch": BATCH})
    def _stream_chat(s, messages, max_new, temperature, top_p, tools=None):
        try:
            s.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            pass
        prompt = _render(messages, tools)
        enc = tok([prompt], return_tensors="pt", padding=True)
        n_prompt = int(enc.input_ids.shape[1])
        gc = build_generation_config(max_new, temperature, top_p)
        q = queue.Queue()
        def work():
            try:
                with LOCK:
                    gen.generate(enc.input_ids, attention_mask=enc.attention_mask,
                                 generation_config=gc,
                                 stopping_criteria=StoppingCriteriaList([_StreamTap(q, n_prompt)]))
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)
        t0 = time.time()
        threading.Thread(target=work, daemon=True).start()
        created = int(time.time())
        cid = f"chatcmpl-{created}"
        s.send_response(200)
        s.send_header("Content-Type", "text/event-stream")
        s.send_header("Cache-Control", "no-cache")
        s.end_headers()
        def emit(delta, fin=None, usage=None):
            o = {"id": cid, "object": "chat.completion.chunk", "created": created,
                 "model": MODEL_NAME,
                 "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
            if usage is not None:
                o["usage"] = usage
            s.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n"); s.wfile.flush()
        stop_ids = {tok.eos_token_id, tok.pad_token_id}
        ids, sent_text, done, failed = [], "", False, False
        try:
            emit({"role": "assistant", "content": ""})
            while not done:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    failed = True
                    print(f"[chat-stream] generate failed: {item!r}", flush=True)
                    break
                for t in item:
                    if t in stop_ids:
                        done = True; break
                    ids.append(t)
                text = tok.decode(ids, skip_special_tokens=True)
                if text.endswith("�") and not done:
                    continue
                if len(text) > len(sent_text):
                    emit({"content": text[len(sent_text):]}); sent_text = text
            dt = time.time() - t0
            ct = len(ids)
            emit({}, fin="error" if failed else "stop",
                 usage={"prompt_tokens": n_prompt, "completion_tokens": ct,
                        "total_tokens": n_prompt + ct})
            s.wfile.write(b"data: [DONE]\n\n"); s.wfile.flush()
            if failed:
                STATS["errs"] += 1
            else:
                STATS["reqs"] += 1; STATS["ptoks"] += n_prompt; STATS["ctoks"] += ct; STATS["secs"] += dt
                print(f"[chat-stream] pt={n_prompt} ct={ct} {ct/max(dt,1e-6):.1f}tok/s {dt:.2f}s", flush=True)
        except Exception:
            pass
    def do_POST(s):
        if not READY.is_set():
            return s._j(503, {"error": {"message": "model loading; retry shortly"}})
        try:
            n = int(s.headers.get("Content-Length", 0))
            body = json.loads(s.rfile.read(n) or b"{}")
            msgs = body.get("messages")
            if not msgs:
                pr = body.get("prompt", "")
                if not pr: return s._j(400, {"error": {"message": "missing 'messages'"}})
                msgs = [{"role": "user", "content": pr}]
            max_new = max(1, min(int(body.get("max_tokens", 256)), 1024))
            temp = float(body.get("temperature", 0.7))
            top_p = float(body.get("top_p", 0.95))
            tools = body.get("tools")
            if body.get("stream"):
                return s._stream_chat(msgs, max_new, temp, top_p, tools)
            text, pt, ct = run_chat(msgs, max_new, temp, top_p, tools)
            s._j(200, {"id": f"chatcmpl-{int(time.time())}", "object": "chat.completion",
                       "created": int(time.time()), "model": MODEL_NAME,
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                    "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                                 "total_tokens": pt + ct}})
        except Exception as e:
            STATS["errs"] += 1
            import traceback; traceback.print_exc()
            try: s._j(500, {"error": {"message": repr(e)}})
            except Exception: pass

threading.Thread(target=boot, daemon=True).start()
print(f"hawkalphaquick backend listening on 127.0.0.1:{PORT} (loading)", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

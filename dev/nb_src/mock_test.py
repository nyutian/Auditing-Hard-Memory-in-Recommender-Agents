"""End-to-end test of the model-call path against a local fake endpoint.

Runs the notebook source in a throwaway directory (never the project), with the API base pointed
at a local server that imitates the chat-completions API. The fake "model" is a crude heuristic;
its outputs are synthetic and exist only to exercise the client, cache, parser and tables.
"""
import hashlib, json, os, re, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = Path(sys.argv[1])
sys.path.insert(0, str(HERE))
import tools

CALLS = {"n": 0}

def fake_answer(prompt):
    acc = [l for l in prompt.splitlines() if l.strip().endswith("-> ACCEPTED")]
    langs = [set(x.strip() for x in re.search(r"language: (.*?) \|", l).group(1).split(",")) for l in acc]
    ctrs = [set(x.strip() for x in re.search(r"country of origin: (.*?) ->", l).group(1).split(",")) for l in acc]
    L = set.intersection(*langs) if langs else set()
    C = set.intersection(*ctrs) if ctrs else set()
    tier = '"status"' in prompt
    evid = "Standard of evidence" in prompt
    h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
    rules = []
    for v in sorted(L):
        r = {"field": "language", "value": v}
        if tier:
            r["status"] = "established" if (h % 3 == 0 or not evid) else "plausible"
        rules.append(r)
    for v in sorted(C):
        if evid and not tier and h % 4:      # hard-only + evidence: usually abstains on country
            continue
        r = {"field": "country", "value": v.lower()}             # lower case: exercises normalisation
        if tier:
            r["status"] = "plausible" if evid or h % 5 else "established"
        rules.append(r)
    if "genre, or period" in prompt and h % 7 == 0:
        rules.append({"field": "genre", "value": "Poetry", "status": "plausible"})
    body = json.dumps({"rules": rules})
    if h % 11 == 0:
        return "```json\n" + body + "\n```"                       # fenced
    if h % 53 == 0:
        return "I think the reader likes books."                  # not JSON at all
    if h % 13 == 0:
        return "Here is the memory:\n" + body + "\nDone."         # prose around the JSON
    return body

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def do_GET(self):
        data = json.dumps({"data": [{"id": "openai/gpt-oss-120b"}, {"id": "zai-org/GLM-5.3-Flash"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(data)
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        CALLS["n"] += 1
        if CALLS["n"] % 97 == 0:                                   # occasional rate limit: exercises the retry
            self.send_response(429); self.end_headers(); self.wfile.write(b"slow down"); return
        assert self.headers["Authorization"] == "Bearer test-key"
        out = {"choices": [{"message": {"content": fake_answer(req["messages"][0]["content"]), "reasoning_content": "..."},
                            "finish_reason": "stop"}], "usage": {"total_tokens": 1}}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(data)

srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
port = srv.server_address[1]

tmp = Path(tempfile.mkdtemp(prefix="mocktest_"))
(tmp / "data").mkdir()
os.symlink(PROJECT / "data" / "catalog.json", tmp / "data" / "catalog.json")
os.chdir(tmp)
os.environ["MPLBACKEND"] = "Agg"
os.environ["BASETEN_API_KEY"] = "test-key"
ns = sys.modules["__main__"].__dict__
cells = [src for kind, src in tools.load(tools.DEFAULT_PARTS) if kind == "code"]
t0 = time.time()
import io, contextlib
for n, src in enumerate(cells):
    if "def sim_update" in src:
        break                                                      # the model path ends before the update experiment
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, f"<cell {n}>", "exec"), ns)
    if n == 0:
        ns["CFG"]["api_base"] = f"http://127.0.0.1:{port}/v1"
        ns["CFG"]["llm_workers"] = 16
    out = buf.getvalue()
    if "model calls planned" in out:
        print("\n".join(l for l in out.splitlines() if "model calls" in l or "finished" in l or "arms with" in l or "unparsed" in l.lower() or "could not" in l)[:1500])

pd = ns["pd"]
print(f"\nran in {time.time() - t0:.0f}s; fake endpoint served {CALLS['n']} requests; cache lines: "
      f"{sum(1 for _ in open(tmp / 'cache' / 'llm_responses.jsonl'))}")
with pd.option_context("display.width", 250, "display.max_colwidth", 60):
    print("\nEXP1 (first columns):"); print(ns["EXP1"].iloc[:, :5].to_string())
    print("\nEXP1 unparsed column:", list(ns["EXP1"].get("unparsed outputs, all calls of this arm", [])))
    print("\nEXP1_TESTS:"); print(ns["EXP1_TESTS"].head(6).to_string())
    print("\nEXP1_BY_PAIR:"); print(ns["EXP1_BY_PAIR"].head(4).to_string())
    print("\nEXP1_LENGTH:"); print(ns["EXP1_LENGTH"].to_string())
    print("\nEXP2 (model rows):"); print(ns["EXP2"].iloc[6:10, :8].to_string())
    print("\nEXP3 (model rows):"); print(ns["EXP3"].iloc[12:18, :6].to_string())
    print("\nEXP4 (model rows):"); print(ns["EXP4"].iloc[10:14, [0, 1, 2, 3, 6]].to_string())
# second pass must make no new calls
before = CALLS["n"]
ns["run_llm"](ns["JOBS"])
print("second pass made", CALLS["n"] - before, "new requests")
print("temp dir:", tmp)

"""Measure the prompts the notebook plans to send, and turn them into a cost range.
Usage: python -I cost_estimate.py <project_dir>      (no model calls are made)
"""
import contextlib, io, os, statistics, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
PROJECT = Path(sys.argv[1])
sys.path.insert(0, str(HERE))
import tools

tmp = Path(tempfile.mkdtemp(prefix="cost_"))
(tmp / "data").mkdir()
os.symlink(PROJECT / "data" / "catalog.json", tmp / "data" / "catalog.json")
os.chdir(tmp)
os.environ["MPLBACKEND"] = "Agg"; os.environ.pop("BASETEN_API_KEY", None)
ns = {"__name__": "__cost__"}
for kind, src in tools.load(["p1_core.py", "p2_writers.py"]):
    if kind != "code":
        continue
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "<cell>", "exec"), ns)
    if "JOBS = plan_jobs()" in src:
        break
jobs = ns["JOBS"]
models = list(ns["CFG"]["models"])
per_model = {m: [len(p) for (mm, p, s) in jobs.values() if mm == m] for m in models}
for m, L in per_model.items():
    print(f"{m}: {len(L)} calls | prompt characters: mean {statistics.mean(L):.0f}, median {statistics.median(L):.0f}, "
          f"min {min(L)}, max {max(L)}, total {sum(L):,}")
L = per_model[models[0]]
lo, hi = sum(L) / 4.0, sum(L) / 3.2
print(f"input tokens per model, all calls: {lo / 1e6:.2f}M to {hi / 1e6:.2f}M (4.0 and 3.2 characters per token)")
print(f"input tokens per call: {lo / len(L):.0f} to {hi / len(L):.0f}")
# by history length
W_, u_ = ns["WRITE_USERS"][0]
for k in ns["CFG"]["prefixes"]:
    c = [len(ns["build_prompt"](W, ns["prefix"](ns["HIST"][(W.pidx, u, "L")], k), "D")) for W, u in ns["WRITE_USERS"]]
    print(f"  arm D, {k} visits: mean {statistics.mean(c):.0f} characters")

"""Robustness check: what changes if a book that was already proposed to a user is never
proposed to that user again?

Runs the notebook source twice in throwaway namespaces and directories (never the project):
once as it is, once with a per-trajectory "seen" set. No model calls (no key is set).
Usage: python -I norepeat_check.py <project_dir>
"""
import contextlib, io, os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = Path(sys.argv[1])
sys.path.insert(0, str(HERE))
import tools

PATCHES = [
    # memoryless histories
    ('''    mem, tie = Memory(), random.Random(0)
    for cell in W.requests(uidx, n_visits):
        _, rec, _, _ = run_visit(W, W.order(uidx, cell, 0), mem, MEMORYLESS, truth, CFG["B"], tie)
        hist.append({"cell": cell, "props": rec})''',
     '''    mem, tie = Memory(), random.Random(0)
    seen = {i for i, _ in hist[0]["props"]}
    for cell in W.requests(uidx, n_visits):
        _, rec, _, _ = run_visit(W, [i for i in W.order(uidx, cell, 0) if i not in seen], mem, MEMORYLESS, truth, CFG["B"], tie)
        seen.update(i for i, _ in rec)
        hist.append({"cell": cell, "props": rec})'''),
    # closed-loop serving: feasibility is judged on what is still unseen
    ('''    for cell in W.requests(u, n_visits, stream):
        ok, rec, q, _ = run_visit(W, W.order(u, cell, cat), mem, pol, truth, CFG["B"], tie)
        o["q"] += q
        if W.feasible(cell, cat, truth):''',
     '''    seen = {i for v in prefix(HIST[(W.pidx, u, utype)], K) for i, _ in v["props"]}
    ta_ = [W.sat(k) for k in truth]
    for cell in W.requests(u, n_visits, stream):
        avail = [i for i in W.order(u, cell, cat) if i not in seen]
        feas_ = any(all(a[i] for a in ta_) for i in avail)
        ok, rec, q, _ = run_visit(W, avail, mem, pol, truth, CFG["B"], tie)
        seen.update(i for i, _ in rec)
        o["q"] += q
        if feas_:'''),
    # entrenchment
    ('''    for cell in W.requests(u, END)[START:END]:
        asked_before = W.C in mem.asked
        _, rec, qq, _ = run_visit(W, W.order(u, cell, 0), mem, pol, truth, CFG["B"], tie)''',
     '''    seen = {i for v in prefix(HIST[(W.pidx, u, "L")], START) for i, _ in v["props"]}
    for cell in W.requests(u, END)[START:END]:
        asked_before = W.C in mem.asked
        _, rec, qq, _ = run_visit(W, [i for i in W.order(u, cell, 0) if i not in seen], mem, pol, truth, CFG["B"], tie)
        seen.update(i for i, _ in rec)'''),
    # update rules
    ('''            mem = Memory({W.L: "hard", W.C: "hard"})
''',
     '''            mem = Memory({W.L: "hard", W.C: "hard"})
            seen = set()
'''),
    ('''                ok, rec, _, _ = run_visit(W, W.order(u, cell, 1), mem, pol, truth, CFG["B"], rng, flip)''',
     '''                ok, rec, _, _ = run_visit(W, [i for i in W.order(u, cell, 1) if i not in seen], mem, pol, truth, CFG["B"], rng, flip)
                seen.update(i for i, _ in rec)'''),
]


def run_variant(patch):
    tmp = Path(tempfile.mkdtemp(prefix="norepeat_"))
    (tmp / "data").mkdir()
    os.symlink(PROJECT / "data" / "catalog.json", tmp / "data" / "catalog.json")
    os.chdir(tmp)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ.pop("BASETEN_API_KEY", None)
    ns = {"__name__": "__variant__"}
    used = [0] * len(PATCHES)
    cells = [src for kind, src in tools.load(["p1_core.py", "p2_writers.py", "p3_experiments.py", "p4_update_fig.py"]) if kind == "code"]
    for n, src in enumerate(cells):
        if "def sim_update" in src:
            src = src.split("\nSYM = (")[0]                    # definitions only, not the full grid
        elif "UPDATE_RULES" in src or "ENTRENCH_CURVES[" in src and "plt." in src or "plt." in src:
            continue                                           # later update tables and the figure
        if patch:
            for j, (a, b) in enumerate(PATCHES):
                if a in src:
                    assert src.count(a) == 1
                    src = src.replace(a, b); used[j] += 1
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"<cell {n}>", "exec"), ns)
        if "CFG = {" in src and OVERRIDE:
            ns["CFG"].update(OVERRIDE)
        if "def sim_update" in src:
            break
    if patch:
        assert used == [1] * len(PATCHES), used
    return ns


def report(ns, title):
    pd, np = ns["pd"], ns["np"]
    print("=" * 110); print(title); print("=" * 110)
    with pd.option_context("display.width", 260, "display.max_colwidth", 70, "display.max_columns", 30):
        print("\nAMBIGUITY"); print(ns["AMBIGUITY"].to_string())
        lt = ns["LOGICAL_TABLE"]
        print("\nLOGICAL (8 visits):", {c: (int(lt[c].sum()) if c != "ambiguous" and c != "pair" else list(lt[c])) for c in lt.columns if c != "pair"})
        print("\nEXP3_POLICIES (two-rule memory, N=320)"); print(ns["EXP3_POLICIES"].to_string())
        print("\nUTILITY"); print(ns["UTILITY"].to_string())
        e3 = ns["EXP3"]; e3 = e3[~e3["memory"].str.contains(r"\|")]
        print("\nEXP3 (reference memories, 80 users)"); print(e3.to_string())
        e4 = ns["EXP4"]; e4 = e4[e4["policy"] != "not run"]
        print("\nEXP4 entrenchment"); print(e4.drop(columns=["users"]).to_string())
        rows = []
        for scope in ("feasible", "all"):
            for noise in ("5%", "10%"):
                for rule in ("Update-1", "Update-3", "Net-2", "Net-3"):
                    recs = ns["sim_update"]("LC", rule, noise, 16, scope, None, n_per_pair=500, snaps=(16,))
                    s = ns["summarize"](recs, "LC", 16, (16,))
                    recsL = ns["sim_update"]("L", rule, noise, 16, "feasible", None, n_per_pair=500, snaps=(16,))
                    sL = ns["summarize"](recsL, "L", 16, (16,))
                    rows.append({"cells": scope, "noise": noise, "rule": rule,
                                 "L+C: false deletion by 16, %": round(100 * s["false deletion by 16"], 1),
                                 "L+C: success to 16, %": round(100 * s["success to 16"], 1),
                                 "L: proxy removed by 16, %": round(100 * sL["proxy removed by 16"], 1),
                                 "L: language rule lost by 16, %": round(100 * sL["false deletion by 16"], 1)})
        print("\nUPDATE (16 visits, 500 users per pair)"); print(pd.DataFrame(rows).to_string())
        # how much inventory is there?
        inv = []
        for W in ns["WORLDS"]:
            nL = [sum(W.isL[i] for i in W.cand[c].get(cell, [])) for c in (0, 1) for cell in W.cells]
            nJ = [sum(W.isL[i] and W.isC[i] for i in W.cand[0].get(cell, [])) for cell in W.cells]
            inv.append({"pair": f"{W.lang}/{W.country}", "cells": len(W.cells),
                        "joint works per feasible cell in C0 (median, max)": (int(np.median(nJ)), max(nJ)),
                        "cells with exactly one joint work in C0": sum(x == 1 for x in nJ),
                        "target-language works per cell, both catalogs (median)": int(np.median(nL))})
        print("\nINVENTORY"); print(pd.DataFrame(inv).to_string())


OVERRIDE = {}
tag = ""
if len(sys.argv) > 2 and sys.argv[2] == "thick":
    # two English pairs with many works per request cell
    n_t = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    OVERRIDE = {"pairs": [("English", "United States", n_t), ("English", "United Kingdom", n_t)], "catalog_size": 2 * n_t}
    tag = f" | THICK: English/US and English/UK, {n_t} target works in a catalog of {2 * n_t}"
base = run_variant(False)
report(base, "AS IN THE NOTEBOOK (a book can be proposed again)" + tag)
nore = run_variant(True)
report(nore, "NO REPEAT (a book proposed once to a user is never proposed to that user again)" + tag)

# %% [markdown]
# # When Catalogs Become Preferences — experiment notebook
#
# A from-scratch re-implementation of the controlled study, written on 2026-11-15.
# It follows the design described in the paper draft and adds the experiments the
# review asked for. **Numbers here replace the ones in draft v5; they are not expected
# to match them**, because the original seeds and construction details were not available.
#
# | Section | What it does | Needs model calls |
# |---|---|---|
# | 1–3 | Catalog, paired sub-catalogs, users, interaction engine | no |
# | 4 | Step 0: equivalence check, logical reference, naive rule | no |
# | 5 | Writers: five prompt arms, cached model calls | **yes** |
# | 6 | Experiment 1: unbundling the writing interface | **yes** (reference rows: no) |
# | 7 | Experiment 2: scoring against three user types | **yes** (reference rows: no) |
# | 8 | Experiment 3: closed loop, written memory served on both catalogs | partly |
# | 9 | Experiment 4: entrenchment curves | partly |
# | 10 | Experiment 5: update rules under noise | no |
# | 11 | Figure and summary | no |
#
# **Choices the draft leaves open, made here** (each is one line in `CFG` or one function):
# a history is one bootstrap acceptance plus k visits; the proposal order is frozen per user,
# not per catalog; the four periods are before 1950, 1950-1979, 1980-1999 and 2000 onward; the few
# works from the paired country in another language are all kept in the background, so that a
# country-only requirement can be told apart from a language-and-country one; relaxation also
# applies when every book that passes the filters has been rejected.
#
# **Known limitation, found after this run (2026-11-15).** The engine lets a book that was
# already proposed to a user be proposed to that user again when a request repeats, and the
# constructed catalogs are thin (a request typically matches one or two target-language books).
# A check without repeated proposals changes two findings of sections 8-9: soft ranking no
# longer leaves the country question unresolved (37.2% resolved at visit 16, against 46.9%
# without memory and 12.8% under the hard filter), and adding the country rule to the language
# rule costs 2.5 points of success on the learning catalog instead of 0. The horizons of 64 and
# 256 visits in section 10 exceed what these catalogs can supply without repeats. Treat those
# numbers as provisional until the engine is changed.
#
# **To enable the model calls** put `BASETEN_API_KEY=...` in a file named `.env` next to
# this notebook (or export it) and re-run. Responses are cached in `cache/`, so a re-run
# only pays for calls it has not made before. Without a key every model-dependent cell
# reports "not run" and the rest of the notebook still executes.

# %%
import collections, hashlib, itertools, json, math, os, random, re, statistics, sys, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)

ROOT = Path.cwd()
DATA, RESULTS, FIGS, CACHE = ROOT / "data", ROOT / "results", ROOT / "figures", ROOT / "cache"
for d in (DATA, RESULTS, FIGS, CACHE):
    d.mkdir(exist_ok=True)

CFG = {
    "seed": 20261115,
    # (target language, paired country, number of target-language works per catalog)
    "pairs": [("Spanish", "Spain", 90), ("French", "France", 80), ("Russian", "Russia", 50), ("German", "Germany", 40)],
    "catalog_size": 1200,
    "skew_learn": 0.90,      # share of target-language works from the paired country, learning catalog C0
    "skew_shift": 0.10,      # same share after the shift, catalog C1
    "periods": [("before 1950", None, 1949), ("1950-1979", 1950, 1979), ("1980-1999", 1980, 1999), ("2000 onward", 2000, None)],
    "B": 5,                  # proposals per visit
    "history_visits": 16, "prefixes": [4, 8, 16], "main_prefix": 8,
    "n_write_users": 20,     # per pair: histories sent to the model writers
    "n_serve_users": 80,     # per pair: users in the serving audit (the first 20 are the writing users)
    "cont_visits": 16,       # continuation visits in the serving audit
    "n_update_users": 500,   # per pair: users per cell of the update stress test
    "show_titles": True,
    "models": {"GPT-OSS-120B": "openai/gpt-oss-120b", "GLM-5.3-Flash": "zai-org/GLM-5.3-Flash"},
    "api_base": "https://inference.baseten.co/v1",
    "temperature": 0.1, "max_tokens": 6000, "n_samples": 1, "llm_workers": 8, "max_new_calls": 8000,
}
PERIOD_LABELS = [p[0] for p in CFG["periods"]]
print("python", sys.version.split()[0], "| numpy", np.__version__, "| pandas", pd.__version__, "| matplotlib", matplotlib.__version__)

# %% [markdown]
# ## 1. Catalog
# The source is the τ-Rec-Books catalog (`books-wikidata-v2`, CC0), pinned to one commit and
# checked against the SHA-256 that the benchmark's own manifest records.

# %%
CATALOG_URL = ("https://raw.githubusercontent.com/nbharaths/tau-rec/e96771e0d4ee7ddaff94f74e5080d4c3a184a4b8/"
               "data/tracks/books-wikidata-v2/catalog.json")
CATALOG_SHA256 = "19e9fa3a4db03de8f8bb8cdd8f7b231346a44f21cefca18947b7b10485d7093f"

def load_catalog():
    path = DATA / "catalog.json"
    if not path.exists():
        print("downloading catalog ...")
        urllib.request.urlretrieve(CATALOG_URL, path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == CATALOG_SHA256, f"catalog hash mismatch: {digest}"
    return json.loads(raw)

CATALOG = load_catalog()
POOL = [w for w in CATALOG if w["languages"] and w["countries"]]
print(f"catalog: {len(CATALOG):,} works | with language and country recorded: {len(POOL):,} | sha256 ok")

def period_of(year):
    for k, (_, lo, hi) in enumerate(CFG["periods"]):
        if (lo is None or year >= lo) and (hi is None or year <= hi):
            return k
    raise ValueError(year)

# %% [markdown]
# ## 2. Paired sub-catalogs
# For each language/country pair we build a learning catalog **C0** and a shifted catalog **C1**
# of 1,200 works each. Both hold the same number of target-language works and the same
# background; only the share of target-language works from the paired country changes
# (90% in C0, 10% in C1). The two catalogs are nested as far as possible, so the shift swaps
# works instead of redrawing them.
#
# All randomness is derived from SHA-256 of the seed and a label, so the construction does
# not depend on a library's random stream.

# %%
def hseed(*parts):
    s = "|".join(str(p) for p in (CFG["seed"],) + parts)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")

def hsorted(works, *parts):
    """Deterministic shuffle: order works by a seeded hash of their id."""
    tag = "|".join(str(p) for p in (CFG["seed"],) + parts)
    return sorted(works, key=lambda w: hashlib.sha256(f"{tag}|{w['id']}".encode()).digest())

class World:
    """One language/country pair: the union of C0 and C1, with flags and lookup tables."""

    def __init__(self, pidx, lang, country, n_t):
        self.pidx, self.lang, self.country, self.n_t = pidx, lang, country, n_t
        self.L, self.C = ("language", lang), ("country", country)
        target = [w for w in POOL if lang in w["languages"]]
        joint = hsorted([w for w in target if country in w["countries"]], "joint", pidx)
        other = hsorted([w for w in target if country not in w["countries"]], "other", pidx)
        self.natural = (len(joint), len(target))
        nj0 = round(CFG["skew_learn"] * n_t); no0 = n_t - nj0
        nj1 = round(CFG["skew_shift"] * n_t); no1 = n_t - nj1
        assert len(joint) >= nj0 and len(other) >= no1, (lang, len(joint), nj0, len(other), no1)
        J0, O1 = joint[:nj0], other[:no1]          # C0's joint works, C1's other-country works
        J1, O0 = J0[:nj1], O1[:no0]                # nested: C1's joint ⊂ C0's, C0's other ⊂ C1's
        self.spare_other = len(other) - no1
        bg_pool = [w for w in POOL if lang not in w["languages"]]
        forced = [w for w in bg_pool if country in w["countries"]]   # paired country, other language
        rest = hsorted([w for w in bg_pool if country not in w["countries"]], "bg", pidx)
        bg = forced + rest[: CFG["catalog_size"] - n_t - len(forced)]
        self.n_country_other_lang = len(forced)
        self.items = J0 + O1 + bg
        ids = [{w["id"] for w in J0 + O0 + bg}, {w["id"] for w in J1 + O1 + bg}]
        assert len(ids[0]) == len(ids[1]) == CFG["catalog_size"]
        n = len(self.items)
        self.in_cat = [[w["id"] in ids[c] for w in self.items] for c in (0, 1)]
        self.isL = [lang in w["languages"] for w in self.items]
        self.isC = [country in w["countries"] for w in self.items]
        self.period = [period_of(w["publication_year"]) for w in self.items]
        self.genres = [tuple(w["genres"]) for w in self.items]
        self._sat, self._ord, self._feas = {}, {}, {}
        # request cells: (genre, period index) -> item indices, per catalog
        self.cand = []
        for c in (0, 1):
            d = collections.defaultdict(list)
            for i in range(n):
                if self.in_cat[c][i]:
                    for g in self.genres[i]:
                        d[(g, self.period[i])].append(i)
            self.cand.append(dict(d))
        allc = sorted(set(self.cand[0]) | set(self.cand[1]))
        self.union = {c: sorted(set(self.cand[0].get(c, [])) | set(self.cand[1].get(c, []))) for c in allc}
        # feasible for a language-only user: a joint work in C0 and a target-language work in C1
        self.cells = [c for c in allc
                      if any(self.isL[i] and self.isC[i] for i in self.cand[0].get(c, []))
                      and any(self.isL[i] for i in self.cand[1].get(c, []))]
        self.joint0 = [i for i in range(n) if self.in_cat[0][i] and self.isL[i] and self.isC[i]]

    def sat(self, key):
        """Boolean list over items: does the item satisfy rule (field, value)?"""
        a = self._sat.get(key)
        if a is None:
            f, v = key
            if f == "language":
                a = [v in w["languages"] for w in self.items]
            elif f == "country":
                a = [v in w["countries"] for w in self.items]
            elif f == "genre":
                a = [v in w["genres"] for w in self.items]
            elif f == "period":
                a = [PERIOD_LABELS[p] == v for p in self.period]
            else:
                a = [False] * len(self.items)
            self._sat[key] = a
        return a

    def order(self, uidx, cell, cat):
        """Request-matching items of one catalog in this user's frozen proposal order.
        The order is drawn once per (user, cell) over the union of both catalogs, so a work
        keeps its relative position when the catalog shifts."""
        o = self._ord.get((uidx, cell))
        if o is None:
            u = self.union.get(cell, [])
            r = random.Random(hseed("order", self.pidx, uidx, cell[0], cell[1]))
            ranked = [i for _, i in sorted((r.random(), i) for i in u)]
            o = ([i for i in ranked if self.in_cat[0][i]], [i for i in ranked if self.in_cat[1][i]])
            self._ord[(uidx, cell)] = o
        return o[cat]

    def requests(self, uidx, n, stream="hist", cells=None):
        cells = self.cells if cells is None else cells
        r = random.Random(hseed("req", stream, self.pidx, uidx))
        return [cells[r.randrange(len(cells))] for _ in range(n)]

    def bootstrap(self, uidx):
        """One accepted joint work that every history starts from."""
        r = random.Random(hseed("boot", self.pidx, uidx))
        i = self.joint0[r.randrange(len(self.joint0))]
        g = self.genres[i][r.randrange(len(self.genres[i]))]
        return {"cell": (g, self.period[i]), "props": [(i, True)]}

    def truth(self, utype):
        """Standing requirements of a user type: language only, both, or country only."""
        return {"L": (self.L,), "LC": (self.L, self.C), "C": (self.C,)}[utype]

    def feasible(self, cell, cat, truth):
        """Does the catalog hold a book that meets the request and the user's requirements?"""
        key = (cell, cat, truth)
        f = self._feas.get(key)
        if f is None:
            ta = [self.sat(k) for k in truth]
            f = self._feas[key] = any(all(a[i] for a in ta) for i in self.cand[cat].get(cell, []))
        return f

WORLDS = [World(i, *p) for i, p in enumerate(CFG["pairs"])]

rows = []
for W in WORLDS:
    def share(c):
        t = [i for i in range(len(W.items)) if W.in_cat[c][i] and W.isL[i]]
        return sum(W.isC[i] for i in t) / len(t)
    other_c1 = [W.items[i] for i in range(len(W.items)) if W.in_cat[1][i] and W.isL[i] and not W.isC[i]]
    top_other = collections.Counter(c for w in other_c1 for c in w["countries"]).most_common(2)
    rows.append({
        "pair": f"{W.lang}/{W.country}", "target works": W.n_t,
        "from paired country, C0": f"{share(0):.0%}", "C1": f"{share(1):.0%}",
        "natural share in pool": f"{W.natural[0]}/{W.natural[1]} = {W.natural[0] / W.natural[1]:.1%}",
        "unused other-country works": W.spare_other,
        "paired country, other language": W.n_country_other_lang,
        "feasible request cells": len(W.cells),
        "C1 other-country works come from": ", ".join(f"{c} {n}" for c, n in top_other),
    })
CATALOG_TABLE = pd.DataFrame(rows)
print("total feasible request cells:", sum(len(W.cells) for W in WORLDS))
CATALOG_TABLE

# %% [markdown]
# ## 3. Interaction engine
# One function runs a visit for every condition in the notebook. A memory is a set of rules
# `(field, value)` with a status; a policy says how statuses are used.
#
# * **hard / established / confirmed** rules filter under `enforce`; **plausible** rules only order books.
# * `soft` treats every stored rule as an ordering signal and excludes nothing.
# * `relax`: when nothing passes the filters, or everything that passes has been rejected and the
#   visit still has budget, drop one blocking rule for the rest of this visit. If several single
#   rules would unblock it, a seeded coin flip picks which to drop first and the others are tried in
#   turn. Confirmed rules are never dropped. The stored memory is not changed.
# * `update`: `("count", k)` deletes a rule after k accepted books that violate it;
#   `("net", m)` adds a point when a book violating only that rule is rejected, removes one when a
#   violating book is accepted, caps the score at +m and deletes at −m;
#   `("demote", k)` turns a rule into a plausible one at the first violating acceptance and deletes it at k.
# * `ask="neutral"`: one yes/no question per visit about one stored, unconfirmed rule.
#   `ask="oracle"`: one question that reveals the full standing requirement (an upper bound).
# * `probe`: once per rule, propose first a book that breaks only that rule.

# %%
FIRM = ("hard", "established", "confirmed")

class Memory:
    __slots__ = ("status", "acc", "net", "probed", "asked", "oracle_done")

    def __init__(self, rules=None):
        self.status = dict(rules or {})
        self.acc, self.net = collections.Counter(), collections.Counter()
        self.probed, self.asked, self.oracle_done = set(), set(), False

class Policy:
    def __init__(self, name, mode="enforce", relax=False, update=None, ask=None, probe=False):
        self.name, self.mode, self.relax, self.update, self.ask, self.probe = name, mode, relax, update, ask, probe

MEMORYLESS = Policy("memoryless", mode="none")

def relax_tail(order, filt, rank, st, sat, tie):
    """Books to fall back on once the filtered list is exhausted: for each filter rule that is not
    confirmed, the books that break that rule and no other filter. Rules are tried in a seeded
    random order."""
    opts = []
    for k in filt:
        if st.get(k) == "confirmed":
            continue
        ak, others = sat(k), [sat(j) for j in filt if j != k]
        e = [i for i in order if not ak[i] and all(a[i] for a in others)]
        if e:
            opts.append(e)
    if len(opts) > 1:
        tie.shuffle(opts)
    ra = [sat(k) for k in rank]
    out = []
    for e in opts:
        out += sorted(e, key=lambda i: -sum(a[i] for a in ra)) if ra else e
    return out

def run_visit(W, order, mem, pol, truth, B, tie, flip=None):
    """Run one visit. `order` is the request-matching part of the active catalog in the user's
    frozen order. `truth` is the tuple of rules the user really requires. `flip` is None or
    (rng, p_false_accept, p_false_reject). Returns (success, [(item, verdict)], questions, relaxed)."""
    st = mem.status
    sat = W.sat
    q = 0
    if pol.ask == "oracle":
        if not mem.oracle_done:
            mem.oracle_done, q = True, 1
            mem.status = st = {k: "confirmed" for k in truth}
    elif pol.ask == "neutral":
        pend = [k for k, s in st.items() if s != "confirmed" and k not in mem.asked]
        if pend:
            k = pend[tie.randrange(len(pend))]
            mem.asked.add(k); q = 1
            if k in truth:
                st[k] = "confirmed"
            else:
                del st[k]
    if pol.mode == "enforce":
        filt = [k for k, s in st.items() if s in FIRM]
        rank = [k for k, s in st.items() if s == "plausible"]
    elif pol.mode == "soft":
        filt, rank = [], list(st)
    else:
        filt, rank = [], []
    if filt:
        fa = [sat(k) for k in filt]
        elig = [i for i in order if all(a[i] for a in fa)]
    else:
        elig = list(order)
    if rank and len(elig) > 1:
        ra = [sat(k) for k in rank]
        elig = sorted(elig, key=lambda i: -sum(a[i] for a in ra))      # stable: frozen order within ties
    if pol.probe:
        todo = [k for k, s in st.items() if s != "confirmed" and k not in mem.probed]
        if todo:
            tie.shuffle(todo)
            for k in todo:
                ak = sat(k); oth = [sat(j) for j in st if j != k]
                c = [i for i in order if not ak[i] and all(a[i] for a in oth)]
                if c:
                    mem.probed.add(k)
                    elig = [c[0]] + [i for i in elig if i != c[0]]
                    break
    ta = [sat(k) for k in truth]
    upd = pol.update
    rec, success = [], False
    props, relaxed, j = elig[:B], False, 0
    while True:
        if j == len(props):
            # everything that passes the filters has been proposed (or nothing passes them)
            if relaxed or not (pol.relax and filt) or j >= B:
                break
            relaxed = True
            props = props + relax_tail(order, filt, rank, st, sat, tie)[: B - j]
            if j == len(props):
                break
        i = props[j]; j += 1
        ok = all(a[i] for a in ta)
        if flip is not None:
            x = flip[0].random()
            if ok:
                if x < flip[2]:
                    ok = False
            elif x < flip[1]:
                ok = True
        rec.append((i, ok))
        if upd is not None and st:
            viol = [k for k in st if not sat(k)[i]]
            if viol:
                kind, thr = upd
                if ok:
                    for k in viol:
                        mem.acc[k] += 1
                        mem.net[k] -= 1
                        if kind == "count":
                            if mem.acc[k] >= thr:
                                del st[k]
                        elif kind == "demote":
                            if mem.acc[k] >= thr:
                                del st[k]
                            elif st[k] in ("hard", "established"):
                                st[k] = "plausible"
                        elif kind == "net":
                            if mem.net[k] <= -thr:
                                del st[k]
                elif kind == "net" and len(viol) == 1:
                    k = viol[0]
                    mem.net[k] = min(thr, mem.net[k] + 1)
        if ok:
            success = True
            break
    return success, rec, q, relaxed

# ---- memoryless histories -------------------------------------------------------------
def gen_history(W, uidx, utype, n_visits=None):
    """Bootstrap acceptance, then `n_visits` visits on C0 with no memory."""
    n_visits = n_visits or CFG["history_visits"]
    truth = W.truth(utype)
    hist = [W.bootstrap(uidx)]
    mem, tie = Memory(), random.Random(0)
    for cell in W.requests(uidx, n_visits):
        _, rec, _, _ = run_visit(W, W.order(uidx, cell, 0), mem, MEMORYLESS, truth, CFG["B"], tie)
        hist.append({"cell": cell, "props": rec})
    return hist

def prefix(hist, k):
    """The bootstrap acceptance plus the first k visits."""
    return hist[: k + 1]

def has_counterexample(W, hist, rule):
    """An accepted book that violates `rule`: direct proof the rule is not required."""
    a = W.sat(rule)
    return any(ok and not a[i] for v in hist for i, ok in v["props"])

UTYPES = ("L", "LC", "C")
USERS = [(W, u) for W in WORLDS for u in range(CFG["n_serve_users"])]                   # 320
WRITE_USERS = [(W, u) for W in WORLDS for u in range(CFG["n_write_users"])]             # 80
HIST = {(W.pidx, u, t): gen_history(W, u, t) for W, u in USERS for t in UTYPES}
print(f"users: {len(USERS)} in the serving audit, of which {len(WRITE_USERS)} are sent to the model writers")

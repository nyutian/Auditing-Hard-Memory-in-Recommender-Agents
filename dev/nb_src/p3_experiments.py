# %% [markdown]
# ## 8. Experiment 3 — closed loop: written memory, served on both catalogs
# Each memory is served for 16 continuation visits on the learning catalog C0 and, with the same
# requests and the same frozen proposal order, on the shifted catalog C1.
#
# * **First accept**: the first proposed book is accepted. With five proposals per visit most
#   policies reach 100% success, so this is the measure that separates them.
# * **Success@5**: a book is accepted within the visit's five proposals.
# * For users who require language *and* country, some C1 visits have no acceptable book at all.
#   Success is computed on the feasible visits; on the others every proposal is **wasted**.

# %%
POL = {
    "enforce": Policy("enforce: firm rules filter, plausible rules rank", "enforce"),
    "soft": Policy("soft: every rule only ranks", "soft"),
    "relax": Policy("enforce + transient relaxation", "enforce", relax=True),
    "update1": Policy("Update-1", "enforce", relax=True, update=("count", 1)),
    "update3": Policy("Update-3", "enforce", relax=True, update=("count", 3)),
    "probe": Policy("probe once", "enforce", relax=True, update=("count", 1), probe=True),
    "ask": Policy("ask about each stored rule", "enforce", relax=True, ask="neutral"),
    "oracle": Policy("ask, oracle (upper bound)", "enforce", ask="oracle"),
}

def serve_user(W, u, utype, rules, pol, cat, n_visits=None, stream="cont"):
    n_visits = n_visits or CFG["cont_visits"]
    truth = W.truth(utype)
    mem = Memory(rules)
    tie = random.Random(hseed("tie", W.pidx, u, pol.name))
    o = {"n": n_visits, "feas": 0, "s1": 0, "sB": 0, "rej": 0, "wasted": 0, "q": 0}
    for cell in W.requests(u, n_visits, stream):
        ok, rec, q, _ = run_visit(W, W.order(u, cell, cat), mem, pol, truth, CFG["B"], tie)
        o["q"] += q
        if W.feasible(cell, cat, truth):
            o["feas"] += 1; o["s1"] += bool(rec and rec[0][1]); o["sB"] += ok; o["rej"] += len(rec) - ok
        else:
            o["wasted"] += len(rec)
    o["false_firm"] = any(mem.status.get(k) in FIRM for k in (W.L, W.C) if k not in truth)
    o["true_lost"] = any(k not in mem.status for k in truth)
    return o

def serve_many(src, pol, utype, cat, users):
    outs = []
    for W, u in users:
        rules = src(W, u, utype)
        if rules is None:
            return None
        outs.append(serve_user(W, u, utype, rules, pol, cat))
    return outs

def ratio_ci(outs, num, den, n_boot=2000):
    """Pooled rate sum(num)/sum(den) with a user-level bootstrap interval."""
    a = np.array([o[num] for o in outs], float)
    b = np.array([(o[den] if isinstance(den, str) else den(o)) for o in outs], float)
    if b.sum() == 0:
        return None
    rng = np.random.default_rng(hseed("bootstrap", num) % (2**32))
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    s = a[idx].sum(1) / np.maximum(b[idx].sum(1), 1e-12)
    return a.sum() / b.sum(), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))

def show(r, ci=False, scale=100, nd=1):
    if r is None:
        return "n/a"
    return f"{scale * r[0]:.{nd}f}" + (f" [{scale * r[1]:.{nd}f}, {scale * r[2]:.{nd}f}]" if ci else "")

def ref_source(fn):
    return lambda W, u, t: fn(W, prefix(HIST[(W.pidx, u, t)], K))

SOURCES = {
    "no memory": lambda W, u, t: {},
    "true requirement": lambda W, u, t: {k: "hard" for k in W.truth(t)},
    "two rules, hand-set": lambda W, u, t: {W.L: "hard", W.C: "hard"},
    "naive shared-attribute rule": ref_source(ref_naive),
    "logical, hard-only": ref_source(ref_logical_hard),
    "logical, two-tier": ref_source(ref_version_space),
}
for model in CFG["models"]:
    for arm in ARMS:
        SOURCES[f"{model} | {arm}: {ARMS[arm]['label']}"] = lambda W, u, t, m=model, a=arm: written(m, a, W, u, t)

infeasible = lambda o: o["n"] - o["feas"]
rows = []
for sname, src in SOURCES.items():
    for pkey in ("enforce", "soft"):
        row = {"memory": sname, "serving": pkey}
        L0 = serve_many(src, POL[pkey], "L", 0, WRITE_USERS)
        if L0 is None:
            row["L: C0 first accept"] = "not run"
            rows.append(row)
            continue
        L1 = serve_many(src, POL[pkey], "L", 1, WRITE_USERS)
        row["L: C0 first accept"] = show(ratio_ci(L0, "s1", "feas"))
        row["L: C0 success@5"] = show(ratio_ci(L0, "sB", "feas"))
        row["L: C1 first accept"] = show(ratio_ci(L1, "s1", "feas"), ci=True)
        row["L: C1 success@5"] = show(ratio_ci(L1, "sB", "feas"), ci=True)
        row["L: C1 rejections/visit"] = show(ratio_ci(L1, "rej", "feas"), scale=1, nd=2)
        LC1 = serve_many(src, POL[pkey], "LC", 1, WRITE_USERS)
        if LC1 is None:
            row["L+C: C1 success@5, feasible visits"] = "not run"
        else:
            row["L+C: C1 success@5, feasible visits"] = show(ratio_ci(LC1, "sB", "feas"), ci=True)
            row["L+C: wasted proposals per infeasible visit"] = show(ratio_ci(LC1, "wasted", infeasible), scale=1, nd=2)
        rows.append(row)
EXP3 = pd.DataFrame(rows).astype(object).fillna("")
EXP3.to_csv(RESULTS / "exp3_closed_loop.csv", index=False)
print(f"closed loop on the {len(WRITE_USERS)} writing users; rates in %, 95% user-level bootstrap intervals")
EXP3

# %%
# Serving policies with the hand-set two-rule memory (the setting of the draft's Table 2), N = 320.
rows = []
for pkey, pol in POL.items():
    src = SOURCES["two rules, hand-set"]
    L0, L1 = serve_many(src, pol, "L", 0, USERS), serve_many(src, pol, "L", 1, USERS)
    LC1 = serve_many(src, pol, "LC", 1, USERS)
    rows.append({
        "policy": pol.name,
        "L: C0 success@5": show(ratio_ci(L0, "sB", "feas")),
        "L: C1 success@5 [95% CI]": show(ratio_ci(L1, "sB", "feas"), ci=True),
        "L: C1 first accept": show(ratio_ci(L1, "s1", "feas")),
        "L: rejections per 16 visits": f"{np.mean([o['rej'] for o in L1]):.3f}",
        "L: questions": f"{np.mean([o['q'] for o in L1]):.2f}",
        "L: country still firm": f"{100 * np.mean([o['false_firm'] for o in L1]):.0f}%",
        "L+C: C1 success@5, feasible visits": show(ratio_ci(LC1, "sB", "feas")),
        "L+C: wasted per infeasible visit": show(ratio_ci(LC1, "wasted", infeasible), scale=1, nd=2),
        "L+C: a true rule lost": f"{100 * np.mean([o['true_lost'] for o in LC1]):.0f}%",
    })
EXP3_POLICIES = pd.DataFrame(rows)
EXP3_POLICIES.to_csv(RESULTS / "exp3_policies_two_rule_memory.csv", index=False)
feas_LC = np.mean([W.feasible(c, 1, W.truth("LC")) for W, u in USERS for c in W.requests(u, CFG["cont_visits"], "cont")])
print(f"share of shifted-catalog visits whose request cell holds a joint work: {100 * feas_LC:.1f}% "
      "(this is the hard filter's success rate by construction)")
by_pair = {f"{W.lang}/{W.country}": show(ratio_ci(serve_many(SOURCES["two rules, hand-set"], POL["enforce"], "L", 1,
                                                             [(W, u) for u in range(CFG["n_serve_users"])]), "sB", "feas"))
           for W in WORLDS}
print("hard-filter success on the shifted catalog by pair, %:", by_pair)
EXP3_POLICIES

# %%
# Utility of single rules on the learning catalog (language-only users, hard filter, N = 320).
def success_per_user(rules_fn, cat=0):
    return np.array([serve_user(W, u, "L", rules_fn(W), POL["enforce"], cat)["sB"] / CFG["cont_visits"] for W, u in USERS])

s_none = success_per_user(lambda W: {})
s_C = success_per_user(lambda W: {W.C: "hard"})
s_L = success_per_user(lambda W: {W.L: "hard"})
s_LC = success_per_user(lambda W: {W.L: "hard", W.C: "hard"})
rows = []
for name, d in (("no memory (level)", s_none), ("country rule alone vs no memory", s_C - s_none),
                ("language rule alone vs no memory", s_L - s_none), ("country rule added to the language rule", s_LC - s_L)):
    m, lo, hi = boot_ci(d, tag=name)
    rows.append({"comparison on C0": name, "success@5, percentage points": f"{100 * m:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"})
UTILITY = pd.DataFrame(rows)
UTILITY.to_csv(RESULTS / "exp3_rule_utility.csv", index=False)
UTILITY

# %% [markdown]
# ## 9. Experiment 4 — entrenchment
# Start from the memory written after 4 visits and keep serving on the learning catalog up to
# visit 16, with the same requests as the memoryless histories. A user's country question is
# **resolved** once the record holds an accepted target-language book from another country, or an
# answer to a direct question about the country rule. The memoryless curve is the reference: it is
# what an agent that never used its memory would have learned from the same requests.
#
# The last column of the table is the ceiling for any policy that learns only from proposals: the
# share of users who, at some visit, ask for a cell that contains such a book in C0.

# %%
START, END = 4, CFG["history_visits"]

def entrench_user(W, u, rules, pol):
    truth = W.truth("L")
    mem = Memory(rules)
    tie = random.Random(hseed("tie-entrench", W.pidx, u, pol.name))
    resolved = has_counterexample(W, prefix(HIST[(W.pidx, u, "L")], START), W.C)
    curve, q = [resolved], 0
    for cell in W.requests(u, END)[START:END]:
        asked_before = W.C in mem.asked
        _, rec, qq, _ = run_visit(W, W.order(u, cell, 0), mem, pol, truth, CFG["B"], tie)
        q += qq
        if any(ok and W.isL[i] and not W.isC[i] for i, ok in rec) or (W.C in mem.asked and not asked_before):
            resolved = True
        curve.append(resolved)
    return curve, q, mem.status.get(W.C) in FIRM

def memoryless_curve(W, u):
    h = HIST[(W.pidx, u, "L")]
    return [has_counterexample(W, prefix(h, t), W.C) for t in range(START, END + 1)]

def ceiling(W, u):
    cells = W.requests(u, END)[START:END]
    return has_counterexample(W, prefix(HIST[(W.pidx, u, "L")], START), W.C) or any(
        any(W.isL[i] and not W.isC[i] for i in W.cand[0].get(c, [])) for c in cells)

def ref_source_at(fn, k):
    return lambda W, u, t: fn(W, prefix(HIST[(W.pidx, u, t)], k))

ENTRENCH_SOURCES = {
    "naive shared-attribute rule, 4 visits": (ref_source_at(ref_naive, START), USERS),
    "logical two-tier, 4 visits": (ref_source_at(ref_version_space, START), USERS),
}
for model in CFG["models"]:
    for arm in ("A", "E"):
        ENTRENCH_SOURCES[f"{model} | {arm}, 4 visits"] = (lambda W, u, t, m=model, a=arm: written(m, a, W, u, t, START), WRITE_USERS)
ENTRENCH_POLICIES = ("enforce", "soft", "update1", "probe", "ask")

ENTRENCH_CURVES, rows = {}, []
ref_all = np.mean([memoryless_curve(W, u) for W, u in USERS], axis=0)
ENTRENCH_CURVES[("memoryless reference", "all users")] = ref_all
for sname, (src, users) in ENTRENCH_SOURCES.items():
    if any(src(W, u, "L") is None for W, u in users):
        rows.append({"memory at visit 4": sname, "policy": "not run"})
        continue
    ref = np.mean([memoryless_curve(W, u) for W, u in users], axis=0)
    ceil = np.mean([ceiling(W, u) for W, u in users])
    holds = np.mean([rule_state(src(W, u, "L"), W.C) == "firm" for W, u in users])
    for pkey in ENTRENCH_POLICIES:
        res = [entrench_user(W, u, src(W, u, "L"), POL[pkey]) for W, u in users]
        curve = np.mean([r[0] for r in res], axis=0)
        ENTRENCH_CURVES[(sname, pkey)] = curve
        rows.append({
            "memory at visit 4": sname, "policy": POL[pkey].name, "users": str(len(users)),
            "country firm at visit 4": f"{100 * holds:.1f}%",
            "resolved at visit 4": f"{100 * curve[0]:.1f}%", "at visit 8": f"{100 * curve[4]:.1f}%",
            "at visit 16": f"{100 * curve[-1]:.1f}%",
            "memoryless at visit 16": f"{100 * ref[-1]:.1f}%",
            "questions per user": f"{np.mean([r[1] for r in res]):.2f}",
            "country still firm at visit 16": f"{100 * np.mean([r[2] for r in res]):.1f}%",
            "ceiling for proposal-only learning": f"{100 * ceil:.1f}%",
        })
EXP4 = pd.DataFrame(rows).astype(object).fillna("")
EXP4.to_csv(RESULTS / "exp4_entrenchment.csv", index=False)
EXP4

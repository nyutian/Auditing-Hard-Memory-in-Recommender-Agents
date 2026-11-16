# %% [markdown]
# ## 10. Experiment 5 — update rules under noisy feedback
# Every user starts on the shifted catalog with the two-rule memory (language and country, both
# firm), served by filtering with transient relaxation. What differs is the rule that decides when
# a stored rule is removed.
#
# * **Update-k**: delete a rule after k accepted books that violate it.
# * **Net-m**: a running score per rule. A rejected book that violates only that rule adds a point
#   (capped at +m); an accepted book that violates it removes a point; the rule is deleted at −m.
#   This is a sequential likelihood-ratio test in its simplest form.
# * **Demote-k**: the first violating acceptance turns the rule into a plausible one (it stops
#   filtering); the k-th deletes it.
#
# Verdicts flip independently with the stated probability. **False deletion**: a rule the user
# really has is deleted. **Proxy removal**: the rule the user does not have is deleted.
#
# Two request regimes for users who require both language and country:
# *feasible cells* (the draft's setting: every request has an acceptable book, so the filter only
# fails after a noisy rejection) and *all cells* (requests are drawn as for everyone else, so about
# two thirds of visits have no acceptable book and relaxation is triggered directly).

# %%
UPDATE_RULES = {
    "Update-1": ("count", 1), "Update-2": ("count", 2), "Update-3": ("count", 3), "Update-4": ("count", 4),
    "Update-5": ("count", 5), "Net-2": ("net", 2), "Net-3": ("net", 3), "Demote-2": ("demote", 2), "Demote-3": ("demote", 3),
}
NOISE = {"0%": (0.0, 0.0), "2%": (0.02, 0.02), "5%": (0.05, 0.05), "10%": (0.10, 0.10), "20%": (0.20, 0.20),
         "10%, false accepts only": (0.10, 0.0), "10%, false rejects only": (0.0, 0.10)}

def sim_update(utype, rule, noise, T, scope="feasible", change_at=None, n_per_pair=None, snaps=(16, 64)):
    """One cell of the stress test. Returns one record per user:
    (visit at which the language rule was deleted, same for country, visit at which each stopped
    being firm, {horizon: (proposals the user should reject, of which accepted, rejections, successes)})."""
    pol = Policy(rule, "enforce", relax=True, update=UPDATE_RULES[rule])
    eps_fa, eps_fr = NOISE[noise]
    n_per_pair = n_per_pair or CFG["n_update_users"]
    recs = []
    for W in WORLDS:
        truth0, truthL = W.truth(utype), W.truth("L")
        cells = [c for c in W.cells if W.feasible(c, 1, truth0)] if scope == "feasible" else W.cells
        if not cells:
            continue
        for u in range(n_per_pair):
            mem = Memory({W.L: "hard", W.C: "hard"})
            rng = random.Random(hseed("update", W.pidx, u, utype, rule, noise, scope, change_at))
            flip = (rng, eps_fa, eps_fr) if (eps_fa or eps_fr) else None
            gone, soft, snap = {}, {}, {}
            doomed = facc = rej = succ = 0
            for t, cell in enumerate(W.requests(u, T, f"update-{utype}-{scope}", cells), 1):
                truth = truth0 if (change_at is None or t < change_at) else truthL
                before = dict(mem.status)
                ok, rec, _, _ = run_visit(W, W.order(u, cell, 1), mem, pol, truth, CFG["B"], rng, flip)
                st = mem.status
                if st != before:
                    for k, s in before.items():
                        if k not in st:
                            gone[k] = t; soft.setdefault(k, t)
                        elif s in FIRM and st[k] not in FIRM:
                            soft[k] = t
                ts = [W.sat(k) for k in truth]
                for i, v in rec:
                    if not all(a[i] for a in ts):
                        doomed += 1; facc += v
                succ += ok; rej += len(rec) - ok
                if t in snaps:
                    snap[t] = (doomed, facc, rej, succ)
            recs.append((gone.get(W.L), gone.get(W.C), soft.get(W.L), soft.get(W.C), snap))
    return recs

def summarize(recs, utype, T, snaps=(16, 64)):
    """Aggregate one cell. Positions: 0 = language rule, 1 = country rule."""
    true_pos = {"L": [0], "LC": [0, 1], "C": [1]}[utype]
    false_pos = {"L": [1], "LC": [], "C": [0]}[utype]
    n = len(recs)
    out = {"users": n}
    by = lambda r, pos, h, off=0: any(r[p + off] is not None and r[p + off] <= h for p in pos)
    for h in snaps:
        if true_pos:
            k = sum(by(r, true_pos, h) for r in recs)
            out[f"false deletion by {h}"] = k / n
            out[f"false deletion by {h} (k)"] = k
            out[f"true rule not firm by {h}"] = sum(by(r, true_pos, h, 2) for r in recs) / n
        if false_pos:
            out[f"proxy removed by {h}"] = sum(by(r, false_pos, h) for r in recs) / n
        out[f"doomed proposals by {h}"] = np.mean([r[4][h][0] for r in recs])
        out[f"false accepts by {h}"] = np.mean([r[4][h][1] for r in recs])
        out[f"rejections per visit to {h}"] = np.mean([r[4][h][2] for r in recs]) / h
        out[f"success to {h}"] = np.mean([r[4][h][3] for r in recs]) / h
    if false_pos:
        times = [r[false_pos[0]] for r in recs]
        out["mean visits to proxy removal (censored)"] = np.mean([t if t is not None else T for t in times])
        done = sorted(t for t in times if t is not None)
        out["median visits to proxy removal"] = done[len(times) // 2] if len(done) > len(times) // 2 else None
    return out

def run_cell(c):
    utype, rule, noise, T, scope, change_at = c
    snaps = (16, 64) if T == 64 else (16, 64, 256)
    recs = sim_update(utype, rule, noise, T, scope, change_at, snaps=snaps)
    if change_at is None:
        s = summarize(recs, utype, T, snaps)
    else:
        # the user stops requiring the country at visit `change_at`: how long until the rule goes?
        s = summarize(recs, "L", T, snaps)
        gone = [r[1] for r in recs]
        s["country rule deleted before the change"] = np.mean([g is not None and g < change_at for g in gone])
        s["visits from change to deletion (censored)"] = np.mean([max(0, (g if g is not None else T) - change_at + 1) for g in gone])
        s["adapted within 8 visits of the change"] = np.mean([g is not None and g < change_at + 8 for g in gone])
    s.update({"user type": utype, "rule": rule, "noise": noise, "T": T, "cells": scope, "change at": change_at})
    return s

def pmap(fn, items):
    """Run cells in parallel worker processes; fall back to a plain loop if that is not possible."""
    import multiprocessing as mp
    try:
        with mp.get_context("fork").Pool(min(8, os.cpu_count() or 1)) as pool:
            return pool.map(fn, items, chunksize=1)
    except Exception as e:
        print("parallel run not available, running serially:", repr(e))
        return [fn(x) for x in items]

SYM = ("0%", "2%", "5%", "10%", "20%")
CONDS = []
for rule in UPDATE_RULES:
    for noise in NOISE:
        CONDS += [("LC", rule, noise, 64, "feasible", None), ("L", rule, noise, 64, "feasible", None)]
    CONDS += [("LC", rule, noise, 64, "all", None) for noise in SYM]
    CONDS += [("C", rule, noise, 64, "feasible", None) for noise in ("0%", "5%", "10%")]
    CONDS += [("LC", rule, noise, 64, "all", 9) for noise in ("0%", "5%")]
for rule in ("Update-1", "Update-3", "Net-2", "Net-3"):
    for noise in ("5%", "10%"):
        CONDS += [("LC", rule, noise, 256, "feasible", None), ("LC", rule, noise, 256, "all", None), ("L", rule, noise, 256, "feasible", None)]

t0 = time.time()
UPD = pd.DataFrame(pmap(run_cell, CONDS))
UPD.to_csv(RESULTS / "exp5_update_rules_all_cells.csv", index=False)
n_users = CFG["n_update_users"] * len(WORLDS)
print(f"{len(CONDS)} cells, {n_users} users each, in {time.time() - t0:.0f}s")

def upd_table(utype, scope, col, noises=SYM, T=64, change=None, fmt="{:.1f}", scale=100):
    d = UPD[(UPD["user type"] == utype) & (UPD["cells"] == scope) & (UPD["T"] == T) &
            (UPD["change at"].isna() if change is None else UPD["change at"] == change)]
    t = d.pivot(index="rule", columns="noise", values=col).reindex(index=list(UPDATE_RULES), columns=list(noises))
    return t.map(lambda x: "" if pd.isna(x) else fmt.format(scale * x))

print("\nFalse deletion of a true rule by visit 16, % of users — users who require language and country, feasible cells")
upd_table("LC", "feasible", "false deletion by 16", noises=list(NOISE))

# %%
print("False deletion by visit 64, % — language-and-country users, feasible cells")
print(upd_table("LC", "feasible", "false deletion by 64", noises=list(NOISE)).to_string())
print("\nFalse deletion by visit 16, % — language-and-country users, all cells (about two thirds of visits have no acceptable book)")
print(upd_table("LC", "all", "false deletion by 16").to_string())
print("\nFalse deletion by visit 64, % — language-and-country users, all cells")
print(upd_table("LC", "all", "false deletion by 64").to_string())
print("\nTrue rule no longer firm by visit 64, % — the same users and cells (differs from deletion only for Demote-k)")
print(upd_table("LC", "all", "true rule not firm by 64").to_string())
print("\nWasted proposals: rejections per visit to visit 64 — language-and-country users, all cells")
print(upd_table("LC", "all", "rejections per visit to 64", fmt="{:.2f}", scale=1).to_string())

# %%
print("Proxy (country rule) removed by visit 16, % — language-only users")
print(upd_table("L", "feasible", "proxy removed by 16", noises=list(NOISE)).to_string())
print("\nMean visits until the proxy is removed (64 if never) — language-only users")
print(upd_table("L", "feasible", "mean visits to proxy removal (censored)", noises=list(NOISE), fmt="{:.1f}", scale=1).to_string())
print("\nA true rule (language) deleted by visit 64, % — language-only users")
print(upd_table("L", "feasible", "false deletion by 64", noises=list(NOISE)).to_string())
print("\nCountry-only users: the language rule is the proxy. Proxy removed by visit 16, % / true country rule deleted by visit 64, %")
print(upd_table("C", "feasible", "proxy removed by 16", noises=("0%", "5%", "10%")).to_string())
print(upd_table("C", "feasible", "false deletion by 64", noises=("0%", "5%", "10%")).to_string())

# %%
# Why false deletion grows faster than the noise rate: exposure itself grows with noise.
d = UPD[(UPD["user type"] == "LC") & (UPD["rule"] == "Update-1") & (UPD["T"] == 64) & UPD["change at"].isna()]
rows = []
for scope in ("feasible", "all"):
    for noise in SYM:
        r = d[(d["cells"] == scope) & (d["noise"] == noise)].iloc[0]
        k, n = int(r["false deletion by 16 (k)"]), int(r["users"])
        rows.append({"cells": scope, "noise": noise,
                     "proposals the user should reject, per user, visits 1-16": f"{r['doomed proposals by 16']:.2f}",
                     "of which accepted by mistake": f"{r['false accepts by 16']:.3f}",
                     "Update-1 false deletion by visit 16": pct(k, n)})
EXPOSURE = pd.DataFrame(rows)
EXPOSURE.to_csv(RESULTS / "exp5_exposure.csv", index=False)
EXPOSURE

# %%
# Longer horizons, and a real change of preference.
d = UPD[(UPD["T"] == 256) & (UPD["user type"] == "LC")]
rows = []
for _, r in d.sort_values(["cells", "noise", "rule"]).iterrows():
    rows.append({"cells": r["cells"], "noise": r["noise"], "rule": r["rule"],
                 "false deletion by 16": f"{100 * r['false deletion by 16']:.1f}%",
                 "by 64": f"{100 * r['false deletion by 64']:.1f}%", "by 256": f"{100 * r['false deletion by 256']:.1f}%"})
HORIZON = pd.DataFrame(rows)
HORIZON.to_csv(RESULTS / "exp5_horizon.csv", index=False)
print("False deletion as the horizon grows — language-and-country users")
print(HORIZON.to_string(index=False))

d = UPD[UPD["change at"] == 9]
rows = []
for rule in UPDATE_RULES:
    row = {"rule": rule}
    for noise in ("0%", "5%"):
        r = d[(d["rule"] == rule) & (d["noise"] == noise)].iloc[0]
        row[f"{noise}: deleted before the change"] = f"{100 * r['country rule deleted before the change']:.1f}%"
        row[f"{noise}: visits from change to deletion"] = f"{r['visits from change to deletion (censored)']:.1f}"
        row[f"{noise}: adapted within 8 visits"] = f"{100 * r['adapted within 8 visits of the change']:.1f}%"
    rows.append(row)
CHANGE = pd.DataFrame(rows)
CHANGE.to_csv(RESULTS / "exp5_preference_change.csv", index=False)
print("\nThe user drops the country requirement at visit 9 (all cells)")
CHANGE

# %% [markdown]
# ## 11. Figure and summary
# Panel (a) uses the naive shared-attribute memory after 4 visits, which needs no model. Panels (b)
# and (c) use language-and-country users whose requests are drawn from all cells, at 5% noise.

# %%
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # validated for colour-vision deficiency on white
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, "ps.fonttype": 42, "xtick.major.size": 2.5, "ytick.major.size": 0,
})
FIG_SCOPE, FIG_NOISE = "all", "5%"
RING = dict(markeredgecolor=SURFACE, markeredgewidth=1.0)

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75), constrained_layout=True, gridspec_kw={"width_ratios": [1.12, 1, 1]})
MS = {"o": 5.6, "s": 5.2, "^": 6.4}

# (a) entrenchment
ax = axes[0]
src = "naive shared-attribute rule, 4 visits"
xs = np.arange(START, END + 1)
series = [
    ("no memory (reference)", ENTRENCH_CURVES[("memoryless reference", "all users")], MUTED, "o"),
    ("filter, soft rank or Update-1", ENTRENCH_CURVES[(src, "enforce")], BLUE, "o"),
    ("probe once", ENTRENCH_CURVES[(src, "probe")], ORANGE, "s"),
    ("ask about each rule", ENTRENCH_CURVES[(src, "ask")], AQUA, "^"),
]
assert np.allclose(ENTRENCH_CURVES[(src, "enforce")], ENTRENCH_CURVES[(src, "soft")])
assert np.allclose(ENTRENCH_CURVES[(src, "enforce")], ENTRENCH_CURVES[(src, "update1")])
h_a = []
for name, y, col, mk in series:
    ax.plot(xs, 100 * y, color=col, lw=1.6, solid_capstyle="round")
    h_a.append(ax.plot(xs[-1], 100 * y[-1], mk, color=col, ms=MS[mk], **RING, clip_on=False, label=name, linestyle="none")[0])
    ax.annotate(f"{100 * y[-1]:.0f}%", (xs[-1], 100 * y[-1]), xytext=(5, 0), textcoords="offset points",
                va="center", ha="left", color=INK2, fontsize=7)
ax.set_xlim(START, 18.6); ax.set_ylim(0, 106); ax.set_xticks([4, 8, 12, 16]); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel("visit on the learning catalog"); ax.set_ylabel("country question resolved (% of users)")
ax.set_title("(a) Is the proxy ever tested?", loc="left", color=INK)

# (b) trade-off at visit 64
ax = axes[1]
fam = {"Update": (BLUE, "o", "Update-k: count violating acceptances"),
       "Net": (ORANGE, "s", "Net-m: also credit violating rejections"),
       "Demote": (AQUA, "^", "Demote-k: demote at 1, delete at k")}
lab_off = {"Update": (-6, -6, "right", "top"), "Net": (0, 6, "center", "bottom"), "Demote": (0, 6, "center", "bottom")}
dL = UPD[(UPD["user type"] == "L") & (UPD["T"] == 64) & (UPD["noise"] == FIG_NOISE) & UPD["change at"].isna()].set_index("rule")
dLC = UPD[(UPD["user type"] == "LC") & (UPD["cells"] == FIG_SCOPE) & (UPD["T"] == 64) & (UPD["noise"] == FIG_NOISE)
          & UPD["change at"].isna()].set_index("rule")
h_b = []
for f, (col, mk, lab) in fam.items():
    rules = [r for r in UPDATE_RULES if r.startswith(f)]
    x = [dL.loc[r, "mean visits to proxy removal (censored)"] for r in rules]
    y = [100 * dLC.loc[r, "false deletion by 64"] for r in rules]
    ax.plot(x, y, color=col, lw=1.0, alpha=0.55, zorder=2)
    h_b.append(ax.plot(x, y, mk, color=col, ms=MS[mk], **RING, zorder=3, label=lab, linestyle="none")[0])
    dx, dy, ha, va = lab_off[f]
    for r, xi, yi in zip(rules, x, y):
        ax.annotate(r.split("-")[1], (xi, yi), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=INK2, fontsize=6.8)
ax.set_xlim(0, 9); ax.set_ylim(-5, 114); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_xticks([0, 2, 4, 6, 8])
ax.set_xlabel("visits until the proxy is removed (mean)"); ax.set_ylabel("true rule deleted (% of users)")
ax.set_title("(b) Cost of each rule at visit 64", loc="left", color=INK)

# (c) horizon
ax = axes[2]
d = UPD[(UPD["T"] == 256) & (UPD["user type"] == "LC") & (UPD["cells"] == FIG_SCOPE) & (UPD["noise"] == FIG_NOISE)].set_index("rule")
hz = [16, 64, 256]
for r in ("Update-1", "Update-3", "Net-2", "Net-3"):
    col, mk, _ = fam[r.split("-")[0]]
    y = [100 * d.loc[r, f"false deletion by {h}"] for h in hz]
    ax.plot(hz, y, color=col, lw=1.6, solid_capstyle="round")
    ax.plot(hz, y, mk, color=col, ms=MS[mk], **RING, linestyle="none", clip_on=False)
ax.annotate("Update-1", (16, 100 * d.loc["Update-1", "false deletion by 16"]), xytext=(2, 17), textcoords="offset points",
            ha="left", va="bottom", color=INK2, fontsize=6.8)
ax.annotate("Update-3", (64, 100 * d.loc["Update-3", "false deletion by 64"]), xytext=(7, -9), textcoords="offset points",
            ha="left", va="top", color=INK2, fontsize=6.8)
ax.annotate("Net-2 and Net-3", (256, 100 * d.loc["Net-2", "false deletion by 256"]), xytext=(0, 6), textcoords="offset points",
            ha="right", va="bottom", color=INK2, fontsize=6.8)
ax.set_xscale("log", base=2); ax.set_xticks(hz); ax.set_xticklabels([str(h) for h in hz]); ax.minorticks_off()
ax.set_xlim(13, 310); ax.set_ylim(-5, 114); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xlabel("visits, shifted catalog (log scale)"); ax.set_ylabel("true rule deleted (% of users)")
ax.set_title("(c) The same rules over time", loc="left", color=INK)

fig.legend(handles=h_a, loc="outside lower left", ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.2)
fig.legend(handles=h_b, loc="outside lower right", ncol=1, frameon=False, handletextpad=0.3, borderaxespad=0.2)
fig.savefig(FIGS / "interface_audit.pdf"); fig.savefig(FIGS / "interface_audit.png", dpi=220)
plt.show()

# %%
# Headline numbers, collected in one place and written to results/summary.json.
def cell(utype, rule, noise, scope="feasible", T=64, change=None):
    d = UPD[(UPD["user type"] == utype) & (UPD["rule"] == rule) & (UPD["noise"] == noise) & (UPD["cells"] == scope) & (UPD["T"] == T)
            & (UPD["change at"].isna() if change is None else UPD["change at"] == change)]
    return d.iloc[0]

two = SOURCES["two rules, hand-set"]
hard = serve_many(two, POL["enforce"], "L", 1, USERS); soft = serve_many(two, POL["soft"], "L", 1, USERS)
src = "naive shared-attribute rule, 4 visits"
SUMMARY = {
    "setup": {"seed": CFG["seed"], "catalog_sha256": CATALOG_SHA256, "pairs": CFG["pairs"],
              "feasible_cells": {f"{W.lang}/{W.country}": len(W.cells) for W in WORLDS},
              "writing_users": len(WRITE_USERS), "serving_users": len(USERS),
              "update_users_per_cell": CFG["n_update_users"] * len(WORLDS)},
    "ambiguity_writing_users": {k: sum(not has_counterexample(W, prefix(HIST[(W.pidx, u, "L")], k), W.C) for W, u in WRITE_USERS)
                                for k in CFG["prefixes"]},
    "logical_reference_8_visits": {
        "language_established": sum(ref_version_space(W, prefix(HIST[(W.pidx, u, "L")], K)).get(W.L) == "established" for W, u in WRITE_USERS),
        "country_established": sum(ref_version_space(W, prefix(HIST[(W.pidx, u, "L")], K)).get(W.C) == "established" for W, u in WRITE_USERS)},
    "model_arms_complete": [f"{m}/{a}" for (m, a), ok in READY.items() if ok],
    "serving_two_rule_memory_shifted_catalog": {
        "hard_filter_success_pct": round(100 * ratio_ci(hard, "sB", "feas")[0], 1),
        "soft_rank_success_pct": round(100 * ratio_ci(soft, "sB", "feas")[0], 1),
        "soft_rank_first_accept_pct": round(100 * ratio_ci(soft, "s1", "feas")[0], 1)},
    "utility_on_learning_catalog_points": {"no_memory_level": round(100 * float(s_none.mean()), 1),
                                           "country_alone": round(100 * float((s_C - s_none).mean()), 1),
                                           "language_alone": round(100 * float((s_L - s_none).mean()), 1),
                                           "country_given_language": round(100 * float((s_LC - s_L).mean()), 1)},
    "entrenchment_resolved_at_visit_16_pct": {
        "memoryless": round(100 * float(ENTRENCH_CURVES[("memoryless reference", "all users")][-1]), 1),
        "filter_or_soft_rank": round(100 * float(ENTRENCH_CURVES[(src, "enforce")][-1]), 1),
        "probe_once": round(100 * float(ENTRENCH_CURVES[(src, "probe")][-1]), 1),
        "ask": round(100 * float(ENTRENCH_CURVES[(src, "ask")][-1]), 1)},
    "update_false_deletion_pct": {
        f"{scope} cells, {noise}, by visit {h}": {r: round(100 * float(cell("LC", r, noise, scope)[f"false deletion by {h}"]), 2)
                                                 for r in ("Update-1", "Update-3", "Net-2", "Net-3")}
        for scope in ("feasible", "all") for noise in ("5%", "10%") for h in (16, 64)},
    "update_mean_visits_to_proxy_removal_5pct": {r: round(float(cell("L", r, "5%")["mean visits to proxy removal (censored)"]), 2)
                                                for r in UPDATE_RULES},
    "preference_change_visits_to_deletion_0pct": {r: round(float(cell("LC", r, "0%", "all", change=9)["visits from change to deletion (censored)"]), 1)
                                                 for r in UPDATE_RULES},
}
(RESULTS / "summary.json").write_text(json.dumps(SUMMARY, indent=1, ensure_ascii=False))
print(json.dumps(SUMMARY, indent=1, ensure_ascii=False))

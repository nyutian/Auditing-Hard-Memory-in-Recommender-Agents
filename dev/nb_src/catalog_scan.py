"""Scan the unmodified catalog: which language/country pairs could carry the audit, and how common
are naturally concentrated attribute pairs?  Usage: python -I catalog_scan.py <catalog.json>
"""
import collections, json, math, sys

cat = json.load(open(sys.argv[1]))
pool = [w for w in cat if w["languages"] and w["countries"]]
print("works:", len(cat), "| with language and country:", len(pool))

PERIODS = [("before 1950", -10**9, 1949), ("1950-1979", 1950, 1979), ("1980-1999", 1980, 1999), ("2000 onward", 2000, 10**9)]
def period(y):
    if y is None:
        return None
    for name, lo, hi in PERIODS:
        if lo <= y <= hi:
            return name

def vals(w, f):
    if f == "period":
        p = period(w.get("publication_year"))
        return [p] if p else []
    return w.get(f) or []

# ---- 1. language/country pairs that fit the notebook's construction --------------------------
# target size n needs round(0.9 n) joint works and round(0.9 n) other-country works
byL = collections.defaultdict(list)
for w in pool:
    for l in w["languages"]:
        byL[l].append(w)
rows = []
for l, ws in byL.items():
    cc = collections.Counter(c for w in ws for c in w["countries"])
    for c, joint in cc.items():
        other = len(ws) - joint
        cap = min(joint, other)
        if cap >= 36:
            n = int(cap / 0.9)
            rows.append((cap, l, c, joint, other, len(ws), n))
rows.sort(reverse=True)
print("\n1. language/country pairs with at least 36 works on both sides (joint, other-country):")
print(f"{'language':<12}{'country':<28}{'joint':>6}{'other':>6}{'total':>6}{'natural share':>15}{'max target size':>17}")
for cap, l, c, j, o, t, n in rows:
    print(f"{l:<12}{c:<28}{j:>6}{o:>6}{t:>6}{j / t:>14.1%}{n:>17}")
print("pairs:", len(rows), "| distinct languages:", len({r[1] for r in rows}))

# ---- 2. how common are concentrated attribute pairs in the unmodified catalog? ---------------
FIELDS = ["languages", "countries", "genres", "period"]
MIN_SUPPORT = 50
print(f"\n2. natural concentration, anchor values with at least {MIN_SUPPORT} works (all 20,000 works; a work counts")
print("   for a pair only if both fields are filled)")
summary = []
examples = []
for fa in FIELDS:
    for fb in FIELDS:
        if fa == fb:
            continue
        sup = collections.Counter(); co = collections.defaultdict(collections.Counter)
        for w in cat:
            A, B = vals(w, fa), vals(w, fb)
            if not A or not B:
                continue
            for a in A:
                sup[a] += 1
                for b in set(B):
                    co[a][b] += 1
        anchors = [a for a, n in sup.items() if n >= MIN_SUPPORT]
        tops = []
        for a in anchors:
            b, n = co[a].most_common(1)[0]
            tops.append((n / sup[a], a, b, n, sup[a]))
        tops.sort(reverse=True)
        n80 = sum(t[0] >= 0.8 for t in tops); n90 = sum(t[0] >= 0.9 for t in tops)
        summary.append((fa, fb, len(anchors), n80, n90))
        examples.append((fa, fb, tops[:6]))
print(f"{'anchor field':<12}{'other field':<12}{'anchors':>8}{'top share >= 80%':>18}{'>= 90%':>8}")
for fa, fb, n, n80, n90 in summary:
    print(f"{fa:<12}{fb:<12}{n:>8}{n80:>12} ({n80 / max(n, 1):.0%}){n90:>8}")
for fa, fb, tops in examples:
    if (fa, fb) in (("languages", "countries"), ("countries", "languages"), ("languages", "period"), ("genres", "period"), ("genres", "languages"), ("countries", "period")):
        print(f"\n   {fa} -> {fb}, most concentrated:")
        for s, a, b, n, t in tops:
            print(f"      {a} -> {b}: {n}/{t} = {s:.1%}")

# share of works whose language has a >= 80% country (and the reverse)
def covered(fa, fb, thr):
    sup = collections.Counter(); co = collections.defaultdict(collections.Counter)
    for w in cat:
        A, B = vals(w, fa), vals(w, fb)
        if not A or not B:
            continue
        for a in A:
            sup[a] += 1
            for b in set(B):
                co[a][b] += 1
    hot = {a for a, n in sup.items() if n >= MIN_SUPPORT and co[a].most_common(1)[0][1] / n >= thr}
    tot = sum(1 for w in cat if vals(w, fa) and vals(w, fb))
    hit = sum(1 for w in cat if vals(w, fa) and vals(w, fb) and any(a in hot for a in vals(w, fa)))
    return hit, tot, len(hot)
for fa, fb in (("languages", "countries"), ("countries", "languages"), ("genres", "period"), ("languages", "period")):
    for thr in (0.8, 0.9):
        h, t, k = covered(fa, fb, thr)
        print(f"   works whose {fa} value has a {fb} value at >= {thr:.0%}: {h}/{t} = {h / t:.1%}  ({k} anchor values)")

# ---- 3. chance that n accepted books of a language-only reader all share the country ----------
print("\n3. unmodified catalog: probability that n uniformly drawn works in the language all come from the country")
def p_all(j, t, n):
    return math.comb(j, n) / math.comb(t, n) if j >= n else 0.0
for l, c in (("French", "France"), ("German", "Germany"), ("Russian", "Russia"), ("Spanish", "Spain"), ("English", "United States"), ("English", "United Kingdom"),
             ("Italian", "Italy"), ("Japanese", "Japan"), ("Polish", "Poland"), ("Swedish", "Sweden"), ("Portuguese", "Brazil"), ("Chinese", "People's Republic of China")):
    ws = byL.get(l, [])
    j = sum(c in w["countries"] for w in ws)
    if ws:
        print(f"   {l}/{c}: {j}/{len(ws)} = {j / len(ws):.1%} | n=3: {p_all(j, len(ws), 3):.0%}  n=5: {p_all(j, len(ws), 5):.0%}  n=9: {p_all(j, len(ws), 9):.0%}")

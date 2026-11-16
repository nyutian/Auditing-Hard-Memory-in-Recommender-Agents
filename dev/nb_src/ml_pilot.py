"""Feasibility pilot on real rating histories (MovieLens latest-small + Wikidata metadata).

Question: after a user's first n liked films, how often do all of them share a country of origin
(a rule a memory writer could store), and how often does the same user's later behaviour refute it?
Usage: python -I ml_pilot.py <ml-latest-small dir> <wd_meta.json>
"""
import collections, json, re, statistics, sys
import pandas as pd

d, meta_path = sys.argv[1], sys.argv[2]
ratings = pd.read_csv(f"{d}/ratings.csv")
movies = pd.read_csv(f"{d}/movies.csv")
links = pd.read_csv(f"{d}/links.csv", dtype={"imdbId": str})
wd = json.load(open(meta_path))

imdb = {int(m): "tt" + str(i).zfill(7) for m, i in zip(links.movieId, links.imdbId) if isinstance(i, str)}
country, language, decade, genres = {}, {}, {}, {}
for m, title, g in zip(movies.movieId, movies.title, movies.genres):
    w = wd.get(imdb.get(int(m), ""), {})
    country[m] = frozenset(w.get("countries", []))
    language[m] = frozenset(w.get("languages", []))
    y = re.search(r"\((\d{4})\)\s*$", title)
    decade[m] = frozenset([f"{int(y.group(1)) // 10 * 10}s"]) if y else frozenset()
    genres[m] = frozenset(x for x in g.split("|") if x and x != "(no genres listed)")
FIELD = {"country": country, "language": language, "decade": decade, "genre": genres}

print("ratings", len(ratings), "| users", ratings.userId.nunique(), "| films", len(movies),
      "| with country", sum(bool(v) for v in country.values()), "| with language", sum(bool(v) for v in language.values()))
LIKE = 4.0
liked_all = ratings[ratings.rating >= LIKE]
print(f"liked (rating >= {LIKE}): {len(liked_all)} = {len(liked_all) / len(ratings):.1%} of ratings")
for f in ("country", "language"):
    c = collections.Counter(v for m in liked_all.movieId for v in FIELD[f][m])
    known = sum(bool(FIELD[f][m]) for m in liked_all.movieId)
    print(f"  liked ratings by {f} (share of liked ratings with known {f}):", ", ".join(f"{k} {n / known:.1%}" for k, n in c.most_common(6)))

# per-user chronological sequences
users = {}
for u, g in ratings.sort_values(["userId", "timestamp", "movieId"]).groupby("userId"):
    users[u] = list(zip(g.movieId, g.rating, g.timestamp))

def analyse(field, n, anchor=None, min_future=20):
    """History = first n liked films with known `field` (and, if anchor=(field2, value), restricted to
    users whose n liked films all carry that anchor value). Rule = value shared by all n."""
    F = FIELD[field]
    out = collections.Counter(); blocked, first_idx, gaps, vals = [], [], [], collections.Counter()
    later_day = collections.Counter()
    for u, seq in users.items():
        hist, pos = [], None
        for j, (m, r, t) in enumerate(seq):
            if r >= LIKE and F[m] and (anchor is None or FIELD[anchor[0]][m]):
                hist.append((m, t))
                if len(hist) == n:
                    pos = j; break
        if pos is None:
            continue
        if anchor is not None and not all(anchor[1] in FIELD[anchor[0]][m] for m, _ in hist):
            continue
        out["eligible"] += 1
        shared = frozenset.intersection(*[F[m] for m, _ in hist])
        if not shared:
            continue
        out["unanimous"] += 1
        v = sorted(shared)[0] if len(shared) == 1 else max(shared, key=lambda x: GLOBAL[field][x])
        vals[v] += 1
        gaps.append((hist[-1][1] - hist[0][1]) / 3600)
        t_n = hist[-1][1]
        fut = [(m, t) for m, r, t in seq[pos + 1:] if r >= LIKE and F[m] and (anchor is None or anchor[1] in FIELD[anchor[0]][m])]
        if len(fut) < min_future:
            continue
        out["with future"] += 1
        viol = [k for k, (m, t) in enumerate(fut) if v not in F[m]]
        blocked.append(len(viol) / len(fut))
        if viol:
            out["refuted"] += 1; first_idx.append(viol[0] + 1)
        fut_late = [(m, t) for m, t in fut if t - t_n >= 86400]
        if len(fut_late) >= 5:
            later_day["n"] += 1
            later_day["refuted"] += any(v not in F[m] for m, t in fut_late)
    return out, blocked, first_idx, gaps, vals, later_day

GLOBAL = {f: collections.Counter(v for m in liked_all.movieId for v in FIELD[f][m]) for f in FIELD}

def line(name, field, n, anchor=None):
    o, blocked, first_idx, gaps, vals, late = analyse(field, n, anchor)
    if not o["eligible"]:
        return
    top = ", ".join(f"{k} {c}" for k, c in vals.most_common(2))
    s = (f"{name:<34} n={n:<2} unanimous {o['unanimous']}/{o['eligible']} = {o['unanimous'] / o['eligible']:.1%} ({top})")
    if o["with future"]:
        s += (f" | with >=20 later likes: {o['with future']}; refuted {o['refuted']}/{o['with future']} = {o['refuted'] / o['with future']:.1%}"
              f"; later likes a filter would block: mean {statistics.mean(blocked):.1%}, median {statistics.median(blocked):.1%}"
              f"; first refuting like: median #{statistics.median(first_idx) if first_idx else float('nan'):.0f}")
    if late["n"]:
        s += f" | on later days only (>=5 likes): refuted {late['refuted']}/{late['n']}"
    if gaps:
        s += f" | hours from 1st to n-th like: median {statistics.median(gaps):.2f}"
    print(s)

print("\nA. Any field, all users")
for field in ("country", "language", "decade", "genre"):
    for n in (5, 8, 12):
        line(f"shared {field}", field, n)

print("\nB. The paper's pair: users whose first n liked films are all English-language; rule = shared country;")
print("   the future counts only later liked English-language films (so a refutation violates the country rule alone)")
for n in (5, 8, 12):
    line("country | all-English history", "country", n, anchor=("language", "English"))

print("\nC. Expected under catalog composition alone: p^n with p = share among liked ratings")
for f, v in (("country", "United States"), ("language", "English")):
    known = sum(bool(FIELD[f][m]) for m in liked_all.movieId)
    p = GLOBAL[f][v] / known
    print(f"  {f} = {v}: p = {p:.3f}; p^5 = {p ** 5:.1%}, p^8 = {p ** 8:.1%}, p^12 = {p ** 12:.1%}")
eng = [m for m in liked_all.movieId if "English" in language[m] and country[m]]
p = sum("United States" in country[m] for m in eng) / len(eng)
print(f"  United States among liked English-language ratings: p = {p:.3f}; p^5 = {p ** 5:.1%}, p^8 = {p ** 8:.1%}, p^12 = {p ** 12:.1%}")

"""Look up country of origin (P495) and original language (P364) on Wikidata for the IMDb ids in a
MovieLens links.csv.  Usage: python -I wd_join.py <links.csv> <out.json>
Only public film identifiers are sent; nothing about users.
"""
import csv, json, sys, time, urllib.parse, urllib.request

links, out = sys.argv[1], sys.argv[2]
ids = []
with open(links, newline="") as f:
    for r in csv.DictReader(f):
        if r["imdbId"]:
            ids.append("tt" + r["imdbId"].zfill(7))
print("imdb ids:", len(ids))

Q = """SELECT ?imdb ?cl ?ll WHERE {
  VALUES ?imdb { %s }
  ?film wdt:P345 ?imdb .
  OPTIONAL { ?film wdt:P495 ?c . ?c rdfs:label ?cl . FILTER(LANG(?cl) = "en") }
  OPTIONAL { ?film wdt:P364 ?l . ?l rdfs:label ?ll . FILTER(LANG(?ll) = "en") }
}"""

def ask(batch):
    q = Q % " ".join(f'"{i}"' for i in batch)
    data = urllib.parse.urlencode({"query": q}).encode()
    req = urllib.request.Request("https://query.wikidata.org/sparql", data=data, headers={
        "Accept": "application/sparql-results+json", "User-Agent": "catalog-metadata-join/0.1 (one-off research script)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=70) as r:
                return json.load(r)["results"]["bindings"]
        except Exception as e:
            print("  retry", attempt, repr(e)[:150]); time.sleep(5 * (attempt + 1))
    return None

res, failed = {}, 0
B = 250
t0 = time.time()
for k in range(0, len(ids), B):
    rows = ask(ids[k:k + B])
    if rows is None:
        failed += 1
        continue
    for b in rows:
        d = res.setdefault(b["imdb"]["value"], {"countries": set(), "languages": set()})
        if "cl" in b:
            d["countries"].add(b["cl"]["value"])
        if "ll" in b:
            d["languages"].add(b["ll"]["value"])
    time.sleep(0.6)
    if (k // B) % 8 == 0:
        print(f"  {k + B}/{len(ids)} ids, {len(res)} matched, {time.time() - t0:.0f}s")
json.dump({k: {"countries": sorted(v["countries"]), "languages": sorted(v["languages"])} for k, v in res.items()}, open(out, "w"))
n_c = sum(1 for v in res.values() if v["countries"]); n_l = sum(1 for v in res.values() if v["languages"])
print(f"matched {len(res)}/{len(ids)} | with country {n_c} | with language {n_l} | failed batches {failed} | {time.time() - t0:.0f}s")

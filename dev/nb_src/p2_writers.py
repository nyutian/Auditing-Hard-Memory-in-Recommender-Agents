# %% [markdown]
# ## 4. Step 0 — checks that need no model
# * **Equivalence.** A history is *ambiguous* for the country rule when it contains no accepted
#   target-language book from another country. For every such history we check that a user who
#   requires language *and* country would have produced exactly the same history.
# * **Logical reference.** For each history we compute, by rule, which predicates belong to every
#   consistent set of requirements (*established*), to some (*plausible*), or to none.
# * **Naive rule.** Store every attribute value shared by all accepted books.

# %%
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))

def pct(k, n, ci=True):
    if n == 0:
        return "n/a"
    s = f"{k}/{n} ({100 * k / n:.1f}%)"
    if ci:
        lo, hi = wilson(k, n)
        s += f" [{100 * lo:.1f}, {100 * hi:.1f}]"
    return s

def boot_ci(values, n_boot=5000, tag="boot"):
    """User-level percentile bootstrap for a mean. `values` holds one number per user."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(hseed("bootstrap", tag) % (2**32))
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

FIELDS2 = ("language", "country")
FIELDS4 = ("language", "country", "genre", "period")

def item_preds(W, i, fields):
    w = W.items[i]; s = set()
    if "language" in fields:
        s |= {("language", v) for v in w["languages"]}
    if "country" in fields:
        s |= {("country", v) for v in w["countries"]}
    if "genre" in fields:
        s |= {("genre", v) for v in w["genres"]}
    if "period" in fields:
        s.add(("period", PERIOD_LABELS[W.period[i]]))
    return s

def version_space(W, hist, fields=FIELDS2):
    """Logical reference. S = predicates every accepted book satisfies. A predicate is
    established when some rejected book violates it and nothing else in S; otherwise it is
    plausible. Returns ({rule: status}, number of rejections S cannot explain)."""
    S = None
    for v in hist:
        for i, ok in v["props"]:
            if ok:
                p = item_preds(W, i, fields)
                S = p if S is None else (S & p)
    S = S or set()
    est, unexplained = set(), 0
    for v in hist:
        for i, ok in v["props"]:
            if not ok:
                V = S - item_preds(W, i, fields)
                if len(V) == 1:
                    est |= V
                elif not V:
                    unexplained += 1
    # sorted, so that rule order (and every seeded tie-break that depends on it) is reproducible
    return {k: ("established" if k in est else "plausible") for k in sorted(S)}, unexplained

def ref_version_space(W, hist):
    return version_space(W, hist)[0]

def ref_logical_hard(W, hist):
    """What a hard-only writer should store under the evidence rule: established predicates only."""
    return {k: "hard" for k, s in version_space(W, hist)[0].items() if s == "established"}

def ref_naive(W, hist):
    """Shared-attribute rule: every language/country value common to all accepted books."""
    return {k: "hard" for k in version_space(W, hist)[0]}

def rule_state(rules, key):
    s = rules.get(key)
    return "absent" if s is None else ("firm" if s in FIRM else "plausible")

# --- ambiguity by history length, and the equivalence check ---------------------------------
rows, n_checked = [], 0
for k in CFG["prefixes"]:
    for name, users in (("writing users (N=80)", WRITE_USERS), ("all users (N=320)", USERS)):
        amb = 0
        for W, u in users:
            hL, hLC = prefix(HIST[(W.pidx, u, "L")], k), prefix(HIST[(W.pidx, u, "LC")], k)
            ambiguous = not has_counterexample(W, hL, W.C)
            assert ambiguous == (hL == hLC), "ambiguous histories must be identical for L and L+C users"
            n_checked += 1
            amb += ambiguous
        rows.append({"visits": k, "population": name, "ambiguous": pct(amb, len(users))})
print(f"equivalence check passed on {n_checked} (user, length) pairs: a history is ambiguous exactly when a "
      "language-and-country user would have produced the same history")
AMBIGUITY = pd.DataFrame(rows).pivot(index="visits", columns="population", values="ambiguous")
AMBIGUITY

# %%
K = CFG["main_prefix"]
AMBIG = {(W.pidx, u): not has_counterexample(W, prefix(HIST[(W.pidx, u, "L")], K), W.C) for W, u in USERS}
n_amb_w = sum(AMBIG[(W.pidx, u)] for W, u in WRITE_USERS)
rows = []
for W in WORLDS:
    us = [(W, u) for u in range(CFG["n_write_users"])]
    vs = [version_space(W, prefix(HIST[(W.pidx, u, "L")], K)) for _, u in us]
    assert all(un == 0 for _, un in vs)
    rows.append({
        "pair": f"{W.lang}/{W.country}",
        "ambiguous": f"{sum(AMBIG[(W.pidx, u)] for _, u in us)}/{len(us)}",
        "language established": sum(r.get(W.L) == "established" for r, _ in vs),
        "language plausible": sum(r.get(W.L) == "plausible" for r, _ in vs),
        "country established": sum(r.get(W.C) == "established" for r, _ in vs),
        "country plausible": sum(r.get(W.C) == "plausible" for r, _ in vs),
        "country absent": sum(W.C not in r for r, _ in vs),
        "C-only differs from L+C": sum(prefix(HIST[(W.pidx, u, "C")], K) != prefix(HIST[(W.pidx, u, "LC")], K) for _, u in us),
    })
print(f"logical reference on the {len(WRITE_USERS)} writing users, language-only truth, {K} visits "
      f"({n_amb_w} ambiguous, {len(WRITE_USERS) - n_amb_w} identifiable)")
LOGICAL_TABLE = pd.DataFrame(rows)
LOGICAL_TABLE

# %% [markdown]
# ## 5. Model writers
# Five prompt arms share one base text and one history rendering. Neighbouring arms differ in
# exactly one thing.
#
# | Arm | Schema | Fields | Evidence rule | Role |
# |---|---|---|---|---|
# | A | hard rules only | language, country | no | forced condition |
# | B | hard rules only | language, country | yes | instruction alone |
# | C | established / plausible | language, country | no | default behaviour, main baseline |
# | D | established / plausible | language, country | yes | tier plus instruction |
# | E | established / plausible | language, country, genre, period | yes | D with four fields |
#
# Every arm tells the model how its output will be used, and every arm allows an empty list.

# %%
BASE_PROMPT = """You maintain the long-term memory of a book recommender about one reader.

On each visit the reader states a request (a genre and a publication period). The recommender proposes books one at a time, and the reader accepts or rejects each book without giving a reason. A reader may also have standing requirements: conditions that a book must meet on every visit, whatever the request. Standing requirements are never stated.

Interaction history of this reader:
{history}

Decide what, if anything, to store in memory about this reader's standing requirements."""

USE_HARD = """How the memory is used: each stored rule is enforced as a filter on every future visit. Books that do not match a stored rule are never proposed to this reader."""

USE_TIER = """How the memory is used: each stored rule has a status.
- "established": the rule is enforced as a filter on every future visit. Books that do not match it are never proposed to this reader.
- "plausible": the rule is used only to order books. Matching books are proposed first, and no book is excluded."""

EVIDENCE_HARD = """Standard of evidence: the reader accepts a book exactly when it meets the visit's request and every standing requirement. Store a rule only if it belongs to every set of standing requirements that is consistent with all the accepted and rejected books above. Never store a rule that an accepted book violates."""

EVIDENCE_TIER = """Standard of evidence: the reader accepts a book exactly when it meets the visit's request and every standing requirement. Mark a rule "established" only if it belongs to every set of standing requirements that is consistent with all the accepted and rejected books above. Mark it "plausible" if it belongs to some but not all of those sets. Never store a rule that an accepted book violates."""

OUT_HARD = """A rule has a field ({fields}) and a value.{note}
Return only JSON in this form. Use an empty list if nothing should be stored.
{{"rules": [{{"field": "<field>", "value": "<value>"}}]}}"""

OUT_TIER = """A rule has a field ({fields}), a value, and a status.{note}
Return only JSON in this form. Use an empty list if nothing should be stored.
{{"rules": [{{"field": "<field>", "value": "<value>", "status": "<established or plausible>"}}]}}"""

ARMS = {
    "A": {"label": "hard-only", "tier": False, "evidence": False, "fields": FIELDS2},
    "B": {"label": "hard-only + evidence rule", "tier": False, "evidence": True, "fields": FIELDS2},
    "C": {"label": "two-tier", "tier": True, "evidence": False, "fields": FIELDS2},
    "D": {"label": "two-tier + evidence rule", "tier": True, "evidence": True, "fields": FIELDS2},
    "E": {"label": "two-tier + evidence rule, 4 fields", "tier": True, "evidence": True, "fields": FIELDS4},
}

def fmt_item(W, i):
    w = W.items[i]
    head = f'"{w["title"][:70]}" ({w["publication_year"]}) | ' if CFG["show_titles"] else f'({w["publication_year"]}) | '
    return (head + f'genre: {", ".join(w["genres"])} | language: {", ".join(w["languages"])} | '
            f'country of origin: {", ".join(w["countries"])}')

def render_history(W, hist):
    lines = []
    for t, v in enumerate(hist, 1):
        g, p = v["cell"]
        lines.append(f"Visit {t}. Request: genre = {g}; publication period = {PERIOD_LABELS[p]}")
        for j, (i, ok) in enumerate(v["props"], 1):
            lines.append(f"  {j}. {fmt_item(W, i)} -> {'ACCEPTED' if ok else 'REJECTED'}")
    return "\n".join(lines)

def build_prompt(W, hist, arm):
    a = ARMS[arm]
    four = len(a["fields"]) == 4
    fields = "language, country, genre, or period" if four else "language or country"
    note = (' For "period" the value is one of: ' + ", ".join(PERIOD_LABELS) + ".") if four else ""
    parts = [BASE_PROMPT.format(history=render_history(W, hist)), USE_TIER if a["tier"] else USE_HARD]
    if a["evidence"]:
        parts.append(EVIDENCE_TIER if a["tier"] else EVIDENCE_HARD)
    parts.append((OUT_TIER if a["tier"] else OUT_HARD).format(fields=fields, note=note))
    return "\n\n".join(parts)

_W, _u = WRITE_USERS[0]
print(build_prompt(_W, prefix(HIST[(_W.pidx, _u, "L")], 3), "D"))

# %%
# ---- cached client for the OpenAI-compatible Baseten endpoint -------------------------------
def read_api_key():
    k = os.environ.get("BASETEN_API_KEY")
    env = ROOT / ".env"
    if not k and env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("BASETEN_API_KEY") and "=" in line:
                k = line.split("=", 1)[1].strip().strip('"').strip("'")
    return k or None

class JsonlCache:
    """Append-only record of every model call: key, model, sample, prompt, response."""

    def __init__(self, path):
        self.path, self.lock, self.d = path, threading.Lock(), {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.d[r["key"]] = r

    def get(self, key):
        return self.d.get(key)

    def put(self, rec):
        with self.lock:
            self.d[rec["key"]] = rec
            with self.path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

LLM_CACHE = JsonlCache(CACHE / "llm_responses.jsonl")

def job_key(model, prompt, sample):
    blob = json.dumps([CFG["models"][model], CFG["temperature"], CFG["max_tokens"], sample, prompt], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()

def _post(slug, prompt, key):
    body = json.dumps({"model": slug, "messages": [{"role": "user", "content": prompt}],
                       "temperature": CFG["temperature"], "max_tokens": CFG["max_tokens"]}).encode()
    req = urllib.request.Request(CFG["api_base"] + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    err = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
            msg = data["choices"][0]["message"]
            return {"content": msg.get("content") or "",
                    "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                    "finish": data["choices"][0].get("finish_reason"), "usage": data.get("usage")}
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
            if e.code not in (408, 409, 429, 500, 502, 503, 504):
                break
        except Exception as e:                                    # network errors, timeouts
            err = f"{type(e).__name__}: {e}"
        time.sleep(min(60, 2 ** (attempt + 1)))
    return {"error": err}

def list_models():
    key = read_api_key()
    req = urllib.request.Request(CFG["api_base"] + "/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return sorted(m["id"] for m in json.loads(r.read())["data"])

def run_llm(jobs):
    """jobs: {key: (model, prompt, sample)}. Calls the API for keys not yet cached."""
    todo = {k: j for k, j in jobs.items() if LLM_CACHE.get(k) is None}
    key = read_api_key()
    print(f"model calls planned: {len(jobs)} | already cached: {len(jobs) - len(todo)} | to run: {len(todo)}")
    if not todo:
        return
    if not key:
        print("no BASETEN_API_KEY found (environment or .env): model calls are skipped and the cells below report 'not run'")
        return
    if len(todo) > CFG["max_new_calls"]:
        print(f"refusing to run {len(todo)} new calls (> max_new_calls = {CFG['max_new_calls']})")
        return
    try:
        have = set(list_models())
        for name, slug in CFG["models"].items():
            if slug not in have:
                print(f"warning: model id {slug!r} for {name} is not in the endpoint's model list; "
                      f"close matches: {[m for m in have if name.split('-')[0].lower() in m.lower()][:5]}")
    except Exception as e:
        print("could not list models:", e)
    done = fail = 0
    t0 = time.time()
    def work(item):
        k, (model, prompt, sample) = item
        out = _post(CFG["models"][model], prompt, key)
        if "error" in out:
            return k, out["error"]
        LLM_CACHE.put({"key": k, "model": model, "sample": sample, "prompt": prompt, "ts": time.time(), **out})
        return k, None
    with ThreadPoolExecutor(CFG["llm_workers"]) as ex:
        for fut in as_completed([ex.submit(work, it) for it in todo.items()]):
            _, e = fut.result()
            if e:
                fail += 1
                if fail <= 3:
                    print("  call failed:", e)
            else:
                done += 1
            if (done + fail) % 200 == 0:
                print(f"  {done + fail}/{len(todo)} ({time.time() - t0:.0f}s)")
    print(f"finished: {done} new responses, {fail} failures, {time.time() - t0:.0f}s")

# ---- parsing --------------------------------------------------------------------------------
VOCAB = {
    "language": {v.lower(): v for w in CATALOG for v in w["languages"]},
    "country": {v.lower(): v for w in CATALOG for v in w["countries"]},
    "genre": {v.lower(): v for w in CATALOG for v in w["genres"]},
    "period": {v.lower(): v for v in PERIOD_LABELS},
}
FIELD_ALIASES = {"language": "language", "languages": "language", "country": "country", "countries": "country",
                 "country of origin": "country", "country_of_origin": "country", "genre": "genre", "genres": "genre",
                 "period": "period", "publication period": "period", "publication_period": "period"}

def extract_json(text):
    t = re.sub(r"```(?:json)?", "", text or "").strip()
    lo, hi = t.find("{"), t.rfind("}")
    if lo < 0 or hi <= lo:
        return None
    try:
        return json.loads(t[lo: hi + 1])
    except Exception:
        return None

def parse_rules(text, arm):
    """Model output -> {(field, value): status}, or None when it cannot be parsed.
    Rules outside the arm's schema are dropped and counted separately."""
    obj = extract_json(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("rules"), list):
        return None, 0
    a, rules, dropped = ARMS[arm], {}, 0
    for r in obj["rules"]:
        if not isinstance(r, dict):
            dropped += 1; continue
        f = FIELD_ALIASES.get(str(r.get("field", "")).strip().lower())
        v = r.get("value")
        if f is None or f not in a["fields"] or v is None:
            dropped += 1; continue
        v = str(v).strip().replace("–", "-").replace("—", "-")
        v = VOCAB[f].get(v.lower(), v)
        if a["tier"]:
            s = str(r.get("status", "")).strip().lower()
            if s not in ("established", "plausible"):
                dropped += 1; continue
        else:
            s = "hard"
        if rules.get((f, v)) != "established":
            rules[(f, v)] = s
    return rules, dropped

UNPARSED = collections.Counter()      # (model, arm) -> outputs that were not valid JSON in the requested form

def written(model, arm, W, u, utype="L", k=None, sample=0):
    """Memory written by `model` under `arm`; None if that call has not been run.
    An output that cannot be parsed stores nothing, so it is treated as an empty memory and counted."""
    k = k or CFG["main_prefix"]
    key = job_key(model, build_prompt(W, prefix(HIST[(W.pidx, u, utype)], k), arm), sample)
    rec = LLM_CACHE.get(key)
    if rec is None:
        return None
    if "parsed" not in rec:
        rules, _ = parse_rules(rec["content"], arm)
        rec["parsed"] = rules
        if rules is None:
            UNPARSED[(model, arm)] += 1
    return rec["parsed"] if rec["parsed"] is not None else {}

def plan_jobs():
    jobs = {}
    def add(arm, utype, k):
        for model in CFG["models"]:
            for W, u in WRITE_USERS:
                p = build_prompt(W, prefix(HIST[(W.pidx, u, utype)], k), arm)
                for s in range(CFG["n_samples"]):
                    jobs[job_key(model, p, s)] = (model, p, s)
    for arm in ARMS:                                   # Experiment 1: five arms, language-only users, 8 visits
        add(arm, "L", CFG["main_prefix"])
    for k in CFG["prefixes"]:                          # history length (arm A), visit-4 memories (arms A and E)
        add("A", "L", k)
    add("E", "L", 4)
    for arm in ARMS:                                   # language-and-country users: all arms (most histories are shared)
        add(arm, "LC", CFG["main_prefix"])
    for arm in ("A", "B", "E"):                        # country-only users: the three arms scored in Experiment 2
        add(arm, "C", CFG["main_prefix"])
    return jobs

JOBS = plan_jobs()
run_llm(JOBS)

def arm_ready(model, arm, utype="L", k=None):
    """True when every writing user has a parsed output for this cell."""
    return all(written(model, arm, W, u, utype, k) is not None for W, u in WRITE_USERS)

READY = {(m, a): arm_ready(m, a) for m in CFG["models"] for a in ARMS}
print("arms with complete model output (8 visits, language-only users):",
      [f"{m}/{a}" for (m, a), ok in READY.items() if ok] or "none yet")
if UNPARSED:
    print("outputs that could not be parsed (treated as an empty memory):", dict(UNPARSED))
(RESULTS / "prompt_templates.json").write_text(json.dumps({
    "base": BASE_PROMPT, "use_hard": USE_HARD, "use_tier": USE_TIER, "evidence_hard": EVIDENCE_HARD,
    "evidence_tier": EVIDENCE_TIER, "output_hard": OUT_HARD, "output_tier": OUT_TIER,
    "arms": {k: {**v, "fields": list(v["fields"])} for k, v in ARMS.items()}}, indent=1))

# %% [markdown]
# ## 6. Experiment 1 — unbundling the writing interface
# Outcomes are coded for the two target rules. *Firm* means stored as a hard or established rule.
# The key column is the joint outcome on ambiguous histories: the country rule is **not** firm
# **and** the language rule is kept in some form. Reference rows need no model.

# %%
def exp1_row(name, get, two_tier):
    """`get(W, u)` returns the written rules for a writing user's 8-visit history."""
    amb = [(W, u) for W, u in WRITE_USERS if AMBIG[(W.pidx, u)]]
    idn = [(W, u) for W, u in WRITE_USERS if not AMBIG[(W.pidx, u)]]
    R = {(W.pidx, u): get(W, u) for W, u in WRITE_USERS}
    if any(r is None for r in R.values()):
        return {"writer": name, "country firm | ambiguous": "not run"}, None
    cs = lambda W, u: rule_state(R[(W.pidx, u)], W.C)
    ls = lambda W, u: rule_state(R[(W.pidx, u)], W.L)
    agree = tot = 0
    for W, u in WRITE_USERS:
        vs = ref_version_space(W, prefix(HIST[(W.pidx, u, "L")], K))
        for key in (W.L, W.C):
            ref = rule_state(vs, key)
            if not two_tier:                       # a hard-only writer can only store or omit
                ref = "firm" if ref == "firm" else "absent"
            agree += rule_state(R[(W.pidx, u)], key) == ref; tot += 1
    extra = sum(any(k not in (W.L, W.C) for k in R[(W.pidx, u)]) for W, u in WRITE_USERS)
    flags = {
        "country_firm": {(W.pidx, u): cs(W, u) == "firm" for W, u in amb},
        "joint_ok": {(W.pidx, u): cs(W, u) != "firm" and ls(W, u) != "absent" for W, u in amb},
    }
    row = {
        "writer": name,
        "country firm | ambiguous": pct(sum(flags["country_firm"].values()), len(amb)),
        "country plausible | ambiguous": pct(sum(cs(W, u) == "plausible" for W, u in amb), len(amb), ci=False),
        "country kept | identifiable": pct(sum(cs(W, u) != "absent" for W, u in idn), len(idn), ci=False),
        "language firm": pct(sum(ls(W, u) == "firm" for W, u in WRITE_USERS), len(WRITE_USERS), ci=False),
        "language plausible": pct(sum(ls(W, u) == "plausible" for W, u in WRITE_USERS), len(WRITE_USERS), ci=False),
        "language absent": pct(sum(ls(W, u) == "absent" for W, u in WRITE_USERS), len(WRITE_USERS), ci=False),
        "country not firm and language kept | ambiguous": pct(sum(flags["joint_ok"].values()), len(amb)),
        "matches logical label": pct(agree, tot, ci=False),
        "histories with other rules": str(extra),
    }
    return row, flags

REF_WRITERS = {
    "reference: naive shared-attribute rule": (lambda W, u: ref_naive(W, prefix(HIST[(W.pidx, u, "L")], K)), False),
    "reference: logical, hard-only": (lambda W, u: ref_logical_hard(W, prefix(HIST[(W.pidx, u, "L")], K)), False),
    "reference: logical, two-tier": (lambda W, u: ref_version_space(W, prefix(HIST[(W.pidx, u, "L")], K)), True),
}
rows, EXP1_FLAGS = [], {}
for name, (get, tt) in REF_WRITERS.items():
    row, _ = exp1_row(name, get, tt)
    rows.append(row)
for model in CFG["models"]:
    for arm, a in ARMS.items():
        name = f"{model} | {arm}: {a['label']}"
        row, flags = exp1_row(name, lambda W, u, m=model, r=arm: written(m, r, W, u), a["tier"])
        if flags:
            row["unparsed outputs, all calls of this arm"] = str(UNPARSED.get((model, arm), 0))
        rows.append(row)
        if flags:
            EXP1_FLAGS[(model, arm)] = flags
EXP1 = pd.DataFrame(rows).astype(object).fillna("")
EXP1.to_csv(RESULTS / "exp1_writers.csv", index=False)

# the same key outcome by language pair (the Russian pair is the one to watch, see section 2)
rows = []
for model in CFG["models"]:
    for arm in ARMS:
        if READY[(model, arm)]:
            row = {"writer": f"{model} | {arm}"}
            for W in WORLDS:
                amb = [u for u in range(CFG["n_write_users"]) if AMBIG[(W.pidx, u)]]
                row[f"{W.lang}/{W.country}"] = pct(sum(rule_state(written(model, arm, W, u), W.C) == "firm" for u in amb), len(amb), ci=False)
            rows.append(row)
EXP1_BY_PAIR = pd.DataFrame(rows)
EXP1_BY_PAIR.to_csv(RESULTS / "exp1_writers_by_pair.csv", index=False)
if len(EXP1_BY_PAIR):
    print("country rule firm on ambiguous histories, by language pair")
    print(EXP1_BY_PAIR.to_string(index=False), "\n")
EXP1

# %%
# Paired tests between neighbouring arms (same histories), and the history-length analysis.
rows = []
for model in CFG["models"]:
    for a, b, what in (("A", "B", "adds evidence rule"), ("A", "C", "adds tier"), ("C", "D", "adds evidence rule"),
                       ("D", "E", "adds two fields"), ("A", "E", "all three")):
        if (model, a) in EXP1_FLAGS and (model, b) in EXP1_FLAGS:
            for outcome in ("country_firm", "joint_ok"):
                fa, fb = EXP1_FLAGS[(model, a)][outcome], EXP1_FLAGS[(model, b)][outcome]
                only_a = sum(fa[k] and not fb[k] for k in fa); only_b = sum(fb[k] and not fa[k] for k in fa)
                rows.append({"model": model, "contrast": f"{a} vs {b} ({what})", "outcome": outcome,
                             f"true only in first": only_a, "true only in second": only_b,
                             "McNemar exact p": f"{mcnemar_exact(only_a, only_b):.4g}"})
EXP1_TESTS = pd.DataFrame(rows)
EXP1_TESTS.to_csv(RESULTS / "exp1_paired_tests.csv", index=False)
print(EXP1_TESTS.to_string(index=False) if len(EXP1_TESTS) else "paired tests: not run (no model output yet)")

rows = []
for k in CFG["prefixes"]:
    amb = [(W, u) for W, u in WRITE_USERS if not has_counterexample(W, prefix(HIST[(W.pidx, u, "L")], k), W.C)]
    row = {"visits": k, "ambiguous histories": f"{len(amb)}/{len(WRITE_USERS)}",
           "naive rule": pct(sum(rule_state(ref_naive(W, prefix(HIST[(W.pidx, u, 'L')], k)), W.C) == "firm" for W, u in amb), len(amb), ci=False)}
    for model in CFG["models"]:
        if arm_ready(model, "A", "L", k):
            row[f"{model} arm A"] = pct(sum(rule_state(written(model, "A", W, u, "L", k), W.C) == "firm" for W, u in amb), len(amb))
        else:
            row[f"{model} arm A"] = "not run"
    rows.append(row)
print("\ncountry rule stored as a firm rule on ambiguous histories, by history length")
EXP1_LENGTH = pd.DataFrame(rows)
EXP1_LENGTH.to_csv(RESULTS / "exp1_history_length.csv", index=False)
EXP1_LENGTH

# %% [markdown]
# ## 7. Experiment 2 — the same writers scored against three user types
# The stored country rule is wrong for a language-only user and right for the other two types;
# the stored language rule is wrong for a country-only user. For each writer we report how often
# it stores a rule firmly when the rule is false of the user (**false commitment**), how often it
# fails to store a true rule firmly (**miss**), and how often a true rule is absent altogether
# (**lost**). The last columns give the precision of the firm tier for the country rule under
# different mixes of user types.

# %%
MIXES = {"all language-only": (1, 0, 0), "L : L+C = 1:1": (0.5, 0.5, 0), "three types equal": (1 / 3, 1 / 3, 1 / 3)}

def exp2_rows(name, get):
    """`get(W, u, utype)` returns written rules for the 8-visit history of that user type."""
    R = {(W.pidx, u, t): get(W, u, t) for W, u in WRITE_USERS for t in UTYPES}
    if any(r is None for r in R.values()):
        return [{"writer": name, "rule": "not run"}]
    out = []
    firm_rate = {}
    for rk_name, rk in (("country", "C"), ("language", "L")):
        row = {"writer": name, "rule": rk_name}
        for t in UTYPES:
            states = [rule_state(R[(W.pidx, u, t)], getattr(W, rk)) for W, u in WRITE_USERS]
            n = len(states); firm = sum(s == "firm" for s in states); absent = sum(s == "absent" for s in states)
            firm_rate[(rk, t)] = firm / n
            true_rule = rk in t                       # "L" in "LC", "C" in "LC", etc.
            if true_rule:
                row[f"{t}: miss"] = pct(n - firm, n, ci=False); row[f"{t}: lost"] = pct(absent, n, ci=False)
            else:
                row[f"{t}: false commitment"] = pct(firm, n, ci=False)
        for mname, w in MIXES.items():
            num = sum(wi * firm_rate[(rk, t)] for wi, t in zip(w, UTYPES) if rk in t)
            den = sum(wi * firm_rate[(rk, t)] for wi, t in zip(w, UTYPES))
            row[f"firm-tier precision, {mname}"] = "n/a (never firm)" if den == 0 else f"{num / den:.2f}"
        out.append(row)
    return out

rows = []
for name, fn in (("reference: naive shared-attribute rule", ref_naive), ("reference: logical, hard-only", ref_logical_hard),
                 ("reference: logical, two-tier", ref_version_space)):
    rows += exp2_rows(name, lambda W, u, t, f=fn: f(W, prefix(HIST[(W.pidx, u, t)], K)))
for model in CFG["models"]:
    for arm in ("A", "B", "E"):
        rows += exp2_rows(f"{model} | {arm}: {ARMS[arm]['label']}", lambda W, u, t, m=model, r=arm: written(m, r, W, u, t))
EXP2_COLS = ["writer", "rule", "L: false commitment", "L: miss", "L: lost", "LC: miss", "LC: lost",
             "C: false commitment", "C: miss", "C: lost"] + [f"firm-tier precision, {m}" for m in MIXES]
EXP2 = pd.DataFrame(rows).reindex(columns=EXP2_COLS).astype(object).fillna("")
EXP2.to_csv(RESULTS / "exp2_user_types.csv", index=False)
EXP2

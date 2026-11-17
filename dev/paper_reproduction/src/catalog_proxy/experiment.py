from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PERIODS: tuple[tuple[str, int, int], ...] = (
    ("before 1900", -10_000, 1899),
    ("1900--1949", 1900, 1949),
    ("1950--1999", 1950, 1999),
    ("2000 or later", 2000, 10_000),
)


@dataclass(frozen=True)
class Item:
    id: str
    title: str
    genres: tuple[str, ...]
    languages: tuple[str, ...]
    countries: tuple[str, ...]
    publication_year: int
    sitelink_count: int

    @classmethod
    def from_dict(cls, raw: dict) -> "Item | None":
        if not raw.get("countries") or raw.get("publication_year") is None:
            return None
        return cls(
            id=str(raw["id"]),
            title=str(raw.get("title", "")),
            genres=tuple(str(x) for x in raw.get("genres", [])),
            languages=tuple(str(x) for x in raw.get("languages", [])),
            countries=tuple(str(x) for x in raw.get("countries", [])),
            publication_year=int(raw["publication_year"]),
            sitelink_count=int(raw.get("sitelink_count") or 0),
        )


@dataclass(frozen=True, order=True)
class Predicate:
    field: str
    value: str

    def holds(self, item: Item) -> bool:
        return self.value in getattr(item, self.field)


@dataclass(frozen=True)
class Request:
    genre: str
    period_name: str
    year_min: int
    year_max: int

    def holds(self, item: Item) -> bool:
        return self.genre in item.genres and self.year_min <= item.publication_year <= self.year_max

    @property
    def key(self) -> str:
        return f"{self.genre}|{self.period_name}"


@dataclass
class Memory:
    established: set[Predicate] = field(default_factory=set)
    tie: set[Predicate] = field(default_factory=set)

    def copy(self) -> "Memory":
        return Memory(set(self.established), set(self.tie))


@dataclass(frozen=True)
class PairCatalogs:
    language: str
    country: str
    learning: tuple[Item, ...]
    shifted: tuple[Item, ...]
    requests: tuple[Request, ...]
    target_count: int
    learning_both: int
    shifted_both: int


@dataclass
class VisitResult:
    success: int
    proposals: int
    rejected: int
    tie_single_rejects: int
    tie_multiple_rejects: int
    whole_tie_available: int
    events: list[dict] = field(default_factory=list)


@dataclass
class UserRecord:
    user_id: str
    pair: str
    method: str
    branch: str
    phase: str
    visit: int
    success: int
    proposals: int
    rejected: int
    tie_single_rejects: int
    tie_multiple_rejects: int
    whole_tie_available: int
    established_recall: float
    unresolved: int
    excluded_share: float


def _stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sample(items: Sequence[Item], n: int, seed: int, *label: object) -> list[Item]:
    if n > len(items):
        raise ValueError(f"requested {n} from a pool of {len(items)} for {label}")
    rng = random.Random(_stable_int(seed, *label))
    return rng.sample(list(items), n)


def load_items(path: Path) -> list[Item]:
    raw = json.loads(path.read_text())
    items = [Item.from_dict(row) for row in raw]
    return [item for item in items if item is not None]


def _period_for_item(item: Item) -> tuple[str, int, int]:
    for period in PERIODS:
        if period[1] <= item.publication_year <= period[2]:
            return period
    raise AssertionError(item.publication_year)


def build_pair_catalogs(
    items: Sequence[Item],
    *,
    language: str,
    country: str,
    target_count: int,
    catalog_size: int,
    seed: int,
    learning_coverage: float = 0.9,
    shifted_coverage: float = 0.1,
) -> PairCatalogs:
    if not 0 <= learning_coverage <= 1 or not 0 <= shifted_coverage <= 1:
        raise ValueError("coverage must be between zero and one")
    both = [x for x in items if language in x.languages and country in x.countries]
    language_other = [x for x in items if language in x.languages and country not in x.countries]
    country_other = [x for x in items if country in x.countries and language not in x.languages]
    background = [x for x in items if language not in x.languages and country not in x.countries]

    learning_both = round(learning_coverage * target_count)
    shifted_both = round(shifted_coverage * target_count)
    c0_targets = _sample(both, learning_both, seed, language, "c0-both") + _sample(
        language_other, target_count - learning_both, seed, language, "c0-other"
    )
    c1_targets = _sample(both, shifted_both, seed, language, "c1-both") + _sample(
        language_other, target_count - shifted_both, seed, language, "c1-other"
    )

    # Country-only works are retained on both sides. They make the proxy useful
    # but imperfect on C0 and prevent an empty result from trivially revealing
    # the shift on C1.
    retained_country_other = sorted(country_other, key=lambda x: x.id)
    n_background = catalog_size - target_count - len(retained_country_other)
    common_background = _sample(background, n_background, seed, language, "background")
    learning = tuple(sorted(c0_targets + retained_country_other + common_background, key=lambda x: x.id))
    shifted = tuple(sorted(c1_targets + retained_country_other + common_background, key=lambda x: x.id))

    requests: list[Request] = []
    genres = sorted({g for item in c0_targets + c1_targets for g in item.genres})
    for genre in genres:
        for period_name, lo, hi in PERIODS:
            q = Request(genre, period_name, lo, hi)
            c0_language = [x for x in learning if q.holds(x) and language in x.languages]
            c1_language = [x for x in shifted if q.holds(x) and language in x.languages]
            c0_joint = [x for x in c0_language if country in x.countries]
            if c0_language and c1_language and c0_joint:
                requests.append(q)
    if len(requests) < 4:
        raise ValueError(f"only {len(requests)} valid requests for {language}/{country}")

    return PairCatalogs(
        language=language,
        country=country,
        learning=learning,
        shifted=shifted,
        requests=tuple(requests),
        target_count=target_count,
        learning_both=learning_both,
        shifted_both=shifted_both,
    )


def _rank(items: Iterable[Item], *, seed: int, user_id: str, visit: int, branch: str) -> list[Item]:
    return sorted(items, key=lambda x: _stable_int(seed, user_id, visit, branch, x.id))


def _matches(item: Item, predicates: Iterable[Predicate]) -> bool:
    return all(p.holds(item) for p in predicates)


def _acceptable(item: Item, language: str) -> bool:
    return language in item.languages


def _update_after_verdict(memory: Memory, item: Item, accepted: bool) -> None:
    if accepted:
        memory.tie = {p for p in memory.tie if p.holds(item)}
        return
    violated = {p for p in memory.tie if not p.holds(item)}
    if len(violated) == 1:
        established = next(iter(violated))
        memory.tie.remove(established)
        memory.established.add(established)


def _propose_sequence(
    method: str,
    catalog: Sequence[Item],
    request: Request,
    memory: Memory,
    *,
    language: str,
    budget: int,
    seed: int,
    user_id: str,
    visit: int,
    branch: str,
) -> VisitResult:
    eligible = [x for x in catalog if request.holds(x)]
    proposed: set[str] = set()
    proposals = rejected = single_rejects = multiple_rejects = 0
    whole_tie = [x for x in eligible if _matches(x, memory.established | memory.tie)]
    whole_tie_available = int(bool(whole_tie))
    eager_test_used = False
    strict_rejected = False
    tried_relaxations: set[Predicate] = set()
    events: list[dict] = []

    while proposals < budget:
        remaining = [x for x in eligible if x.id not in proposed]
        if not remaining:
            break

        mode = "ordinary"
        relaxed: set[Predicate] = set()
        candidates: list[Item]

        if method == "tie":
            candidates = [x for x in remaining if _matches(x, memory.established | memory.tie)]
            if not candidates and memory.tie:
                single_options: list[tuple[int, Predicate, list[Item]]] = []
                for predicate in sorted(memory.tie):
                    keep = memory.established | (memory.tie - {predicate})
                    pool = [x for x in remaining if _matches(x, keep) and not predicate.holds(x)]
                    if pool:
                        single_options.append((len(pool), predicate, pool))
                if single_options:
                    _, predicate, candidates = max(single_options, key=lambda z: (z[0], z[1]))
                    relaxed = {predicate}
                    mode = "single"
                else:
                    scored = [(sum(not p.holds(x) for p in memory.tie), x) for x in remaining if _matches(x, memory.established)]
                    if not scored:
                        break
                    min_violations = min(score for score, _ in scored)
                    candidates = [x for score, x in scored if score == min_violations]
                    relaxed = {p for p in memory.tie if not p.holds(candidates[0])}
                    mode = "multiple"
            elif not candidates:
                break
        elif method in {"frozen_relax", "transient_relax", "relax_delete"}:
            predicates = memory.established
            candidates = [] if strict_rejected else [x for x in remaining if _matches(x, predicates)]
            if not candidates and predicates:
                options: list[tuple[int, Predicate, list[Item]]] = []
                for predicate in sorted(predicates - tried_relaxations):
                    keep = predicates - {predicate}
                    pool = [x for x in remaining if _matches(x, keep) and not predicate.holds(x)]
                    if pool:
                        options.append((len(pool), predicate, pool))
                if options:
                    _, predicate, candidates = max(options, key=lambda z: (z[0], z[1]))
                    relaxed = {predicate}
                    tried_relaxations.add(predicate)
                    mode = "single"
                else:
                    candidates = remaining
                    mode = "multiple"
        elif method == "soft_rank":
            # Stored rules are ranking features, not eligibility constraints.
            # Recommend from the highest matching tier while retaining all
            # request-matching items as fallbacks after rejections.
            scores = {
                item.id: sum(predicate.holds(item) for predicate in memory.established)
                for item in remaining
            }
            best_score = max(scores.values())
            candidates = [item for item in remaining if scores[item.id] == best_score]
        elif method == "eager" and memory.tie and not eager_test_used:
            options: list[tuple[int, Predicate, list[Item]]] = []
            for predicate in sorted(memory.tie):
                keep = memory.established | (memory.tie - {predicate})
                pool = [x for x in remaining if _matches(x, keep) and not predicate.holds(x)]
                if pool:
                    options.append((len(pool), predicate, pool))
            if options:
                _, predicate, candidates = max(options, key=lambda z: (z[0], z[1]))
                relaxed = {predicate}
                mode = "single"
                eager_test_used = True
            else:
                candidates = [x for x in remaining if _matches(x, memory.established)]
        else:
            predicates = memory.established if method in {
                "passive", "eager", "frozen", "hard_filter", "clarify_once",
            } else set()
            candidates = [x for x in remaining if _matches(x, predicates)]

        if not candidates:
            break
        item = _rank(candidates, seed=seed, user_id=user_id, visit=visit, branch=branch)[0]
        proposed.add(item.id)
        proposals += 1
        accepted = _acceptable(item, language)
        events.append({
            "item_id": item.id,
            "title": item.title,
            "genres": list(item.genres),
            "languages": list(item.languages),
            "countries": list(item.countries),
            "publication_year": item.publication_year,
            "verdict": "accept" if accepted else "reject",
        })
        if method in {"passive", "eager", "tie"}:
            _update_after_verdict(memory, item, accepted)
        elif method == "relax_delete" and accepted:
            # An accepted item is a direct counterexample to every committed
            # predicate that it violates.  The serving fallback therefore
            # becomes a memory update rather than a transient bypass.
            memory.established = {p for p in memory.established if p.holds(item)}
        if accepted:
            return VisitResult(1, proposals, rejected, single_rejects, multiple_rejects, whole_tie_available, events)
        rejected += 1
        if method in {"frozen_relax", "transient_relax", "relax_delete"} and mode == "ordinary":
            strict_rejected = True
        if mode == "single":
            single_rejects += 1
        elif mode == "multiple":
            multiple_rejects += 1

    return VisitResult(0, proposals, rejected, single_rejects, multiple_rejects, whole_tie_available, events)


def _request_schedule(pair: PairCatalogs, *, user_id: str, visits: int, seed: int) -> list[Request]:
    rng = random.Random(_stable_int(seed, user_id, "requests"))
    schedule = [rng.choice(pair.requests) for _ in range(visits)]
    # Guarantee that the shifted branch contains an inventory-forced tie test
    # whenever the source pool provides one.
    forced = [
        q for q in pair.requests
        if not any(q.holds(x) and pair.language in x.languages and pair.country in x.countries for x in pair.shifted)
    ]
    if forced and visits:
        schedule[0] = rng.choice(forced)
    return schedule


def _excluded_share(catalog: Sequence[Item], memory: Memory, language: str, method: str) -> float:
    acceptable = [x for x in catalog if _acceptable(x, language)]
    if not acceptable:
        return math.nan
    if method == "tie":
        predicates = memory.established | memory.tie
    elif method == "soft_rank":
        return 0.0
    elif method in {
        "passive", "eager", "frozen", "frozen_relax", "transient_relax", "relax_delete",
        "hard_filter", "clarify_once",
    }:
        predicates = memory.established
    else:
        predicates = set()
    kept = sum(_matches(x, predicates) for x in acceptable)
    return 1.0 - kept / len(acceptable)


def _state_metrics(memory: Memory, language: str, catalog: Sequence[Item], method: str) -> tuple[float, int, float]:
    true_rule = Predicate("languages", language)
    recall = float(true_rule in memory.established)
    return recall, len(memory.tie), _excluded_share(catalog, memory, language, method)


def run_scripted(
    pair: PairCatalogs,
    *,
    method: str,
    users: int,
    learning_visits: int,
    continuation_visits: int,
    budget: int,
    seed: int,
) -> list[UserRecord]:
    records: list[UserRecord] = []
    for user_index in range(users):
        user_id = f"{pair.language[:3].lower()}-{user_index:03d}"
        candidates = {Predicate("languages", pair.language), Predicate("countries", pair.country)}
        true_rule = Predicate("languages", pair.language)
        memory = Memory(
            established=(
                {true_rule} if method == "clarify_once"
                else set(candidates) if method in {
                    "relax_delete", "hard_filter", "soft_rank", "transient_relax",
                }
                else set()
            ),
            tie=set(candidates) if method in {"passive", "eager", "tie"} else set(),
        )
        learning_schedule = _request_schedule(pair, user_id=user_id + "-learn", visits=learning_visits, seed=seed)
        for visit, request in enumerate(learning_schedule):
            result = _propose_sequence(
                method, pair.learning, request, memory,
                language=pair.language, budget=budget, seed=seed,
                user_id=user_id, visit=visit, branch="learn",
            )
            est, unresolved, excl = _state_metrics(memory, pair.language, pair.learning, method)
            records.append(UserRecord(
                user_id, f"{pair.language}/{pair.country}", method, "learning", "learning", visit,
                result.success, result.proposals, result.rejected, result.tie_single_rejects,
                result.tie_multiple_rejects, result.whole_tie_available, est, unresolved, excl,
            ))

        checkpoint = memory.copy()
        schedule = _request_schedule(pair, user_id=user_id + "-continue", visits=continuation_visits, seed=seed)
        for branch, catalog in (("c0", pair.learning), ("shift", pair.shifted)):
            branch_memory = checkpoint.copy()
            for visit, request in enumerate(schedule):
                result = _propose_sequence(
                    method, catalog, request, branch_memory,
                    language=pair.language, budget=budget, seed=seed,
                    user_id=user_id, visit=visit, branch=branch,
                )
                est, unresolved, excl = _state_metrics(branch_memory, pair.language, catalog, method)
                records.append(UserRecord(
                    user_id, f"{pair.language}/{pair.country}", method, branch, "continuation", visit,
                    result.success, result.proposals, result.rejected, result.tie_single_rejects,
                    result.tie_multiple_rejects, result.whole_tie_available, est, unresolved, excl,
                ))
    return records


def _bootstrap_user_mean(values: dict[str, float], *, reps: int, seed: int) -> dict[str, float]:
    ids = sorted(values)
    observed = float(np.mean([values[x] for x in ids]))
    rng = np.random.default_rng(seed)
    arr = np.asarray([values[x] for x in ids], dtype=float)
    draws = rng.choice(arr, size=(reps, len(arr)), replace=True).mean(axis=1)
    return {"mean": observed, "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975))}


def summarize(records: Sequence[UserRecord], *, bootstrap_replicates: int, seed: int) -> dict:
    summary: dict[str, dict] = {}
    methods = sorted({r.method for r in records})
    for method in methods:
        summary[method] = {}
        learning_subset = [r for r in records if r.method == method and r.branch == "learning"]
        learning_by_user: dict[str, list[UserRecord]] = {}
        for row in learning_subset:
            learning_by_user.setdefault(row.user_id, []).append(row)
        if learning_by_user:
            learning_proposals = {u: float(sum(x.proposals for x in rows)) for u, rows in learning_by_user.items()}
            learning_rejections = {u: float(sum(x.rejected for x in rows)) for u, rows in learning_by_user.items()}
            summary[method]["learning"] = {
                "proposals_per_user": _bootstrap_user_mean(
                    learning_proposals, reps=bootstrap_replicates,
                    seed=_stable_int(seed, method, "learning", "proposals"),
                ),
                "rejections_per_user": _bootstrap_user_mean(
                    learning_rejections, reps=bootstrap_replicates,
                    seed=_stable_int(seed, method, "learning", "rejections"),
                ),
            }
        for branch in ("c0", "shift"):
            subset = [r for r in records if r.method == method and r.branch == branch]
            by_user: dict[str, list[UserRecord]] = {}
            for row in subset:
                by_user.setdefault(row.user_id, []).append(row)
            success = {u: float(np.mean([x.success for x in rows])) for u, rows in by_user.items()}
            proposals = {u: float(sum(x.proposals for x in rows)) for u, rows in by_user.items()}
            rejections = {u: float(sum(x.rejected for x in rows)) for u, rows in by_user.items()}
            single_rejections = {u: float(sum(x.tie_single_rejects for x in rows)) for u, rows in by_user.items()}
            multiple_rejections = {u: float(sum(x.tie_multiple_rejects for x in rows)) for u, rows in by_user.items()}
            final_rows = [max(rows, key=lambda x: x.visit) for rows in by_user.values()]
            summary[method][branch] = {
                "success": _bootstrap_user_mean(success, reps=bootstrap_replicates, seed=_stable_int(seed, method, branch, "success")),
                "proposals_per_user": _bootstrap_user_mean(proposals, reps=bootstrap_replicates, seed=_stable_int(seed, method, branch, "proposals")),
                "rejections_per_user": _bootstrap_user_mean(rejections, reps=bootstrap_replicates, seed=_stable_int(seed, method, branch, "rejections")),
                "established_recall": float(np.mean([x.established_recall for x in final_rows])),
                "unresolved": float(np.mean([x.unresolved for x in final_rows])),
                "excluded_share": float(np.mean([x.excluded_share for x in final_rows])),
                "single_rejects_per_user": float(np.mean([sum(x.tie_single_rejects for x in rows) for rows in by_user.values()])),
                "multiple_rejects_per_user": float(np.mean([sum(x.tie_multiple_rejects for x in rows) for rows in by_user.values()])),
                "single_rejects_per_user_ci": _bootstrap_user_mean(
                    single_rejections, reps=bootstrap_replicates,
                    seed=_stable_int(seed, method, branch, "single-rejections"),
                ),
                "multiple_rejects_per_user_ci": _bootstrap_user_mean(
                    multiple_rejections, reps=bootstrap_replicates,
                    seed=_stable_int(seed, method, branch, "multiple-rejections"),
                ),
                "whole_tie_available": float(np.mean([x.whole_tie_available for x in subset])),
            }
    return summary


def write_records(records: Sequence[UserRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in records:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def run_primary(config_path: Path, catalog_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text())
    items = load_items(catalog_path)
    pairs = [
        build_pair_catalogs(
            items,
            language=spec["language"], country=spec["country"],
            target_count=spec["target_count"], catalog_size=config["catalog_size"], seed=config["seed"],
        )
        for spec in config["pairs"]
    ]
    records: list[UserRecord] = []
    for pair in pairs:
        for method in (
            "no_memory", "hard_filter", "soft_rank", "transient_relax", "clarify_once",
            "relax_delete", "passive", "eager", "tie",
        ):
            records.extend(run_scripted(
                pair, method=method, users=config["users_per_pair"],
                learning_visits=config["learning_visits"], continuation_visits=config["continuation_visits"],
                budget=config["proposal_budget"], seed=config["seed"],
            ))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_records(records, output_dir / "scripted_records.jsonl")
    result = {
        "config": config,
        "source_catalog": f"tau-rec/books-wikidata-v2/{catalog_path.name}",
        "source_catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "catalogs": [
            {
                "pair": f"{p.language}/{p.country}",
                "size": len(p.learning),
                "target_count_each": p.target_count,
                "learning_proxy_coverage": p.learning_both / p.target_count,
                "shifted_proxy_coverage": p.shifted_both / p.target_count,
                "valid_requests": len(p.requests),
            }
            for p in pairs
        ],
        "summary": summarize(records, bootstrap_replicates=config["bootstrap_replicates"], seed=config["seed"]),
    }
    (output_dir / "scripted_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

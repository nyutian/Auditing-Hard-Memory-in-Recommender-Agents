from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from catalog_proxy.experiment import (
    Item,
    PairCatalogs,
    Predicate,
    Request,
    _matches,
    _rank,
    _stable_int,
    build_pair_catalogs,
    load_items,
)


TRUTHS = ("language", "country", "both")
NOISE_LEVELS = (0.0, 0.05, 0.10)
CONFIRMATIONS = (1, 2, 3)


def truth_rules(pair: PairCatalogs, truth: str) -> set[Predicate]:
    language = Predicate("languages", pair.language)
    country = Predicate("countries", pair.country)
    if truth == "language":
        return {language}
    if truth == "country":
        return {country}
    if truth == "both":
        return {language, country}
    raise ValueError(truth)


def acceptable(item: Item, requirements: set[Predicate]) -> bool:
    return _matches(item, requirements)


def request_schedule(
    pair: PairCatalogs,
    catalog: tuple[Item, ...],
    requirements: set[Predicate],
    *,
    user_id: str,
    visits: int,
    seed: int,
    branch: str,
) -> list[Request]:
    valid = [
        request for request in pair.requests
        if any(request.holds(item) and acceptable(item, requirements) for item in catalog)
    ]
    if not valid:
        raise ValueError(f"no valid requests for {pair.language}/{pair.country} {requirements}")
    rng = random.Random(_stable_int(seed, user_id, branch, "stress-requests"))
    schedule = [rng.choice(valid) for _ in range(visits)]
    initial = {
        Predicate("languages", pair.language),
        Predicate("countries", pair.country),
    }
    resolving = [
        request for request in valid
        if not any(request.holds(item) and _matches(item, initial) for item in catalog)
    ]
    if resolving and visits:
        schedule[0] = rng.choice(resolving)
    return schedule


def simulate_user(
    pair: PairCatalogs,
    *,
    truth: str,
    branch: str,
    noise: float,
    confirmations: int,
    user_id: str,
    visits: int,
    budget: int,
    seed: int,
) -> dict:
    catalog = pair.learning if branch == "c0" else pair.shifted
    requirements = truth_rules(pair, truth)
    initial = {
        Predicate("languages", pair.language),
        Predicate("countries", pair.country),
    }
    memory = set(initial)
    counterexamples: dict[Predicate, int] = defaultdict(int)
    schedule = request_schedule(
        pair, catalog, requirements, user_id=user_id, visits=visits,
        seed=seed, branch=branch,
    )
    observed_successes = valid_successes = proposals = rejections = 0
    correction_visit: int | None = None
    for visit, request in enumerate(schedule):
        eligible = [item for item in catalog if request.holds(item)]
        proposed: set[str] = set()
        strict_rejected = False
        tried_relaxations: set[Predicate] = set()
        visit_observed_success = False
        visit_valid_success = False
        while len(proposed) < budget:
            remaining = [item for item in eligible if item.id not in proposed]
            if not remaining:
                break
            candidates = [] if strict_rejected else [item for item in remaining if _matches(item, memory)]
            if not candidates and memory:
                options: list[tuple[int, Predicate, list[Item]]] = []
                for predicate in sorted(memory - tried_relaxations):
                    keep = memory - {predicate}
                    pool = [
                        item for item in remaining
                        if _matches(item, keep) and not predicate.holds(item)
                    ]
                    if pool:
                        options.append((len(pool), predicate, pool))
                if options:
                    _, predicate, candidates = max(options, key=lambda option: (option[0], option[1]))
                    tried_relaxations.add(predicate)
                else:
                    candidates = remaining
            if not candidates:
                break
            item = _rank(
                candidates, seed=seed, user_id=user_id,
                visit=visit, branch=f"stress-{branch}",
            )[0]
            proposed.add(item.id)
            proposals += 1
            true_accept = acceptable(item, requirements)
            flip_rng = random.Random(
                _stable_int(seed, user_id, truth, branch, noise, visit, item.id, "verdict")
            )
            observed_accept = (not true_accept) if flip_rng.random() < noise else true_accept
            if observed_accept:
                visit_observed_success = True
                visit_valid_success = true_accept
                for predicate in list(memory):
                    if not predicate.holds(item):
                        counterexamples[predicate] += 1
                        if counterexamples[predicate] >= confirmations:
                            memory.remove(predicate)
                if correction_visit is None and not ((initial - requirements) & memory):
                    correction_visit = visit + 1
                break
            rejections += 1
            if _matches(item, memory):
                strict_rejected = True
        observed_successes += int(visit_observed_success)
        valid_successes += int(visit_valid_success)

    true_retained = requirements & memory
    false_rules = initial - requirements
    return {
        "pair": f"{pair.language}/{pair.country}",
        "user_id": user_id,
        "truth": truth,
        "branch": branch,
        "noise": noise,
        "confirmations": confirmations,
        "observed_success": observed_successes / visits,
        "valid_success": valid_successes / visits,
        "proposals": proposals,
        "rejections": rejections,
        "false_deletion": float(len(true_retained) < len(requirements)),
        "false_rule_retained": float(bool(false_rules & memory)),
        "exact_memory": float(memory == requirements),
        "true_rule_recall": len(true_retained) / len(requirements),
        "correction_visit": correction_visit,
        "final_memory": [predicate.__dict__ for predicate in sorted(memory)],
    }


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["truth"], row["branch"], row["noise"], row["confirmations"])
        groups[key].append(row)
    output: list[dict] = []
    for (truth, branch, noise, confirmations), selected in sorted(groups.items()):
        output.append({
            "truth": truth,
            "branch": branch,
            "noise": noise,
            "confirmations": confirmations,
            "users": len(selected),
            **{
                metric: float(np.mean([row[metric] for row in selected]))
                for metric in (
                    "observed_success", "valid_success", "proposals", "rejections",
                    "false_deletion", "false_rule_retained", "exact_memory", "true_rule_recall",
                )
            },
        })
    return output


def plot_summary(rows: list[dict], summary: list[dict], output: Path) -> None:
    mpl.rcParams.update({
        "font.size": 7,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.05))
    colors = {1: "#D55E00", 2: "#0072B2", 3: "#009E73"}
    ax = axes[0]
    visit_axis = np.arange(1, 6)
    for confirmations in CONFIRMATIONS:
        selected = [
            row for row in rows
            if row["truth"] == "language" and row["branch"] == "shift"
            and row["noise"] == 0.0 and row["confirmations"] == confirmations
        ]
        ax.plot(
            visit_axis,
            [
                100 * np.mean([
                    row["correction_visit"] is not None and row["correction_visit"] <= visit
                    for row in selected
                ])
                for visit in visit_axis
            ],
            marker="o", markersize=3, linewidth=1.1, color=colors[confirmations],
            label=f"delete after {confirmations}",
        )
    ax.set_title("Correction under deterministic feedback")
    ax.set_ylabel("Users with proxy removed (%)")
    ax.set_xlabel("Shifted-catalog visit")
    ax.set_ylim(-3, 103)
    ax.set_xticks(visit_axis)
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")

    ax = axes[1]
    for confirmations in CONFIRMATIONS:
        points = [
            row for row in summary
            if row["truth"] == "both" and row["branch"] == "shift"
            and row["confirmations"] == confirmations
        ]
        ax.plot(
            [100 * row["noise"] for row in points],
            [100 * row["false_deletion"] for row in points],
            marker="o", markersize=3, linewidth=1.1, color=colors[confirmations],
            label=f"delete after {confirmations}",
        )
    ax.set_title("Deleting a true rule")
    ax.set_ylabel("Users with false deletion (%)")
    ax.set_xlabel("Verdict noise (%)")
    ax.set_ylim(-3, 103)
    ax.set_xticks([0, 5, 10])
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    fig.tight_layout(pad=0.45, w_pad=1.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/primary.json"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/stress/stress_summary.json"))
    parser.add_argument("--figure", type=Path, default=Path("outputs/paper_assets/stress_tradeoff.pdf"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    items = load_items(args.catalog)
    pairs = [
        build_pair_catalogs(
            items, language=spec["language"], country=spec["country"],
            target_count=spec["target_count"], catalog_size=config["catalog_size"],
            seed=config["seed"],
        )
        for spec in config["pairs"]
    ]
    rows = [
        simulate_user(
            pair, truth=truth, branch=branch, noise=noise,
            confirmations=confirmations,
            user_id=f"stress-{pair.language[:3].lower()}-{index:03d}",
            visits=config["continuation_visits"], budget=config["proposal_budget"],
            seed=config["seed"],
        )
        for pair in pairs
        for index in range(config["users_per_pair"])
        for truth in TRUTHS
        for branch in ("c0", "shift")
        for noise in NOISE_LEVELS
        for confirmations in CONFIRMATIONS
    ]
    summary = aggregate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary}, indent=2, sort_keys=True) + "\n")
    with args.output.with_name("stress_records.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    plot_summary(rows, summary, args.figure)
    print(args.output)
    print(args.figure)


if __name__ == "__main__":
    main()

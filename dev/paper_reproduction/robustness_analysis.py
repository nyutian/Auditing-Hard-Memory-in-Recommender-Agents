from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from catalog_proxy.experiment import (
    Memory,
    PairCatalogs,
    Predicate,
    _bootstrap_user_mean,
    _excluded_share,
    _propose_sequence,
    _request_schedule,
    _stable_int,
    build_pair_catalogs,
    load_items,
)
from catalog_proxy.llm_experiment import _logging_history


METHODS = ("plain_notes", "count_gated", "judge_gated")
COVERAGES = (0.1, 0.3, 0.5, 0.7, 0.9)
ROOT = Path(__file__).resolve().parent
FROZEN_DIR = ROOT.parent.parent / "results" / "paper_reproduction"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_memory_inputs() -> dict[str, list[dict]]:
    gpt_output = Path("outputs/llm/llm_memories.jsonl")
    gpt_path = gpt_output if gpt_output.exists() else FROZEN_DIR / "llm_memories_gpt.jsonl"
    glm_outputs = [
        Path("outputs/llm-glm-spanish/llm_memories.jsonl"),
        Path("outputs/llm-glm-french/llm_memories.jsonl"),
        Path("outputs/llm-glm-russian/llm_memories.jsonl"),
        Path("outputs/llm-glm-german/llm_memories.jsonl"),
    ]
    if all(path.exists() for path in glm_outputs):
        glm_rows = [row for path in glm_outputs for row in load_jsonl(path)]
    else:
        glm_rows = [
            row
            for language in ("spanish", "french", "russian", "german")
            for row in load_jsonl(FROZEN_DIR / f"llm_memories_glm_{language}.jsonl")
        ]
    return {
        "GPT-OSS-120B": load_jsonl(gpt_path),
        "GLM-5.3-Flash": glm_rows,
    }


def load_sensitivity_rows(model_slug: str, visits: int) -> list[dict]:
    full = Path(f"outputs/history-{model_slug}-{visits}/llm_memories.jsonl")
    if full.exists():
        return load_jsonl(full)
    split = sorted(Path("outputs").glob(f"history-{model_slug}-{visits}-*/llm_memories.jsonl"))
    if split:
        return [row for path in split for row in load_jsonl(path)]
    checked_in = sorted(FROZEN_DIR.glob(f"history_memories_{model_slug}_{visits}*.jsonl"))
    if checked_in:
        return [row for path in checked_in for row in load_jsonl(path)]
    raise FileNotFoundError(f"no {model_slug} memory inputs for {visits} visits")


def load_history_extraction_inputs(
    base: dict[str, list[dict]],
) -> dict[int, dict[str, list[dict]]]:
    return {
        4: {
            "GPT-OSS-120B": load_sensitivity_rows("gpt", 4),
            "GLM-5.3-Flash": load_sensitivity_rows("glm", 4),
        },
        8: base,
        16: {
            "GPT-OSS-120B": load_sensitivity_rows("gpt", 16),
            "GLM-5.3-Flash": load_sensitivity_rows("glm", 16),
        },
    }


def parse_rules(row: dict) -> set[Predicate]:
    return {Predicate(rule["field"], rule["value"]) for rule in row["rules"]}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if total == 0:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan}
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {"mean": proportion, "lo": center - half_width, "hi": center + half_width}


def evaluate(
    pair: PairCatalogs,
    *,
    user_id: str,
    rules: set[Predicate],
    tie_rules: set[Predicate] | None = None,
    policy: str,
    catalog_name: str,
    schedule: list,
    budget: int,
    seed: int,
) -> dict[str, float]:
    catalog = pair.learning if catalog_name == "c0" else pair.shifted
    memory = Memory(established=set(rules), tie=set(tie_rules or set()))
    successes: list[int] = []
    proposals = 0
    rejections = 0
    for visit, request in enumerate(schedule):
        result = _propose_sequence(
            policy,
            catalog,
            request,
            memory,
            language=pair.language,
            budget=budget,
            seed=seed,
            user_id=user_id,
            visit=visit,
            branch=catalog_name,
        )
        successes.append(result.success)
        proposals += result.proposals
        rejections += result.rejected
    return {
        "success": float(np.mean(successes)),
        "proposals": float(proposals),
        "rejections": float(rejections),
        "excluded_share": _excluded_share(catalog, memory, pair.language, policy),
        "established_recall": float(Predicate("languages", pair.language) in memory.established),
        "unresolved": float(len(memory.tie)),
        "proxy_retained": float(
            Predicate("countries", pair.country) in (memory.established | memory.tie)
        ),
    }


def aggregate(rows: list[dict], *, seed: int, label: str, reps: int) -> dict:
    result: dict[str, dict | int] = {"users": len(rows)}
    for metric in (
        "success", "proposals", "rejections", "excluded_share",
        "established_recall", "unresolved", "proxy_retained",
    ):
        values = {row["key"]: row[metric] for row in rows}
        result[metric] = _bootstrap_user_mean(
            values,
            reps=reps,
            seed=_stable_int(seed, label, metric),
        )
    return result


def paired_contrast(
    left: list[dict],
    right: list[dict],
    *,
    metric: str,
    seed: int,
    label: str,
    reps: int,
) -> dict[str, float]:
    left_values = {row["key"]: row[metric] for row in left}
    right_values = {row["key"]: row[metric] for row in right}
    if left_values.keys() != right_values.keys():
        raise ValueError(f"unpaired contrast for {label}")
    differences = {key: left_values[key] - right_values[key] for key in left_values}
    return _bootstrap_user_mean(differences, reps=reps, seed=_stable_int(seed, label, metric))


def evidence_profile(pair: PairCatalogs, user_id: str, *, visits: int, budget: int, seed: int) -> dict[str, bool]:
    seed_event, history = _logging_history(pair, user_id, visits=visits, budget=budget, seed=seed)
    accepted = [seed_event] + [
        event
        for visit in history
        for event in visit["events"]
        if event["verdict"] == "accept"
    ]
    accepted_counterexample = any(
        pair.language in event["languages"] and pair.country not in event["countries"]
        for event in accepted
    )
    rejected_counterexample = any(
        event["verdict"] == "reject"
        and pair.country in event["countries"]
        and pair.language not in event["languages"]
        for visit in history
        for event in visit["events"]
    )
    return {
        "accepted_counterexample": accepted_counterexample,
        "rejected_counterexample": rejected_counterexample,
    }


def summarize_identifiability(
    memories: list[dict],
    pairs: dict[str, PairCatalogs],
    *,
    visits: int,
    budget: int,
    seed: int,
) -> dict:
    evidence: dict[tuple[str, str], dict[str, bool]] = {}
    for row in memories:
        key = (row["pair"], row["user_id"])
        if key not in evidence:
            evidence[key] = evidence_profile(
                pairs[row["pair"]], row["user_id"], visits=visits, budget=budget, seed=seed,
            )

    output: dict[str, dict] = {}
    for method in METHODS:
        selected = [row for row in memories if row["method"] == method]
        strata: dict[str, dict] = {}
        categories = {
            # An accepted target-language/non-country item proves that country
            # is not required. A country/non-language rejection only rules out
            # country-only; it remains consistent with language AND country.
            "ambiguous_for_country_rule": lambda profile: not profile["accepted_counterexample"],
            "rejection_only_partial_evidence": lambda profile: (
                profile["rejected_counterexample"] and not profile["accepted_counterexample"]
            ),
            "accepted_counterexample": lambda profile: profile["accepted_counterexample"],
            "no_xor_evidence": lambda profile: (
                not profile["accepted_counterexample"] and not profile["rejected_counterexample"]
            ),
        }
        for name, condition in categories.items():
            rows = [
                row for row in selected
                if condition(evidence[(row["pair"], row["user_id"])])
            ]
            commits = sum(bool(row["proxy_committed"]) for row in rows)
            strata[name] = {
                "users": len(rows),
                "proxy_commits": commits,
                "proxy_commit_rate": float(np.mean([row["proxy_committed"] for row in rows])) if rows else None,
                "proxy_commit_wilson95": wilson_interval(commits, len(rows)) if rows else None,
            }

        states = {"both": 0, "true_only": 0, "proxy_only": 0, "neither": 0}
        pair_rates: dict[str, dict] = {}
        for row in selected:
            if row["true_committed"] and row["proxy_committed"]:
                states["both"] += 1
            elif row["true_committed"]:
                states["true_only"] += 1
            elif row["proxy_committed"]:
                states["proxy_only"] += 1
            else:
                states["neither"] += 1
        for pair_name in sorted(pairs):
            pair_rows = [row for row in selected if row["pair"] == pair_name]
            pair_rates[pair_name] = {
                "users": len(pair_rows),
                "true_commit_rate": float(np.mean([row["true_committed"] for row in pair_rows])),
                "proxy_commit_rate": float(np.mean([row["proxy_committed"] for row in pair_rows])),
                "proxy_commit_wilson95": wilson_interval(
                    sum(bool(row["proxy_committed"]) for row in pair_rows), len(pair_rows),
                ),
            }
        output[method] = {"strata": strata, "memory_states": states, "pairs": pair_rates}
    return output


def summarize_history_sensitivity(
    pairs: dict[str, PairCatalogs],
    *,
    budget: int,
    seed: int,
) -> list[dict]:
    points: list[dict] = []
    for visits in (4, 8, 16):
        profiles = [
            evidence_profile(
                pair, f"llm-{pair.language[:3].lower()}-{index:03d}",
                visits=visits, budget=budget, seed=seed,
            )
            for pair in pairs.values()
            for index in range(20)
        ]
        identifiable = sum(profile["accepted_counterexample"] for profile in profiles)
        rejection_only = sum(
            profile["rejected_counterexample"] and not profile["accepted_counterexample"]
            for profile in profiles
        )
        no_xor = sum(
            not profile["accepted_counterexample"] and not profile["rejected_counterexample"]
            for profile in profiles
        )
        ambiguous = len(profiles) - identifiable
        points.append({
            "visits": visits,
            "users": len(profiles),
            "accepted_counterexample": identifiable,
            "rejection_only_partial_evidence": rejection_only,
            "no_xor_evidence": no_xor,
            "ambiguous_for_country_rule": ambiguous,
            "ambiguous_share": ambiguous / len(profiles),
        })
    return points


def summarize_extraction_sensitivity(
    inputs: dict[int, dict[str, list[dict]]],
    pairs: dict[str, PairCatalogs],
    *,
    budget: int,
    seed: int,
) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for visits, models in sorted(inputs.items()):
        output[str(visits)] = {}
        for model, memories in models.items():
            output[str(visits)][model] = summarize_identifiability(
                memories, pairs, visits=visits, budget=budget, seed=seed,
            )
    return output


def memory_state(row: dict) -> str:
    if row["true_committed"] and row["proxy_committed"]:
        return "both"
    if row["true_committed"]:
        return "true_only"
    if row["proxy_committed"]:
        return "proxy_only"
    return "neither"


def summarize_memory_state_outcomes(
    memories: list[dict],
    pairs: dict[str, PairCatalogs],
    *,
    visits: int,
    budget: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    output: dict[str, dict] = {}
    for method in METHODS:
        output[method] = {}
        method_rows = [row for row in memories if row["method"] == method]
        for state in ("both", "true_only", "proxy_only", "neither"):
            selected = [row for row in method_rows if memory_state(row) == state]
            if not selected:
                continue
            state_output: dict[str, object] = {
                "users": len(selected),
                "pair_counts": {
                    pair_name: sum(row["pair"] == pair_name for row in selected)
                    for pair_name in sorted(pairs)
                },
            }
            for branch in ("c0", "shift"):
                evaluated: list[dict] = []
                no_memory: list[dict] = []
                for row in selected:
                    pair = pairs[row["pair"]]
                    schedule = _request_schedule(
                        pair, user_id=row["user_id"] + "-eval", visits=visits, seed=seed,
                    )
                    metrics = evaluate(
                        pair,
                        user_id=row["user_id"],
                        rules=parse_rules(row),
                        policy="frozen",
                        catalog_name=branch,
                        schedule=schedule,
                        budget=budget,
                        seed=seed,
                    )
                    metrics["key"] = row["pair"] + "/" + row["user_id"]
                    evaluated.append(metrics)
                    baseline = evaluate(
                        pair,
                        user_id=row["user_id"],
                        rules=set(),
                        policy="frozen",
                        catalog_name=branch,
                        schedule=schedule,
                        budget=budget,
                        seed=seed,
                    )
                    baseline["key"] = metrics["key"]
                    no_memory.append(baseline)
                state_output[branch] = {
                    "actual": aggregate(
                        evaluated,
                        seed=seed,
                        label=f"state-{method}-{state}-{branch}",
                        reps=bootstrap_replicates,
                    ),
                    "no_memory": aggregate(
                        no_memory,
                        seed=seed,
                        label=f"state-{method}-{state}-{branch}-none",
                        reps=bootstrap_replicates,
                    ),
                    "actual_minus_no_memory_success": paired_contrast(
                        evaluated,
                        no_memory,
                        metric="success",
                        seed=seed,
                        label=f"state-{method}-{state}-{branch}-difference",
                        reps=bootstrap_replicates,
                    ),
                }
            output[method][state] = state_output
    return output


def summarize_relax_baseline(
    memories: list[dict],
    pairs: dict[str, PairCatalogs],
    *,
    visits: int,
    budget: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    output: dict[str, dict] = {}
    for method in METHODS:
        proxy_rows = [row for row in memories if row["method"] == method and row["proxy_committed"]]
        output[method] = {}
        for branch in ("c0", "shift"):
            variants: dict[str, list[dict]] = defaultdict(list)
            for row in proxy_rows:
                pair = pairs[row["pair"]]
                rules = parse_rules(row)
                proxy = Predicate("countries", pair.country)
                true_rule = Predicate("languages", pair.language)
                schedule = _request_schedule(pair, user_id=row["user_id"] + "-eval", visits=visits, seed=seed)
                specifications = (
                    ("strict", rules, "frozen"),
                    ("soft_rank", rules, "soft_rank"),
                    ("relax", rules, "frozen_relax"),
                    ("relax_delete", rules, "relax_delete"),
                    ("clarify_once", {true_rule}, "frozen"),
                    ("proxy_removed", rules - {proxy}, "frozen"),
                    ("no_memory", set(), "frozen"),
                )
                for name, variant_rules, policy in specifications:
                    metrics = evaluate(
                        pair,
                        user_id=row["user_id"],
                        rules=variant_rules,
                        policy=policy,
                        catalog_name=branch,
                        schedule=schedule,
                        budget=budget,
                        seed=seed,
                    )
                    metrics["key"] = row["pair"] + "/" + row["user_id"]
                    variants[name].append(metrics)
            branch_output = {
                name: aggregate(
                    rows,
                    seed=seed,
                    label=f"{method}-{branch}-{name}",
                    reps=bootstrap_replicates,
                )
                for name, rows in variants.items()
            }
            branch_output["contrasts"] = {
                "strict_minus_no_memory_success": paired_contrast(
                    variants["strict"], variants["no_memory"], metric="success",
                    seed=seed, label=f"{method}-{branch}-strict-minus-none",
                    reps=bootstrap_replicates,
                ),
                "relax_minus_strict_success": paired_contrast(
                    variants["relax"], variants["strict"], metric="success",
                    seed=seed, label=f"{method}-{branch}-relax-minus-strict",
                    reps=bootstrap_replicates,
                ),
                "soft_rank_minus_strict_success": paired_contrast(
                    variants["soft_rank"], variants["strict"], metric="success",
                    seed=seed, label=f"{method}-{branch}-soft-minus-strict",
                    reps=bootstrap_replicates,
                ),
                "relax_delete_minus_relax_success": paired_contrast(
                    variants["relax_delete"], variants["relax"], metric="success",
                    seed=seed, label=f"{method}-{branch}-relax-delete-minus-relax",
                    reps=bootstrap_replicates,
                ),
            }
            output[method][branch] = branch_output
    return output


def tie_memory_from_logged_history(
    pair: PairCatalogs,
    user_id: str,
    *,
    visits: int,
    budget: int,
    seed: int,
) -> Memory:
    _, history = _logging_history(pair, user_id, visits=visits, budget=budget, seed=seed)
    memory = Memory(tie={
        Predicate("languages", pair.language),
        Predicate("countries", pair.country),
    })
    for visit in history:
        for event in visit["events"]:
            def event_holds(predicate: Predicate) -> bool:
                return predicate.value in event[predicate.field]

            if event["verdict"] == "accept":
                memory.tie = {predicate for predicate in memory.tie if event_holds(predicate)}
                continue
            if not all(event_holds(predicate) for predicate in memory.established):
                continue
            violated = {predicate for predicate in memory.tie if not event_holds(predicate)}
            if len(violated) == 1:
                established = next(iter(violated))
                memory.tie.remove(established)
                memory.established.add(established)
    return memory


def relax_delete_memory_from_logged_history(
    pair: PairCatalogs,
    user_id: str,
    *,
    visits: int,
    budget: int,
    seed: int,
) -> Memory:
    _, history = _logging_history(pair, user_id, visits=visits, budget=budget, seed=seed)
    memory = Memory(established={
        Predicate("languages", pair.language),
        Predicate("countries", pair.country),
    })
    for visit in history:
        for event in visit["events"]:
            if event["verdict"] == "accept":
                memory.established = {
                    predicate
                    for predicate in memory.established
                    if predicate.value in event[predicate.field]
                }
    return memory


def summarize_same_history_tie(
    pairs: dict[str, PairCatalogs],
    *,
    learning_visits: int,
    continuation_visits: int,
    budget: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    rows_by_branch: dict[str, list[dict]] = defaultdict(list)
    final_states: dict[str, list[dict]] = defaultdict(list)
    for pair_name, pair in pairs.items():
        prefix = pair.language[:3].lower()
        for index in range(20):
            user_id = f"llm-{prefix}-{index:03d}"
            checkpoint = tie_memory_from_logged_history(
                pair,
                user_id,
                visits=learning_visits,
                budget=budget,
                seed=seed,
            )
            schedule = _request_schedule(
                pair,
                user_id=user_id + "-eval",
                visits=continuation_visits,
                seed=seed,
            )
            for branch in ("c0", "shift"):
                metrics = evaluate(
                    pair,
                    user_id=user_id,
                    rules=checkpoint.established,
                    tie_rules=checkpoint.tie,
                    policy="tie",
                    catalog_name=branch,
                    schedule=schedule,
                    budget=budget,
                    seed=seed,
                )
                metrics["key"] = pair_name + "/" + user_id
                rows_by_branch[branch].append(metrics)
                final_states[branch].append({
                    "established_recall": metrics["established_recall"],
                    "unresolved": metrics["unresolved"],
                })
    output: dict[str, dict] = {}
    for branch, rows in rows_by_branch.items():
        output[branch] = aggregate(
            rows,
            seed=seed,
            label=f"same-history-tie-{branch}",
            reps=bootstrap_replicates,
        )
        output[branch]["established_recall"] = float(np.mean([
            row["established_recall"] for row in final_states[branch]
        ]))
        output[branch]["unresolved"] = float(np.mean([
            row["unresolved"] for row in final_states[branch]
        ]))
    return output


def summarize_same_history_relax_delete(
    pairs: dict[str, PairCatalogs],
    *,
    learning_visits: int,
    continuation_visits: int,
    budget: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    rows_by_branch: dict[str, list[dict]] = defaultdict(list)
    for pair_name, pair in pairs.items():
        prefix = pair.language[:3].lower()
        for index in range(20):
            user_id = f"llm-{prefix}-{index:03d}"
            checkpoint = relax_delete_memory_from_logged_history(
                pair, user_id, visits=learning_visits, budget=budget, seed=seed,
            )
            schedule = _request_schedule(
                pair, user_id=user_id + "-eval", visits=continuation_visits, seed=seed,
            )
            for branch in ("c0", "shift"):
                metrics = evaluate(
                    pair,
                    user_id=user_id,
                    rules=checkpoint.established,
                    policy="relax_delete",
                    catalog_name=branch,
                    schedule=schedule,
                    budget=budget,
                    seed=seed,
                )
                metrics["key"] = pair_name + "/" + user_id
                rows_by_branch[branch].append(metrics)
    return {
        branch: aggregate(
            rows,
            seed=seed,
            label=f"same-history-relax-delete-{branch}",
            reps=bootstrap_replicates,
        )
        for branch, rows in rows_by_branch.items()
    }


def common_request_pairs(items: list, config: dict) -> dict[float, dict[str, PairCatalogs]]:
    by_coverage: dict[float, dict[str, PairCatalogs]] = {}
    for coverage in COVERAGES:
        by_coverage[coverage] = {}
        for spec in config["pairs"]:
            pair = build_pair_catalogs(
                items,
                language=spec["language"],
                country=spec["country"],
                target_count=spec["target_count"],
                catalog_size=config["catalog_size"],
                seed=config["seed"],
                shifted_coverage=coverage,
            )
            by_coverage[coverage][f"{pair.language}/{pair.country}"] = pair

    for pair_name in by_coverage[COVERAGES[0]]:
        common = set.intersection(*[
            {request.key for request in by_coverage[coverage][pair_name].requests}
            for coverage in COVERAGES
        ])
        if len(common) < 4:
            raise ValueError(f"only {len(common)} common requests for {pair_name}")
        reference = by_coverage[COVERAGES[0]][pair_name]
        ordered = tuple(request for request in reference.requests if request.key in common)
        for coverage in COVERAGES:
            pair = by_coverage[coverage][pair_name]
            by_coverage[coverage][pair_name] = replace(pair, requests=ordered)
    return by_coverage


def summarize_coverage_scan(
    memories: list[dict],
    pairs_by_coverage: dict[float, dict[str, PairCatalogs]],
    *,
    visits: int,
    budget: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict:
    output: dict[str, list[dict]] = {}
    for method in METHODS:
        both_rows = [
            row for row in memories
            if row["method"] == method and row["true_committed"] and row["proxy_committed"]
        ]
        points: list[dict] = []
        for coverage in COVERAGES:
            variants: dict[str, list[dict]] = defaultdict(list)
            analytic: dict[str, float] = {}
            for row in both_rows:
                pair = pairs_by_coverage[coverage][row["pair"]]
                rules = parse_rules(row)
                proxy = Predicate("countries", pair.country)
                rng = random.Random(_stable_int(seed, row["user_id"] + "-coverage", "requests"))
                schedule = [rng.choice(pair.requests) for _ in range(visits)]
                key = row["pair"] + "/" + row["user_id"]
                analytic[key] = float(np.mean([
                    any(request.holds(item) and all(predicate.holds(item) for predicate in rules)
                        for item in pair.shifted)
                    for request in schedule
                ]))
                specifications = (
                    ("strict", rules, "frozen"),
                    ("relax", rules, "frozen_relax"),
                    ("relax_delete", rules, "relax_delete"),
                    ("proxy_removed", rules - {proxy}, "frozen"),
                )
                for name, variant_rules, policy in specifications:
                    metrics = evaluate(
                        pair,
                        user_id=row["user_id"] + "-coverage",
                        rules=variant_rules,
                        policy=policy,
                        catalog_name="shift",
                        schedule=schedule,
                        budget=budget,
                        seed=seed,
                    )
                    metrics["key"] = key
                    variants[name].append(metrics)
            strict = {row["key"]: row["success"] for row in variants["strict"]}
            max_error = max(abs(strict[key] - analytic[key]) for key in analytic) if analytic else 0.0
            points.append({
                "coverage": coverage,
                "users": len(both_rows),
                "analytic_strict_success": _bootstrap_user_mean(
                    analytic,
                    reps=bootstrap_replicates,
                    seed=_stable_int(seed, method, coverage, "analytic"),
                ),
                "simulation_analytic_max_abs_error": max_error,
                **{
                    name: aggregate(
                        rows,
                        seed=seed,
                        label=f"coverage-{method}-{coverage}-{name}",
                        reps=bootstrap_replicates,
                    )
                    for name, rows in variants.items()
                },
            })
        output[method] = points
    return output


def plot_coverage(points: list[dict], output_dir: Path) -> None:
    mpl.rcParams.update({
        "font.size": 7,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
    })
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    x = np.asarray([100 * point["coverage"] for point in points])
    analytic = np.asarray([100 * point["analytic_strict_success"]["mean"] for point in points])
    strict = np.asarray([100 * point["strict"]["success"]["mean"] for point in points])
    relaxed = np.asarray([100 * point["relax"]["success"]["mean"] for point in points])
    removed = np.asarray([100 * point["proxy_removed"]["success"]["mean"] for point in points])
    ax.plot(x, analytic, color="#D55E00", linewidth=1.2, label="Strict: availability prediction")
    ax.scatter(x, strict, color="#D55E00", marker="x", s=18, linewidths=1.0, label="Strict: simulation")
    ax.plot(x, relaxed, color="#0072B2", marker="o", markersize=2.8, linewidth=1.2, label="Relax on failure")
    ax.plot(x, removed, color="#777777", linestyle="--", linewidth=1.1, label="Proxy removed")
    ax.set_xlim(7, 93)
    ax.set_ylim(0, 103)
    ax.set_xticks([10, 30, 50, 70, 90])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Joint language--country coverage (%)")
    ax.set_ylabel("Shift success (%)")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(loc="lower right", frameon=False, fontsize=5.7)
    fig.tight_layout(pad=0.35)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "coverage_scan.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "coverage_scan.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_audit_summary(result: dict, output_dir: Path) -> None:
    mpl.rcParams.update({
        "font.size": 7,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(
        2, 1, figsize=(3.3, 2.85),
        gridspec_kw={"height_ratios": [1.45, 1.0]},
    )
    ax = axes[0]
    model_methods = [
        ("GPT-OSS-120B", "plain_notes", "GPT-D"),
        ("GPT-OSS-120B", "count_gated", "GPT-C"),
        ("GPT-OSS-120B", "judge_gated", "GPT-J"),
        ("GLM-5.3-Flash", "plain_notes", "GLM-D"),
        ("GLM-5.3-Flash", "count_gated", "GLM-C"),
        ("GLM-5.3-Flash", "judge_gated", "GLM-J"),
    ]
    x = np.asarray([0, 1, 2, 3.6, 4.6, 5.6])
    strata = (
        ("ambiguous_for_country_rule", "Ambiguous ($n=63$)", "#D55E00", "o", -0.10),
        ("accepted_counterexample", "Identifiable ($n=17$)", "#0072B2", "s", 0.10),
    )
    for stratum, label, color, marker, offset in strata:
        intervals = [
            result["models"][model]["identifiability"][method]["strata"][stratum]["proxy_commit_wilson95"]
            for model, method, _ in model_methods
        ]
        means = np.asarray([100 * interval["mean"] for interval in intervals])
        lower = means - np.asarray([100 * interval["lo"] for interval in intervals])
        upper = np.asarray([100 * interval["hi"] for interval in intervals]) - means
        ax.errorbar(
            x + offset, means, yerr=np.vstack([lower, upper]),
            fmt=marker, color=color, markersize=3.2, linewidth=0.9,
            capsize=1.8, label=label,
        )
    ax.axvline(2.8, color="#bbbbbb", linewidth=0.6)
    ax.set_xticks(x, [label for _, _, label in model_methods])
    ax.set_ylim(-4, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Proxy commitment (%)")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(loc="center", bbox_to_anchor=(0.50, 0.55), frameon=False, fontsize=5.8)
    ax.text(0.0, 1.05, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    sensitivity = result["history_sensitivity"]
    visits = np.asarray([point["visits"] for point in sensitivity])
    ambiguous = np.asarray([100 * point["ambiguous_share"] for point in sensitivity])
    ax.plot(
        visits, ambiguous, color="#009E73", marker="o", markersize=3.2,
        linewidth=1.2, linestyle="--", label="Ambiguous share",
    )
    for xv, yv, point in zip(visits, ambiguous, sensitivity):
        ax.annotate(
            f"{point['ambiguous_for_country_rule']}/80",
            (xv, yv), xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=5.8,
        )
    for model, color, label in (
        ("GPT-OSS-120B", "#D55E00", "GPT commitment"),
        ("GLM-5.3-Flash", "#0072B2", "GLM commitment"),
    ):
        direct: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        for visit_count in visits:
            conditions = result["extraction_sensitivity"][str(int(visit_count))][model]
            rates = []
            for method in METHODS:
                states = conditions[method]["memory_states"]
                rates.append(100 * (states["both"] + states["proxy_only"]) / 80)
            direct.append(rates[0])
            lower.append(min(rates))
            upper.append(max(rates))
        ax.fill_between(visits, lower, upper, color=color, alpha=0.14, linewidth=0)
        ax.plot(visits, direct, color=color, marker="s", markersize=2.8, linewidth=1.1, label=label)
    ax.set_xlim(3, 18.2)
    ax.set_ylim(0, 100)
    ax.set_xticks([4, 8, 16])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Exploratory visits")
    ax.set_ylabel("Share / commitment (%)")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.text(16.35, 64, "Ambiguous", color="#009E73", fontsize=5.4, va="center")
    ax.text(16.35, 51, "GPT", color="#D55E00", fontsize=5.4, va="center")
    ax.text(16.35, 28, "GLM", color="#0072B2", fontsize=5.4, va="center")
    ax.text(0.0, 1.05, "(b)", transform=ax.transAxes, fontweight="bold")

    fig.tight_layout(pad=0.35, h_pad=0.55)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "audit_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "audit_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global FROZEN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/primary.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/robustness/robustness_summary.json"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/paper_assets"))
    parser.add_argument("--frozen-dir", type=Path, default=FROZEN_DIR)
    args = parser.parse_args()
    FROZEN_DIR = args.frozen_dir

    config = json.loads(args.config.read_text())
    items = load_items(args.catalog)
    base_pairs_list = [
        build_pair_catalogs(
            items,
            language=spec["language"],
            country=spec["country"],
            target_count=spec["target_count"],
            catalog_size=config["catalog_size"],
            seed=config["seed"],
        )
        for spec in config["pairs"]
    ]
    base_pairs = {f"{pair.language}/{pair.country}": pair for pair in base_pairs_list}
    memories_by_model = load_memory_inputs()
    history_extraction_inputs = load_history_extraction_inputs(memories_by_model)
    coverage_pairs = common_request_pairs(items, config)
    result: dict[str, dict] = {
        "design": {
            "users_per_model": 80,
            "learning_visits": config["learning_visits"],
            "continuation_visits": config["continuation_visits"],
            "proposal_budget": config["proposal_budget"],
            "coverage_levels": list(COVERAGES),
            "coverage_scan_population": "users committing both the true language rule and the country proxy",
        },
        "history_sensitivity": summarize_history_sensitivity(
            base_pairs,
            budget=config["proposal_budget"],
            seed=config["seed"],
        ),
        "extraction_sensitivity": summarize_extraction_sensitivity(
            history_extraction_inputs,
            base_pairs,
            budget=config["proposal_budget"],
            seed=config["seed"],
        ),
        "same_history_tie": summarize_same_history_tie(
            base_pairs,
            learning_visits=config["learning_visits"],
            continuation_visits=config["continuation_visits"],
            budget=config["proposal_budget"],
            bootstrap_replicates=config["bootstrap_replicates"],
            seed=config["seed"],
        ),
        "same_history_relax_delete": summarize_same_history_relax_delete(
            base_pairs,
            learning_visits=config["learning_visits"],
            continuation_visits=config["continuation_visits"],
            budget=config["proposal_budget"],
            bootstrap_replicates=config["bootstrap_replicates"],
            seed=config["seed"],
        ),
        "models": {},
    }
    for model, memories in memories_by_model.items():
        result["models"][model] = {
            "identifiability": summarize_identifiability(
                memories,
                base_pairs,
                visits=config["learning_visits"],
                budget=config["proposal_budget"],
                seed=config["seed"],
            ),
            "relax_baseline": summarize_relax_baseline(
                memories,
                base_pairs,
                visits=config["continuation_visits"],
                budget=config["proposal_budget"],
                bootstrap_replicates=config["bootstrap_replicates"],
                seed=config["seed"],
            ),
            "memory_state_outcomes": summarize_memory_state_outcomes(
                memories,
                base_pairs,
                visits=config["continuation_visits"],
                budget=config["proposal_budget"],
                bootstrap_replicates=config["bootstrap_replicates"],
                seed=config["seed"],
            ),
            "coverage_scan": summarize_coverage_scan(
                memories,
                coverage_pairs,
                visits=config["continuation_visits"],
                budget=config["proposal_budget"],
                bootstrap_replicates=config["bootstrap_replicates"],
                seed=config["seed"],
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot_coverage(result["models"]["GPT-OSS-120B"]["coverage_scan"]["plain_notes"], args.figure_dir)
    plot_audit_summary(result, args.figure_dir)
    print(args.output)
    print(args.figure_dir / "coverage_scan.pdf")
    print(args.figure_dir / "audit_summary.pdf")


if __name__ == "__main__":
    main()

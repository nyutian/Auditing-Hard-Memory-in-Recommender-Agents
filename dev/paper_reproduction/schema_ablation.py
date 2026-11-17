from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from catalog_proxy.experiment import Predicate, build_pair_catalogs, load_items
from catalog_proxy.llm_experiment import (
    CachedBaseten,
    _history_text,
    _logging_history,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float]:
    if total == 0:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan}
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return {"mean": proportion, "lo": center - half, "hi": center + half}


def summarize(rows: list[dict], baseline_rows: list[dict]) -> dict:
    row_keys = {(row["pair"], row["user_id"]) for row in rows}
    direct = {
        (row["pair"], row["user_id"]): row
        for row in baseline_rows
        if row["method"] == "plain_notes" and (row["pair"], row["user_id"]) in row_keys
    }
    if len(direct) != len(rows):
        raise ValueError(f"baseline has {len(direct)} direct rows, expected {len(rows)}")

    conditions: dict[str, dict] = {}
    definitions = {
        "hard_schema_committed": lambda row: bool(direct[(row["pair"], row["user_id"])]["proxy_committed"]),
        "calibrated_established": lambda row: bool(row["proxy_established"]),
        "calibrated_plausible": lambda row: bool(row["proxy_plausible"]),
        "calibrated_mentioned": lambda row: bool(row["proxy_established"] or row["proxy_plausible"]),
    }
    for condition, predicate in definitions.items():
        selected = [row for row in rows if predicate(row)]
        conditions[condition] = {
            "users": len(rows),
            "proxy_users": len(selected),
            "proxy_rate_wilson95": wilson(len(selected), len(rows)),
        }

    strata: dict[str, dict] = {}
    for stratum, match in {
        "ambiguous": lambda row: not row["accepted_counterexample"],
        "identifiable": lambda row: row["accepted_counterexample"],
    }.items():
        subset = [row for row in rows if match(row)]
        strata[stratum] = {"users": len(subset)}
        for condition, predicate in definitions.items():
            count = sum(predicate(row) for row in subset)
            strata[stratum][condition] = {
                "proxy_users": count,
                "proxy_rate_wilson95": wilson(count, len(subset)),
            }

    return {
        "conditions": conditions,
        "strata": strata,
        "true_established_rate": sum(row["true_established"] for row in rows) / len(rows),
        "true_plausible_rate": sum(row["true_plausible"] for row in rows) / len(rows),
        "other_established_rules": sum(len(row["other_established"]) for row in rows),
        "other_plausible_rules": sum(len(row["other_plausible"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--language", action="append", dest="languages")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    items = load_items(args.catalog)
    selected_specs = [
        spec for spec in config["pairs"]
        if not args.languages or spec["language"] in set(args.languages)
    ]
    pairs = [
        build_pair_catalogs(
            items,
            language=spec["language"], country=spec["country"],
            target_count=spec["target_count"], catalog_size=config["catalog_size"],
            seed=config["seed"],
        )
        for spec in selected_specs
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    client = CachedBaseten(
        args.output / "baseten_calls.jsonl",
        model=config["llm_model"], temperature=config["llm_temperature"],
    )
    rows: list[dict] = []
    for pair in pairs:
        true_rule = Predicate("languages", pair.language)
        proxy_rule = Predicate("countries", pair.country)
        for index in range(config["llm_users_per_pair"]):
            user_id = f"llm-{pair.language[:3].lower()}-{index:03d}"
            seed_event, history = _logging_history(
                pair, user_id, visits=config["learning_visits"],
                budget=config["proposal_budget"], seed=config["seed"],
            )
            established, plausible = client.calibrated_notes(
                _history_text(seed_event, history), session_id=user_id + "-calibrated",
            )
            accepted_counterexample = any(
                event["verdict"] == "accept"
                and pair.language in event["languages"]
                and pair.country not in event["countries"]
                for visit in history for event in visit["events"]
            )
            row = {
                "user_id": user_id,
                "pair": f"{pair.language}/{pair.country}",
                "accepted_counterexample": accepted_counterexample,
                "true_established": true_rule in established,
                "true_plausible": true_rule in plausible,
                "proxy_established": proxy_rule in established,
                "proxy_plausible": proxy_rule in plausible,
                "established": [p.__dict__ for p in sorted(established)],
                "plausible": [p.__dict__ for p in sorted(plausible)],
                "other_established": [
                    p.__dict__ for p in sorted(established - {true_rule, proxy_rule})
                ],
                "other_plausible": [
                    p.__dict__ for p in sorted(plausible - {true_rule, proxy_rule})
                ],
            }
            rows.append(row)
            print(
                f"completed {user_id}: established={len(established)}, "
                f"plausible={len(plausible)}", flush=True,
            )

    with (args.output / "schema_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    baseline_rows = [row for path in args.baseline for row in load_jsonl(path)]
    result = {
        "model": config["llm_model"],
        "temperature": config["llm_temperature"],
        "learning_visits": config["learning_visits"],
        "summary": summarize(rows, baseline_rows),
    }
    (args.output / "schema_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

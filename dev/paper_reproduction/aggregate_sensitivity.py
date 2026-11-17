from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from catalog_proxy.experiment import _bootstrap_user_mean, _stable_int


METHODS = ("plain_notes", "count_gated", "judge_gated")
BRANCHES = ("c0", "shift")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261114)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    args = parser.parse_args()

    memories = [row for directory in args.inputs for row in load_jsonl(directory / "llm_memories.jsonl")]
    removal_rows = [row for directory in args.inputs for row in load_jsonl(directory / "paired_removal.jsonl")]
    models = {json.loads((directory / "llm_summary.json").read_text())["model"] for directory in args.inputs}
    if len(models) != 1:
        raise ValueError(f"expected one model, found {sorted(models)}")

    occurrence: dict[str, dict] = {}
    pair_occurrence: dict[str, dict[str, float]] = defaultdict(dict)
    for method in METHODS:
        selected = [row for row in memories if row["method"] == method]
        occurrence[method] = {
            "users": len(selected),
            "true_commit_rate": float(np.mean([row["true_committed"] for row in selected])),
            "proxy_commit_rate": float(np.mean([row["proxy_committed"] for row in selected])),
        }
        for pair in sorted({row["pair"] for row in selected}):
            pair_rows = [row for row in selected if row["pair"] == pair]
            pair_occurrence[method][pair] = float(np.mean([row["proxy_committed"] for row in pair_rows]))

    paired_removal: dict[str, dict] = {}
    for method in METHODS:
        paired_removal[method] = {}
        for branch in BRANCHES:
            rows = [row for row in removal_rows if row["method"] == method and row["branch"] == branch]
            differences = {row["user_id"]: row["difference"] for row in rows}
            utilities = {row["user_id"]: row["proxy_only_utility"] for row in rows}
            paired_removal[method][branch] = {
                "proxy_users": len(rows),
                "full_success": float(np.mean([row["full_success"] for row in rows])),
                "removed_success": float(np.mean([row["removed_success"] for row in rows])),
                "difference": float(np.mean([row["difference"] for row in rows])),
                "difference_ci": _bootstrap_user_mean(
                    differences, reps=args.bootstrap_replicates,
                    seed=_stable_int(args.seed, method, branch, "sensitivity-removal"),
                ),
                "proxy_only_utility": float(np.mean([row["proxy_only_utility"] for row in rows])),
                "proxy_only_utility_ci": _bootstrap_user_mean(
                    utilities, reps=args.bootstrap_replicates,
                    seed=_stable_int(args.seed, method, branch, "sensitivity-utility"),
                ),
            }

    result = {
        "model": models.pop(),
        "occurrence": occurrence,
        "pair_occurrence": pair_occurrence,
        "paired_removal": paired_removal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field

from .experiment import (
    Memory,
    PairCatalogs,
    Predicate,
    UserRecord,
    _excluded_share,
    _bootstrap_user_mean,
    _propose_sequence,
    _request_schedule,
    _stable_int,
    build_pair_catalogs,
    load_items,
    summarize,
)


class Rule(BaseModel):
    field: Literal["languages", "countries"]
    value: str


class NoteOutput(BaseModel):
    rules: list[Rule] = Field(description="Persistent hard-requirement rules to store")


class Decision(BaseModel):
    field: Literal["languages", "countries"]
    value: str
    commit: bool


class JudgeOutput(BaseModel):
    decisions: list[Decision]


class CalibratedRule(BaseModel):
    field: Literal["languages", "countries", "genres", "publication_year"]
    value: str
    status: Literal["established", "plausible"]


class CalibratedOutput(BaseModel):
    rules: list[CalibratedRule] = Field(
        description="Persistent requirements separated into established and plausible rules"
    )


SYSTEM_NOTE = """You maintain persistent preference notes for a book recommender.
The same reader returns across visits. Each visit states a genre and publication
period, which are temporary and must never enter persistent memory. The reader
then accepts or rejects proposed books without explaining why. Infer only
persistent hard requirements about language or country. Store a rule when the
observed history consistently supports treating it as a required filter. A
rejection alone does not identify which field failed. Return only the rules."""


SYSTEM_JUDGE = """You verify proposed persistent hard-requirement rules for a
book recommender. The reader's genre and publication period change by visit and
are not persistent. Accept/reject verdicts are deterministic, but a rejection
does not identify which field failed. For each proposed language or country
rule, decide whether the interaction history establishes that it is required.
Return exactly one decision for every candidate."""


SYSTEM_CALIBRATED = """You maintain persistent preference memory for a book
recommender. The same reader returns across visits. Each visit states a genre
and publication period; those are temporary unless the cross-visit evidence
independently establishes otherwise. The reader then accepts or rejects books
without explaining why. Feedback is deterministic, standing requirements may
be conjunctions, and a rejection does not identify which requirement failed.

For every persistent rule worth retaining, label it ESTABLISHED only when the
complete history logically rules out every allowed conjunctive explanation in
which that rule is not required. Otherwise label it PLAUSIBLE. In particular,
an accepted book supports its attributes but does not by itself make all of
them requirements. A rule is PLAUSIBLE only if it is compatible with every
verdict; omit it as soon as any accepted book violates it. A rejection may
support one rule while leaving another rule possible. Preserve uncertainty
rather than turning a plausible rule into a hard filter. Allowed fields are
language, country, genre, and publication year. Return established and
plausible rules; return an empty list when no persistent rule is supported."""


def _rule_set(rules: Sequence[Rule]) -> set[Predicate]:
    return {Predicate(rule.field, rule.value) for rule in rules}


def _history_text(seed_event: dict, visits: list[dict]) -> str:
    lines = [
        "Earlier accepted book (standing-preference bootstrap):",
        json.dumps(seed_event, ensure_ascii=False, sort_keys=True),
    ]
    for visit in visits:
        lines.append(f"Visit {visit['visit'] + 1}: request={visit['request']}")
        for event in visit["events"]:
            lines.append(json.dumps(event, ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


def _cache_key(kind: str, model: str, temperature: float, messages: list[dict]) -> str:
    raw = json.dumps({"kind": kind, "model": model, "temperature": temperature, "messages": messages}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class CachedBaseten:
    def __init__(self, path: Path, *, model: str, temperature: float) -> None:
        self.path = path
        self.model = model
        self.temperature = temperature
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row
        api_key = os.environ.get("BASETEN_API_KEY")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://inference.baseten.co/v1",
            timeout=90.0,
            max_retries=0,
        ) if api_key else None

    def _call(self, *, kind: str, messages: list[dict], schema: type[BaseModel], session_id: str) -> BaseModel:
        key = _cache_key(kind, self.model, self.temperature, messages)
        if key in self.cache:
            return schema.model_validate(self.cache[key]["parsed"])
        if self.client is None:
            raise RuntimeError("BASETEN_API_KEY is not set and the requested call is not cached")
        error: Exception | None = None
        for attempt in range(5):
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=schema,
                    temperature=self.temperature,
                    max_tokens=4096,
                    extra_headers={"x-session-affinity": session_id},
                )
                message = response.choices[0].message
                if message.refusal:
                    raise RuntimeError(f"model refusal: {message.refusal}")
                if message.parsed is None:
                    raise RuntimeError("no parsed output")
                row = {
                    "key": key,
                    "kind": kind,
                    "model": self.model,
                    "temperature": self.temperature,
                    "messages": messages,
                    "parsed": message.parsed.model_dump(),
                    "usage": response.usage.model_dump() if response.usage else None,
                }
                with self.path.open("a") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                self.cache[key] = row
                return message.parsed
            except Exception as exc:  # network and provider errors are retryable here
                error = exc
                if attempt < 4:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Baseten call failed after retries: {error}")

    def notes(self, history: str, session_id: str) -> set[Predicate]:
        messages = [
            {"role": "system", "content": SYSTEM_NOTE},
            {"role": "user", "content": history},
        ]
        output = self._call(kind="notes", messages=messages, schema=NoteOutput, session_id=session_id)
        assert isinstance(output, NoteOutput)
        return _rule_set(output.rules)

    def judge(self, history: str, candidates: set[Predicate], session_id: str) -> set[Predicate]:
        candidate_text = json.dumps(
            [{"field": p.field, "value": p.value} for p in sorted(candidates)],
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": SYSTEM_JUDGE},
            {"role": "user", "content": history + "\nCandidates: " + candidate_text},
        ]
        output = self._call(kind="judge", messages=messages, schema=JudgeOutput, session_id=session_id)
        assert isinstance(output, JudgeOutput)
        accepted = {Predicate(d.field, d.value) for d in output.decisions if d.commit}
        return accepted & candidates

    def calibrated_notes(
        self, history: str, session_id: str,
    ) -> tuple[set[Predicate], set[Predicate]]:
        messages = [
            {"role": "system", "content": SYSTEM_CALIBRATED},
            {"role": "user", "content": history},
        ]
        output = self._call(
            kind="calibrated-notes", messages=messages,
            schema=CalibratedOutput, session_id=session_id,
        )
        assert isinstance(output, CalibratedOutput)
        established = {
            Predicate(rule.field, rule.value)
            for rule in output.rules if rule.status == "established"
        }
        plausible = {
            Predicate(rule.field, rule.value)
            for rule in output.rules if rule.status == "plausible"
        }
        return established, plausible - established


def count_gate(candidates: set[Predicate], seed_event: dict, visits: list[dict], threshold: int = 2) -> set[Predicate]:
    accepted = [seed_event] + [event for visit in visits for event in visit["events"] if event["verdict"] == "accept"]
    kept: set[Predicate] = set()
    for predicate in candidates:
        values_key = predicate.field
        support = sum(predicate.value in event[values_key] for event in accepted)
        contradicted = any(predicate.value not in event[values_key] for event in accepted)
        if support >= threshold and not contradicted:
            kept.add(predicate)
    return kept


def _seed_event(pair: PairCatalogs, user_id: str, seed: int) -> dict:
    candidates = [
        x for x in pair.learning
        if pair.language in x.languages and pair.country in x.countries
    ]
    item = sorted(candidates, key=lambda x: _stable_int(seed, user_id, "seed", x.id))[0]
    return {
        "item_id": item.id,
        "title": item.title,
        "genres": list(item.genres),
        "languages": list(item.languages),
        "countries": list(item.countries),
        "publication_year": item.publication_year,
        "verdict": "accept",
    }


def _logging_history(pair: PairCatalogs, user_id: str, *, visits: int, budget: int, seed: int) -> tuple[dict, list[dict]]:
    seed_event = _seed_event(pair, user_id, seed)
    schedule = _request_schedule(pair, user_id=user_id + "-logging", visits=visits, seed=seed)
    history: list[dict] = []
    for visit, request in enumerate(schedule):
        result = _propose_sequence(
            "no_memory", pair.learning, request, Memory(),
            language=pair.language, budget=budget, seed=seed,
            user_id=user_id, visit=visit, branch="logging",
        )
        history.append({"visit": visit, "request": request.key, "events": result.events})
    return seed_event, history


def _run_frozen_branch(
    pair: PairCatalogs,
    *,
    user_id: str,
    method: str,
    rules: set[Predicate],
    branch: str,
    visits: int,
    budget: int,
    seed: int,
    policy: str = "frozen",
) -> list[UserRecord]:
    catalog = pair.learning if branch == "c0" else pair.shifted
    schedule = _request_schedule(pair, user_id=user_id + "-eval", visits=visits, seed=seed)
    records: list[UserRecord] = []
    memory = Memory(established=set(rules))
    true_rule = Predicate("languages", pair.language)
    for visit, request in enumerate(schedule):
        result = _propose_sequence(
            policy, catalog, request, memory,
            language=pair.language, budget=budget, seed=seed,
            user_id=user_id, visit=visit, branch=branch,
        )
        records.append(UserRecord(
            user_id=user_id, pair=f"{pair.language}/{pair.country}", method=method,
            branch=branch, phase="frozen", visit=visit, success=result.success,
            proposals=result.proposals, rejected=result.rejected,
            tie_single_rejects=0, tie_multiple_rejects=0, whole_tie_available=0,
            established_recall=float(true_rule in rules), unresolved=0,
            excluded_share=_excluded_share(catalog, memory, pair.language, policy),
        ))
    return records


def run_llm(
    pairs: Sequence[PairCatalogs],
    *,
    client: CachedBaseten,
    users_per_pair: int,
    learning_visits: int,
    continuation_visits: int,
    budget: int,
    bootstrap_replicates: int,
    seed: int,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[UserRecord] = []
    memories: list[dict] = []
    removal_rows: list[dict] = []
    for pair in pairs:
        proxy = Predicate("countries", pair.country)
        true_rule = Predicate("languages", pair.language)
        for index in range(users_per_pair):
            user_id = f"llm-{pair.language[:3].lower()}-{index:03d}"
            seed_event, history = _logging_history(
                pair, user_id, visits=learning_visits, budget=budget, seed=seed,
            )
            text = _history_text(seed_event, history)
            proposals = client.notes(text, session_id=user_id)
            method_rules = {
                "plain_notes": proposals,
                "count_gated": count_gate(proposals, seed_event, history),
                "judge_gated": client.judge(text, proposals, session_id=user_id + "-judge"),
            }
            print(
                f"completed {user_id}: "
                + ", ".join(f"{name}={len(rules)}" for name, rules in method_rules.items()),
                flush=True,
            )
            for method, rules in method_rules.items():
                memories.append({
                    "user_id": user_id,
                    "pair": f"{pair.language}/{pair.country}",
                    "method": method,
                    "rules": [{"field": p.field, "value": p.value} for p in sorted(rules)],
                    "true_committed": true_rule in rules,
                    "proxy_committed": proxy in rules,
                })
                for branch in ("c0", "shift"):
                    full = _run_frozen_branch(
                        pair, user_id=user_id, method=method, rules=rules, branch=branch,
                        visits=continuation_visits, budget=budget, seed=seed,
                    )
                    records.extend(full)
                    if proxy in rules:
                        removed = _run_frozen_branch(
                            pair, user_id=user_id, method=method + "_proxy_removed", rules=rules - {proxy},
                            branch=branch, visits=continuation_visits, budget=budget, seed=seed,
                        )
                        proxy_only = _run_frozen_branch(
                            pair, user_id=user_id, method=method + "_proxy_only", rules={proxy},
                            branch=branch, visits=continuation_visits, budget=budget, seed=seed,
                        )
                        no_memory = _run_frozen_branch(
                            pair, user_id=user_id, method=method + "_no_memory", rules=set(),
                            branch=branch, visits=continuation_visits, budget=budget, seed=seed,
                        )
                        full_success = float(np.mean([x.success for x in full]))
                        removed_success = float(np.mean([x.success for x in removed]))
                        proxy_only_success = float(np.mean([x.success for x in proxy_only]))
                        no_memory_success = float(np.mean([x.success for x in no_memory]))
                        removal_rows.append({
                            "user_id": user_id, "pair": f"{pair.language}/{pair.country}",
                            "method": method, "branch": branch,
                            "full_success": full_success, "removed_success": removed_success,
                            "difference": full_success - removed_success,
                            "proxy_only_success": proxy_only_success,
                            "no_memory_success": no_memory_success,
                            "proxy_only_utility": proxy_only_success - no_memory_success,
                        })

    with (output_dir / "llm_memories.jsonl").open("w") as handle:
        for row in memories:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "llm_records.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row.__dict__, sort_keys=True) + "\n")
    with (output_dir / "paired_removal.jsonl").open("w") as handle:
        for row in removal_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    occurrence: dict[str, dict] = {}
    for method in ("plain_notes", "count_gated", "judge_gated"):
        selected = [x for x in memories if x["method"] == method]
        occurrence[method] = {
            "users": len(selected),
            "true_commit_rate": float(np.mean([x["true_committed"] for x in selected])),
            "proxy_commit_rate": float(np.mean([x["proxy_committed"] for x in selected])),
        }
    removal: dict[str, dict] = {}
    for method in ("plain_notes", "count_gated", "judge_gated"):
        removal[method] = {}
        for branch in ("c0", "shift"):
            rows = [x for x in removal_rows if x["method"] == method and x["branch"] == branch]
            differences = {x["user_id"]: x["difference"] for x in rows}
            proxy_utilities = {x["user_id"]: x["proxy_only_utility"] for x in rows}
            removal[method][branch] = {
                "proxy_users": len(rows),
                "full_success": float(np.mean([x["full_success"] for x in rows])) if rows else None,
                "removed_success": float(np.mean([x["removed_success"] for x in rows])) if rows else None,
                "difference": float(np.mean([x["difference"] for x in rows])) if rows else None,
                "difference_ci": _bootstrap_user_mean(
                    differences, reps=bootstrap_replicates,
                    seed=_stable_int(seed, method, branch, "removal"),
                ) if rows else None,
                "proxy_only_success": float(np.mean([x["proxy_only_success"] for x in rows])) if rows else None,
                "no_memory_success": float(np.mean([x["no_memory_success"] for x in rows])) if rows else None,
                "proxy_only_utility": float(np.mean([x["proxy_only_utility"] for x in rows])) if rows else None,
                "proxy_only_utility_ci": _bootstrap_user_mean(
                    proxy_utilities, reps=bootstrap_replicates,
                    seed=_stable_int(seed, method, branch, "proxy-utility"),
                ) if rows else None,
            }
    result = {
        "model": client.model,
        "temperature": client.temperature,
        "occurrence": occurrence,
        "paired_removal": removal,
        "summary": summarize(records, bootstrap_replicates=bootstrap_replicates, seed=seed),
    }
    (output_dir / "llm_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_from_config(
    config_path: Path,
    catalog_path: Path,
    output_dir: Path,
    *,
    languages: set[str] | None = None,
) -> dict:
    config = json.loads(config_path.read_text())
    items = load_items(catalog_path)
    pair_specs = [
        spec for spec in config["pairs"]
        if languages is None or spec["language"] in languages
    ]
    if not pair_specs:
        raise ValueError(f"no configured pairs match languages={sorted(languages or set())}")
    pairs = [
        build_pair_catalogs(
            items, language=spec["language"], country=spec["country"],
            target_count=spec["target_count"], catalog_size=config["catalog_size"], seed=config["seed"],
        )
        for spec in pair_specs
    ]
    client = CachedBaseten(
        output_dir / "baseten_calls.jsonl",
        model=config["llm_model"], temperature=config["llm_temperature"],
    )
    return run_llm(
        pairs, client=client, users_per_pair=config["llm_users_per_pair"],
        learning_visits=config["learning_visits"], continuation_visits=config["continuation_visits"],
        budget=config["proposal_budget"], bootstrap_replicates=config["bootstrap_replicates"],
        seed=config["seed"], output_dir=output_dir,
    )

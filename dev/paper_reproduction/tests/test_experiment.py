from __future__ import annotations

from catalog_proxy.experiment import (
    Item,
    Memory,
    Predicate,
    Request,
    _propose_sequence,
)
from catalog_proxy.llm_experiment import count_gate


def item(item_id: str, language: str, country: str) -> Item:
    return Item(item_id, item_id, ("Fantasy",), (language,), (country,), 2001, 1)


REQUEST = Request("Fantasy", "2000 or later", 2000, 10_000)


def test_acceptance_deletes_false_tied_candidate() -> None:
    memory = Memory(tie={Predicate("languages", "Spanish"), Predicate("countries", "Spain")})
    result = _propose_sequence(
        "tie", [item("a", "Spanish", "Mexico")], REQUEST, memory,
        language="Spanish", budget=3, seed=1, user_id="u", visit=0, branch="shift",
    )
    assert result.success == 1
    assert Predicate("countries", "Spain") not in memory.tie
    assert Predicate("languages", "Spanish") in memory.tie


def test_single_relaxation_rejection_establishes_true_rule() -> None:
    memory = Memory(tie={Predicate("languages", "Spanish"), Predicate("countries", "Spain")})
    result = _propose_sequence(
        "tie", [item("a", "French", "Spain")], REQUEST, memory,
        language="Spanish", budget=1, seed=1, user_id="u", visit=0, branch="shift",
    )
    assert result.success == 0
    assert result.tie_single_rejects == 1
    assert Predicate("languages", "Spanish") in memory.established


def test_whole_tie_recommendation_is_accepted() -> None:
    memory = Memory(tie={Predicate("languages", "Spanish"), Predicate("countries", "Spain")})
    result = _propose_sequence(
        "tie", [item("a", "Spanish", "Spain"), item("b", "English", "US")], REQUEST, memory,
        language="Spanish", budget=1, seed=2, user_id="u", visit=0, branch="c0",
    )
    assert result.success == 1
    assert result.proposals == 1
    assert result.whole_tie_available == 1


def test_count_gate_rejects_rule_contradicted_by_acceptance() -> None:
    seed = {
        "languages": ["Spanish"], "countries": ["Spain"], "verdict": "accept",
    }
    visits = [{
        "events": [
            {"languages": ["Spanish"], "countries": ["Mexico"], "verdict": "accept"},
            {"languages": ["French"], "countries": ["Spain"], "verdict": "reject"},
        ]
    }]
    candidates = {Predicate("languages", "Spanish"), Predicate("countries", "Spain")}
    assert count_gate(candidates, seed, visits) == {Predicate("languages", "Spanish")}


def test_frozen_relax_recovers_from_empty_hard_filter() -> None:
    memory = Memory(established={Predicate("languages", "Spanish"), Predicate("countries", "Spain")})
    result = _propose_sequence(
        "frozen_relax", [item("a", "Spanish", "Mexico"), item("b", "French", "Spain")],
        REQUEST, memory, language="Spanish", budget=3, seed=1,
        user_id="u", visit=0, branch="shift",
    )
    assert result.success == 1
    assert result.proposals <= 2
    assert result.rejected <= 1
    assert memory.established == {Predicate("languages", "Spanish"), Predicate("countries", "Spain")}


def test_frozen_relax_recovers_after_strict_rejection() -> None:
    memory = Memory(established={Predicate("countries", "Spain")})
    result = _propose_sequence(
        "frozen_relax", [item("a", "French", "Spain"), item("b", "Spanish", "Mexico")],
        REQUEST, memory, language="Spanish", budget=3, seed=1,
        user_id="u", visit=0, branch="shift",
    )
    assert result.success == 1
    assert result.proposals == 2
    assert result.rejected == 1


def test_relax_delete_removes_rule_violated_by_acceptance() -> None:
    language = Predicate("languages", "Spanish")
    proxy = Predicate("countries", "Spain")
    memory = Memory(established={language, proxy})
    result = _propose_sequence(
        "relax_delete", [item("a", "Spanish", "Mexico")],
        REQUEST, memory, language="Spanish", budget=2, seed=1,
        user_id="u", visit=0, branch="shift",
    )
    assert result.success == 1
    assert memory.established == {language}


def test_soft_rank_uses_rules_as_preferences_not_filters() -> None:
    language = Predicate("languages", "Spanish")
    proxy = Predicate("countries", "Spain")
    memory = Memory(established={language, proxy})
    result = _propose_sequence(
        "soft_rank", [item("a", "Spanish", "Mexico")],
        REQUEST, memory, language="Spanish", budget=1, seed=1,
        user_id="u", visit=0, branch="shift",
    )
    assert result.success == 1
    assert memory.established == {language, proxy}

# Paper-matched reproduction

This directory contains the standalone implementation that generated the
reported paper results. It is intentionally separate from `proxy_audit.ipynb`,
which is a later reimplementation with changed seeds, policies, and catalog
construction.

## What is covered

| Paper component | Entry point |
|---|---|
| Hard/soft/transient/update/ask serving audit | `run_experiment.py` |
| Hard-only LLM writer | `run_llm_experiment.py` |
| Established/plausible schema ablation | `schema_ablation.py` |
| Truth × noise × update-threshold stress test | `stress_analysis.py` |
| Pair, history-length, utility, and LLM-memory replay analyses | `robustness_analysis.py` |
| Paper summary figure | `make_interface_report.py` |

The package implementation is under `src/catalog_proxy/`. Frozen parsed model
states and aggregate outputs are under `../../results/paper_reproduction/`.
Raw provider prompts/responses and credentials are not committed.

## Environment and public catalog

From this directory:

```bash
uv sync --frozen
uv run python download_catalog.py
uv run pytest -q
```

The downloader pins τ-Rec commit
`e96771e0d4ee7ddaff94f74e5080d4c3a184a4b8` and verifies catalog SHA-256
`19e9fa3a4db03de8f8bb8cdd8f7b231346a44f21cefca18947b7b10485d7093f`.

## Provider-free reproduction

```bash
uv run python run_experiment.py \
  --catalog ../../data/catalog.json \
  --output outputs/primary-v3

uv run python stress_analysis.py \
  --catalog ../../data/catalog.json

uv run python robustness_analysis.py \
  --catalog ../../data/catalog.json

uv run python make_interface_report.py
uv run python verify_frozen.py
```

The robustness and figure scripts automatically use checked-in parsed LLM
states when fresh provider outputs are absent. Generated scratch outputs remain
under this directory; canonical frozen outputs remain in the repository-level
`results/paper_reproduction/` namespace.

## Re-running provider calls

Provider calls use the OpenAI-compatible Baseten endpoint at temperature 0.1.
Set `BASETEN_API_KEY` in the environment; do not commit it.

```bash
BASETEN_API_KEY=... uv run python run_llm_experiment.py \
  --config configs/primary.json \
  --catalog ../../data/catalog.json \
  --output outputs/llm

BASETEN_API_KEY=... uv run python schema_ablation.py \
  --config configs/primary.json \
  --catalog ../../data/catalog.json \
  --output outputs/schema-gpt \
  --baseline outputs/llm/llm_memories.jsonl
```

GLM uses `configs/sensitivity_glm.json` for the hard-only writer and
`configs/schema_ablation_glm.json` for the calibrated writer. The language
flag can split calls into Spanish, French, Russian, and German runs. Provider
responses are stochastic external dependencies; the checked-in parsed states
are the audit record for the paper's reported samples.

## Headline consistency checks

`results/paper_reproduction/schema_interface_summary.json` records:

- GPT ambiguous-history proxy commitment: 60/63 hard-only, 5/63 established
  under the calibrated interface;
- GLM: 24/63 hard-only, 1/63 calibrated established;
- both calibrated writers retain the true language for all 80 users as either
  established or plausible.

The test suite is credential-free and checks the serving-policy mechanics,
including that soft ranking never filters request-matching candidates.

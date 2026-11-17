# Auditing Hard Memory in Recommender Agents

Research artifact for *When Catalogs Become Preferences: Auditing Hard Memory
in Recommender Agents*.

The repository keeps two deliberately separate experiment tracks:

1. `proxy_audit.ipynb` is an expanded notebook reimplementation with additional
   exploratory experiments. Its design changes and provisional findings are
   documented in `HANDOFF.md`.
2. `dev/paper_reproduction/` is the standalone implementation that generated
   the numbers reported in the paper. Its parsed LLM states and aggregate outputs
   are frozen under `results/paper_reproduction/`.

Start with [the paper-reproduction instructions](dev/paper_reproduction/README.md)
when checking a claim in the submitted paper. The public τ-Rec-Books catalog is
downloaded from a pinned commit and verified by SHA-256. Provider call caches,
API keys, and local paths are intentionally excluded.

## Repository layout

- `dev/nb_src/`: percent-format source for the exploratory notebook
- `dev/paper_reproduction/`: paper-matched package, runners, configs, and tests
- `results/`: frozen tabular and JSON outputs, separated by experiment track
- `figures/`: generated paper and exploratory figures

The paper-matched unit suite currently contains eight tests and requires no
provider credentials.

# Paper-matched frozen outputs

These files are the audit record for the numbers reported in the submitted
paper. They are separate from the expanded notebook results in the parent
directory.

- `schema_interface_summary.json` and `schema_rows_*.jsonl`: writing
  interface audit
- `scripted_summary.json`: hard, soft, transient, update, and ask serving
  policies
- `stress_summary.json`: true-rule, verdict-noise, and update-threshold grid
- `robustness_summary.json`: pair heterogeneity, LLM-memory replay,
  stand-alone proxy utility, and history-length sensitivity
- `llm_memories_*.jsonl` and `history_memories_*.jsonl`: parsed synthetic
  memory states used by the aggregate analyses

The JSONL files contain synthetic user identifiers and parsed rules. They do
not contain API keys, raw prompts, provider responses, or local filesystem
paths.

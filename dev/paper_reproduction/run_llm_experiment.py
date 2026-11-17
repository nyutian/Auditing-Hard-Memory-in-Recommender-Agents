from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from catalog_proxy.llm_experiment import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/primary.json"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/llm"))
    parser.add_argument("--language", action="append", dest="languages")
    args = parser.parse_args()
    result = run_from_config(
        args.config, args.catalog, args.output,
        languages=set(args.languages) if args.languages else None,
    )
    print(result["occurrence"])
    print(result["paired_removal"])


if __name__ == "__main__":
    main()

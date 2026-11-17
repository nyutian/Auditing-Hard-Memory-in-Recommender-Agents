from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from catalog_proxy import run_primary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/primary.json"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/primary"))
    args = parser.parse_args()
    result = run_primary(args.config, args.catalog, args.output)
    for method, branches in result["summary"].items():
        c0 = branches["c0"]["success"]["mean"]
        shift = branches["shift"]["success"]["mean"]
        print(f"{method:12s} c0={c0:.3f} shift={shift:.3f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT.parent.parent / "results" / "paper_reproduction"
CHECKS = {
    "primary": (
        ROOT / "outputs" / "primary-v3" / "scripted_summary.json",
        FROZEN / "scripted_summary.json",
    ),
    "stress": (
        ROOT / "outputs" / "stress" / "stress_summary.json",
        FROZEN / "stress_summary.json",
    ),
    "robustness": (
        ROOT / "outputs" / "robustness" / "robustness_summary.json",
        FROZEN / "robustness_summary.json",
    ),
}


def main() -> None:
    missing = [str(generated) for generated, _ in CHECKS.values() if not generated.exists()]
    if missing:
        raise SystemExit(
            "run the provider-free reproduction commands first; missing:\n"
            + "\n".join(missing)
        )
    failed = False
    for name, (generated, frozen) in CHECKS.items():
        same = json.loads(generated.read_text()) == json.loads(frozen.read_text())
        print(f"{name}: {'MATCH' if same else 'MISMATCH'}")
        failed |= not same
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

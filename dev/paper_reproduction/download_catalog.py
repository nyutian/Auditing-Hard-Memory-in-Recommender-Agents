from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path


CATALOG_URL = (
    "https://raw.githubusercontent.com/nbharaths/tau-rec/"
    "e96771e0d4ee7ddaff94f74e5080d4c3a184a4b8/"
    "data/tracks/books-wikidata-v2/catalog.json"
)
CATALOG_SHA256 = "19e9fa3a4db03de8f8bb8cdd8f7b231346a44f21cefca18947b7b10485d7093f"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "catalog.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and digest(args.output) == CATALOG_SHA256:
        print(f"{args.output} (already verified)")
        return

    with tempfile.NamedTemporaryFile(
        dir=args.output.parent, prefix="catalog-", suffix=".json", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        urllib.request.urlretrieve(CATALOG_URL, temporary)
        actual = digest(temporary)
        if actual != CATALOG_SHA256:
            raise RuntimeError(
                f"catalog hash mismatch: expected {CATALOG_SHA256}, got {actual}"
            )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{args.output} (sha256 verified)")


if __name__ == "__main__":
    main()

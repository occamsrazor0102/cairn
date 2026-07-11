"""Write deterministic hashes and byte sizes for one or more artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from cairn.tocsin_cache import hash_artifacts, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    write_json_atomic(args.out, hash_artifacts(args.paths))


if __name__ == "__main__":
    main()

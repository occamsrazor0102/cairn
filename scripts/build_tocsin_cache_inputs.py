"""Build a deterministic normalized-text union for TOCSIN cache generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from cairn.tocsin_cache import build_union_dataset, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-out", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--datasets", required=True, nargs="+", type=Path)
    args = parser.parse_args()

    manifest = build_union_dataset(args.datasets, args.union_out)
    write_json_atomic(args.manifest_out, manifest)


if __name__ == "__main__":
    main()

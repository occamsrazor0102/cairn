"""Slice a validated union feature cache for one consumer dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from cairn.tocsin_cache import extract_cache_for_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    extract_cache_for_dataset(args.source_cache, args.dataset, args.out)


if __name__ == "__main__":
    main()

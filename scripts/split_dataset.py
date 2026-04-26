"""Stratified train / calib / test split of an EvalSample JSONL.

Calibrating a fusion model and a conformal wrapper on the same data
invalidates the conformal coverage guarantee. This utility produces
the three disjoint files those scripts expect.

The split is stratified by `label` and seeded for reproducibility.

Usage:
    python scripts/split_dataset.py \\
        --in     data/my_corpus.jsonl \\
        --out-dir data \\
        --prefix mydata \\
        --train 0.70 --calib 0.15 --test 0.15 \\
        --seed 42

Produces:
    data/mydata_train.jsonl
    data/mydata_calib.jsonl
    data/mydata_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--prefix", required=True, help="filename prefix for the three splits")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--calib", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    total = args.train + args.calib + args.test
    if not (0.99 <= total <= 1.01):
        raise SystemExit(f"--train + --calib + --test must sum to 1.0 (got {total:.3f})")

    rng = random.Random(args.seed)
    rows = [
        json.loads(line)
        for line in args.src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_label: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[int(r["label"])].append(r)
    for v in by_label.values():
        rng.shuffle(v)

    train: list[dict] = []
    calib: list[dict] = []
    test: list[dict] = []
    for v in by_label.values():
        n = len(v)
        n_train = int(n * args.train)
        n_calib = int(n * args.calib)
        train.extend(v[:n_train])
        calib.extend(v[n_train:n_train + n_calib])
        test.extend(v[n_train + n_calib:])
    rng.shuffle(train)
    rng.shuffle(calib)
    rng.shuffle(test)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("calib", calib), ("test", test)]:
        out = args.out_dir / f"{args.prefix}_{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")
        h = sum(1 for r in split if int(r["label"]) == 0)
        l = sum(1 for r in split if int(r["label"]) == 1)
        print(f"{out}: n={len(split)} (human={h}, llm={l})")


if __name__ == "__main__":
    main()

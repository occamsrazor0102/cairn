"""Deterministic, label-balanced, source-stratified sampler.

Pulls a small holdout slice from one of the calib2000 JSONLs for the
full-coverage-vs-partial-coverage experiment described in the
post-honest-calibration roadmap. Default input is the held-out test
split so the holdout has no overlap with the EBM's training data.

Usage:
    python scripts/sample_calib2000_holdout.py \\
        --input data/calib2000_full_test.jsonl \\
        --output data/calib2000_holdout_500.jsonl \\
        --n 500

Output is JSONL with the same schema as the input
({id, text, label, tier, source, source_id}), ready to feed into
`scripts/cache_pipeline_features.py`.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _stratified_sample(
    rows: list[dict], n: int, seed: int
) -> list[dict]:
    """Sample n rows, label-balanced and source-stratified.

    Splits the request 50/50 across labels {0, 1}; within each label,
    allocates per-source quotas proportional to that source's share of
    the label pool, then rounds and reconciles drift to land on
    exactly n/2 per label.
    """
    rng = random.Random(seed)
    by_label_source: dict[int, dict[str, list[dict]]] = {0: defaultdict(list), 1: defaultdict(list)}
    for r in rows:
        label = int(r["label"])
        if label not in (0, 1):
            continue
        by_label_source[label][r.get("source", "?")].append(r)

    half = n // 2
    picks: list[dict] = []
    for label in (0, 1):
        pool = by_label_source[label]
        total = sum(len(v) for v in pool.values())
        if total == 0:
            raise SystemExit(f"no rows with label={label} in input")
        quotas: dict[str, int] = {}
        for src, items in pool.items():
            quotas[src] = max(1, round(half * len(items) / total)) if items else 0
        total_quota = sum(quotas.values())
        adj_order = sorted(quotas.keys(), key=lambda s: -quotas[s])
        i = 0
        while total_quota > half and i < 10_000:
            src = adj_order[i % len(adj_order)]
            if quotas[src] > 1:
                quotas[src] -= 1
                total_quota -= 1
            i += 1
        i = 0
        while total_quota < half and i < 10_000:
            src = adj_order[i % len(adj_order)]
            if quotas[src] < len(pool[src]):
                quotas[src] += 1
                total_quota += 1
            i += 1
        for src, count in quotas.items():
            items = pool[src]
            count = min(count, len(items))
            picks.extend(rng.sample(items, count))

    rng.shuffle(picks)
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", type=Path,
                    default=Path("data/calib2000_full_test.jsonl"),
                    help="Source JSONL. Default is the test split — no overlap "
                         "with EBM training data.")
    ap.add_argument("--output", type=Path,
                    default=Path("data/calib2000_holdout_500.jsonl"))
    ap.add_argument("--n", type=int, default=500,
                    help="Target sample size. Must be even (split 50/50 across labels).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed. Same seed → same holdout.")
    args = ap.parse_args()

    if args.n % 2 != 0:
        raise SystemExit("--n must be even (label-balanced sample)")

    rows = _load_rows(args.input)
    print(f"loaded {len(rows)} rows from {args.input}", flush=True)

    picks = _stratified_sample(rows, args.n, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in picks:
            f.write(json.dumps(r) + "\n")

    n0 = sum(1 for r in picks if int(r["label"]) == 0)
    n1 = sum(1 for r in picks if int(r["label"]) == 1)
    n_sources = len({r.get("source", "?") for r in picks})
    print(
        f"wrote {len(picks)} rows to {args.output} "
        f"(label 0: {n0}, label 1: {n1}, {n_sources} unique sources)",
        flush=True,
    )


if __name__ == "__main__":
    main()

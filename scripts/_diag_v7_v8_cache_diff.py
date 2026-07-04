"""Diff v7 vs v8 feature caches on shared sample hashes.

Open question from docs/benchmarks/robust_v2_validation/HYPOTHESIS_INVESTIGATION.md:
the ~0.14 RAID AUROC gap between robust_v1 (v7-trained) and robust_v2i
(v8-trained, same EBM + calibrator schema as v1) is unexplained by feature
schema, calibrator schema, or the 4-row corpus delta. The remaining suspect
is detector-code or signal-value drift between the May 13 v7 build and the
May 18 v8 build.

Cache record schema (confirmed via scripts/_diag_v7_v8_inspect.py):
  {"hash": sha256_of_text, "results": [LayerResult_dict, ...], "router": {...}}
The hash is content-derived, so a sample present in both caches is the
same input text. Same input -> same expected output, so any signal-value
delta is real detector-code drift.

This script joins on hash, then for each detector layer that exists in
both caches, reports the per-field delta distribution (raw_score, p_llm,
confidence). Sorts loudest drift first.

Throwaway diagnostic.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load_cache(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            h = rec.get("hash")
            if h is None:
                continue
            out[h] = rec
    return out


def _as_float(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        f = float(x)
        return None if math.isnan(f) else f
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7", type=Path, default=Path("data/all_features_v7.jsonl"))
    ap.add_argument("--v8", type=Path, default=Path("data/all_features_v8.jsonl"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=1e-6)
    args = ap.parse_args()

    print(f"[load] v7 cache: {args.v7}")
    v7 = load_cache(args.v7)
    print(f"  -> {len(v7)} records")
    print(f"[load] v8 cache: {args.v8}")
    v8 = load_cache(args.v8)
    print(f"  -> {len(v8)} records")

    shared = sorted(set(v7) & set(v8))
    only_v8 = len(set(v8) - set(v7))
    only_v7 = len(set(v7) - set(v8))
    print(f"[shared] {len(shared)} hashes in both caches  "
          f"(only-v7: {only_v7}, only-v8: {only_v8})")

    rng = random.Random(args.seed)
    sample = shared if len(shared) <= args.n else rng.sample(shared, args.n)
    print(f"[sample] using N={len(sample)}")

    if not sample:
        print("[ERROR] no shared hashes")
        return

    # Discover the union of (layer_id, field) keys across the sampled set.
    layer_fields_seen: dict[str, set[str]] = defaultdict(set)
    for h in sample:
        for rec in (v7[h], v8[h]):
            for r in rec.get("results", []) or []:
                lid = r.get("layer_id")
                if not lid:
                    continue
                for k in ("raw_score", "p_llm", "confidence"):
                    if _as_float(r.get(k)) is not None:
                        layer_fields_seen[lid].add(k)

    # Track NaN-pattern asymmetry (missing/None in one but present in the other).
    deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
    nan_diffs: dict[tuple[str, str], int] = defaultdict(int)
    layer_only_v7: dict[str, int] = defaultdict(int)
    layer_only_v8: dict[str, int] = defaultdict(int)

    for h in sample:
        res7 = {r["layer_id"]: r for r in v7[h].get("results", []) or [] if "layer_id" in r}
        res8 = {r["layer_id"]: r for r in v8[h].get("results", []) or [] if "layer_id" in r}
        layers = set(res7) | set(res8)
        for lid in layers:
            if lid in res7 and lid not in res8:
                layer_only_v7[lid] += 1
                continue
            if lid in res8 and lid not in res7:
                layer_only_v8[lid] += 1
                continue
            r7, r8 = res7[lid], res8[lid]
            for field in layer_fields_seen.get(lid, ()):
                x7 = _as_float(r7.get(field))
                x8 = _as_float(r8.get(field))
                if x7 is None and x8 is None:
                    continue
                if (x7 is None) != (x8 is None):
                    nan_diffs[(lid, field)] += 1
                    continue
                deltas[(lid, field)].append(x8 - x7)

    rows = []
    for (lid, field), ds in deltas.items():
        nan_d = nan_diffs.get((lid, field), 0)
        n = len(ds)
        mean = statistics.fmean(ds) if n else 0.0
        stdev = statistics.pstdev(ds) if n > 1 else 0.0
        max_abs = max(abs(x) for x in ds) if ds else 0.0
        shifted = sum(1 for x in ds if abs(x) > args.threshold)
        rows.append((lid, field, n, nan_d, mean, stdev, max_abs, shifted))
    # also surface field-less rows where only nan_diffs fired
    for (lid, field), nd in nan_diffs.items():
        if (lid, field) not in deltas:
            rows.append((lid, field, 0, nd, 0.0, 0.0, 0.0, 0))

    rows.sort(key=lambda r: (r[6], r[3]), reverse=True)

    print()
    print(f"{'layer':<24} {'field':<12} {'n':>6} {'nanD':>6} {'mean':>12} {'std':>10} {'maxAbs':>10} {'shifted':>8}")
    print("-" * 105)
    for lid, field, n, nan_d, mean, stdev, max_abs, shifted in rows:
        if max_abs <= args.threshold and nan_d == 0:
            continue
        marker = " <<<" if max_abs > 1e-3 or nan_d > 0 else ""
        print(f"{lid:<24} {field:<12} {n:>6} {nan_d:>6} {mean:>+12.4g} {stdev:>10.4g} {max_abs:>10.4g} {shifted:>8}{marker}")

    print()
    if layer_only_v7:
        print(f"Layers present only in v7: {dict(layer_only_v7)}")
    if layer_only_v8:
        print(f"Layers present only in v8: {dict(layer_only_v8)}")
    drifted = sum(1 for r in rows if r[6] > args.threshold or r[3] > 0)
    print(f"Summary: {drifted} (layer, field) pairs drifted beyond threshold {args.threshold}")


if __name__ == "__main__":
    main()

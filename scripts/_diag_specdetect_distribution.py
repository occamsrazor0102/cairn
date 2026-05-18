"""Diagnostic for H2: does the SpecDetect heuristic table cover the
empirical dft_total_energy distribution?

Reads the v8 feature cache, extracts every specdetect record's
dft_total_energy signal value, and reports:
  - empirical percentiles vs heuristic table breakpoints
  - per-label (human/LLM) distributions
  - per-router-domain distributions
  - p_llm distribution after calibration

If the empirical distribution clusters outside the heuristic table's
0.5-5.0 anchor range, that's evidence H2 is real -- the table doesn't
discriminate well in the actual data regime.

One-off scratch invocation (underscore-prefix).

Usage:
    python scripts/_diag_specdetect_distribution.py \\
        --cache data/all_features_v8.jsonl \\
        --dataset data/master_minus_arxiv_robust_v2_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


_HEURISTIC_BREAKPOINTS = [(0.5, 0.92), (1.0, 0.78), (1.5, 0.55), (2.0, 0.30), (3.0, 0.10), (5.0, 0.05)]


def percentiles(values: list[float], ps: list[float]) -> list[float]:
    if not values:
        return [float("nan")] * len(ps)
    s = sorted(values)
    n = len(s)
    return [s[min(int(p / 100 * n), n - 1)] for p in ps]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    args = ap.parse_args()

    # Load label by text-hash via the dataset
    from cairn.evaluation.dataset import load_dataset
    from cairn.feature_cache import text_hash
    from cairn.pipeline import normalize_text
    samples = load_dataset(args.dataset)
    label_by_hash: dict[str, int] = {}
    for s in samples:
        label_by_hash[text_hash(normalize_text(s.text))] = int(s.label)
    print(f"loaded labels for {len(label_by_hash):,} unique texts from {args.dataset}", flush=True)

    by_label: dict[int, list[float]] = defaultdict(list)
    by_router: dict[str, list[float]] = defaultdict(list)
    p_llm_values: list[float] = []
    n_records = 0
    n_with_specdetect = 0

    with args.cache.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_records += 1
            rec = json.loads(line)
            router_domain = rec.get("router", {}).get("domain", "prose")
            label = label_by_hash.get(rec.get("hash"))
            for r in rec.get("results", []):
                if r.get("layer_id") != "specdetect":
                    continue
                signals = r.get("signals") or {}
                energy = signals.get("dft_total_energy")
                if energy is None:
                    continue
                n_with_specdetect += 1
                e = float(energy)
                if label is not None:
                    by_label[label].append(e)
                by_router[router_domain].append(e)
                p_llm = r.get("p_llm")
                if p_llm is not None:
                    p_llm_values.append(float(p_llm))

    print(f"\n=== Coverage ===")
    print(f"  records read: {n_records:,}")
    print(f"  records with specdetect: {n_with_specdetect:,} ({100*n_with_specdetect/n_records:.1f}%)")

    print(f"\n=== Heuristic table anchor range: {_HEURISTIC_BREAKPOINTS[0][0]:.2f} to {_HEURISTIC_BREAKPOINTS[-1][0]:.2f} ===")

    print(f"\n=== Empirical dft_total_energy percentiles ===")
    print(f"{'group':<20} {'n':>8} {'p1':>8} {'p5':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8} {'p99':>8}")
    groups = []
    if 0 in by_label and 1 in by_label:
        groups.append(("label=human", by_label[0]))
        groups.append(("label=LLM", by_label[1]))
    for dom, vs in by_router.items():
        groups.append((f"router={dom}", vs))
    all_vs: list[float] = []
    if 0 in by_label or 1 in by_label:
        all_vs = by_label.get(0, []) + by_label.get(1, [])
    if not all_vs:
        all_vs = sum(by_router.values(), [])
    groups.insert(0, ("OVERALL", all_vs))

    ps = [1, 5, 25, 50, 75, 95, 99]
    for name, vs in groups:
        if not vs:
            continue
        pvs = percentiles(vs, ps)
        n = len(vs)
        print(f"{name:<20} {n:>8} " + " ".join(f"{p:>8.3f}" for p in pvs))

    above_max = sum(1 for v in all_vs if v > _HEURISTIC_BREAKPOINTS[-1][0])
    below_min = sum(1 for v in all_vs if v < _HEURISTIC_BREAKPOINTS[0][0])
    print(f"\n  off-table-high (energy > {_HEURISTIC_BREAKPOINTS[-1][0]}): "
          f"{above_max:,} ({100*above_max/max(1,len(all_vs)):.1f}%)")
    print(f"  off-table-low  (energy < {_HEURISTIC_BREAKPOINTS[0][0]}): "
          f"{below_min:,} ({100*below_min/max(1,len(all_vs)):.1f}%)")

    print(f"\n=== Calibrated specdetect_p_llm distribution ===")
    if p_llm_values:
        pvs = percentiles(p_llm_values, ps)
        print(f"{'p_llm':<20} {len(p_llm_values):>8} " + " ".join(f"{p:>8.3f}" for p in pvs))

    if 0 in by_label and 1 in by_label:
        from sklearn.metrics import roc_auc_score
        labels = [0]*len(by_label[0]) + [1]*len(by_label[1])
        scores = [-v for v in by_label[0]] + [-v for v in by_label[1]]
        auroc = roc_auc_score(labels, scores)
        print(f"\n=== Raw dft_total_energy AUROC (lower energy = LLM, heuristic direction): {auroc:.4f} ===")
        scores_flipped = [v for v in by_label[0]] + [v for v in by_label[1]]
        auroc_flipped = roc_auc_score(labels, scores_flipped)
        print(f"=== Raw dft_total_energy AUROC (HIGHER energy = LLM, opposite direction): {auroc_flipped:.4f} ===")

    return 0


if __name__ == "__main__":
    sys.exit(main())

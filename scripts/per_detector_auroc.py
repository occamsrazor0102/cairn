"""Per-detector AUROC across one or more datasets, using a feature cache.

For each detector and each dataset, loads the cached LayerResults,
applies a fitted calibrator (so we evaluate calibrated p_llm, matching
what the live pipeline would produce), and computes AUROC against the
EvalSample labels. Useful for diagnosing which detectors transfer to
out-of-distribution data and which don't.

Usage:
    python scripts/per_detector_auroc.py \\
        --calibrator models/calib2000/ablation/full/detector_calibration.json \\
        --datasets in:data/calib2000_full_test.jsonl \\
                   m4:data/ood_m4.jsonl \\
                   raid:data/ood_raid.jsonl \\
                   mage:data/ood_mage.jsonl \\
        --features-caches in:data/calib2000_features.jsonl \\
                          m4:data/ood_features.jsonl \\
                          raid:data/ood_features.jsonl \\
                          mage:data/ood_features.jsonl

The --datasets and --features-caches args use NAME:PATH pairs so each
dataset can read from its own cache (in-domain test uses the calib2000
cache; OOD splits use the OOD cache).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import roc_auc_score

from cairn.calibration import DetectorCalibrator
from cairn.contracts import LayerResult
from cairn.dotenv import load_dotenv
from cairn.feature_cache import load_cache, text_hash
from cairn.pipeline import normalize_text


def load_dataset_with_text(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def parse_pairs(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for it in items:
        if ":" not in it:
            raise SystemExit(f"expected NAME:PATH, got {it!r}")
        k, v = it.split(":", 1)
        out[k] = Path(v)
    return out


def per_detector_auroc(
    samples: list[dict],
    cache: dict,
    calibrator: DetectorCalibrator | None,
) -> tuple[dict[str, float], dict[str, int]]:
    score_per_det: dict[str, list[float]] = defaultdict(list)
    label_per_det: dict[str, list[int]] = defaultdict(list)
    for s in samples:
        h = text_hash(normalize_text(s["text"]))
        rec = cache.get(h)
        if rec is None:
            continue
        for cached_result in rec["results"]:
            result = LayerResult(**cached_result)
            if calibrator is not None:
                calibrator.apply(result)
            raw = result.raw_score
            if raw is None or not math.isfinite(float(raw)):
                continue
            if result.determination == "SKIP":
                continue
            score_per_det[result.layer_id].append(float(result.p_llm))
            label_per_det[result.layer_id].append(int(s["label"]))

    auroc: dict[str, float] = {}
    n_per_det: dict[str, int] = {}
    for det_id, scores in score_per_det.items():
        labels = label_per_det[det_id]
        if len(set(labels)) < 2 or len(labels) < 10:
            auroc[det_id] = float("nan")
        else:
            auroc[det_id] = float(roc_auc_score(labels, scores))
        n_per_det[det_id] = len(labels)
    return auroc, n_per_det


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--calibrator", required=True, type=Path)
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="NAME:PATH pairs (e.g. m4:data/ood_m4.jsonl)")
    ap.add_argument("--features-caches", nargs="+", required=True,
                    help="NAME:PATH pairs mapping each dataset name to its feature cache file")
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    datasets = parse_pairs(args.datasets)
    caches_paths = parse_pairs(args.features_caches)
    if set(datasets) != set(caches_paths):
        raise SystemExit(
            f"--datasets keys {sorted(datasets)} != --features-caches keys {sorted(caches_paths)}"
        )

    cal = DetectorCalibrator()
    cal.load(args.calibrator)

    cache_blobs: dict[str, dict] = {}
    seen_paths: dict[Path, dict] = {}
    for name, p in caches_paths.items():
        if p in seen_paths:
            cache_blobs[name] = seen_paths[p]
        else:
            print(f"loading cache {p} for {name}...", flush=True)
            blob = load_cache(p)
            seen_paths[p] = blob
            cache_blobs[name] = blob
            print(f"  {len(blob)} records", flush=True)

    results: dict[str, dict[str, float]] = {}
    for name, ds_path in datasets.items():
        rows = load_dataset_with_text(ds_path)
        auroc, _n = per_detector_auroc(rows, cache_blobs[name], cal)
        results[name] = auroc
        n_with = sum(1 for v in auroc.values() if not math.isnan(v))
        print(f"{name}: {len(rows)} samples, {n_with} detectors", flush=True)

    all_dets = sorted({d for ds in results.values() for d in ds})
    ds_names = list(datasets.keys())

    print()
    header = f"{'detector':<25} " + " ".join(f"{n:>10}" for n in ds_names)
    print(header)
    print("-" * len(header))
    for det in all_dets:
        row = f"{det:<25} "
        for n in ds_names:
            v = results[n].get(det, float("nan"))
            row += f"{v:>10.4f} " if not math.isnan(v) else f"{'n/a':>10} "
        print(row)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
            f.write("detector," + ",".join(ds_names) + "\n")
            for det in all_dets:
                vals = []
                for n in ds_names:
                    v = results[n].get(det, float("nan"))
                    vals.append(f"{v:.4f}" if not math.isnan(v) else "")
                f.write(det + "," + ",".join(vals) + "\n")
        print(f"\nwrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()

"""Pre-compute and cache pipeline.analyze_detailed outputs for a dataset.

Run once per text corpus. The cached features are calibrator-INDEPENDENT
(they record raw_score, signals, etc.); each downstream calibration
script applies its own calibrator at read-time. This means the same
cache can power N ablation runs that swap calibrators / fusion models /
conformal thresholds, paying the heavy detector-inference cost only
once.

Usage:
    python scripts/cache_pipeline_features.py \\
        --dataset data/calib2000_full.jsonl \\
        --config configs/default.yaml \\
        --out data/calib2000_full_features.jsonl

Tip: pass --config configs/default.yaml (NOT a calibrated profile) so
the cache is written with no calibrator applied. Calibrators are
applied at read-time via cairn.feature_cache.CachedPipeline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import load_dataset
from cairn.feature_cache import text_hash, write_record
from cairn.pipeline import CairnPipeline, normalize_text

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(it, **_kw):
        return it


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path,
                    help="Pass an UNCALIBRATED profile (e.g. default.yaml) so "
                         "cached p_llm equals the detector's heuristic mapping. "
                         "Calibrators are applied at read-time.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    pipeline = CairnPipeline.from_config_file(args.config)
    if pipeline._detector_calibrator is not None:
        print(
            "WARNING: config has a detector_calibration_path set. The cache "
            "will store calibrated p_llm values, which defeats the point of "
            "ablation-friendly caching. Pass --config configs/default.yaml "
            "instead.",
            flush=True,
        )

    samples = load_dataset(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n_dup = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for s in tqdm(samples, desc="caching", unit="sample"):
            text = normalize_text(s.text)
            key = text_hash(text)
            if key in seen:
                # Same text appears under multiple sample ids (e.g. same
                # human answer reused across providers). Cache once.
                n_dup += 1
                continue
            seen.add(key)
            _det, results, rd = pipeline.analyze_detailed(text)
            write_record(out, key, results, rd)

    print(
        f"wrote {len(seen)} unique features to {args.out} "
        f"(skipped {n_dup} duplicate texts)",
        flush=True,
    )


if __name__ == "__main__":
    main()

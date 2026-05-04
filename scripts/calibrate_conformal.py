"""Calibrate conformal prediction threshold from a held-out dataset.

Usage:
    python scripts/calibrate_conformal.py \
        --dataset data/seed_conformal.jsonl \
        --config configs/default.yaml \
        --alpha 0.05 \
        --out models/conformal_cal.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import load_dataset
from cairn.feature_cache import CachedPipeline
from cairn.fusion.conformal import ConformalWrapper
from cairn.pipeline import CairnPipeline

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(it, **_kw):
        return it


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--features-cache", type=Path, default=None,
                    help="Path to a precomputed feature cache (built by "
                         "scripts/cache_pipeline_features.py).")
    args = ap.parse_args()

    pipeline = (
        CachedPipeline(args.features_cache, args.config)
        if args.features_cache
        else CairnPipeline.from_config_file(args.config)
    )
    samples = load_dataset(args.dataset)

    scores: list[float] = []
    labels: list[int] = []
    for s in tqdm(samples, desc="scoring", unit="sample"):
        det = pipeline.analyze(s.text)
        scores.append(float(det.p_llm))
        labels.append(int(s.label))

    wrapper = ConformalWrapper(alpha=args.alpha)
    wrapper.calibrate(np.asarray(scores), np.asarray(labels))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wrapper.save(args.out)
    print(
        f"calibrated on {len(scores)} samples (alpha={args.alpha}); "
        f"per-class={wrapper.threshold_per_class}; "
        f"wrote {args.out}"
    )


if __name__ == "__main__":
    main()

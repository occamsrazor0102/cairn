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
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import tempfile

import numpy as np
import sklearn

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


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
    ap.add_argument(
        "--manifest-out", type=Path, default=None,
        help="Optional path for an atomic calibration provenance manifest.",
    )
    args = ap.parse_args()

    pipeline = (
        CachedPipeline(args.features_cache, args.config)
        if args.features_cache
        else CairnPipeline.from_config_file(args.config)
    )
    samples = load_dataset(args.dataset)

    scores: list[float] = []
    labels: list[int] = []
    router_classes: list[str] = []
    for s in tqdm(samples, desc="scoring", unit="sample"):
        det, _results, rd = pipeline.analyze_detailed(s.text)
        scores.append(float(det.p_llm))
        labels.append(int(s.label))
        router_classes.append(rd.domain)

    wrapper = ConformalWrapper(alpha=args.alpha)
    wrapper.calibrate(
        np.asarray(scores), np.asarray(labels),
        router_classes=np.asarray(router_classes),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wrapper.save(args.out)
    if args.manifest_out is not None:
        class_counts = Counter(labels)
        router_counts = Counter(router_classes)
        threshold_per_class = wrapper.threshold_per_class or {}
        threshold_per_router = wrapper.threshold_per_router or {}
        manifest: dict[str, object] = {
            "dataset_path": str(args.dataset),
            "dataset_sha256": _sha256_file(args.dataset),
            "features_cache_path": (
                str(args.features_cache)
                if args.features_cache is not None
                else None
            ),
            "features_cache_sha256": (
                _sha256_file(args.features_cache)
                if args.features_cache is not None
                else None
            ),
            "alpha": float(wrapper.alpha),
            "n_calibration": len(scores),
            "class_counts": {
                str(label): int(count)
                for label, count in sorted(class_counts.items())
            },
            "router_counts": {
                str(router): int(count)
                for router, count in sorted(router_counts.items())
            },
            "threshold_per_class": {
                str(label): float(threshold)
                for label, threshold in sorted(threshold_per_class.items())
            },
            "threshold_per_router": {
                str(router): {
                    str(label): float(threshold)
                    for label, threshold in sorted(thresholds.items())
                }
                for router, thresholds in sorted(threshold_per_router.items())
            },
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(args.manifest_out, manifest)
    print(
        f"calibrated on {len(scores)} samples (alpha={args.alpha}); "
        f"per-class={wrapper.threshold_per_class}; "
        f"per-router={list((wrapper.threshold_per_router or {}).keys())}; "
        f"wrote {args.out}"
    )
    if args.manifest_out is not None:
        print(f"wrote {args.manifest_out}")


if __name__ == "__main__":
    main()

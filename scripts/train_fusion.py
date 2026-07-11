"""Train the EBM fusion model from a labeled JSONL dataset.

Usage:
    python scripts/train_fusion.py \
        --dataset data/seed_dataset.jsonl \
        --config configs/default.yaml \
        --out models/ebm_v1.pkl
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from cairn.config import load_config
from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import load_dataset
from cairn.feature_cache import CachedPipeline
from cairn.fusion.ebm import FeatureAssembler
from cairn.fusion.serialization import save_model as save_model_skops
from cairn.fusion.training import train_ebm
from cairn.pipeline import CairnPipeline

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(it, **_kw):
        return it


def build_training_assembler(
    config: dict,
    *,
    drop_features: set[str],
) -> FeatureAssembler:
    clamp = (config.get("fusion") or {}).get("word_count_clamp")
    return FeatureAssembler(
        drop_features=drop_features,
        word_count_clamp=int(clamp) if clamp is not None else None,
    )


def source_holdout_split(
    X: list[dict],
    y: list[int],
    samples: list,
    holdout_source: str,
) -> tuple[list[dict], list[dict], list[int], list[int]]:
    idx_train = [
        i
        for i, sample in enumerate(samples)
        if (sample.source or "").lower() != holdout_source.lower()
    ]
    idx_test = [
        i
        for i, sample in enumerate(samples)
        if (sample.source or "").lower() == holdout_source.lower()
    ]
    if not idx_train or not idx_test:
        raise SystemExit(
            f"--holdout-source {holdout_source!r} produced an empty split "
            f"(train={len(idx_train)}, test={len(idx_test)})"
        )
    return (
        [X[i] for i in idx_train],
        [X[i] for i in idx_test],
        [y[i] for i in idx_train],
        [y[i] for i in idx_test],
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--features-cache", type=Path, default=None,
                    help="Path to a precomputed feature cache (built by "
                         "scripts/cache_pipeline_features.py). When set, "
                         "the cache supplies detector outputs and the "
                         "config's calibrator is applied at read-time.")
    split_group = ap.add_mutually_exclusive_group()
    split_group.add_argument(
        "--holdout-source", default=None,
        help="LLM source to hold out entirely from training (zero-shot eval).",
    )
    split_group.add_argument(
        "--train-all", action="store_true",
        help="Train on the full dataset without creating an evaluation holdout.",
    )
    ap.add_argument(
        "--interactions", type=int, default=15,
        help="Number of pairwise EBM interaction terms. Default 15.",
    )
    ap.add_argument(
        "--drop-features", default="",
        help="Comma-separated FeatureAssembler keys to drop from the EBM "
             "input vector. Used for ablation experiments (e.g. "
             "'specdetect_total_energy,specdetect_low_band,...' to test "
             "whether the spectral signal features are noise).",
    )
    args = ap.parse_args()

    config = load_config(args.config)
    pipeline = (
        CachedPipeline(args.features_cache, args.config)
        if args.features_cache
        else CairnPipeline(config)
    )
    samples = load_dataset(args.dataset)
    drop_set: set[str] = {k.strip() for k in args.drop_features.split(",") if k.strip()}
    assembler = build_training_assembler(config, drop_features=drop_set)
    if drop_set:
        print(f"dropping {len(drop_set)} EBM features: {sorted(drop_set)}", flush=True)

    X: list[dict] = []
    y: list[int] = []
    for s in tqdm(samples, desc="featurizing", unit="sample"):
        _, results, rd = pipeline.analyze_detailed(s.text)
        vec = assembler.assemble(results, word_count=rd.word_count, domain=rd.domain)
        X.append(vec)
        y.append(int(s.label))

    if args.train_all:
        X_train, y_train = X, y
        X_test, y_test = [], []
    elif args.holdout_source:
        X_train, X_test, y_train, y_test = source_holdout_split(
            X, y, samples, args.holdout_source
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42
        )
    model = train_ebm(X_train, y_train, interactions=args.interactions)

    feature_names = list(
        getattr(model, "_cairn_feature_names", None) or X_train[0].keys()
    )
    clamp = (config.get("fusion") or {}).get("word_count_clamp")
    metadata = {
        "dataset_path": str(args.dataset),
        "dataset_sha256": _sha256_file(args.dataset),
        "features_cache_path": (
            str(args.features_cache) if args.features_cache is not None else None
        ),
        "features_cache_sha256": (
            _sha256_file(args.features_cache)
            if args.features_cache is not None
            else None
        ),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_balance_train": sum(y_train) / len(y_train),
        "label_balance_test": (
            sum(y_test) / len(y_test) if y_test else None
        ),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "interactions": int(args.interactions),
        "max_rounds": int(getattr(model, "max_rounds", 300)),
        "random_state": int(getattr(model, "random_state", 42)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "interpret_version": _distribution_version("interpret"),
        "skops_version": _distribution_version("skops"),
        "feature_names": feature_names,
        "dropped_features": sorted(drop_set),
        "word_count_clamp": int(clamp) if clamp is not None else None,
        "schema_version": "ebm-v8",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model_skops(model, args.out, metadata=metadata)
    print(f"trained on {len(X_train)}, held out {len(X_test)}, wrote {args.out}")


if __name__ == "__main__":
    main()

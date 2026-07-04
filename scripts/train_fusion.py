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
from pathlib import Path

from sklearn.model_selection import train_test_split

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
    ap.add_argument(
        "--holdout-source", default=None,
        help="LLM source to hold out entirely from training (zero-shot eval).",
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

    pipeline = (
        CachedPipeline(args.features_cache, args.config)
        if args.features_cache
        else CairnPipeline.from_config_file(args.config)
    )
    samples = load_dataset(args.dataset)
    drop_set: set[str] = {k.strip() for k in args.drop_features.split(",") if k.strip()}
    assembler = FeatureAssembler(drop_features=drop_set)
    if drop_set:
        print(f"dropping {len(drop_set)} EBM features: {sorted(drop_set)}", flush=True)

    X: list[dict] = []
    y: list[int] = []
    for s in tqdm(samples, desc="featurizing", unit="sample"):
        _, results, rd = pipeline.analyze_detailed(s.text)
        vec = assembler.assemble(results, word_count=rd.word_count, domain=rd.domain)
        X.append(vec)
        y.append(int(s.label))

    if args.holdout_source:
        idx_train = [i for i, s in enumerate(samples)
                     if (s.source or "").lower() != args.holdout_source.lower()]
        idx_test = [i for i, s in enumerate(samples)
                    if (s.source or "").lower() == args.holdout_source.lower()]
        if not idx_train or not idx_test:
            raise SystemExit(
                f"--holdout-source {args.holdout_source!r} produced an empty split "
                f"(train={len(idx_train)}, test={len(idx_test)})"
            )
        X_train = [X[i] for i in idx_train]
        y_train = [y[i] for i in idx_train]
        X_test = [X[i] for i in idx_test]
        y_test = [y[i] for i in idx_test]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42
        )
    model = train_ebm(X_train, y_train, interactions=args.interactions)

    dataset_bytes = args.dataset.read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()[:16]
    metadata = {
        "dataset_path": str(args.dataset),
        "dataset_hash": dataset_hash,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_balance_train": sum(y_train) / max(1, len(y_train)),
        "label_balance_test": sum(y_test) / max(1, len(y_test)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "interactions": getattr(model, "interactions", 15),
        "max_rounds": getattr(model, "max_rounds", 300),
        "schema_version": "ebm-v8",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model_skops(model, args.out, metadata=metadata)
    print(f"trained on {len(X_train)}, held out {len(X_test)}, wrote {args.out}")


if __name__ == "__main__":
    main()

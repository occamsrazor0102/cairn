"""Train the EBM fusion model from a labeled JSONL dataset.

Usage:
    python scripts/train_fusion.py \
        --dataset data/seed_dataset.jsonl \
        --config configs/default.yaml \
        --out models/ebm_v1.pkl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import load_dataset
from cairn.feature_cache import CachedPipeline
from cairn.fusion.ebm import FeatureAssembler
from cairn.fusion.training import save_model, train_ebm
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
    args = ap.parse_args()

    pipeline = (
        CachedPipeline(args.features_cache, args.config)
        if args.features_cache
        else CairnPipeline.from_config_file(args.config)
    )
    samples = load_dataset(args.dataset)
    assembler = FeatureAssembler()

    X: list[dict] = []
    y: list[int] = []
    for s in tqdm(samples, desc="featurizing", unit="sample"):
        _, results, rd = pipeline.analyze_detailed(s.text)
        vec = assembler.assemble(results, word_count=rd.word_count, domain=rd.domain)
        X.append(vec)
        y.append(int(s.label))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    model = train_ebm(X_train, y_train)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.out)
    print(f"trained on {len(X_train)}, held out {len(X_test)}, wrote {args.out}")


if __name__ == "__main__":
    main()

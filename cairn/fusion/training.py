# cairn/fusion/training.py
"""Utilities for training and persisting the EBM fusion model."""
import pickle
from pathlib import Path
from typing import Any

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    import numpy as np
    _HAS_INTERPRET = True
except ImportError:
    _HAS_INTERPRET = False


def train_ebm(X: list[dict], y: list[int], feature_names: list[str] | None = None) -> Any:
    """
    Train EBM on assembled feature vectors.
    X: list of feature dicts (from FeatureAssembler.assemble)
    y: list of labels (1=LLM, 0=human)
    Returns: trained EBM model

    Computes per-column means from finite training values and imputes
    NaNs with those means (instead of a global 0.5). The means are
    attached to the model as `_cairn_column_means` so inference can
    reuse the same imputation contract.
    """
    if not _HAS_INTERPRET:
        raise ImportError("interpretML not installed. Run: pip install 'cairn[fusion]'")

    if feature_names is None:
        feature_names = list(X[0].keys())

    X_arr = np.array([[row.get(f, float("nan")) for f in feature_names] for row in X])
    col_means: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        col = X_arr[:, j]
        finite = col[~np.isnan(col)]
        col_means[name] = float(finite.mean()) if finite.size else 0.5
    for j, name in enumerate(feature_names):
        # Idiomatic in-place NaN fill: index directly into X_arr so the write
        # is unambiguous (no intermediate `col` view that may or may not be
        # a copy depending on layout). After the loop X_arr must be NaN-free.
        X_arr[np.isnan(X_arr[:, j]), j] = col_means[name]
    assert not np.isnan(X_arr).any(), "NaN imputation incomplete before model.fit"
    y_arr = np.array(y)

    model = ExplainableBoostingClassifier(
        feature_names=feature_names,
        max_rounds=300,
        # 15 pairwise interactions (was 5). Length-bucket × per-detector
        # p_llm interactions are the prime motivator — verb=low sweep
        # showed length-conditional behaviour the previous 5-interaction
        # budget couldn't model.
        interactions=15,
        random_state=42,
    )
    model.fit(X_arr, y_arr)
    model._cairn_feature_names = feature_names  # attach for later use
    model._cairn_column_means = col_means
    return model


def save_model(model: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

# cairn/fusion/training.py
"""Utilities for training and persisting the EBM fusion model."""
import logging
import pickle
from pathlib import Path
from typing import Any

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    import numpy as np
    _HAS_INTERPRET = True
except ImportError:
    _HAS_INTERPRET = False


logger = logging.getLogger("cairn.fusion")


# Features whose meaning is monotonically associated with "more LLM-like":
# higher preamble severity / more matches, more LLM-fingerprint vocabulary
# hits per 1k tokens, more NSSI signals active, and so on. Wiring these as
# monotone-up constraints stops the EBM from learning non-monotonic shapes
# that contradict the underlying detector semantics — e.g. "very high
# preamble severity is somehow less LLM-like than medium severity" — and
# improves interpretability. Other features (e.g. *_p_llm, *_missing,
# domain indicators, length features) are intentionally unconstrained.
_MONOTONE_UP_FEATURES = (
    "preamble_severity",
    "preamble_n_matches",
    "fingerprint_hits_per_1000",
    "fingerprint_bigram_hits",
    "nssi_n_signals_active",
    "nssi_formulaic_density",
    "idi_score",
    "ps_meta_design_hits",
)


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

    monotone = [1 if name in _MONOTONE_UP_FEATURES else 0 for name in feature_names]
    try:
        model = ExplainableBoostingClassifier(
            feature_names=feature_names,
            max_rounds=300,
            # 15 pairwise interactions (was 5). Length-bucket × per-detector
            # p_llm interactions are the prime motivator — verb=low sweep
            # showed length-conditional behaviour the previous 5-interaction
            # budget couldn't model.
            interactions=15,
            random_state=42,
            monotone_constraints=monotone,
        )
    except TypeError as exc:
        # Only swallow the specific "unknown kwarg monotone_constraints" TypeError
        # raised by Python's argument dispatch on legacy interpret-ml builds.
        # Unrelated TypeErrors (numpy dtype mismatch, malformed feature_names,
        # etc.) must propagate so they don't masquerade as a missing-feature
        # warning.
        if "monotone_constraints" not in str(exc):
            raise
        import interpret
        logger.warning(
            "monotone_constraints unsupported in interpret-ml=%s; training without",
            getattr(interpret, "__version__", "unknown"),
        )
        model = ExplainableBoostingClassifier(
            feature_names=feature_names,
            max_rounds=300,
            interactions=15,
            random_state=42,
        )
        monotone = None
    model.fit(X_arr, y_arr)
    model._cairn_feature_names = feature_names  # attach for later use
    model._cairn_column_means = col_means
    # Always attach: None on the legacy-fallback path is the explicit
    # "constraints not applied by design" sentinel. Without this,
    # `getattr(model, "_cairn_monotone_constraints", None)` cannot
    # distinguish "trained-without-by-design" from "pre-Phase-5 artifact".
    model._cairn_monotone_constraints = monotone
    return model


def save_model(model: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

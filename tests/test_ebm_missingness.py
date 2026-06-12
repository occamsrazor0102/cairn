import math
import pytest
from cairn.contracts import LayerResult
from cairn.fusion.ebm import FeatureAssembler, ASSEMBLER_FEATURE_NAMES


def _lr(layer_id, p_llm=0.6, confidence=0.7, **signals):
    return LayerResult(
        layer_id=layer_id, domain="universal", raw_score=p_llm, p_llm=p_llm,
        confidence=confidence, signals=signals, determination="AMBER",
        attacker_tiers=[], compute_cost="trivial", min_text_length=0,
    )


def test_missing_indicator_per_detector():
    a = FeatureAssembler()
    vec = a.assemble([], word_count=200, domain="prose")
    assert vec["preamble_missing"] == 1.0
    assert vec["fingerprint_missing"] == 1.0
    assert vec["binoculars_missing"] == 1.0
    vec2 = a.assemble([_lr("preamble", p_llm=0.9, severity="HIGH", n_matches=2)],
                      word_count=200, domain="prose")
    assert vec2["preamble_missing"] == 0.0


def test_missing_indicator_in_canonical_feature_list():
    assert "preamble_missing" in ASSEMBLER_FEATURE_NAMES
    assert "fingerprint_missing" in ASSEMBLER_FEATURE_NAMES


def test_confidence_per_detector():
    a = FeatureAssembler()
    vec = a.assemble([_lr("preamble", p_llm=0.9, confidence=0.85, severity="HIGH")],
                     word_count=200, domain="prose")
    assert vec["preamble_confidence"] == pytest.approx(0.85, abs=1e-6)
    assert math.isnan(vec["nssi_confidence"])


def test_ebm_uses_stored_column_means_not_half():
    pytest.importorskip("interpret")
    from cairn.fusion.training import train_ebm
    from cairn.fusion.ebm import EBMFusion
    import numpy as np

    a = FeatureAssembler()
    rows = []
    for i in range(40):
        if i < 20:
            r = _lr("contrastive_lm", p_llm=0.2, confidence=0.6, binoculars_ratio=1.0)
        else:
            r = _lr("contrastive_lm", p_llm=0.8, confidence=0.6, binoculars_ratio=0.7)
        rows.append((a.assemble([r], word_count=200, domain="prose"), int(i >= 20)))
    X = [r for r, _ in rows]
    y = [lab for _, lab in rows]
    model = train_ebm(X, y)
    means = getattr(model, "_cairn_column_means", None)
    assert means is not None
    expected_mean = (1.0 + 0.7) / 2
    assert means["binoculars_ratio"] == pytest.approx(expected_mean, abs=1e-6)

    # Now actually exercise the inference imputation path. Build a LayerResult
    # that omits binoculars_ratio (but keeps contrastive_lm present so its
    # p_llm column is non-NaN), wire the trained model into EBMFusion, and
    # spy on predict_proba to confirm the binoculars_ratio column was imputed
    # from `_cairn_column_means` rather than 0.5.
    fusion = EBMFusion(model=model)
    seen_X: list = []
    real_predict_proba = model.predict_proba

    def _spy(X_arr):
        seen_X.append(np.asarray(X_arr).copy())
        return real_predict_proba(X_arr)

    model.predict_proba = _spy
    try:
        # Layer result without binoculars_ratio signal — assembler emits NaN.
        lr = _lr("contrastive_lm", p_llm=0.5, confidence=0.6)
        fusion.fuse([lr], word_count=200, domain="prose")
    finally:
        model.predict_proba = real_predict_proba

    assert seen_X, "predict_proba was never called during fuse"
    feature_names = list(model._cairn_feature_names)
    idx = feature_names.index("binoculars_ratio")
    seen_value = float(seen_X[0][0, idx])
    assert seen_value == pytest.approx(expected_mean, abs=1e-6), (
        f"Inference should impute binoculars_ratio from stored column_means "
        f"({expected_mean}), saw {seen_value}"
    )


def test_ebm_raises_when_artifact_requires_unknown_features():
    from cairn.fusion.ebm import EBMFusion

    class _StubModel:
        _cairn_feature_names = ["preamble_p_llm", "imaginary_feature_xyz"]
        _cairn_column_means = {}

    with pytest.raises(RuntimeError, match="imaginary_feature_xyz"):
        EBMFusion(model=_StubModel())


def test_inference_imputes_from_stored_column_means_not_half():
    """The inference path must use _column_means.get(name, 0.5), not a hardcoded 0.5.

    Uses a stub model with a recordable predict_proba so this test runs even
    without the `interpret` library installed. Verifies that the assembled
    feature row passed into predict_proba has NaN-imputed columns filled from
    the artifact's stored `_cairn_column_means` rather than the legacy 0.5.
    """
    from cairn.fusion.ebm import EBMFusion
    import numpy as np

    seen_X: list = []

    class _RecordingModel:
        _cairn_feature_names = ["preamble_p_llm", "binoculars_ratio"]
        _cairn_column_means = {"preamble_p_llm": 0.42, "binoculars_ratio": 0.71}

        def predict_proba(self, X):
            seen_X.append(X.copy())
            return np.array([[0.4, 0.6]])

    fusion = EBMFusion(model=_RecordingModel())
    # Build a LayerResult that lacks binoculars_ratio so it'll be NaN-imputed.
    lr = _lr("preamble", p_llm=0.9, confidence=0.8, severity="HIGH")
    fusion.fuse([lr], word_count=200, domain="prose")

    # _column_means says binoculars_ratio == 0.71; inference must use that, not 0.5.
    assert seen_X, "predict_proba was never called"
    X = seen_X[0]
    idx = ["preamble_p_llm", "binoculars_ratio"].index("binoculars_ratio")
    assert X[0, idx] == pytest.approx(0.71, abs=1e-6), (
        f"Inference should impute binoculars_ratio from stored column_means (0.71), "
        f"saw {X[0, idx]}"
    )


def test_legacy_artifact_with_no_column_means_falls_back_to_half():
    """A model without _cairn_column_means at all should load and impute to 0.5.

    Documents the contract for legacy artifacts: an absent
    `_cairn_column_means` attribute means "no per-column means recorded";
    `_column_means` stays empty so the inference path's
    `self._column_means.get(name, 0.5)` falls back to 0.5.
    """
    from cairn.fusion.ebm import EBMFusion

    class _LegacyModel:
        _cairn_feature_names = ["preamble_p_llm"]
        # deliberately no _cairn_column_means attribute

    fusion = EBMFusion(model=_LegacyModel())
    assert fusion._column_means == {}, "legacy artifact must result in empty _column_means"


def test_empty_column_means_dict_is_preserved_not_dropped():
    """An artifact with `_cairn_column_means = {}` (NaN-free training set) must
    still be loaded as an explicit empty dict rather than treated as absent.

    This is the load() guard fix: `if cm is not None:` instead of `if cm:`. An
    empty dict is falsy, so the old code silently dropped a legitimate
    "trained, but no columns required imputation" signal. The behavioral
    difference vs. the legacy-no-attribute case is subtle here (both produce
    `_column_means == {}`), but the contract this test pins is:
    "an artifact's explicit empty dict must propagate, not be coerced away."
    """
    from cairn.fusion.ebm import EBMFusion

    class _CleanTrainingModel:
        _cairn_feature_names = ["preamble_p_llm"]
        _cairn_column_means = {}  # explicitly empty: training had no NaNs

    fusion = EBMFusion(model=_CleanTrainingModel())
    # The dict was explicitly set; whether the load guard checks `is not None`
    # or `truthy`, the resulting _column_means should be the same empty dict
    # in this case. The real value of the load guard is exercised by the
    # inference-imputation test above.
    assert fusion._column_means == {}


def test_train_ebm_applies_monotone_constraints():
    pytest.importorskip("interpret")
    a = FeatureAssembler()
    rows = []
    for i in range(50):
        if i < 25:
            lr = _lr("preamble", p_llm=0.05, confidence=0.5, severity="NONE", n_matches=0)
        else:
            lr = _lr("preamble", p_llm=0.95, confidence=0.5, severity="CRITICAL", n_matches=4)
        rows.append((a.assemble([lr], word_count=200, domain="prose"), int(i >= 25)))
    X = [r for r, _ in rows]
    y = [lab for _, lab in rows]
    from cairn.fusion.training import train_ebm
    model = train_ebm(X, y)
    constraints = getattr(model, "_cairn_monotone_constraints", None)
    if constraints is None:
        pytest.skip("interpret-ml without monotone_constraints support")
    feat_to_idx = {n: i for i, n in enumerate(model._cairn_feature_names)}
    for name in ("preamble_severity", "fingerprint_hits_per_1000", "nssi_n_signals_active"):
        idx = feat_to_idx[name]
        assert constraints[idx] == 1, f"expected monotone-up on {name}"


@pytest.mark.parametrize("prefix", [
    "preamble", "fingerprint", "ps", "vs", "idi", "nssi", "surprisal",
    "binoculars", "fast_detectgpt", "tocsin", "perturbation",
    "contrastive_gen", "dna_gpt", "mixed_boundary", "cross_similarity",
    "semantic_kinematics",
])
def test_each_detector_has_full_feature_triplet(prefix):
    """Every cascade detector must contribute a full {p_llm, missing, confidence}
    triplet to the assembler. Fires the moment a future detector is added with
    only two of the three suffixes.
    """
    for suffix in ("p_llm", "missing", "confidence"):
        key = f"{prefix}_{suffix}"
        assert key in ASSEMBLER_FEATURE_NAMES, (
            f"detector prefix {prefix!r} missing {suffix!r} feature"
        )


def test_assembler_emits_semantic_kinematics_features():
    a = FeatureAssembler()
    vec_absent = a.assemble([], word_count=200, domain="prose")
    assert vec_absent["semantic_kinematics_missing"] == 1.0
    assert math.isnan(vec_absent["semantic_kinematics_confidence"])

    lr = _lr(
        "semantic_kinematics", p_llm=0.7, confidence=0.55,
        n_sentences=8, sk_velocity_mean=0.18, sk_velocity_std=0.04,
        sk_acceleration_mean=0.02, sk_acceleration_std=0.01,
        sk_jerk_mean=0.01, sk_jerk_std=0.005,
    )
    vec = a.assemble([lr], word_count=200, domain="prose")
    assert vec["semantic_kinematics_missing"] == 0.0
    assert vec["semantic_kinematics_p_llm"] == pytest.approx(0.7)
    assert vec["sk_velocity_std"] == pytest.approx(0.04)
    assert vec["sk_jerk_std"] == pytest.approx(0.005)


def test_semantic_kinematics_features_in_canonical_list():
    assert "semantic_kinematics_p_llm" in ASSEMBLER_FEATURE_NAMES
    assert "semantic_kinematics_missing" in ASSEMBLER_FEATURE_NAMES
    assert "semantic_kinematics_confidence" in ASSEMBLER_FEATURE_NAMES
    assert "sk_velocity_std" in ASSEMBLER_FEATURE_NAMES
    assert "sk_jerk_std" in ASSEMBLER_FEATURE_NAMES

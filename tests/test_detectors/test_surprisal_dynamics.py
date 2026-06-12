import math

import pytest

# Pure-Python feature helpers do not need torch — test them unconditionally.
from cairn.detectors.surprisal_dynamics import (
    _diveye_features,
    _moments,
    _lag1_autocorr,
)


def test_moments_basic():
    mean, std, skew, kurt = _moments([1.0, 2.0, 3.0, 4.0])
    assert mean == pytest.approx(2.5)
    assert std > 0
    assert isinstance(skew, float) and isinstance(kurt, float)


def test_moments_constant_series_is_degenerate():
    mean, std, skew, kurt = _moments([3.0, 3.0, 3.0])
    assert mean == pytest.approx(3.0)
    assert std == 0.0 and skew == 0.0 and kurt == 0.0


def test_lag1_autocorr_in_range():
    ac = _lag1_autocorr([1.0, 2.0, 1.0, 2.0, 1.0])
    assert -1.0 <= ac <= 1.0


def test_diveye_emits_nine_named_features():
    surprisals = [float(i % 5) + 0.3 for i in range(40)]
    feats = _diveye_features(surprisals)
    expected = {
        "surprisal_skew", "surprisal_kurtosis",
        "surprisal_d1_std", "surprisal_d1_absmean", "surprisal_d1_autocorr",
        "surprisal_d2_std", "surprisal_d2_absmean",
        "surprisal_moment_mean", "surprisal_moment_std",
    }
    assert set(feats) == expected
    assert all(isinstance(v, float) and not math.isnan(v) for v in feats.values())


# ---- Model-backed behavior (skipped without torch/transformers) ----

pytest.importorskip("transformers", reason="transformers not installed — skip Tier 2 tests")
pytest.importorskip("torch", reason="torch not installed — skip Tier 2 tests")

from cairn.detectors.surprisal_dynamics import SurprisalDynamicsDetector  # noqa: E402
from tests.fixtures.llm_samples import A0_CLINICAL_TASK  # noqa: E402
from tests.fixtures.human_samples import FORMAL_SOP  # noqa: E402


@pytest.fixture(scope="module")
def detector():
    # Uses GPT-2 small — downloads ~500MB on first run
    return SurprisalDynamicsDetector(model_name="gpt2")


def test_result_suppresses_heuristic_and_exposes_features(detector):
    result = detector.analyze(FORMAL_SOP, {})
    assert result.layer_id == "surprisal_dynamics"
    # Heuristic point estimate is suppressed (it inverted OOD); features feed
    # the EBM instead. UNCERTAIN — NOT SKIP — so the assembler keeps the row.
    assert result.determination == "UNCERTAIN"
    assert result.p_llm == 0.5
    assert result.confidence == 0.0
    # Legacy + DivEye signals both present.
    assert "late_volatility_ratio" in result.signals
    assert "surprisal_d1_std" in result.signals
    assert "surprisal_kurtosis" in result.signals


def test_compute_cost_is_moderate(detector):
    result = detector.analyze(FORMAL_SOP, {})
    assert result.compute_cost == "moderate"


def test_llm_and_human_both_produce_feature_floats(detector):
    llm_result = detector.analyze(A0_CLINICAL_TASK, {})
    human_result = detector.analyze(FORMAL_SOP, {})
    assert isinstance(llm_result.signals["late_volatility_ratio"], float)
    assert isinstance(human_result.signals["surprisal_d1_autocorr"], float)

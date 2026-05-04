import math
import json
import numpy as np
import pytest
from pathlib import Path

from cairn.fusion.conformal import ConformalWrapper


def _mk_scores(n_human=200, n_llm=200, seed=0):
    rng = np.random.default_rng(seed)
    human = rng.beta(2, 8, n_human)
    llm = rng.beta(8, 2, n_llm)
    scores = np.concatenate([human, llm])
    labels = np.concatenate([np.zeros(n_human, dtype=int), np.ones(n_llm, dtype=int)])
    return scores, labels


def test_calibrate_populates_per_class_thresholds():
    w = ConformalWrapper(alpha=0.1)
    scores, labels = _mk_scores()
    w.calibrate(scores, labels)
    assert w.threshold_per_class is not None
    assert set(w.threshold_per_class.keys()) == {0, 1}


def test_per_class_calibration_rejects_class_with_too_few_points():
    w = ConformalWrapper(alpha=0.1)
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.beta(2, 8, 200), rng.beta(8, 2, 5)])
    labels = np.concatenate([np.zeros(200, dtype=int), np.ones(5, dtype=int)])
    with pytest.raises(ValueError, match="too few class-1 calibration points"):
        w.calibrate(scores, labels)


def test_predict_set_uses_per_class_thresholds():
    w = ConformalWrapper(alpha=0.1)
    scores, labels = _mk_scores()
    w.calibrate(scores, labels)
    assert w.threshold_per_class[0] != pytest.approx(w.threshold_per_class[1], abs=1e-6)
    assert w.predict_set(0.99) == ["LLM"]
    assert w.predict_set(0.01) == ["human"]


def test_save_and_load_round_trips_mondrian(tmp_path):
    w = ConformalWrapper(alpha=0.1)
    scores, labels = _mk_scores()
    w.calibrate(scores, labels)
    p = tmp_path / "cal.json"
    w.save(p)
    w2 = ConformalWrapper(alpha=0.5)
    w2.load(p)
    assert w2.threshold_per_class is not None
    assert w2.predict_set(0.5) == w.predict_set(0.5)


def test_load_v1_artifact_lifts_threshold_to_both_classes(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(
        '{"alpha": 0.05, "threshold": 0.42, "n_calibration": 100, '
        '"schema": "binary-split-conformal-v1"}'
    )
    w = ConformalWrapper()
    w.load(p)
    assert w.calibrated
    assert w.threshold_per_class == {0: 0.42, 1: 0.42}


def test_calibrate_failure_leaves_wrapper_uncalibrated():
    """A failed calibration must not leave the wrapper in a half-calibrated state."""
    w = ConformalWrapper(alpha=0.1)
    rng = np.random.default_rng(0)
    # 200 human, 5 LLM — class 1 too few
    scores = np.concatenate([rng.beta(2, 8, 200), rng.beta(8, 2, 5)])
    labels = np.concatenate([np.zeros(200, dtype=int), np.ones(5, dtype=int)])
    with pytest.raises(ValueError, match="too few class-1 calibration points"):
        w.calibrate(scores, labels)
    assert not w.calibrated, "wrapper must report uncalibrated after a failed calibrate()"
    assert w.threshold_per_class is None, "per-class thresholds must remain None after failure"
    # predict_set should fall back to the all-labels honest abstention, NOT silently
    # use a marginal threshold from a partial calibration
    assert set(w.predict_set(0.5)) == {"human", "LLM"}


def test_load_v2_artifact_without_threshold_per_class_raises(tmp_path):
    """A v2-schema artifact missing the threshold_per_class key is malformed; load() must raise."""
    p = tmp_path / "malformed_v2.json"
    p.write_text(json.dumps({
        "alpha": 0.05,
        "threshold": 0.42,
        "n_calibration": 100,
        "schema": "binary-mondrian-conformal-v2",
        # threshold_per_class deliberately absent
    }))
    w = ConformalWrapper()
    with pytest.raises(ValueError, match="threshold_per_class"):
        w.load(p)


def test_load_v2_artifact_with_empty_threshold_per_class_loads_empty(tmp_path):
    """Empty threshold_per_class dict must not silently fall back to v1-lift.

    Documents the empty-dict edge case: with `if tpc is not None` (rather than
    truthy `if tpc`), an empty dict in a v2 artifact loads as an empty per-class
    map rather than silently being filled by the v1-lift `{0: t, 1: t}`.
    """
    p = tmp_path / "empty_tpc_v2.json"
    p.write_text(json.dumps({
        "alpha": 0.05,
        "threshold": 0.42,
        "n_calibration": 100,
        "threshold_per_class": {},  # empty dict — must NOT fall through to v1-lift
        "schema": "binary-mondrian-conformal-v2",
    }))
    w = ConformalWrapper()
    w.load(p)
    # Empty dict means we have NO per-class data; the wrapper should NOT silently invent
    # {0: t, 1: t} from the global threshold (that's the v1-lift behavior, only for v1 schema).
    assert w.threshold_per_class == {}


# --- Phase 3: router-conditional Mondrian conformal -----------------------


def test_calibrate_with_router_classes_populates_per_router_thresholds():
    w = ConformalWrapper(alpha=0.1)
    rng = np.random.default_rng(0)
    n = 200
    scores = np.concatenate([
        rng.beta(2, 8, n), rng.beta(8, 2, n),
        rng.beta(2, 6, n), rng.beta(6, 2, n),
    ])
    labels = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)]).astype(int)
    routers = np.array(["prose"]*(2*n) + ["prompt"]*(2*n))
    w.calibrate(scores, labels, router_classes=routers)
    assert w.threshold_per_router is not None
    assert set(w.threshold_per_router.keys()) == {"prose", "prompt"}
    for rc in ("prose", "prompt"):
        assert set(w.threshold_per_router[rc].keys()) == {0, 1}


def test_underpowered_router_cell_falls_back_to_class_only():
    w = ConformalWrapper(alpha=0.1)
    rng = np.random.default_rng(0)
    n = 200
    scores = np.concatenate([rng.beta(2, 8, n), rng.beta(8, 2, n), rng.beta(2, 8, 10)])
    labels = np.concatenate([np.zeros(n), np.ones(n), np.zeros(10)]).astype(int)
    routers = np.array(["prose"]*(2*n) + ["mixed"]*10)
    w.calibrate(scores, labels, router_classes=routers)
    assert "mixed" not in (w.threshold_per_router or {})

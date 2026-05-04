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

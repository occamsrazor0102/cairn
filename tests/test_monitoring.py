"""Drift monitor + meta-detector tests."""
import sqlite3

import pytest

from cairn.contracts import LayerResult
from cairn.monitoring.drift import DriftMonitor, _kl_symmetric
from cairn.monitoring.meta_detector import MetaDetector
from cairn.monitoring.store import DriftStore


def _layer(p, det_id="preamble", det="AMBER"):
    return LayerResult(
        layer_id=det_id, domain="universal", raw_score=p, p_llm=p,
        confidence=0.8, signals={}, determination=det,
        attacker_tiers=["A0"], compute_cost="cheap", min_text_length=0,
    )


def test_kl_symmetric_zero_for_identical():
    p = [1.0, 2.0, 3.0, 4.0]
    assert _kl_symmetric(p, p) == pytest.approx(0.0, abs=1e-6)


def test_kl_symmetric_positive_for_different():
    assert _kl_symmetric([10, 0, 0, 0], [0, 0, 0, 10]) > 1.0


def test_drift_monitor_fires_on_population_shift(tmp_path):
    import time as _time
    mon = DriftMonitor(tmp_path, {"drift_monitoring": {"window_size": 10}})
    alerts = []
    for _ in range(10):
        alerts.extend(mon.record(0.95, "RED", {}))
    assert any("POPULATION_DRIFT" in a for a in alerts)
    # Alert file is written on a daemon thread; poll briefly.
    path = tmp_path / "alerts.jsonl"
    for _ in range(50):  # up to ~500ms
        if path.exists() and path.stat().st_size > 0:
            break
        _time.sleep(0.01)
    assert path.exists()


def test_drift_monitor_no_alerts_on_balanced(tmp_path):
    mon = DriftMonitor(tmp_path, {"drift_monitoring": {"window_size": 20}})
    alerts = []
    for i in range(20):
        alerts.extend(mon.record(0.35 + (0.02 * (i % 5)), "YELLOW", {}))
    assert not any("POPULATION_DRIFT" in a for a in alerts)


def test_drift_monitor_feature_drift(tmp_path):
    mon = DriftMonitor(tmp_path, {"drift_monitoring": {"window_size": 10, "kl_threshold": 0.05}})
    baseline = [{"x": 0.1} for _ in range(50)]
    mon.set_baseline(baseline)
    for _ in range(10):
        mon.record(0.5, "AMBER", {"x": 0.9})
    alerts = mon.check_drift()  # window may have flushed; call explicitly if needed
    # After flush, _observations is empty — re-populate and check:
    for _ in range(10):
        mon._observations.append({"p_llm": 0.5, "determination": "AMBER",
                                  "features": {"x": 0.9}, "confirmed": None,
                                  "timestamp": "2026-01-01T00:00:00"})
    alerts = mon.check_drift()
    assert any("FEATURE_DRIFT" in a for a in alerts)


def test_drift_monitor_calibration_drift(tmp_path):
    mon = DriftMonitor(tmp_path, {"drift_monitoring": {"window_size": 100, "ece_threshold": 0.10}})
    # Predictions 0.9 with true label 0 → miscalibrated
    for _ in range(30):
        mon._observations.append({"p_llm": 0.9, "determination": "RED",
                                  "features": {}, "confirmed": 0,
                                  "timestamp": "2026-01-01T00:00:00"})
    alerts = mon.check_drift()
    assert any("CALIBRATION_DRIFT" in a for a in alerts)


def test_meta_detector_stable_verdict():
    meta = MetaDetector(window_size=50)
    meta.set_baseline({"preamble": 0.5})
    for _ in range(30):
        meta.record(_layer(0.55))
    health = meta.health()
    assert health["preamble"].verdict == "stable"


def test_meta_detector_flags_drift():
    meta = MetaDetector(window_size=50, mean_shift_threshold=0.10)
    meta.set_baseline({"preamble": 0.5})
    for _ in range(30):
        meta.record(_layer(0.85))
    health = meta.health()
    assert health["preamble"].verdict == "drifting"


def test_meta_detector_flags_degradation():
    meta = MetaDetector(window_size=100, min_accuracy=0.80)
    # High-confidence LLM prediction but all confirmed human
    for _ in range(30):
        meta.record(_layer(0.9), confirmed_label=0)
    health = meta.health()
    assert health["preamble"].verdict == "degrading"
    assert health["preamble"].accuracy_vs_confirmed is not None


def test_meta_detector_ignores_skip():
    meta = MetaDetector()
    meta.record(_layer(0.5, det="SKIP"))
    assert meta.health() == {}


# --- Persistent DriftMonitor (B2) -----------------------------------------

def test_drift_observations_survive_reinstantiation(tmp_path):
    """Sidecar restart simulation: recorded observations persist across DriftMonitor instances."""
    db_path = tmp_path / "drift.sqlite3"
    cfg = {"drift_monitoring": {"window_size": 1000}}

    mon1 = DriftMonitor(tmp_path, cfg, db_path=db_path)
    for i in range(50):
        mon1.record(0.5 + (i % 5) * 0.01, "AMBER", {"x": 0.4 + (i % 3) * 0.05})
    # Sanity: first instance sees its writes back through check_drift hydration.
    mon1._observations = []  # force hydration path
    alerts1 = mon1.check_drift()
    assert mon1._observations, "first DriftMonitor should hydrate from sqlite"
    assert len(mon1._observations) == 50
    # Hand-control over to a freshly-constructed monitor against the same db.
    mon2 = DriftMonitor(tmp_path, cfg, db_path=db_path)
    assert mon2._observations == [], "fresh monitor starts with empty in-memory list"
    # check_drift hydrates from sqlite before evaluating; no new records.
    mon2.check_drift()
    assert len(mon2._observations) == 50, (
        "fresh monitor must see the 50 persisted observations before any new record()"
    )
    # And the alerts logic still runs over the persisted window.
    _ = alerts1  # alerts may or may not fire; the persistence is the contract.


def test_failed_drift_write_does_not_raise(tmp_path, monkeypatch):
    """Best-effort: a sqlite OperationalError inside the store must not surface."""
    db_path = tmp_path / "drift.sqlite3"
    mon = DriftMonitor(tmp_path, {"drift_monitoring": {"window_size": 1000}}, db_path=db_path)

    def _boom(self, observation):
        raise sqlite3.OperationalError("synthetic disk error")

    monkeypatch.setattr(DriftStore, "record", _boom)
    # Must not raise.
    alerts = mon.record(0.7, "AMBER", {"x": 0.42})
    assert isinstance(alerts, list)


def test_in_memory_fallback_when_store_path_none(tmp_path):
    """Regression: db_path=None preserves the original list-backed behavior."""
    mon = DriftMonitor(tmp_path, {"drift_monitoring": {"window_size": 1000}})
    assert mon._store is None
    for i in range(10):
        mon.record(0.5, "AMBER", {"x": 0.1 * i})
    assert len(mon._observations) == 10
    # check_drift reads directly from the list — no hydration step.
    mon.check_drift()
    assert len(mon._observations) == 10

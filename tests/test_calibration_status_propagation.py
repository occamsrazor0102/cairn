"""Regression: Determination.calibration_status must reflect the pipeline that
produced it.

decision.py::decide() builds every Determination without setting
calibration_status, so it falls back to the dataclass default ("heuristic").
The pipeline computes the real status (heuristic / fusion-only / calibrated)
but historically never stamped it onto the object handed back to callers — so
a fully calibrated EBM + conformal run reported calibration_status="heuristic"
on its determination (and in report.py), contradicting /health. These tests
pin the invariant: the determination carries the pipeline's status.
"""
from pathlib import Path

from cairn.config import load_config
from cairn.pipeline import CairnPipeline

_CFG = Path(__file__).parent.parent / "configs" / "screening.yaml"
_TEXT = "This is a sufficiently long sample sentence for the cascade to run. " * 3


class _LoadedConformal:
    calibrated = True


class _LoadedFusion:
    _model = object()
    _conformal = _LoadedConformal()


def test_determination_reflects_pipeline_calibration_status():
    """If the pipeline loaded calibrated artifacts, the determination must say
    so — not the dataclass default. We simulate the loaded state by setting the
    pinned status (artifacts are heavy; the propagation is what's under test)."""
    p = CairnPipeline(load_config(_CFG))
    p._calibration_status = "calibrated"
    det = p.analyze(_TEXT)
    assert det.calibration_status == "calibrated", (
        "Determination.calibration_status must mirror the pipeline status, "
        f"got {det.calibration_status!r}"
    )


def test_determination_calibration_status_consistent_by_default():
    """A heuristic pipeline (no model) must report heuristic on the
    determination too — and always agree with the pipeline's own status."""
    p = CairnPipeline(load_config(_CFG))
    det = p.analyze(_TEXT)
    assert det.calibration_status == p._calibration_status == "heuristic"


def test_declared_detector_calibration_load_failure_degrades_aggregate_status(
    tmp_path,
    monkeypatch,
):
    bad = tmp_path / "bad-calibration.json"
    bad.write_text("{not-json", encoding="utf-8")
    cfg = load_config(_CFG)
    cfg["fusion"] = {"detector_calibration_path": str(bad)}
    monkeypatch.setattr(
        CairnPipeline,
        "_build_fusion",
        staticmethod(lambda _config: _LoadedFusion()),
    )

    pipeline = CairnPipeline(cfg)

    assert pipeline._detector_calibrator is None
    assert pipeline.detector_calibration_state == {
        "state": "load-failed",
        "path": str(bad),
        "error": pipeline.detector_calibration_state["error"],
    }
    assert "Expecting property name" in pipeline.detector_calibration_state["error"]
    assert pipeline._calibration_status == "degraded"


def test_no_detector_calibration_path_preserves_existing_status_semantics(monkeypatch):
    cfg = load_config(_CFG)
    cfg["fusion"] = {}
    monkeypatch.setattr(
        CairnPipeline,
        "_build_fusion",
        staticmethod(lambda _config: _LoadedFusion()),
    )

    pipeline = CairnPipeline(cfg)

    assert pipeline.detector_calibration_state == {
        "state": "not-configured",
        "path": None,
        "error": None,
    }
    assert pipeline._calibration_status == "calibrated"

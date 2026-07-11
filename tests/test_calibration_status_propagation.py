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
import json
from pathlib import Path

import pytest

from cairn.calibration import CalibrationMetadata, DetectorCalibrator
from cairn.config import load_config
from cairn.contracts import LayerResult
from cairn.detectors.token_cohesiveness import TOCSIN_COMBINED_SIGNATURE
from cairn.pipeline import CairnPipeline

_CFG = Path(__file__).parent.parent / "configs" / "screening.yaml"
_TEXT = "This is a sufficiently long sample sentence for the cascade to run. " * 3


class _LoadedConformal:
    calibrated = True


class _LoadedFusion:
    _model = object()
    _conformal = _LoadedConformal()


def _write_detector_calibration(
    path: Path,
    *,
    token_schema: str | None,
    token_signature: str = TOCSIN_COMBINED_SIGNATURE,
) -> None:
    calibrator = DetectorCalibrator()
    calibrator.fit("preamble", [0.0, 1.0], [0, 1])
    if token_schema in {"v1", "v1-extra-identity"}:
        calibrator.fit("token_cohesiveness", [-0.4, 0.1], [0, 1])
    elif token_schema == "v2":
        calibrator.fit(
            "token_cohesiveness",
            [-0.4, -0.2, 0.0, 0.1],
            [0, 0, 1, 1],
            metadata=CalibrationMetadata(
                detector_signature=token_signature,
                raw_channel="combined_w",
                training_dataset_hash="sha256:test",
                weighting_policy={"policy": "uniform"},
            ),
        )
    calibrator.save(path)
    if token_schema == "v1-extra-identity":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["token_cohesiveness"].update(
            {
                "detector_signature": TOCSIN_COMBINED_SIGNATURE,
                "raw_channel": "combined_w",
            }
        )
        path.write_text(json.dumps(payload), encoding="utf-8")


def _unsigned_preamble() -> LayerResult:
    return LayerResult(
        layer_id="preamble",
        domain="universal",
        raw_score=0.5,
        p_llm=0.5,
        confidence=1.0,
        signals={},
        determination="AMBER",
        attacker_tiers=["A0"],
        compute_cost="test",
        min_text_length=0,
    )


def _signed_tocsin() -> LayerResult:
    return LayerResult(
        layer_id="token_cohesiveness",
        domain="universal",
        raw_score=0.0,
        p_llm=0.37,
        confidence=1.0,
        signals={
            "calibration_signature": TOCSIN_COMBINED_SIGNATURE,
            "calibration_raw_channel": "combined_w",
        },
        determination="AMBER",
        attacker_tiers=["A0"],
        compute_cost="test",
        min_text_length=0,
    )


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


@pytest.mark.parametrize(
    ("token_schema", "token_signature"),
    [
        ("v1", TOCSIN_COMBINED_SIGNATURE),
        ("v1-extra-identity", TOCSIN_COMBINED_SIGNATURE),
        ("v2", "stale-tocsin-signature"),
    ],
)
def test_enabled_tocsin_incompatible_artifact_degrades_without_disabling_legacy_entries(
    tmp_path,
    monkeypatch,
    token_schema,
    token_signature,
):
    artifact = tmp_path / "mixed-calibration.json"
    _write_detector_calibration(
        artifact,
        token_schema=token_schema,
        token_signature=token_signature,
    )
    cfg = load_config(_CFG)
    cfg["detectors"]["token_cohesiveness"]["enabled"] = True
    cfg.setdefault("fusion", {})["detector_calibration_path"] = str(artifact)
    monkeypatch.setattr(
        CairnPipeline,
        "_build_fusion",
        staticmethod(lambda _config: _LoadedFusion()),
    )

    pipeline = CairnPipeline(cfg)

    assert pipeline.detector_calibration_state["state"] == "loaded-incompatible"
    assert pipeline._calibration_status == "degraded"
    assert pipeline.calibration_health() == {
        "model_loaded": True,
        "conformal_loaded": True,
        "calibration_status": "degraded",
        "detector_calibration": {
            "state": "loaded-incompatible",
            "error": "artifact-incompatible",
        },
    }

    # The incompatible identity-aware entry remains fail-closed, while an
    # unrelated unsigned v1 calibration in the same artifact still works.
    preamble = _unsigned_preamble()
    assert pipeline._detector_calibrator.apply(preamble) is True
    assert preamble.calibrator_applied is True
    tocsin = _signed_tocsin()
    assert pipeline._detector_calibrator.apply(tocsin) is False
    assert tocsin.determination == "SKIP"
    assert "calibrator_signature_mismatch" in tocsin.signals


def test_matching_v2_tocsin_artifact_retains_calibrated_status(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "matching-calibration.json"
    _write_detector_calibration(artifact, token_schema="v2")
    cfg = load_config(_CFG)
    cfg["detectors"]["token_cohesiveness"]["enabled"] = True
    cfg.setdefault("fusion", {})["detector_calibration_path"] = str(artifact)
    monkeypatch.setattr(
        CairnPipeline,
        "_build_fusion",
        staticmethod(lambda _config: _LoadedFusion()),
    )

    pipeline = CairnPipeline(cfg)

    assert pipeline.detector_calibration_state["state"] == "loaded"
    assert pipeline._calibration_status == "calibrated"


@pytest.mark.parametrize("mode", ["disabled", "entry-absent"])
def test_tocsin_compatibility_check_is_scoped_to_enabled_present_entry(
    tmp_path,
    monkeypatch,
    mode,
):
    artifact = tmp_path / "scoped-calibration.json"
    _write_detector_calibration(
        artifact,
        token_schema="v1" if mode == "disabled" else None,
    )
    cfg = load_config(_CFG)
    cfg["detectors"]["token_cohesiveness"]["enabled"] = mode != "disabled"
    cfg.setdefault("fusion", {})["detector_calibration_path"] = str(artifact)
    monkeypatch.setattr(
        CairnPipeline,
        "_build_fusion",
        staticmethod(lambda _config: _LoadedFusion()),
    )

    pipeline = CairnPipeline(cfg)

    assert pipeline.detector_calibration_state["state"] == "loaded"
    assert pipeline._calibration_status == "calibrated"

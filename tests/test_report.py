import json
import pytest
from cairn.report import build_json_report, build_text_report
from cairn.contracts import Determination

@pytest.fixture
def sample_determination():
    return Determination(label="AMBER", p_llm=0.62, confidence_interval=(0.48, 0.73),
        prediction_set=["YELLOW", "AMBER"], reason="High binoculars ratio",
        top_features=[("binoculars_ratio", 1.4)], override_applied=False,
        detectors_run=["preamble", "fingerprint_vocab"], cascade_phases=[1], mixed_report=None)

def test_json_report_is_serializable(sample_determination):
    report = build_json_report(sample_determination, submission_id="sub_001")
    json_str = json.dumps(report)
    assert "AMBER" in json_str and "sub_001" in json_str

def test_text_report_is_string(sample_determination):
    report = build_text_report(sample_determination)
    assert "AMBER" in report and "0.62" in report

def test_json_report_has_required_keys(sample_determination):
    report = build_json_report(sample_determination)
    for key in ["determination", "p_llm", "confidence_interval", "detectors_run",
                "layer_results", "feature_contributions"]:
        assert key in report

def test_json_report_serializes_layer_results():
    from cairn.contracts import LayerResult
    lr = LayerResult(layer_id="preamble", domain="universal", raw_score=0.8,
        p_llm=0.9, confidence=0.7, signals={"severity": "HIGH"},
        determination="RED", attacker_tiers=["A0"], compute_cost="trivial",
        min_text_length=0,
        spans=[{"start": 0, "end": 10, "kind": "preamble", "note": "match"}])
    det = Determination(label="RED", p_llm=0.9, confidence_interval=(0.8, 0.95),
        prediction_set=["RED"], reason="t", top_features=[("preamble", 1.0)],
        override_applied=True, detectors_run=["preamble"], cascade_phases=[1],
        mixed_report=None, layer_results=[lr],
        feature_contributions={"preamble": 1.0, "fingerprint_vocab": 0.2})
    report = build_json_report(det)
    assert len(report["layer_results"]) == 1
    layer = report["layer_results"][0]
    for key in ["layer_id", "domain", "raw_score", "p_llm", "confidence",
                "determination", "signals", "compute_cost"]:
        assert key in layer
    assert layer["layer_id"] == "preamble"
    assert layer["signals"] == {"severity": "HIGH"}
    assert layer["spans"] == [{"start": 0, "end": 10, "kind": "preamble", "note": "match"}]
    assert report["feature_contributions"]["preamble"] == 1.0
    assert report["feature_contributions"]["fingerprint_vocab"] == 0.2


def test_json_report_emits_calibration_fields_for_calibrated_singleton():
    """Calibrated singleton: coverage applies, no abstention, calibrated status."""
    det = Determination(
        label="RED",
        p_llm=0.91,
        confidence_interval=(0.85, 0.96),
        prediction_set=["RED"],
        reason="calibrated singleton",
        top_features=[("ebm_score", 1.2)],
        override_applied=False,
        detectors_run=["preamble", "fingerprint_vocab"],
        cascade_phases=[1, 2],
        mixed_report=None,
        conformal_set=["LLM"],
        conformal_alpha=0.1,
        fusion_mode="ebm",
        coverage_guarantee_applies=True,
        abstention_reason=None,
        calibration_status="calibrated",
        warnings=[],
    )
    report = build_json_report(det, submission_id="sub_calibrated")
    # The fix: all four honesty fields must reach the JSON envelope.
    assert "coverage_guarantee_applies" in report
    assert "abstention_reason" in report
    assert "calibration_status" in report
    assert "warnings" in report
    assert report["coverage_guarantee_applies"] is True
    assert report["abstention_reason"] is None
    assert report["calibration_status"] == "calibrated"
    assert report["warnings"] == []
    # Round-trips through json.dumps without coercion surprises.
    assert json.loads(json.dumps(report))["abstention_reason"] is None


def test_json_report_emits_calibration_fields_for_conformal_empty():
    """Conformal-empty (OOD) path: coverage voided, abstention reason populated."""
    det = Determination(
        label="UNCERTAIN",
        p_llm=0.5,
        confidence_interval=(0.4, 0.6),
        prediction_set=["UNCERTAIN"],
        reason="empty conformal set - OOD",
        top_features=[],
        override_applied=False,
        detectors_run=["preamble"],
        cascade_phases=[1],
        mixed_report=None,
        conformal_set=[],
        conformal_alpha=0.1,
        fusion_mode="ebm",
        coverage_guarantee_applies=False,
        abstention_reason="conformal_empty",
        calibration_status="heuristic",
        warnings=["empty conformal set; abstaining"],
    )
    report = build_json_report(det, submission_id="sub_ood")
    assert "coverage_guarantee_applies" in report
    assert "abstention_reason" in report
    assert "calibration_status" in report
    assert "warnings" in report
    assert report["coverage_guarantee_applies"] is False
    assert report["abstention_reason"] == "conformal_empty"
    assert report["calibration_status"] == "heuristic"
    assert report["warnings"] == ["empty conformal set; abstaining"]
    # Defensive copy: mutating the input list must not bleed into the report.
    det.warnings.append("mutated after build")
    assert report["warnings"] == ["empty conformal set; abstaining"]

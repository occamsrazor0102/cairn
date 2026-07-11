"""Behavior tests for detector-calibration compatibility guards."""

import copy
import json
import math

import pytest

from cairn.calibration import CalibrationMetadata, DetectorCalibrator
from cairn.contracts import LayerResult
from cairn.detectors.token_cohesiveness import TOCSIN_COMBINED_SIGNATURE


def layer(
    *,
    raw: float,
    layer_id: str = "token_cohesiveness",
    signature: str | None = TOCSIN_COMBINED_SIGNATURE,
    raw_channel: str | None = "combined_w",
    determination: str = "AMBER",
) -> LayerResult:
    signals = {"native_evidence": "preserve-me"}
    if signature is not None:
        signals["calibration_signature"] = signature
    if raw_channel is not None:
        signals["calibration_raw_channel"] = raw_channel
    return LayerResult(
        layer_id=layer_id,
        domain="universal",
        raw_score=raw,
        p_llm=0.37,
        confidence=0.8,
        signals=signals,
        determination=determination,
        attacker_tiers=["A0"],
        compute_cost="moderate",
        min_text_length=40,
    )


def fitted_versioned() -> DetectorCalibrator:
    calibrator = DetectorCalibrator()
    calibrator.fit(
        "token_cohesiveness",
        [-0.4, -0.2, 0.0, 0.1],
        [0, 0, 1, 1],
        metadata=CalibrationMetadata(
            detector_signature=TOCSIN_COMBINED_SIGNATURE,
            raw_channel="combined_w",
            training_dataset_hash="sha256:abc",
            weighting_policy={"policy": "source-length-balanced-v1"},
        ),
    )
    return calibrator


def write_versioned_artifact(path, identity: dict[str, object]) -> None:
    entry = {
        "X": [-0.4, -0.2, 0.0, 0.1],
        "y": [0.0, 0.0, 1.0, 1.0],
        "schema": "isotonic-thresholds-v2",
        "support_min": -0.4,
        "support_max": 0.1,
        "training_rows": 4,
        "class_counts": {"0": 2, "1": 2},
        "training_dataset_hash": "sha256:abc",
        "weighting_policy": {"policy": "source-length-balanced-v1"},
        **identity,
    }
    path.write_text(
        json.dumps({"token_cohesiveness": entry}),
        encoding="utf-8",
    )


def test_current_signature_rejects_legacy_entry_before_clipping():
    legacy = DetectorCalibrator()
    legacy.fit("token_cohesiveness", [0.0, 0.01], [0, 1])
    result = layer(raw=0.005)

    assert legacy.apply(result) is False
    assert result.calibrator_applied is False
    assert result.determination == "SKIP"
    assert result.signals["calibrator_signature_mismatch"] == {
        "result_signature": TOCSIN_COMBINED_SIGNATURE,
        "artifact_signature": None,
        "result_raw_channel": "combined_w",
        "artifact_raw_channel": None,
    }
    assert result.raw_score == pytest.approx(0.005)
    assert result.p_llm == pytest.approx(0.37)
    assert result.signals["native_evidence"] == "preserve-me"


@pytest.mark.parametrize(
    ("signature", "raw_channel"),
    [
        (None, "combined_w"),
        (TOCSIN_COMBINED_SIGNATURE, None),
        ("wrong", "combined_w"),
        (TOCSIN_COMBINED_SIGNATURE, "wrong"),
    ],
)
def test_versioned_entry_rejects_missing_or_mismatched_result_identity(
    signature, raw_channel
):
    result = layer(raw=0.0, signature=signature, raw_channel=raw_channel)

    assert fitted_versioned().apply(result) is False
    assert result.calibrator_applied is False
    assert result.determination == "SKIP"
    assert result.signals["calibrator_signature_mismatch"] == {
        "result_signature": signature,
        "artifact_signature": TOCSIN_COMBINED_SIGNATURE,
        "result_raw_channel": raw_channel,
        "artifact_raw_channel": "combined_w",
    }


@pytest.mark.parametrize(
    ("artifact_signature", "artifact_channel"),
    [
        (None, None),
        ("", "combined_w"),
        (TOCSIN_COMBINED_SIGNATURE, ""),
    ],
)
def test_malformed_versioned_fit_cannot_apply_as_legacy(
    artifact_signature, artifact_channel
):
    calibrator = DetectorCalibrator()
    calibrator.fit(
        "token_cohesiveness",
        [-0.4, -0.2, 0.0, 0.1],
        [0, 0, 1, 1],
        metadata=CalibrationMetadata(
            detector_signature=artifact_signature,
            raw_channel=artifact_channel,
            training_dataset_hash="sha256:abc",
            weighting_policy={"policy": "source-length-balanced-v1"},
        ),
    )
    result = layer(
        raw=0.0,
        signature=artifact_signature,
        raw_channel=artifact_channel,
    )

    assert calibrator.apply(result) is False
    assert result.calibrator_applied is False
    assert result.determination == "SKIP"
    assert result.signals["calibrator_signature_mismatch"] == {
        "result_signature": artifact_signature,
        "artifact_signature": artifact_signature,
        "result_raw_channel": artifact_channel,
        "artifact_raw_channel": artifact_channel,
    }
    assert result.raw_score == pytest.approx(0.0)
    assert result.p_llm == pytest.approx(0.37)
    assert result.signals["native_evidence"] == "preserve-me"


@pytest.mark.parametrize(
    ("identity", "artifact_signature", "artifact_channel"),
    [
        ({}, None, None),
        (
            {"detector_signature": None, "raw_channel": None},
            None,
            None,
        ),
        (
            {"detector_signature": "", "raw_channel": "combined_w"},
            "",
            "combined_w",
        ),
        (
            {
                "detector_signature": TOCSIN_COMBINED_SIGNATURE,
                "raw_channel": "",
            },
            TOCSIN_COMBINED_SIGNATURE,
            "",
        ),
    ],
)
def test_malformed_loaded_v2_cannot_apply_as_legacy(
    tmp_path,
    identity,
    artifact_signature,
    artifact_channel,
):
    path = tmp_path / "malformed-v2.json"
    write_versioned_artifact(path, identity)
    calibrator = DetectorCalibrator()
    calibrator.load(path)
    result = layer(
        raw=0.0,
        signature=artifact_signature,
        raw_channel=artifact_channel,
    )

    assert calibrator.apply(result) is False
    assert result.calibrator_applied is False
    assert result.determination == "SKIP"
    assert result.signals["calibrator_signature_mismatch"] == {
        "result_signature": artifact_signature,
        "artifact_signature": artifact_signature,
        "result_raw_channel": artifact_channel,
        "artifact_raw_channel": artifact_channel,
    }
    assert result.raw_score == pytest.approx(0.0)
    assert result.p_llm == pytest.approx(0.37)
    assert result.signals["native_evidence"] == "preserve-me"


def test_matching_entry_rejects_beyond_support_tolerance():
    result = layer(raw=0.226)

    assert fitted_versioned().apply(result) is False
    assert result.calibrator_applied is False
    assert result.determination == "SKIP"
    assert result.signals["calibrator_out_of_domain"] == {
        "raw_score": pytest.approx(0.226),
        "support_min": pytest.approx(-0.4),
        "support_max": pytest.approx(0.1),
        "tolerance": pytest.approx(0.125),
    }


def test_matching_entry_allows_score_on_tolerance_boundary():
    result = layer(raw=0.225)

    assert fitted_versioned().apply(result) is True
    assert result.calibrator_applied is True


@pytest.mark.parametrize(
    ("raw", "raw_repr"),
    [
        (float("nan"), "nan"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
    ],
)
def test_nonfinite_raw_is_rejected_without_mutating_native_evidence(
    raw, raw_repr
):
    result = layer(raw=raw)
    native_signals = copy.deepcopy(result.signals)

    assert fitted_versioned().apply(result) is False
    assert result.calibrator_applied is False
    assert result.determination == "SKIP"
    assert result.signals == {
        **native_signals,
        "calibrator_nonfinite_raw": {"raw_score": raw_repr},
    }
    if math.isnan(raw):
        assert math.isnan(result.raw_score)
    else:
        assert result.raw_score == raw
    assert result.p_llm == pytest.approx(0.37)


def test_legacy_entry_still_applies_to_unsigned_legacy_result():
    legacy = DetectorCalibrator()
    legacy.fit("preamble", [0.0, 1.0], [0, 1])
    result = layer(
        raw=0.5,
        layer_id="preamble",
        signature=None,
        raw_channel=None,
    )

    assert legacy.apply(result) is True
    assert result.calibrator_applied is True


def test_preexisting_skip_is_ignored_without_mutation():
    result = layer(
        raw=999.0,
        signature="wrong",
        raw_channel="wrong",
        determination="SKIP",
    )
    before = copy.deepcopy(result)

    assert fitted_versioned().apply(result) is False
    assert result == before

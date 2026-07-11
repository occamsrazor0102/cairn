import json
from pathlib import Path

import pytest

from cairn.detectors.token_cohesiveness import TOCSIN_COMBINED_SIGNATURE
from cairn.pipeline import CairnPipeline


PROFILE = Path("configs/.profile_prompts_only_tocsin_v2.yaml")
pytestmark = pytest.mark.skipif(
    not PROFILE.exists(),
    reason="local TOCSIN v2 candidate has not been generated",
)


def test_candidate_profile_loads_coherent_artifacts():
    pipeline = CairnPipeline.from_config_file(PROFILE)
    assert pipeline._calibration_status == "calibrated"
    assert pipeline._detector_calibrator.has("token_cohesiveness")
    assert pipeline._fusion._model is not None
    assert pipeline._fusion._conformal.calibrated
    assert pipeline._fusion._assembler._word_count_clamp == 200

    names = set(pipeline._fusion._model._cairn_feature_names)
    assert {
        "tocsin_p_llm",
        "tocsin_cohesiveness_u",
        "tocsin_combined_w",
        "tocsin_missing",
    } <= names
    assert "tocsin_mean_deletion_impact" not in names

    entry = json.loads(
        Path(
            "models/calib2000/profiles/prompts_only_tocsin_v2/"
            "detector_calibration.json"
        ).read_text()
    )["token_cohesiveness"]
    assert entry["detector_signature"] == TOCSIN_COMBINED_SIGNATURE
    assert entry["raw_channel"] == "combined_w"

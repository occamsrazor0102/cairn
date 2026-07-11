import json
import math
from pathlib import Path

import pytest

from cairn.contracts import RouterDecision
from cairn.detectors.token_cohesiveness import TOCSIN_COMBINED_SIGNATURE
from cairn.feature_cache import CachedPipeline, text_hash, write_record
from cairn.normalizer import normalize_text
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


def test_candidate_cached_replay_reports_loaded_determination_tier(tmp_path):
    text = normalize_text("candidate cached replay calibration status")
    cache_path = tmp_path / "features.jsonl"
    router = RouterDecision(
        domain="prose",
        confidence=1.0,
        prompt_score=0.0,
        prose_score=1.0,
        word_count=100,
        recommended_detectors=[],
        skip_detectors=[],
    )
    with cache_path.open("w", encoding="utf-8") as handle:
        write_record(handle, text_hash(text), [], router)

    pipeline = CachedPipeline(cache_path, PROFILE)
    health = pipeline._inner.calibration_health()

    assert health == {
        "model_loaded": True,
        "conformal_loaded": True,
        "calibration_status": "calibrated-router-conditional",
        "detector_calibration": {"state": "loaded", "error": None},
    }
    assert pipeline._inner._detector_calibrator.has("token_cohesiveness")
    assert pipeline._inner._fusion._model is not None

    determination, _results, _router = pipeline.analyze_detailed(text)

    assert math.isfinite(determination.p_llm)
    assert determination.cascade_phases
    # Per-determination semantics intentionally use the pinned four-tier
    # pipeline status; router-conditional detail remains a health concern.
    assert determination.calibration_status == "calibrated"

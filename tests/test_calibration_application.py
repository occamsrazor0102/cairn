"""Integration coverage for calibration at runtime and replay boundaries."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from cairn.calibration import CalibrationMetadata, DetectorCalibrator
from cairn.config import load_config
from cairn.contracts import LayerResult, RouterDecision
from cairn.detectors.token_cohesiveness import TOCSIN_COMBINED_SIGNATURE
from cairn.feature_cache import CachedPipeline, text_hash, write_record
from cairn.fusion.ebm import FeatureAssembler
from cairn.pipeline import CairnPipeline, normalize_text
from scripts.per_detector_auroc import per_detector_auroc
from scripts.run_loss_filter import score_corpus


ROOT = Path(__file__).parent.parent
SCREENING_CONFIG = ROOT / "configs" / "screening.yaml"


def make_versioned_calibrator() -> DetectorCalibrator:
    calibrator = DetectorCalibrator()
    calibrator.fit(
        "token_cohesiveness",
        [-0.4, -0.2, 0.0, 0.1],
        [0, 0, 1, 1],
        metadata=CalibrationMetadata(
            detector_signature=TOCSIN_COMBINED_SIGNATURE,
            raw_channel="combined_w",
            training_dataset_hash="sha256:test",
            weighting_policy={"policy": "source-length-balanced-v1"},
        ),
    )
    return calibrator


def make_tocsin_result(
    raw: float,
    signature: str,
    determination: str,
) -> LayerResult:
    return LayerResult(
        layer_id="token_cohesiveness",
        domain="universal",
        raw_score=raw,
        p_llm=0.37,
        confidence=0.8,
        signals={
            "native_evidence": "preserve-me",
            "calibration_signature": signature,
            "calibration_raw_channel": "combined_w",
        },
        determination=determination,
        attacker_tiers=["A0"],
        compute_cost="moderate",
        min_text_length=40,
    )


class _SeedDetector:
    def __init__(self, seed: LayerResult) -> None:
        self._seed = seed

    def analyze(self, text: str, config: dict) -> LayerResult:
        return copy.deepcopy(self._seed)


class _BatchSeedDetector:
    def __init__(self, seed: LayerResult) -> None:
        self._seed = seed

    def analyze_batch(
        self,
        texts: dict[str, str],
        config: dict,
    ) -> dict[str, LayerResult]:
        return {key: copy.deepcopy(self._seed) for key in texts}


def run_stubbed_live_phase(
    calibrator: DetectorCalibrator,
    seed: LayerResult,
) -> LayerResult:
    config = load_config(SCREENING_CONFIG)
    config["detectors"]["token_cohesiveness"]["enabled"] = True
    pipeline = CairnPipeline(config)
    pipeline._detector_calibrator = calibrator
    pipeline._detectors["token_cohesiveness"] = _SeedDetector(seed)
    skipped = [
        detector_id
        for detector_id in pipeline._cascade.detectors_for_phase(2)
        if detector_id != "token_cohesiveness"
    ]

    results = pipeline._run_phase(2, "stub text", config, skipped)

    assert len(results) == 1
    return results[0]


def run_stubbed_cached_pipeline(
    tmp_path: Path,
    calibrator: DetectorCalibrator,
    seed: LayerResult,
) -> LayerResult:
    text = normalize_text("stub text")
    cache_path = tmp_path / "features.jsonl"
    router = RouterDecision(
        domain="prose",
        confidence=1.0,
        prompt_score=0.0,
        prose_score=1.0,
        word_count=100,
        recommended_detectors=["token_cohesiveness"],
        skip_detectors=[],
    )
    with cache_path.open("w", encoding="utf-8") as handle:
        write_record(handle, text_hash(text), [copy.deepcopy(seed)], router)

    pipeline = CachedPipeline(cache_path, SCREENING_CONFIG)
    pipeline._inner._detector_calibrator = calibrator
    _determination, results, _router = pipeline.analyze_detailed(text)

    assert len(results) == 1
    assert results[0] is not seed
    return results[0]


@pytest.mark.parametrize(
    "raw,signature,determination,expected_applied,rejection_key",
    [
        (0.0, TOCSIN_COMBINED_SIGNATURE, "AMBER", True, None),
        (
            0.0,
            "wrong-signature",
            "AMBER",
            False,
            "calibrator_signature_mismatch",
        ),
        (
            0.226,
            TOCSIN_COMBINED_SIGNATURE,
            "AMBER",
            False,
            "calibrator_out_of_domain",
        ),
        (999.0, "wrong-signature", "SKIP", False, None),
    ],
)
def test_live_and_cached_pipeline_share_calibration_guard(
    tmp_path,
    monkeypatch,
    raw,
    signature,
    determination,
    expected_applied,
    rejection_key,
):
    calibrator = make_versioned_calibrator()
    seed = make_tocsin_result(raw, signature, determination)
    apply_calls: list[LayerResult] = []
    original_apply = calibrator.apply

    def spy_apply(result: LayerResult) -> bool:
        apply_calls.append(result)
        return original_apply(result)

    monkeypatch.setattr(calibrator, "apply", spy_apply)

    live = run_stubbed_live_phase(calibrator, seed)
    assert apply_calls == [live]
    cached = run_stubbed_cached_pipeline(tmp_path, calibrator, seed)
    assert apply_calls == [live, cached]

    assert cached.p_llm == pytest.approx(live.p_llm)
    assert cached.calibrator_applied is live.calibrator_applied
    assert cached.determination == live.determination
    assert cached.signals == live.signals
    assert live.calibrator_applied is expected_applied
    if rejection_key is not None:
        assert rejection_key in live.signals
        vector = FeatureAssembler().assemble(
            [live], word_count=100, domain="prose"
        )
        assert vector["tocsin_missing"] == 1.0


def test_batch_cross_similarity_uses_guard_and_preserves_rejection_evidence(
    monkeypatch,
):
    config = {
        "detectors": {"cross_similarity": {"enabled": True}},
        "fusion": {},
        "cascade": {},
        "decision": {
            "red_threshold": 0.75,
            "amber_threshold": 0.50,
            "yellow_threshold": 0.25,
            "abstention": {"enabled": False},
        },
    }
    pipeline = CairnPipeline(config)
    calibrator = DetectorCalibrator()
    calibrator.fit(
        "cross_similarity",
        [0.0, 0.3, 0.7, 1.0],
        [0, 0, 1, 1],
        metadata=CalibrationMetadata(
            detector_signature="cross-similarity-v1",
            raw_channel="max_jaccard",
            training_dataset_hash="sha256:test",
            weighting_policy={"policy": "uniform"},
        ),
    )
    pipeline._detector_calibrator = calibrator
    cross_seed = LayerResult(
        layer_id="cross_similarity",
        domain="universal",
        raw_score=0.99,
        p_llm=0.99,
        confidence=1.0,
        signals={
            "calibration_signature": "stale-signature",
            "calibration_raw_channel": "max_jaccard",
        },
        determination="RED",
        attacker_tiers=["A0"],
        compute_cost="cheap",
        min_text_length=0,
    )
    pipeline._detectors["cross_similarity"] = _BatchSeedDetector(cross_seed)
    router = RouterDecision(
        domain="prose",
        confidence=1.0,
        prompt_score=0.0,
        prose_score=1.0,
        word_count=100,
        recommended_detectors=[],
        skip_detectors=[],
    )

    def fake_analyze_detailed(text: str, task_metadata=None):
        base = LayerResult(
            layer_id="preamble",
            domain="universal",
            raw_score=0.2,
            p_llm=0.2,
            confidence=1.0,
            signals={},
            determination="GREEN",
            attacker_tiers=["A0"],
            compute_cost="trivial",
            min_text_length=0,
        )
        fusion = pipeline._fusion.fuse(
            [base], word_count=router.word_count, domain=router.domain
        )
        determination = pipeline._decision.decide(fusion, [base])
        determination.cascade_phases = [1]
        return determination, [base], router

    monkeypatch.setattr(pipeline, "analyze_detailed", fake_analyze_detailed)
    apply_calls: list[LayerResult] = []
    original_apply = calibrator.apply

    def spy_apply(result: LayerResult) -> bool:
        apply_calls.append(result)
        return original_apply(result)

    monkeypatch.setattr(calibrator, "apply", spy_apply)

    outputs = pipeline.analyze_batch({"a": "first", "b": "second"})

    assert len(apply_calls) == 2
    assert all(result.layer_id == "cross_similarity" for result in apply_calls)
    for determination in outputs.values():
        cross = next(
            result
            for result in determination.layer_results
            if result.layer_id == "cross_similarity"
        )
        assert cross.determination == "SKIP"
        assert cross.calibrator_applied is False
        assert "calibrator_signature_mismatch" in cross.signals
        assert determination.p_llm == pytest.approx(0.2)


def test_per_detector_auroc_applies_guard_and_excludes_rejected_results(
    monkeypatch,
):
    calibrator = make_versioned_calibrator()
    samples = []
    cache = {}
    for index in range(13):
        text = normalize_text(f"sample {index}")
        label = 0 if index < 6 else 1
        signature = (
            "stale-signature"
            if index == 12
            else TOCSIN_COMBINED_SIGNATURE
        )
        raw = -0.3 if label == 0 else 0.05
        samples.append({"text": text, "label": label})
        cache[text_hash(text)] = {
            "results": [
                asdict(make_tocsin_result(raw, signature, "AMBER"))
            ],
            "router": {},
        }
    apply_calls: list[LayerResult] = []
    original_apply = calibrator.apply

    def spy_apply(result: LayerResult) -> bool:
        apply_calls.append(result)
        return original_apply(result)

    monkeypatch.setattr(calibrator, "apply", spy_apply)

    auroc, counts = per_detector_auroc(samples, cache, calibrator)

    assert len(apply_calls) == 13
    assert counts["token_cohesiveness"] == 12
    assert auroc["token_cohesiveness"] == pytest.approx(1.0)
    rejected = apply_calls[-1]
    assert rejected.determination == "SKIP"
    assert "calibrator_signature_mismatch" in rejected.signals


@pytest.mark.parametrize("calibrator_present", [False, True])
def test_per_detector_auroc_filters_invalid_raw_after_optional_guard(
    monkeypatch,
    calibrator_present,
):
    calibrator = DetectorCalibrator() if calibrator_present else None
    samples = []
    cache = {}
    for index in range(12):
        text = normalize_text(f"valid raw sample {index}")
        label = 0 if index < 6 else 1
        result = make_tocsin_result(
            -0.3 if label == 0 else 0.05,
            TOCSIN_COMBINED_SIGNATURE,
            "AMBER",
        )
        result.p_llm = 0.1 if label == 0 else 0.9
        samples.append({"text": text, "label": label})
        cache[text_hash(text)] = {
            "results": [asdict(result)],
            "router": {},
        }
    for offset, raw in enumerate(
        [None, float("nan"), float("inf"), float("-inf")]
    ):
        text = normalize_text(f"invalid raw sample {offset}")
        result = make_tocsin_result(
            raw,
            TOCSIN_COMBINED_SIGNATURE,
            "AMBER",
        )
        result.p_llm = 0.99
        samples.append({"text": text, "label": offset % 2})
        cache[text_hash(text)] = {
            "results": [asdict(result)],
            "router": {},
        }

    apply_calls: list[LayerResult] = []
    if calibrator is not None:
        original_apply = calibrator.apply

        def spy_apply(result: LayerResult) -> bool:
            apply_calls.append(result)
            return original_apply(result)

        monkeypatch.setattr(calibrator, "apply", spy_apply)

    auroc, counts = per_detector_auroc(samples, cache, calibrator)

    assert counts["token_cohesiveness"] == 12
    assert auroc["token_cohesiveness"] == pytest.approx(1.0)
    if calibrator is not None:
        assert len(apply_calls) == len(samples)
        assert len({id(result) for result in apply_calls}) == len(samples)


def test_per_detector_auroc_matching_calibrator_excludes_missing_raw():
    calibrator = make_versioned_calibrator()
    samples = []
    cache = {}
    for index in range(12):
        text = normalize_text(f"matching calibrator sample {index}")
        label = 0 if index < 6 else 1
        result = make_tocsin_result(
            -0.3 if label == 0 else 0.05,
            TOCSIN_COMBINED_SIGNATURE,
            "AMBER",
        )
        samples.append({"text": text, "label": label})
        cache[text_hash(text)] = {
            "results": [asdict(result)],
            "router": {},
        }
    missing_text = normalize_text("matching calibrator missing raw")
    missing = make_tocsin_result(
        None,
        TOCSIN_COMBINED_SIGNATURE,
        "AMBER",
    )
    missing.p_llm = 0.99
    samples.append({"text": missing_text, "label": 1})
    cache[text_hash(missing_text)] = {
        "results": [asdict(missing)],
        "router": {},
    }

    auroc, counts = per_detector_auroc(samples, cache, calibrator)

    assert counts["token_cohesiveness"] == 12
    assert auroc["token_cohesiveness"] == pytest.approx(1.0)


class _RecordingEBM:
    _cairn_feature_names = ["tocsin_missing"]

    def __init__(self) -> None:
        self.rows = []

    def predict_proba(self, rows):
        self.rows.append(rows.copy())
        return np.array([[0.1, 0.9]])


def test_loss_filter_replay_applies_guard_before_feature_assembly(monkeypatch):
    calibrator = make_versioned_calibrator()
    text = normalize_text("loss-filter sample")
    seed = make_tocsin_result(0.0, "stale-signature", "AMBER")
    router = RouterDecision(
        domain="prose",
        confidence=1.0,
        prompt_score=0.0,
        prose_score=1.0,
        word_count=100,
        recommended_detectors=["token_cohesiveness"],
        skip_detectors=[],
    )
    cache = {
        text_hash(text): {
            "results": [asdict(seed)],
            "router": asdict(router),
        }
    }
    apply_calls: list[LayerResult] = []
    original_apply = calibrator.apply

    def spy_apply(result: LayerResult) -> bool:
        apply_calls.append(result)
        return original_apply(result)

    monkeypatch.setattr(calibrator, "apply", spy_apply)
    ebm = _RecordingEBM()

    predictions = score_corpus(
        [{"text": text, "label": 1}], cache, calibrator, ebm
    )

    assert predictions == pytest.approx([0.9])
    assert len(apply_calls) == 1
    assert apply_calls[0].determination == "SKIP"
    assert "calibrator_signature_mismatch" in apply_calls[0].signals
    assert ebm.rows[0][0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "relative_path,guard_call",
    [
        ("cairn/pipeline.py", "self._detector_calibrator.apply("),
        ("cairn/feature_cache.py", "cal.apply("),
        ("scripts/per_detector_auroc.py", "calibrator.apply("),
        ("scripts/run_loss_filter.py", "calibrator.apply("),
        ("scripts/_formb_profile_grid.py", "cal.apply("),
        ("scripts/_formb_profile_grid_auc.py", "cal.apply("),
        ("scripts/_stage1_cpu.py", "cal.apply("),
        ("scripts/_stage1_cpu_clampoff.py", "cal.apply("),
    ],
)
def test_runtime_and_replay_callers_use_guard(relative_path, guard_call):
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert guard_call in source
    assert "calibrator.transform(" not in source
    assert "cal.transform(" not in source
    assert "_detector_calibrator.transform(" not in source

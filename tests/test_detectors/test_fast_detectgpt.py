"""Behavior tests for the fast_detectgpt detector.

The no-torch paths (short-text SKIP, tier2-unavailable SKIP, heuristic
monotonicity) run everywhere; the real-model path is exercised via a
monkeypatched curvature so CI doesn't download GPT-2 to assert wiring.
"""
from cairn.detectors.fast_detectgpt import FastDetectGPTDetector, DETECTOR


def test_skips_short_text():
    d = FastDetectGPTDetector()
    result = d.analyze("only a few words here", {})
    assert result.determination == "SKIP"
    assert result.signals["skipped"] == "too_short"


def test_skips_when_tier2_unavailable_or_runs():
    # In an environment without torch/transformers the detector must SKIP
    # cleanly (not raise); where they're present it produces a real verdict.
    d = FastDetectGPTDetector()
    long_text = " ".join(["word"] * 80)
    result = d.analyze(long_text, {})
    if result.determination == "SKIP":
        assert result.signals["skipped"] in {"too_short", "tier2_unavailable"}
        assert result.p_llm == 0.5
        assert result.confidence == 0.0
    else:
        assert "curvature" in result.signals
        assert 0.0 <= result.p_llm <= 1.0


def test_curvature_maps_into_unit_range_and_surfaces_signal(monkeypatch):
    d = FastDetectGPTDetector()
    # Bypass model load: force availability and a fixed curvature.
    monkeypatch.setattr(d._adapter, "is_available", lambda: True)
    monkeypatch.setattr(d._adapter, "curvature", lambda text: 2.5)
    result = d.analyze(" ".join(["word"] * 100), {})
    assert result.layer_id == "fast_detectgpt"
    assert result.determination != "SKIP"
    assert result.signals["curvature"] == 2.5
    assert 0.0 <= result.p_llm <= 1.0
    # High positive curvature → LLM-leaning.
    assert result.p_llm > 0.5
    # raw_score carries the uncalibrated curvature for the detector calibrator.
    assert result.raw_score == 2.5


def test_negative_curvature_leans_human(monkeypatch):
    d = FastDetectGPTDetector()
    monkeypatch.setattr(d._adapter, "is_available", lambda: True)
    monkeypatch.setattr(d._adapter, "curvature", lambda text: -1.5)
    result = d.analyze(" ".join(["word"] * 100), {})
    assert result.p_llm < 0.5


def test_module_detector_id():
    assert DETECTOR.id == "fast_detectgpt"
    assert DETECTOR.domain == "universal"

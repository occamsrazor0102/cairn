"""Behavior tests for the faithful TOCSIN token_cohesiveness detector.

No-torch paths and the pure-Python cohesiveness mapping run everywhere; the
BARTScore path is exercised via monkeypatched scoring so CI need not
download facebook/bart-base to assert wiring.
"""
from cairn.detectors.token_cohesiveness import (
    TokenCohesivenessDetector,
    _heuristic_cohesiveness_to_p_llm,
    DETECTOR,
)


def test_skips_short_text():
    d = TokenCohesivenessDetector()
    result = d.analyze("too short to score", {})
    assert result.determination == "SKIP"
    assert result.signals["skipped"] == "too_short"


def test_skips_cleanly_when_tier2_absent_or_runs():
    d = TokenCohesivenessDetector()
    long_text = " ".join(["word"] * 80)
    result = d.analyze(long_text, {})
    # Without torch this must SKIP, not raise.
    if result.determination == "SKIP":
        assert result.signals["skipped"] in {
            "too_short", "tier2_unavailable", "bart_load_failed",
        }


def test_cohesiveness_heuristic_is_monotone_increasing():
    # More cohesive (u closer to 0) → higher p(LLM).
    lo = _heuristic_cohesiveness_to_p_llm(-4.0)
    mid = _heuristic_cohesiveness_to_p_llm(-1.5)
    hi = _heuristic_cohesiveness_to_p_llm(0.0)
    assert lo < mid < hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_combined_channel_surfaces_signals(monkeypatch):
    d = TokenCohesivenessDetector()
    # Force torch present, BART score fixed, base channel present.
    monkeypatch.setattr(
        "cairn.detectors.token_cohesiveness._try_import_torch", lambda: object()
    )
    monkeypatch.setattr(d, "_ensure_bart", lambda: None)
    monkeypatch.setattr(d, "_bart_score", lambda src, tgt: -1.0)
    monkeypatch.setattr(d._base, "is_available", lambda: True)
    monkeypatch.setattr(d._base, "curvature", lambda text: 1.2)

    result = d.analyze(" ".join(["word"] * 100), {})
    assert result.layer_id == "token_cohesiveness"
    assert result.determination != "SKIP"
    assert result.signals["cohesiveness_u"] == -1.0
    assert result.signals["base_v"] == 1.2
    assert "combined_w" in result.signals
    assert 0.0 <= result.p_llm <= 1.0


def test_standalone_fallback_when_base_unavailable(monkeypatch):
    d = TokenCohesivenessDetector()
    monkeypatch.setattr(
        "cairn.detectors.token_cohesiveness._try_import_torch", lambda: object()
    )
    monkeypatch.setattr(d, "_ensure_bart", lambda: None)
    monkeypatch.setattr(d, "_bart_score", lambda src, tgt: -0.5)
    monkeypatch.setattr(d._base, "is_available", lambda: False)

    result = d.analyze(" ".join(["word"] * 100), {})
    assert result.determination != "SKIP"
    assert result.signals["base_v"] is None
    assert result.signals["cohesiveness_u"] == -0.5
    assert 0.0 <= result.p_llm <= 1.0


def test_module_detector_id():
    assert DETECTOR.id == "token_cohesiveness"

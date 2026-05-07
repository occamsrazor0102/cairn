"""Unit tests for ``cairn.auto_routing.recommend_profile`` (B3).

The function is pure and decoupled from disk; these tests instantiate
``RouterDecision`` directly with controlled inputs to keep the
recommendation logic isolated from the live router heuristics.
"""
from __future__ import annotations

import pytest

from cairn.auto_routing import ProfileRecommendation, recommend_profile
from cairn.contracts import RouterDecision


# A baseline list of profile names used by most tests; mirrors a host
# that has both prose and prompt calibration artifacts loaded.
_FULL_AVAILABLE = [
    "default",
    "master_minus_arxiv",
    "prompts_only",
    "calibrated_calib2000",
    "general_hc3",
]


def _decision(domain: str, *, word_count: int = 100) -> RouterDecision:
    """Construct a ``RouterDecision`` with sensible defaults.

    Score fields are set to deterministic values; the recommender does
    not consume them today, but the dataclass requires them.
    """
    return RouterDecision(
        domain=domain,
        confidence=0.8,
        prompt_score=0.5 if domain == "prompt" else 0.05,
        prose_score=0.5 if domain == "prose" else 0.05,
        word_count=word_count,
        recommended_detectors=[],
        skip_detectors=[],
    )


def test_prompt_routes_to_prompts_only():
    """Domain=prompt -> prompts_only with high confidence and no caveats."""
    rec = recommend_profile(
        _decision("prompt", word_count=80), None, _FULL_AVAILABLE
    )
    assert isinstance(rec, ProfileRecommendation)
    assert rec.profile == "prompts_only"
    assert rec.confidence >= 0.9
    assert rec.caveats == ()
    assert "prompt" in rec.reason.lower()


def test_long_prose_routes_to_master_minus_arxiv():
    """Long prose (wc=200) -> master_minus_arxiv, no caveats."""
    rec = recommend_profile(
        _decision("prose", word_count=200), None, _FULL_AVAILABLE
    )
    assert rec.profile == "master_minus_arxiv"
    assert rec.confidence >= 0.85
    assert rec.caveats == ()


def test_short_prose_carries_short_text_caveat():
    """Short prose (wc=20) -> master_minus_arxiv with short_text caveat."""
    rec = recommend_profile(
        _decision("prose", word_count=20), None, _FULL_AVAILABLE
    )
    assert rec.profile == "master_minus_arxiv"
    assert "short_text" in rec.caveats
    # Confidence is intentionally lower than the long-prose path.
    assert rec.confidence < 0.7


def test_mixed_input_carries_mixed_caveat():
    """Domain=mixed -> master_minus_arxiv with mixed_input caveat."""
    rec = recommend_profile(
        _decision("mixed", word_count=120), None, _FULL_AVAILABLE
    )
    assert rec.profile == "master_minus_arxiv"
    assert "mixed_input" in rec.caveats
    assert rec.confidence < 0.85


def test_insufficient_input_carries_insufficient_caveat():
    """Domain=insufficient -> master_minus_arxiv with insufficient_text caveat."""
    rec = recommend_profile(
        _decision("insufficient", word_count=10), None, _FULL_AVAILABLE
    )
    assert rec.profile == "master_minus_arxiv"
    assert "insufficient_text" in rec.caveats
    # Lowest confidence band among present-but-uncertain cases.
    assert rec.confidence <= 0.5


def test_non_english_carries_non_english_caveat():
    """task_metadata.language='es' -> non-english caveat retained alongside profile."""
    rec = recommend_profile(
        _decision("prose", word_count=120),
        {"language": "es"},
        _FULL_AVAILABLE,
    )
    assert rec.profile == "master_minus_arxiv"
    assert "non-english" in rec.caveats
    # Long prose, so the short_text caveat should not be present.
    assert "short_text" not in rec.caveats


@pytest.mark.parametrize("lang", ["en", "english", "EN", "  English  ", "", None])
def test_english_or_unspecified_does_not_add_caveat(lang):
    """English or unspecified language must not trigger non-english."""
    metadata = {"language": lang} if lang is not None else None
    rec = recommend_profile(
        _decision("prose", word_count=120), metadata, _FULL_AVAILABLE
    )
    assert "non-english" not in rec.caveats


def test_fallback_when_recommended_unavailable():
    """When the desired profile is missing, fall back and tag with caveat."""
    # Recommend prompts_only -> available list lacks it -> falls back to
    # master_minus_arxiv (highest priority among the present entries).
    rec = recommend_profile(
        _decision("prompt", word_count=80),
        None,
        ["master_minus_arxiv", "default"],
    )
    assert rec.profile == "master_minus_arxiv"
    assert "fallback" in rec.caveats
    # Confidence drops by 0.2 from the original 0.95 baseline.
    assert rec.confidence == pytest.approx(0.75, abs=1e-9)
    assert "prompts_only" in rec.reason
    assert "master_minus_arxiv" in rec.reason


def test_fallback_walks_priority_order():
    """When master_minus_arxiv is missing, fall through to general_hc3."""
    rec = recommend_profile(
        _decision("prose", word_count=200),
        None,
        ["general_hc3", "default"],
    )
    assert rec.profile == "general_hc3"
    assert "fallback" in rec.caveats


def test_no_profiles_available_returns_default_with_zero_confidence():
    """Empty available list -> default placeholder with zero confidence."""
    rec = recommend_profile(_decision("prose", word_count=200), None, [])
    assert rec.profile == "default"
    assert rec.confidence == 0.0
    assert rec.caveats == ("no_profiles",)
    assert rec.reason == "no profiles available"


def test_non_english_combines_with_short_text_caveat():
    """Caveats compose: short non-English prose carries both flags."""
    rec = recommend_profile(
        _decision("prose", word_count=20),
        {"language": "fr"},
        _FULL_AVAILABLE,
    )
    assert rec.profile == "master_minus_arxiv"
    assert "short_text" in rec.caveats
    assert "non-english" in rec.caveats


def test_recommendation_is_frozen_dataclass():
    """ProfileRecommendation must be immutable to be safely passed around."""
    rec = recommend_profile(
        _decision("prose", word_count=200), None, _FULL_AVAILABLE
    )
    with pytest.raises(Exception):
        rec.profile = "something_else"  # type: ignore[misc]


def test_confidence_clamped_to_unit_interval():
    """Confidence must always be in [0, 1] after the function returns."""
    for domain in ("prompt", "prose", "mixed", "insufficient"):
        rec = recommend_profile(
            _decision(domain, word_count=100), None, _FULL_AVAILABLE
        )
        assert 0.0 <= rec.confidence <= 1.0


def test_unknown_domain_falls_through_safely():
    """A garbage domain string returns a defensive recommendation, not an exception."""
    rec = recommend_profile(
        _decision("alien_text"),  # type: ignore[arg-type]
        None,
        _FULL_AVAILABLE,
    )
    assert rec.profile == "master_minus_arxiv"
    assert "insufficient_text" in rec.caveats
    assert "alien_text" in rec.reason

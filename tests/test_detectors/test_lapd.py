"""LAPD detector tests.

Runtime behavior (loading the Falcon-7B pair) isn't exercised here --
weights are ~14 GB and would dominate CI. We test the gates,
tokenizer-mismatch detection, the sign of S on mocked log-probs, the
perturbation-variance abstention, and the signals schema. Pattern
mirrors `tests/test_detectors/test_perturbation.py`.
"""
from unittest.mock import patch, MagicMock

import pytest

from cairn.detectors.lapd import (
    LAPDDetector,
    _HEURISTIC_S_LAPD_TO_P_LLM,
)


@pytest.fixture
def detector():
    return LAPDDetector()


def _fake_input_ids(n_tokens: int = 150):
    """Tensor-like stand-in that satisfies analyze()'s attribute access.

    The real code paths that would touch this object (_score,
    _perturb_ids) are mocked in the helper below, so this only needs
    to provide `.shape[1]` for the n_tokens / truncated calculation.
    """
    obj = MagicMock()
    obj.shape = (1, n_tokens)
    return obj


def test_lapd_registered():
    import cairn.detectors as registry
    assert "lapd" in registry.get_all_detectors()


def test_lapd_placed_in_phase3():
    from cairn.cascade import PHASE3_DETECTORS
    assert "lapd" in PHASE3_DETECTORS


def test_lapd_skips_short_text(detector):
    r = detector.analyze("short.", {})
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "too_short"


def test_lapd_skip_when_torch_unavailable(detector):
    long_text = " ".join(["word"] * 200)
    import cairn.detectors.lapd as mod
    with patch.object(mod, "_HAS_TORCH", False):
        r = detector.analyze(long_text, {})
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "torch_unavailable"
    assert "tier2" in r.signals.get("hint", "")


def test_lapd_skip_when_bitsandbytes_unavailable_with_quantize_requested(detector):
    long_text = " ".join(["word"] * 200)
    import cairn.detectors.lapd as mod
    with patch.object(mod, "_HAS_TORCH", True), \
         patch.object(mod, "_HAS_BITSANDBYTES", False):
        r = detector.analyze(long_text, {"quantize_4bit": True})
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "bitsandbytes_unavailable"


def test_lapd_skip_on_tokenizer_mismatch(detector):
    long_text = " ".join(["word"] * 200)
    import cairn.detectors.lapd as mod

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {"input_ids": _fake_input_ids(150)}
    detector._aligned_tokenizer = fake_tokenizer
    detector._base_tokenizer = fake_tokenizer

    with patch.object(mod, "_HAS_TORCH", True), \
         patch.object(mod, "_HAS_BITSANDBYTES", True), \
         patch.object(detector, "_ensure_loaded", return_value=None), \
         patch.object(LAPDDetector, "_tokenizer_fingerprint",
                      side_effect=["aligned_fp", "base_fp"]):
        r = detector.analyze(long_text, {"quantize_4bit": False})

    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "tokenizer_mismatch"
    assert r.signals.get("aligned_fingerprint") == "aligned_fp"
    assert r.signals.get("base_fingerprint") == "base_fp"


def _run_analyze_with_mocked_scores(
    detector, scores, n_perturbations=5, n_tokens=150, max_tokens=512,
):
    """Drive analyze() end-to-end with mocked `_score` returns.

    The `scores` list is consumed in order: first element is the raw S(x),
    the rest are perturbation samples. Tokenizer, model load, and
    perturbation are all stubbed so no torch/transformers calls fire.
    """
    long_text = " ".join(["word"] * 200)
    import cairn.detectors.lapd as mod

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {"input_ids": _fake_input_ids(n_tokens)}
    detector._aligned_tokenizer = fake_tokenizer
    detector._base_tokenizer = fake_tokenizer

    with patch.object(mod, "_HAS_TORCH", True), \
         patch.object(mod, "_HAS_BITSANDBYTES", True), \
         patch.object(detector, "_ensure_loaded", return_value=None), \
         patch.object(LAPDDetector, "_tokenizer_fingerprint", return_value="same"), \
         patch.object(detector, "_score") as mock_score, \
         patch.object(detector, "_perturb_ids", side_effect=lambda x, *a, **kw: x):
        mock_score.side_effect = scores
        return detector.analyze(long_text, {
            "n_perturbations": n_perturbations,
            "quantize_4bit": False,
            "max_tokens": max_tokens,
        })


def test_lapd_sign_on_llm_like_input(detector):
    """LLM-like: raw S much higher than perturbation mean. s_lapd >> 0, p_llm > 0.5."""
    scores = [8.0, 2.5, 3.0, 3.5, 2.0, 4.0]
    r = _run_analyze_with_mocked_scores(detector, scores, n_perturbations=5)
    assert r.determination != "SKIP"
    assert r.signals["s_lapd"] > 1.0
    assert r.p_llm > 0.5


def test_lapd_sign_on_human_like_input(detector):
    """Human-like: raw S near perturbation mean. s_lapd ~ 0, p_llm low."""
    scores = [2.0, 1.5, 2.5, 1.8, 2.2, 2.0]
    r = _run_analyze_with_mocked_scores(detector, scores, n_perturbations=5)
    assert r.determination != "SKIP"
    assert abs(r.signals["s_lapd"]) < 1.0
    assert r.p_llm < 0.30


def test_lapd_perturbation_variance_floor(detector):
    """All perturbations land at the same S -> sigma_tilde ~ 0 -> SKIP."""
    scores = [5.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    r = _run_analyze_with_mocked_scores(detector, scores, n_perturbations=5)
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "insufficient_perturbation_variance"


def test_lapd_signals_schema(detector):
    scores = [8.0, 2.5, 3.0, 3.5, 2.0, 4.0]
    r = _run_analyze_with_mocked_scores(detector, scores, n_perturbations=5)
    expected_keys = {
        "raw_s", "s_lapd", "mu_tilde", "sigma_tilde",
        "n_tokens", "n_perturbations",
        "model_pair", "quantized", "device", "truncated",
    }
    missing = expected_keys - set(r.signals.keys())
    assert not missing, f"missing signals keys: {missing}"
    assert isinstance(r.signals["raw_s"], float)
    assert isinstance(r.signals["n_tokens"], int)
    assert isinstance(r.signals["truncated"], bool)


def test_lapd_truncation_flag(detector):
    """When the tokenizer returns max_tokens tokens, truncated must be True."""
    scores = [8.0, 2.5, 3.0, 3.5, 2.0, 4.0]
    r = _run_analyze_with_mocked_scores(
        detector, scores, n_perturbations=5, n_tokens=512, max_tokens=512,
    )
    assert r.signals["truncated"] is True


def test_lapd_no_truncation_when_short(detector):
    scores = [8.0, 2.5, 3.0, 3.5, 2.0, 4.0]
    r = _run_analyze_with_mocked_scores(
        detector, scores, n_perturbations=5, n_tokens=120, max_tokens=512,
    )
    assert r.signals["truncated"] is False
    assert r.signals["n_tokens"] == 120


@pytest.mark.parametrize(
    "s_lapd, expected_bounds",
    [
        (-2.0, (0.0, 0.05)),
        (0.0, (0.05, 0.15)),
        (1.0, (0.50, 0.60)),
        (3.0, (0.85, 0.95)),
        (10.0, (0.95, 1.0)),
    ],
)
def test_heuristic_s_lapd_to_p_llm_mapping_is_monotone(s_lapd, expected_bounds):
    from cairn.detectors._utils import interpolate
    p = interpolate(s_lapd, _HEURISTIC_S_LAPD_TO_P_LLM)
    lo, hi = expected_bounds
    assert lo <= p <= hi, f"p_llm={p} for s_lapd={s_lapd} outside [{lo}, {hi}]"


def test_tokenizer_fingerprint_handles_unknown_tokenizer():
    """If a tokenizer doesn't expose get_vocab(), fingerprint must still return a string."""
    broken = MagicMock()
    broken.get_vocab.side_effect = RuntimeError("no vocab")
    fp = LAPDDetector._tokenizer_fingerprint(broken)
    assert fp == "unknown"

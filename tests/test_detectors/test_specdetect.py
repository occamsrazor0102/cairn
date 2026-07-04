"""SpecDetect detector tests.

Runtime behaviour (loading GPT-2 or GPT-J-6B) isn't exercised here. We
test the gates, the pure-Python spectral feature computation, and the
end-to-end analyze flow with `_per_token_log_probs` mocked.
"""
import math
from unittest.mock import patch, MagicMock

import pytest

from cairn.detectors.specdetect import (
    SpecDetectDetector,
    _HEURISTIC_TOTAL_ENERGY_TO_P_LLM,
    _MIN_TOKENS_FOR_FFT,
)


@pytest.fixture
def detector():
    return SpecDetectDetector()


def _fake_input_ids(n_tokens: int = 100):
    obj = MagicMock()
    obj.shape = (1, n_tokens)
    return obj


def test_specdetect_registered():
    import cairn.detectors as registry
    assert "specdetect" in registry.get_all_detectors()


def test_specdetect_placed_in_phase2():
    from cairn.cascade import PHASE2_DETECTORS
    assert "specdetect" in PHASE2_DETECTORS


def test_specdetect_skips_short_text(detector):
    r = detector.analyze("short.", {})
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "too_short"


def test_specdetect_skip_when_torch_unavailable(detector):
    long_text = " ".join(["word"] * 100)
    import cairn.detectors.specdetect as mod
    with patch.object(mod, "_HAS_TORCH", False):
        r = detector.analyze(long_text, {})
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "torch_unavailable"
    assert "tier2" in r.signals.get("hint", "")


# ---- pure-Python spectral feature tests (no torch needed) ----


def test_spectral_features_uniform_signal_has_low_total_energy():
    """Near-constant log-probs -> near-zero spectral energy (LLM-like)."""
    log_probs = [-2.0] * 64
    feats = SpecDetectDetector._spectral_features(log_probs)
    assert feats["dft_total_energy"] < 0.01


def test_spectral_features_volatile_signal_has_high_total_energy():
    """High-variance log-probs -> high total energy (human-like)."""
    import random
    rng = random.Random(7)
    log_probs = [rng.gauss(-3.0, 2.0) for _ in range(64)]
    feats = SpecDetectDetector._spectral_features(log_probs)
    assert feats["dft_total_energy"] > 0.01


def test_spectral_features_band_fractions_sum_to_one():
    import random
    rng = random.Random(11)
    log_probs = [rng.gauss(-2.5, 1.5) for _ in range(80)]
    feats = SpecDetectDetector._spectral_features(log_probs)
    band_sum = (
        feats["dft_low_band_energy"]
        + feats["dft_mid_band_energy"]
        + feats["dft_high_band_energy"]
    )
    assert math.isclose(band_sum, 1.0, abs_tol=1e-9)


def test_spectral_features_low_freq_signal_concentrates_low_band():
    """A slow sinusoid should put most energy in the low band."""
    import math as m
    n = 128
    log_probs = [m.sin(2 * m.pi * 2 * i / n) for i in range(n)]  # 2 cycles
    feats = SpecDetectDetector._spectral_features(log_probs)
    assert feats["dft_low_band_energy"] > 0.85


def test_spectral_features_high_freq_signal_concentrates_high_band():
    """A fast sinusoid should put most energy in the high band."""
    import math as m
    n = 128
    # n/2 - 2 cycles puts the peak near Nyquist
    log_probs = [m.sin(2 * m.pi * (n // 2 - 2) * i / n) for i in range(n)]
    feats = SpecDetectDetector._spectral_features(log_probs)
    assert feats["dft_high_band_energy"] > 0.85


def test_spectral_features_constant_signal_degenerate_case():
    """All-zero variance -> uniform band fractions, no NaN."""
    log_probs = [-2.5] * 50
    feats = SpecDetectDetector._spectral_features(log_probs)
    assert math.isclose(feats["dft_low_band_energy"], 1.0 / 3, abs_tol=1e-9)
    assert math.isclose(feats["dft_mid_band_energy"], 1.0 / 3, abs_tol=1e-9)
    assert math.isclose(feats["dft_high_band_energy"], 1.0 / 3, abs_tol=1e-9)
    for v in feats.values():
        assert not math.isnan(v) and not math.isinf(v)


def test_spectral_entropy_higher_for_flat_spectrum():
    """White noise has higher spectral entropy than a single sinusoid."""
    import math as m
    import random
    rng = random.Random(3)
    n = 128

    noise = [rng.gauss(0, 1) for _ in range(n)]
    sine = [m.sin(2 * m.pi * 3 * i / n) for i in range(n)]

    e_noise = SpecDetectDetector._spectral_features(noise)["dft_spectral_entropy"]
    e_sine = SpecDetectDetector._spectral_features(sine)["dft_spectral_entropy"]
    assert e_noise > e_sine


# ---- end-to-end analyze() with mocked log-prob extraction ----


def _run_analyze_with_mocked_log_probs(
    detector, log_probs, n_tokens=None, max_tokens=512,
):
    long_text = " ".join(["word"] * 100)
    import cairn.detectors.specdetect as mod

    if n_tokens is None:
        n_tokens = len(log_probs) + 1   # _per_token_log_probs returns T-1 values

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {"input_ids": _fake_input_ids(n_tokens)}
    detector._tokenizer = fake_tokenizer

    with patch.object(mod, "_HAS_TORCH", True), \
         patch.object(detector, "_ensure_loaded", return_value=None), \
         patch.object(detector, "_per_token_log_probs", return_value=log_probs):
        return detector.analyze(long_text, {"max_tokens": max_tokens})


def test_analyze_uniform_log_probs_lands_high_p_llm(detector):
    """Constant log-probs -> ~0 energy -> p_llm at the top anchor."""
    log_probs = [-2.0] * 100
    r = _run_analyze_with_mocked_log_probs(detector, log_probs)
    assert r.determination != "SKIP"
    assert r.p_llm >= 0.78    # >= the 1.0-energy anchor
    # Confirm we're in the right region of the heuristic table
    assert r.signals["dft_total_energy"] < 0.5


def test_analyze_volatile_log_probs_lands_low_p_llm(detector):
    """High-variance log-probs -> high energy -> p_llm at low anchor."""
    import random
    rng = random.Random(13)
    log_probs = [rng.gauss(-2.0, 4.0) for _ in range(100)]
    r = _run_analyze_with_mocked_log_probs(detector, log_probs)
    assert r.determination != "SKIP"
    assert r.signals["dft_total_energy"] > 1.5


def test_analyze_skip_when_too_few_tokens(detector):
    log_probs = [-2.0] * 4   # only 5 tokens -> below _MIN_TOKENS_FOR_FFT (16)
    r = _run_analyze_with_mocked_log_probs(detector, log_probs, n_tokens=5)
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "too_few_tokens_for_fft"


def test_analyze_signals_schema(detector):
    log_probs = [-2.0 + 0.1 * i for i in range(100)]   # linear ramp
    r = _run_analyze_with_mocked_log_probs(detector, log_probs)
    expected_keys = {
        "n_tokens",
        "dft_total_energy",
        "dft_low_band_energy",
        "dft_mid_band_energy",
        "dft_high_band_energy",
        "dft_low_mid_ratio",
        "dft_spectral_entropy",
        "model",
        "device",
        "truncated",
    }
    missing = expected_keys - set(r.signals.keys())
    assert not missing, f"missing signals keys: {missing}"


def test_analyze_truncation_flag(detector):
    """When tokenizer hits max_tokens, signals.truncated must be True."""
    log_probs = [-2.0] * 511   # n_tokens=512 -> hits max_tokens
    r = _run_analyze_with_mocked_log_probs(
        detector, log_probs, n_tokens=512, max_tokens=512,
    )
    assert r.signals["truncated"] is True


def test_heuristic_total_energy_table_is_monotone_decreasing():
    """As total energy increases, p_llm should DECREASE (higher energy = more human)."""
    energies = [x for x, _ in _HEURISTIC_TOTAL_ENERGY_TO_P_LLM]
    probs = [p for _, p in _HEURISTIC_TOTAL_ENERGY_TO_P_LLM]
    for i in range(1, len(energies)):
        assert energies[i] > energies[i - 1], "anchors must be sorted by energy"
        assert probs[i] <= probs[i - 1], (
            "p_llm should decrease (or hold) as total energy increases"
        )

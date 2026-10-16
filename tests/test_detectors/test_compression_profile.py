"""Tests for the compression_profile detector.

Behavioural, not tautological: these assert that redundant text scores
above dense text, that spans land on real offsets, that the SKIP
contract holds, and — the point of the detector's design — that the
heuristic table cannot be saturated by any constructible input. They do
not assert the specific p_llm values the heuristic table happens to
produce (see CLAUDE.md on tautological tests).
"""
import pytest

from cairn.detectors._utils import interpolate
from cairn.detectors.compression_profile import (
    DETECTOR,
    _HEURISTIC_REDUNDANCY_TO_P_LLM,
    _MIN_WORDS,
    CompressionProfileDetector,
)
from tests.fixtures.human_samples import CASUAL_SHORT, FORMAL_SOP
from tests.fixtures.llm_samples import A0_PREAMBLE


@pytest.fixture
def detector():
    return CompressionProfileDetector()


@pytest.fixture
def config():
    return {}


DENSE = " ".join(
    f"word{i} thing{i} idea{i} concept{i} matter{i}" for i in range(40)
)
REDUNDANT = "The key point here is important. " * 40


def test_compression_profile_registered():
    import cairn.detectors as registry
    assert "compression_profile" in registry.get_all_detectors()


def test_compression_profile_placed_in_phase1():
    from cairn.cascade import PHASE1_DETECTORS
    assert "compression_profile" in PHASE1_DETECTORS


def test_result_has_required_fields(detector, config):
    r = detector.analyze(A0_PREAMBLE, config)
    assert r.layer_id == "compression_profile"
    assert r.domain == "universal"
    assert r.compute_cost == "trivial"
    assert "A0" in r.attacker_tiers
    assert 0.0 <= r.p_llm <= 1.0
    assert 0.0 <= r.confidence <= 1.0


def test_signals_dict_has_all_keys(detector, config):
    r = detector.analyze(A0_PREAMBLE, config)
    expected = {
        "zlib_ratio", "lzma_ratio", "mattr_100", "distinct_4",
        "max_ngram_repeat", "redundancy", "n_words",
    }
    missing = expected - set(r.signals.keys())
    assert not missing, f"missing signals keys: {missing}"


def test_skips_below_word_floor(detector, config):
    r = detector.analyze(" ".join(["word"] * (_MIN_WORDS - 5)), config)
    assert r.determination == "SKIP"
    assert r.signals["skipped"] == "too_short"
    assert r.p_llm == 0.5
    assert r.confidence == 0.0


def test_redundant_text_scores_above_dense_text(detector, config):
    """The behavioural claim the detector exists to make.

    Summarize-then-re-expand output restates its own content; unique
    prose does not. Whatever the heuristic anchors are, the ordering
    must hold.
    """
    redundant = detector.analyze(REDUNDANT, config)
    dense = detector.analyze(DENSE, config)
    assert redundant.determination != "SKIP"
    assert dense.determination != "SKIP"
    assert redundant.p_llm > dense.p_llm
    assert redundant.signals["redundancy"] > dense.signals["redundancy"]


def test_ratio_signals_are_bounded(detector, config):
    for text in (DENSE, REDUNDANT, A0_PREAMBLE, FORMAL_SOP):
        r = detector.analyze(text, config)
        if r.determination == "SKIP":
            continue
        for key in ("zlib_ratio", "lzma_ratio", "mattr_100", "distinct_4",
                    "max_ngram_repeat", "redundancy"):
            assert 0.0 <= r.signals[key] <= 1.0, f"{key} out of range on {text[:30]!r}"


def test_heuristic_table_cannot_saturate():
    """The SpecDetect regression guard.

    `_HEURISTIC_TOTAL_ENERGY_TO_P_LLM` was anchored over a guessed band
    (0.5..5.0) while the real corpus median sat at 5.75, so 70.9% of
    samples clamped to one endpoint and isotonic calibration could not
    recover the lost information. This detector's score is bounded in
    [0, 1] by construction, so the table must span that full range for
    the same failure to be unreachable.
    """
    lo = _HEURISTIC_REDUNDANCY_TO_P_LLM[0][0]
    hi = _HEURISTIC_REDUNDANCY_TO_P_LLM[-1][0]
    assert lo <= 0.0 and hi >= 1.0, (
        "heuristic table must span the score's full constructible range"
    )


def test_heuristic_table_is_monotone_increasing():
    xs = [x for x, _ in _HEURISTIC_REDUNDANCY_TO_P_LLM]
    ys = [y for _, y in _HEURISTIC_REDUNDANCY_TO_P_LLM]
    assert xs == sorted(xs)
    assert ys == sorted(ys)
    assert interpolate(0.0, _HEURISTIC_REDUNDANCY_TO_P_LLM) < interpolate(
        1.0, _HEURISTIC_REDUNDANCY_TO_P_LLM
    )


def test_analyze_is_deterministic(detector, config):
    """Pins the invariant `token_cohesiveness` violated.

    That detector drew its sample positions from the global `random`
    module, so identical text produced different output across runs —
    silently miscalibrating production against its own training cache
    (docs/benchmarks/robust_v2_validation/HYPOTHESIS_INVESTIGATION.md).
    """
    a = detector.analyze(REDUNDANT, config)
    b = detector.analyze(REDUNDANT, config)
    assert a.p_llm == b.p_llm
    assert a.signals == b.signals
    assert a.spans == b.spans


def test_spans_are_valid_and_disjoint(detector, config):
    r = detector.analyze(REDUNDANT, config)
    assert r.spans, "repeated phrases should produce highlights"
    last_end = -1
    for s in r.spans:
        assert {"start", "end", "kind", "note"} <= set(s.keys())
        assert s["kind"] == "repetition"
        assert 0 <= s["start"] < s["end"] <= len(REDUNDANT)
        assert REDUNDANT[s["start"]:s["end"]]
        # Sliding 4-grams overlap by construction; the detector filters
        # them so the UI does not report "N overlap dropped".
        assert s["start"] >= last_end
        last_end = s["end"]


def test_no_spans_without_repetition(detector, config):
    r = detector.analyze(DENSE, config)
    assert r.spans == []


def test_module_singleton_is_wired():
    assert DETECTOR.id == "compression_profile"


def test_handles_unicode_without_crashing(detector, config):
    text = ("emoji 🙂 and accents café naïve " * 12)
    r = detector.analyze(text, config)
    assert r.determination in ("RED", "AMBER", "YELLOW", "GREEN", "SKIP")


def test_casual_human_text_is_not_flagged_red(detector, config):
    """Not a tautology on the table: a short human note has no repeated
    4-grams and a high type-token ratio, so nothing in the score should
    push it to the top band."""
    r = detector.analyze(CASUAL_SHORT, config)
    if r.determination != "SKIP":
        assert r.determination != "RED"

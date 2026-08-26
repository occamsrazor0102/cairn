"""Tests for the typographic_signature detector.

The load-bearing tests here are the two that guard its design premise:
every signal must survive `normalize_text` (because detectors never see
raw text, and the feature cache is keyed on the normalized form), and no
ratio may saturate on a single observation in short input.
"""
import pytest

from cairn.detectors._utils import interpolate
from cairn.detectors.typographic_signature import (
    DETECTOR,
    _HEURISTIC_TYPOGRAPHY_TO_P_LLM,
    _MIN_EVENTS_FOR_RATIO,
    _MIN_WORDS,
    TypographicSignatureDetector,
)
from cairn.normalizer import normalize_text


@pytest.fixture
def detector():
    return TypographicSignatureDetector()


@pytest.fixture
def config():
    return {}


# Detectors always receive normalized text (cairn/pipeline.py), so the
# fixtures are normalized here too — testing on raw text would assert
# behaviour the detector never actually sees.
LLM_TYPOGRAPHY = normalize_text("""## Key Considerations for Migration

The migration path depends on three factors — throughput, cost, and operational risk.

• **Throughput**: the new cluster sustains roughly twice the write rate.
• **Cost**: reserved capacity lowers the monthly spend, and the savings compound.
• **Operational Risk**: a staged cutover limits blast radius, exposure, and rollback time.

## Recommended Sequence

We recommend a phased approach — staged, reversible, and observable — with a
two-week soak between stages.
""")

HUMAN_TYPOGRAPHY = normalize_text("""ok so i looked at the migration thing again
today - the write throughput numbers from last week were wrong, i'd been reading
the p50 not the p99. redid the test with a bigger sample and it's about 1.7x not
2x. still worth doing but the cost case is weaker than i told you. also the
rollback plan needs work - right now we'd be stuck for maybe 40 mins if the
cutover goes bad. i'll write it up properly tomorrow - sorry for the churn.
""")


def test_typographic_signature_registered():
    import cairn.detectors as registry
    assert "typographic_signature" in registry.get_all_detectors()


def test_typographic_signature_placed_in_phase1():
    from cairn.cascade import PHASE1_DETECTORS
    assert "typographic_signature" in PHASE1_DETECTORS


def test_result_has_required_fields(detector, config):
    r = detector.analyze(LLM_TYPOGRAPHY, config)
    assert r.layer_id == "typographic_signature"
    assert r.domain == "universal"
    assert r.compute_cost == "trivial"
    assert "A0" in r.attacker_tiers
    assert 0.0 <= r.p_llm <= 1.0
    assert 0.0 <= r.confidence <= 1.0


def test_signals_dict_has_all_keys(detector, config):
    r = detector.analyze(LLM_TYPOGRAPHY, config)
    expected = {
        "dash_style_ratio", "n_typographic_dashes", "curly_quote_ratio",
        "fancy_bullet_ratio", "oxford_comma_ratio", "heading_title_case",
        "typography_score", "n_headings", "n_words",
    }
    missing = expected - set(r.signals.keys())
    assert not missing, f"missing signals keys: {missing}"


def test_skips_below_word_floor(detector, config):
    r = detector.analyze(" ".join(["word"] * (_MIN_WORDS - 5)), config)
    assert r.determination == "SKIP"
    assert r.signals["skipped"] == "too_short"
    assert r.p_llm == 0.5
    assert r.confidence == 0.0


def test_typographic_text_outscores_plain_text(detector, config):
    llm = detector.analyze(LLM_TYPOGRAPHY, config)
    human = detector.analyze(HUMAN_TYPOGRAPHY, config)
    assert llm.determination != "SKIP"
    assert human.determination != "SKIP"
    assert llm.p_llm > human.p_llm


def test_every_signal_survives_normalization(detector, config):
    """The detector's central design constraint.

    `cairn/pipeline.py` normalizes before any detector runs and
    `cairn/feature_cache.py` keys the cache on the normalized form, so a
    signal that normalization destroys is a signal the detector can
    never observe. Scoring the raw and the normalized form must
    therefore agree.
    """
    raw = """## Key Considerations for Migration

The path depends on three factors — throughput, cost, and operational risk.

• **Throughput**: the cluster sustains twice the write rate.
• **Cost**: reserved capacity lowers spend, and the savings compound.
• **Operational Risk**: staged cutover limits radius, exposure, and rollback.

## Recommended Sequence

A phased approach — staged, reversible, and observable — with a soak period.
"""
    a = detector.analyze(raw, config)
    b = detector.analyze(normalize_text(raw), config)
    assert a.signals["typography_score"] == b.signals["typography_score"]
    assert a.p_llm == b.p_llm


def test_single_observation_cannot_saturate_a_ratio(detector, config):
    """A rate-based version of this detector would call one em-dash in a
    30-word note 33-per-1000-words and clamp its term to 1.0 — the exact
    length artefact that sank SpecDetect. Ratios gated on a minimum
    event count make that unreachable."""
    text = normalize_text(
        "we shipped the thing today — it took longer than planned but the "
        "rollout was clean and nobody had to page anyone about it overnight, "
        "so on balance i think we came out of the week ahead of where we "
        "expected to be when we started planning this back in the spring"
    )
    r = detector.analyze(text, config)
    assert r.determination != "SKIP"
    assert r.signals["n_typographic_dashes"] == 1
    assert r.signals["dash_style_ratio"] == 0.0, (
        "a single dash is an incident, not a habit"
    )


def test_ratios_report_zero_when_evidence_is_thin(detector, config):
    text = normalize_text(" ".join(["plain words here"] * 15))
    r = detector.analyze(text, config)
    for key in ("dash_style_ratio", "curly_quote_ratio",
                "fancy_bullet_ratio", "oxford_comma_ratio",
                "heading_title_case"):
        assert r.signals[key] == 0.0


def test_all_ratio_signals_are_bounded(detector, config):
    for text in (LLM_TYPOGRAPHY, HUMAN_TYPOGRAPHY):
        r = detector.analyze(text, config)
        for key in ("dash_style_ratio", "curly_quote_ratio",
                    "fancy_bullet_ratio", "oxford_comma_ratio",
                    "heading_title_case", "typography_score"):
            assert 0.0 <= r.signals[key] <= 1.0


def test_oxford_ratio_counts_both_comma_styles(detector, config):
    """Regression: the denominator must be every coordination, not just
    the bare ones — otherwise fully-Oxford text divides by zero-ish and
    reports 0.0."""
    oxford = normalize_text(
        "We measured latency, throughput, and cost. We tracked errors, "
        "retries, and timeouts. We logged reads, writes, and deletes across "
        "the whole fleet for a week to be sure the numbers were stable."
    )
    r = detector.analyze(oxford, config)
    assert r.signals["oxford_comma_ratio"] > 0.0


def test_heuristic_table_cannot_saturate():
    lo = _HEURISTIC_TYPOGRAPHY_TO_P_LLM[0][0]
    hi = _HEURISTIC_TYPOGRAPHY_TO_P_LLM[-1][0]
    assert lo <= 0.0 and hi >= 1.0


def test_heuristic_table_is_monotone_increasing():
    xs = [x for x, _ in _HEURISTIC_TYPOGRAPHY_TO_P_LLM]
    ys = [y for _, y in _HEURISTIC_TYPOGRAPHY_TO_P_LLM]
    assert xs == sorted(xs)
    assert ys == sorted(ys)
    assert interpolate(0.0, _HEURISTIC_TYPOGRAPHY_TO_P_LLM) < interpolate(
        1.0, _HEURISTIC_TYPOGRAPHY_TO_P_LLM
    )


def test_analyze_is_deterministic(detector, config):
    a = detector.analyze(LLM_TYPOGRAPHY, config)
    b = detector.analyze(LLM_TYPOGRAPHY, config)
    assert a.p_llm == b.p_llm
    assert a.signals == b.signals
    assert a.spans == b.spans


def test_spans_are_valid(detector, config):
    r = detector.analyze(LLM_TYPOGRAPHY, config)
    assert r.spans
    for s in r.spans:
        assert {"start", "end", "kind", "note"} <= set(s.keys())
        assert s["kind"] == "typography"
        assert 0 <= s["start"] < s["end"] <= len(LLM_TYPOGRAPHY)
        assert LLM_TYPOGRAPHY[s["start"]:s["end"]]


def test_min_events_constant_is_above_one():
    assert _MIN_EVENTS_FOR_RATIO >= 2


def test_module_singleton_is_wired():
    assert DETECTOR.id == "typographic_signature"

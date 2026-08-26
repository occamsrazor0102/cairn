"""Tests for the reasoning_artifact detector (roadmap Track A3).

Behavioural: deliberation-marked text must outscore the same-length
prose without markers, spans must land on the marker that caused the
score, and the SKIP / determinism / saturation contracts must hold.
"""
import pytest

from cairn.detectors._utils import interpolate
from cairn.detectors.reasoning_artifact import (
    DETECTOR,
    _HEURISTIC_ARTIFACT_TO_P_LLM,
    _MIN_WORDS,
    ReasoningArtifactDetector,
)
from tests.fixtures.human_samples import FORMAL_SOP


@pytest.fixture
def detector():
    return ReasoningArtifactDetector()


@pytest.fixture
def config():
    return {}


REASONING = """Let me think about this carefully. The user is asking about the
tradeoff between latency and throughput in a batched inference server.

First, I'll establish the baseline. With a batch size of one, each request is
served immediately, so latency is minimal but GPU utilisation is poor. The goal
here is to find the point where added queueing delay is still acceptable.

Actually, wait. That framing assumes requests arrive uniformly, which is rarely
true in production. Let me reconsider the arrival process.

Step 1: Model the arrival process as Poisson with rate lambda.
Step 2: Compute expected queue depth at each batch size.
Step 3: Solve for the batch size that meets the p99 latency target.

What we need is the smallest batch that saturates the GPU without breaching the
service objective. My approach is to sweep batch sizes and measure both metrics
directly rather than trusting the analytical model.
"""

PLAIN = """The tradeoff between latency and throughput in a batched inference
server depends on the arrival process. With a batch size of one, each request is
served immediately, so latency is minimal but GPU utilisation is poor. Larger
batches raise utilisation at the cost of queueing delay.

Arrival rates in production are bursty rather than uniform, which weakens the
usual analytical treatment. A Poisson model gives a tractable first estimate of
queue depth at each batch size, but measured sweeps are more reliable.

The smallest batch that saturates the GPU without breaching the service
objective is the operating point. Measuring both metrics across a sweep beats
trusting the closed-form result.
"""


def test_reasoning_artifact_registered():
    import cairn.detectors as registry
    assert "reasoning_artifact" in registry.get_all_detectors()


def test_reasoning_artifact_placed_in_phase2():
    from cairn.cascade import PHASE2_DETECTORS
    assert "reasoning_artifact" in PHASE2_DETECTORS


def test_result_has_required_fields(detector, config):
    r = detector.analyze(REASONING, config)
    assert r.layer_id == "reasoning_artifact"
    assert r.domain == "prose"
    assert r.compute_cost == "cheap"
    assert "A0" in r.attacker_tiers
    assert 0.0 <= r.p_llm <= 1.0
    assert 0.0 <= r.confidence <= 1.0


def test_signals_dict_has_all_keys(detector, config):
    r = detector.analyze(REASONING, config)
    expected = {
        "deliberation_density", "self_correction_density", "intent_density",
        "enum_scaffold_density", "register_uniformity", "artifact_score",
        "n_sentences", "n_words",
    }
    missing = expected - set(r.signals.keys())
    assert not missing, f"missing signals keys: {missing}"


def test_skips_below_word_floor(detector, config):
    r = detector.analyze(" ".join(["word"] * (_MIN_WORDS - 10)), config)
    assert r.determination == "SKIP"
    assert r.signals["skipped"] == "too_short"
    assert r.p_llm == 0.5
    assert r.confidence == 0.0


def test_deliberation_markers_raise_the_score(detector, config):
    """The behavioural claim: same topic, same length, markers removed."""
    marked = detector.analyze(REASONING, config)
    plain = detector.analyze(PLAIN, config)
    assert marked.determination != "SKIP"
    assert plain.determination != "SKIP"
    assert marked.p_llm > plain.p_llm
    assert marked.signals["deliberation_density"] > 0
    assert plain.signals["deliberation_density"] == 0


def test_self_correction_is_detected(detector, config):
    r = detector.analyze(REASONING, config)
    assert r.signals["self_correction_density"] > 0


def test_prose_without_markers_reports_zero_densities(detector, config):
    r = detector.analyze(PLAIN, config)
    assert r.signals["deliberation_density"] == 0
    assert r.signals["self_correction_density"] == 0
    assert r.signals["enum_scaffold_density"] == 0


def test_register_uniformity_is_bounded(detector, config):
    for text in (REASONING, PLAIN, FORMAL_SOP):
        r = detector.analyze(text, config)
        if r.determination == "SKIP":
            continue
        assert 0.0 <= r.signals["register_uniformity"] <= 1.0
        assert 0.0 <= r.signals["artifact_score"] <= 1.0


def test_heuristic_table_cannot_saturate():
    """See test_compression_profile for why this guard exists."""
    lo = _HEURISTIC_ARTIFACT_TO_P_LLM[0][0]
    hi = _HEURISTIC_ARTIFACT_TO_P_LLM[-1][0]
    assert lo <= 0.0 and hi >= 1.0


def test_heuristic_table_is_monotone_increasing():
    xs = [x for x, _ in _HEURISTIC_ARTIFACT_TO_P_LLM]
    ys = [y for _, y in _HEURISTIC_ARTIFACT_TO_P_LLM]
    assert xs == sorted(xs)
    assert ys == sorted(ys)
    assert interpolate(0.0, _HEURISTIC_ARTIFACT_TO_P_LLM) < interpolate(
        1.0, _HEURISTIC_ARTIFACT_TO_P_LLM
    )


def test_analyze_is_deterministic(detector, config):
    a = detector.analyze(REASONING, config)
    b = detector.analyze(REASONING, config)
    assert a.p_llm == b.p_llm
    assert a.signals == b.signals
    assert a.spans == b.spans


def test_spans_are_valid_and_disjoint(detector, config):
    r = detector.analyze(REASONING, config)
    assert r.spans, "deliberation markers should produce highlights"
    last_end = -1
    for s in r.spans:
        assert {"start", "end", "kind", "note"} <= set(s.keys())
        assert s["kind"] == "reasoning"
        assert 0 <= s["start"] < s["end"] <= len(REASONING)
        assert REASONING[s["start"]:s["end"]].strip()
        assert s["start"] >= last_end
        last_end = s["end"]


def test_no_spans_without_markers(detector, config):
    r = detector.analyze(PLAIN, config)
    assert r.spans == []


def test_is_first_phase2_span_emitter():
    """Guards the UI contract: `reasoning` must be a known span kind.

    Every other Phase-2 detector leaves `spans` empty, so this is the
    first time the highlight palette is exercised from Phase 2.
    """
    from pathlib import Path
    html = Path("cairn/gui/static/index.html").read_text(encoding="utf-8")
    assert "span-reasoning" in html
    assert "reasoning: \"reasoning artifact\"" in html


def test_module_singleton_is_wired():
    assert DETECTOR.id == "reasoning_artifact"

"""Token cohesiveness detector tests.

The detector embeds the full text plus n_samples deletion-of-one-word
variants and reports cosine-distance statistics. The encoder is mocked
here so the suite stays light; runtime correctness against MiniLM is
covered by the calibrated-profile benchmarks, not unit tests.
"""
import hashlib
import random

import pytest


np = pytest.importorskip("numpy")


@pytest.fixture
def detector(monkeypatch):
    import cairn.detectors.token_cohesiveness as mod

    class FakeST:
        def __init__(self, *a, **k):
            pass

        def encode(self, text, convert_to_numpy=True):
            h = hashlib.sha256(text.encode("utf-8")).digest()
            return np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0

    monkeypatch.setattr(mod, "SentenceTransformer", FakeST)
    monkeypatch.setattr(mod, "_HAS_ST", True)
    return mod.TokenCohesivenessDetector(n_samples=10)


_SAMPLE_TEXT = (
    "The mitochondrial membrane potential collapses when uncoupling proteins "
    "activate under cold exposure, releasing stored chemical energy as heat. "
    "Brown adipose tissue is the primary thermogenic depot in mammals, and "
    "its activity scales with surface-area-to-volume ratio across species. "
    "Recent studies in beige adipocytes have shown that creatine cycling "
    "contributes substantially to non-shivering thermogenesis in adults."
)


def test_token_cohesiveness_is_deterministic_across_calls(detector):
    r1 = detector.analyze(_SAMPLE_TEXT, {})
    r2 = detector.analyze(_SAMPLE_TEXT, {})
    assert r1.raw_score == r2.raw_score
    assert r1.p_llm == r2.p_llm
    assert r1.confidence == r2.confidence
    assert r1.signals == r2.signals


def test_token_cohesiveness_unaffected_by_global_rng(detector):
    # Future regression: if someone re-introduces a global-RNG call,
    # poisoning random's global state between analyze() calls will
    # produce different output -- this test catches that.
    random.seed(1)
    r1 = detector.analyze(_SAMPLE_TEXT, {})
    random.seed(999_999)
    for _ in range(50):
        random.random()
    r2 = detector.analyze(_SAMPLE_TEXT, {})
    assert r1.signals == r2.signals
    assert r1.p_llm == r2.p_llm

"""Cross-submission similarity detector tests."""
import pytest

from cairn.detectors.cross_similarity import DETECTOR


def test_registered_in_registry():
    import cairn.detectors as registry
    assert "cross_similarity" in registry.get_all_detectors()


def test_single_text_returns_skip():
    r = DETECTOR.analyze("some text", {})
    assert r.determination == "SKIP"


def test_batch_duplicates_produce_high_similarity():
    shared = "the quick brown fox jumps over the lazy dog"
    texts = {
        "a": shared + " one two three",
        "b": shared + " four five six",
        "c": "completely unrelated text about weather patterns and climate",
    }
    out = DETECTOR.analyze_batch(texts, {"shingle_k": 3, "jaccard_threshold": 0.2})
    assert set(out.keys()) == {"a", "b", "c"}
    # a and b share a long prefix → high Jaccard
    assert out["a"].signals["max_jaccard"] > 0.3
    assert out["b"].signals["max_jaccard"] > 0.3
    # c is unrelated → low max
    assert out["c"].signals["max_jaccard"] < 0.2


def test_batch_singleton_skips():
    out = DETECTOR.analyze_batch({"solo": "just one submission"}, {})
    assert out["solo"].determination == "SKIP"


def test_batch_empty_returns_empty():
    assert DETECTOR.analyze_batch({}, {}) == {}


def test_high_similarity_promotes_to_red_or_amber():
    texts = {f"s{i}": "template line one. template line two. template line three." for i in range(3)}
    out = DETECTOR.analyze_batch(texts, {"shingle_k": 3, "jaccard_threshold": 0.3})
    for sid, r in out.items():
        assert r.p_llm >= 0.75
        assert r.determination in ("RED", "AMBER")


def test_cross_similarity_uses_retrieval_index_when_present():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from cairn.detectors.cross_similarity import CrossSimilarityDetector
    from cairn.index.retrieval import RetrievalIndex

    rx = RetrievalIndex()
    rx.add("historical_llm_1", "Once upon a time the model produced text. " * 10)
    rx.add("historical_human_1", "I remember when I was a kid playing in the garden. " * 10)

    det = CrossSimilarityDetector()
    config = {
        "use_retrieval_index": True,
        "_index_handle": rx,
        "neighbor_score_threshold": 0.85,
    }
    text = "Once upon a time the model produced text. " * 10
    res = det.analyze(text, config)
    assert res.determination != "SKIP"
    assert res.signals["top_neighbor_id"] == "historical_llm_1"
    assert res.signals["top_neighbor_score"] > 0.85


def test_cross_similarity_falls_back_to_skip_when_no_index():
    from cairn.detectors.cross_similarity import CrossSimilarityDetector
    det = CrossSimilarityDetector()
    res = det.analyze("Some text. " * 30, {"use_retrieval_index": False})
    assert res.determination == "SKIP"

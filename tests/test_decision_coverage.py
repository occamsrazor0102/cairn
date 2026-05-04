import pytest
from cairn.contracts import FusionResult, LayerResult
from cairn.decision import DecisionEngine


def _engine():
    cfg = {"decision": {"red_threshold": 0.75, "amber_threshold": 0.50,
                        "yellow_threshold": 0.25,
                        "abstention": {"enabled": True, "max_prediction_set_size": 3},
                        "mixed": {"enabled": True, "min_mixed_probability": 0.60}}}
    return DecisionEngine(cfg)


def _fr(p, conformal_set):
    return FusionResult(
        p_llm=p, confidence_interval=(max(0, p - 0.1), min(1, p + 0.1)),
        prediction_set=["YELLOW"], feature_contributions={}, top_contributors=[],
        conformal_set=conformal_set, conformal_alpha=0.1, fusion_mode="ebm",
    )


def test_empty_conformal_set_yields_uncertain_with_reason():
    eng = _engine()
    det = eng.decide(_fr(0.42, []), layer_results=[])
    assert det.label == "UNCERTAIN"
    assert det.abstention_reason == "conformal_empty"
    assert det.coverage_guarantee_applies is False
    assert ("out-of-distribution" in det.reason.lower()) or ("neither" in det.reason.lower())


def test_singleton_conformal_set_uses_that_label():
    eng = _engine()
    det = eng.decide(_fr(0.82, ["LLM"]), layer_results=[])
    assert det.label == "RED"
    assert det.coverage_guarantee_applies is True


def test_two_class_conformal_set_yields_uncertain():
    eng = _engine()
    det = eng.decide(_fr(0.5, ["human", "LLM"]), layer_results=[])
    assert det.label == "UNCERTAIN"
    assert det.abstention_reason == "conformal_set"
    assert det.coverage_guarantee_applies is True


def test_preamble_critical_override_marks_coverage_off():
    eng = _engine()
    crit = LayerResult(
        layer_id="preamble", domain="universal", raw_score=1.0, p_llm=0.95,
        confidence=0.95, signals={"severity": "CRITICAL"}, determination="RED",
        attacker_tiers=[], compute_cost="trivial", min_text_length=0,
    )
    det = eng.decide(_fr(0.95, ["LLM"]), layer_results=[crit])
    assert det.label == "RED"
    assert det.override_applied is True
    assert det.coverage_guarantee_applies is False


def test_mixed_override_marks_coverage_off():
    eng = _engine()
    mb = LayerResult(
        layer_id="mixed_boundary", domain="universal", raw_score=0.7, p_llm=0.7,
        confidence=0.7, signals={"mixed_probability": 0.85, "n_boundaries": 3,
                                 "segment_determinations": []},
        determination="AMBER", attacker_tiers=[], compute_cost="moderate", min_text_length=0,
    )
    det = eng.decide(_fr(0.5, ["human", "LLM"]), layer_results=[mb])
    assert det.label == "MIXED"
    assert det.coverage_guarantee_applies is False

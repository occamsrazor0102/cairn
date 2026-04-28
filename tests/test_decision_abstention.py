import pytest
from beet.decision import DecisionEngine
from beet.contracts import FusionResult


def _engine(max_set=2):
    return DecisionEngine({
        "decision": {
            "red_threshold": 0.75,
            "amber_threshold": 0.50,
            "yellow_threshold": 0.25,
            "abstention": {"enabled": True, "max_prediction_set_size": max_set},
        }
    })


def test_abstention_uses_conformal_set_when_present():
    engine = _engine(max_set=2)
    fusion = FusionResult(
        p_llm=0.50,
        confidence_interval=(0.30, 0.70),
        prediction_set=["AMBER"],
        conformal_set=["human", "LLM"],
        conformal_alpha=0.10,
        feature_contributions={},
        top_contributors=[],
    )
    det = engine.decide(fusion, layer_results=[])
    assert det.label == "UNCERTAIN"
    assert det.abstention_reason == "conformal_set"


def test_no_abstention_when_conformal_set_is_single_class():
    engine = _engine(max_set=2)
    fusion = FusionResult(
        p_llm=0.82,
        confidence_interval=(0.70, 0.91),
        prediction_set=["RED"],
        conformal_set=["LLM"],
        conformal_alpha=0.10,
        feature_contributions={},
        top_contributors=[],
    )
    det = engine.decide(fusion, layer_results=[])
    assert det.label != "UNCERTAIN"
    assert det.abstention_reason is None


def test_heuristic_fallback_when_conformal_set_is_none():
    engine = _engine(max_set=3)
    fusion = FusionResult(
        p_llm=0.50,
        confidence_interval=(0.30, 0.70),
        prediction_set=["YELLOW", "AMBER", "RED"],
        conformal_set=None,
        feature_contributions={},
        top_contributors=[],
    )
    det = engine.decide(fusion, layer_results=[])
    assert det.label == "UNCERTAIN"
    assert det.abstention_reason == "prediction_set"


def test_no_heuristic_abstention_with_narrow_prediction_set():
    engine = _engine(max_set=3)
    fusion = FusionResult(
        p_llm=0.82,
        confidence_interval=(0.70, 0.91),
        prediction_set=["RED"],
        conformal_set=None,
        feature_contributions={},
        top_contributors=[],
    )
    det = engine.decide(fusion, layer_results=[])
    assert det.label == "RED"
    assert det.abstention_reason is None

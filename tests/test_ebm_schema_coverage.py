import math
import pytest
from cairn.fusion.ebm import FeatureAssembler, ASSEMBLER_FEATURE_NAMES
from cairn.cascade import PHASE1_DETECTORS, PHASE2_DETECTORS, PHASE3_DETECTORS, PHASE4_DETECTORS
from cairn.contracts import LayerResult

_EXCLUDED = {"contributor_graph"}
ALL_CASCADE_DETECTORS = (
    set(PHASE1_DETECTORS) | set(PHASE2_DETECTORS) |
    set(PHASE3_DETECTORS) | set(PHASE4_DETECTORS)
) - _EXCLUDED

_ALIAS = {
    "fingerprint_vocab": "fingerprint_p_llm",
    "prompt_structure": "ps_p_llm",
    "voice_spec": "vs_p_llm",
    "instruction_density": "idi_p_llm",
    "surprisal_dynamics": "surprisal_p_llm",
    "contrastive_lm": "binoculars_p_llm",
    "token_cohesiveness": "tocsin_p_llm",
}


def _layer(det_id: str, p_llm: float = 0.5) -> LayerResult:
    return LayerResult(
        layer_id=det_id, domain="prose", raw_score=p_llm, p_llm=p_llm,
        confidence=0.8, signals={}, determination="AMBER",
        attacker_tiers=[], compute_cost="cheap", min_text_length=0,
    )


@pytest.mark.parametrize("det_id", sorted(ALL_CASCADE_DETECTORS))
def test_feature_key_exists_for_every_cascade_detector(det_id):
    assembler = FeatureAssembler()
    vec = assembler.assemble([_layer(det_id)], word_count=100, domain="prose")
    expected_key = _ALIAS.get(det_id, f"{det_id}_p_llm")
    assert expected_key in vec, f"No feature key for detector '{det_id}'"
    assert not math.isnan(vec[expected_key]), "Present detector must not produce NaN"


def test_absent_detector_produces_nan():
    assembler = FeatureAssembler()
    vec = assembler.assemble([_layer("preamble")], word_count=50, domain="prose")
    assert not math.isnan(vec["preamble_p_llm"])
    assert math.isnan(vec["dna_gpt_p_llm"])
    assert math.isnan(vec["perturbation_p_llm"])


def test_assembler_feature_names_constant_matches_assemble_output():
    assembler = FeatureAssembler()
    vec = assembler.assemble([], word_count=0, domain="prose")
    assert tuple(vec.keys()) == ASSEMBLER_FEATURE_NAMES


def test_assembler_reads_p_llm_not_raw_score():
    assembler = FeatureAssembler()
    layer = _layer("preamble", p_llm=0.77)
    layer.raw_score = 0.10
    vec = assembler.assemble([layer], word_count=50, domain="prose")
    assert vec["preamble_p_llm"] == pytest.approx(0.77), (
        "FeatureAssembler must read p_llm, not raw_score"
    )

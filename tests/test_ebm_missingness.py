import math
import pytest
from cairn.contracts import LayerResult
from cairn.fusion.ebm import FeatureAssembler, ASSEMBLER_FEATURE_NAMES


def _lr(layer_id, p_llm=0.6, confidence=0.7, **signals):
    return LayerResult(
        layer_id=layer_id, domain="universal", raw_score=p_llm, p_llm=p_llm,
        confidence=confidence, signals=signals, determination="AMBER",
        attacker_tiers=[], compute_cost="trivial", min_text_length=0,
    )


def test_missing_indicator_per_detector():
    a = FeatureAssembler()
    vec = a.assemble([], word_count=200, domain="prose")
    assert vec["preamble_missing"] == 1.0
    assert vec["fingerprint_missing"] == 1.0
    assert vec["binoculars_missing"] == 1.0
    vec2 = a.assemble([_lr("preamble", p_llm=0.9, severity="HIGH", n_matches=2)],
                      word_count=200, domain="prose")
    assert vec2["preamble_missing"] == 0.0


def test_missing_indicator_in_canonical_feature_list():
    assert "preamble_missing" in ASSEMBLER_FEATURE_NAMES
    assert "fingerprint_missing" in ASSEMBLER_FEATURE_NAMES


def test_confidence_per_detector():
    a = FeatureAssembler()
    vec = a.assemble([_lr("preamble", p_llm=0.9, confidence=0.85, severity="HIGH")],
                     word_count=200, domain="prose")
    assert vec["preamble_confidence"] == pytest.approx(0.85, abs=1e-6)
    assert math.isnan(vec["nssi_confidence"])


def test_ebm_uses_stored_column_means_not_half():
    pytest.importorskip("interpret")
    from cairn.fusion.training import train_ebm
    a = FeatureAssembler()
    rows = []
    for i in range(40):
        if i < 20:
            r = _lr("contrastive_lm", p_llm=0.2, confidence=0.6, binoculars_ratio=1.0)
        else:
            r = _lr("contrastive_lm", p_llm=0.8, confidence=0.6, binoculars_ratio=0.7)
        rows.append((a.assemble([r], word_count=200, domain="prose"), int(i >= 20)))
    X = [r for r, _ in rows]
    y = [lab for _, lab in rows]
    model = train_ebm(X, y)
    means = getattr(model, "_cairn_column_means", None)
    assert means is not None
    assert means["binoculars_ratio"] == pytest.approx((1.0 + 0.7) / 2, abs=1e-6)


def test_ebm_raises_when_artifact_requires_unknown_features():
    from cairn.fusion.ebm import EBMFusion

    class _StubModel:
        _cairn_feature_names = ["preamble_p_llm", "imaginary_feature_xyz"]
        _cairn_column_means = {}

    with pytest.raises(RuntimeError, match="imaginary_feature_xyz"):
        EBMFusion(model=_StubModel())

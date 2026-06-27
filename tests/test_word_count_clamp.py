"""OOD-length guard: FeatureAssembler.word_count_clamp caps the length features.

On prompts_only the EBM's word_count shape function saturates to ~+2.1 (LLM) in
the sparse long tail (>~340 words), so prompts far longer than the training
distribution get flagged on length alone (see
docs/benchmarks/lighthouse_prompts_calibration.md). The clamp caps word_count
before word_count/log_word_count/is_* are computed. Default None = no change.
"""
import math

from cairn.fusion.ebm import EBMFusion, FeatureAssembler


def test_clamp_caps_length_features():
    vec = FeatureAssembler(word_count_clamp=200).assemble([], word_count=600, domain="prompt")
    assert vec["word_count"] == 200.0
    assert vec["log_word_count"] == math.log1p(200)
    # 200 falls in the medium bucket, not long
    assert vec["is_long"] == 0.0
    assert vec["is_medium"] == 1.0
    assert vec["is_short"] == 0.0


def test_no_clamp_is_default_behaviour():
    vec = FeatureAssembler().assemble([], word_count=600, domain="prompt")
    assert vec["word_count"] == 600.0
    assert vec["log_word_count"] == math.log1p(600)
    assert vec["is_long"] == 1.0


def test_clamp_inactive_when_below_threshold():
    vec = FeatureAssembler(word_count_clamp=200).assemble([], word_count=50, domain="prompt")
    assert vec["word_count"] == 50.0
    assert vec["is_short"] == 1.0


def test_clamp_plumbs_through_ebmfusion():
    f = EBMFusion(word_count_clamp=200)
    assert f._assembler._word_count_clamp == 200
    assert FeatureAssembler()._word_count_clamp is None

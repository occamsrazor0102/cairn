"""Behavior tests for the faithful TOCSIN token_cohesiveness detector.

No-torch paths and the pure-Python cohesiveness mapping run everywhere; the
BARTScore path is exercised via monkeypatched scoring so CI need not
download facebook/bart-base to assert wiring.
"""
import pytest

import cairn.detectors.token_cohesiveness as tc
from cairn.detectors.token_cohesiveness import (
    DETECTOR,
    TOCSIN_COHESIVENESS_SIGNATURE,
    TOCSIN_COMBINED_SIGNATURE,
    TOCSIN_SIGNATURE_TO_RAW_CHANNEL,
    TokenCohesivenessDetector,
    _heuristic_cohesiveness_to_p_llm,
)


class FakeTensor:
    def __init__(self):
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


class FakeTokenizer:
    def __init__(self):
        self.tensors = []

    def __call__(self, _text, **_kwargs):
        tensor = FakeTensor()
        self.tensors.append(tensor)
        return {"input_ids": tensor}


class FakeLoss:
    def item(self):
        return 1.25


class FakeBartOutput:
    loss = FakeLoss()


class FakeBart:
    def __init__(self):
        self.to_calls = []
        self.eval_called = False

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True

    def __call__(self, **_kwargs):
        return FakeBartOutput()


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, _exc_type, _exc, _traceback):
        return False


class FakeCudaTorch:
    class cuda:
        @staticmethod
        def is_available():
            return True

    @staticmethod
    def no_grad():
        return FakeNoGrad()


class FakeCpuTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False


def test_skips_short_text():
    d = TokenCohesivenessDetector()
    result = d.analyze("too short to score", {})
    assert result.determination == "SKIP"
    assert result.signals["skipped"] == "too_short"


def test_skips_cleanly_when_tier2_absent(monkeypatch):
    monkeypatch.setattr(tc, "_try_import_torch", lambda: None)
    d = TokenCohesivenessDetector()
    long_text = " ".join(["word"] * 80)
    result = d.analyze(long_text, {})
    assert result.determination == "SKIP"
    assert result.signals["skipped"] == "tier2_unavailable"


def test_cohesiveness_heuristic_is_monotone_increasing():
    # More cohesive (u closer to 0) → higher p(LLM).
    lo = _heuristic_cohesiveness_to_p_llm(-4.0)
    mid = _heuristic_cohesiveness_to_p_llm(-1.5)
    hi = _heuristic_cohesiveness_to_p_llm(0.0)
    assert lo < mid < hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_combined_channel_surfaces_signals(monkeypatch):
    d = TokenCohesivenessDetector()
    # Force torch present, BART score fixed, base channel present.
    monkeypatch.setattr(
        "cairn.detectors.token_cohesiveness._try_import_torch", lambda: object()
    )
    monkeypatch.setattr(d, "_ensure_bart", lambda: None)
    monkeypatch.setattr(d, "_bart_score", lambda src, tgt: -1.0)
    monkeypatch.setattr(d._base, "is_available", lambda: True)
    monkeypatch.setattr(d._base, "curvature", lambda text: 1.2)

    result = d.analyze(" ".join(["word"] * 100), {})
    assert result.layer_id == "token_cohesiveness"
    assert result.determination != "SKIP"
    assert result.signals["cohesiveness_u"] == -1.0
    assert result.signals["base_v"] == 1.2
    assert "combined_w" in result.signals
    assert result.signals["calibration_signature"] == TOCSIN_COMBINED_SIGNATURE
    assert result.signals["calibration_raw_channel"] == "combined_w"
    assert result.signals["calibration_identity"] == {
        "raw_channel": "combined_w",
        "tocsin_algorithm": "random-deletion-bartscore-v1",
        "bart_model": "facebook/bart-base",
        "base_algorithm": "fast-detectgpt-curvature-v1",
        "base_model": "gpt2",
    }
    assert result.raw_score == pytest.approx(
        result.signals["combined_w"], abs=1e-5
    )
    assert 0.0 <= result.p_llm <= 1.0


def test_standalone_fallback_when_base_unavailable(monkeypatch):
    d = TokenCohesivenessDetector()
    monkeypatch.setattr(
        "cairn.detectors.token_cohesiveness._try_import_torch", lambda: object()
    )
    monkeypatch.setattr(d, "_ensure_bart", lambda: None)
    monkeypatch.setattr(d, "_bart_score", lambda src, tgt: -0.5)
    monkeypatch.setattr(d._base, "is_available", lambda: False)

    result = d.analyze(" ".join(["word"] * 100), {})
    assert result.determination != "SKIP"
    assert result.signals["base_v"] is None
    assert result.signals["cohesiveness_u"] == -0.5
    assert (
        result.signals["calibration_signature"]
        == TOCSIN_COHESIVENESS_SIGNATURE
    )
    assert result.signals["calibration_raw_channel"] == "cohesiveness_u"
    assert result.signals["calibration_identity"] == {
        "raw_channel": "cohesiveness_u",
        "tocsin_algorithm": "random-deletion-bartscore-v1",
        "bart_model": "facebook/bart-base",
    }
    assert result.raw_score == pytest.approx(
        result.signals["cohesiveness_u"], abs=1e-5
    )
    assert 0.0 <= result.p_llm <= 1.0


def test_module_detector_id():
    assert DETECTOR.id == "token_cohesiveness"


def test_signature_raw_channel_mapping_is_public_and_stable():
    assert TOCSIN_COMBINED_SIGNATURE == (
        "tocsin-bart-fastdetectgpt-combined-w-v1"
    )
    assert TOCSIN_COHESIVENESS_SIGNATURE == "tocsin-bart-cohesiveness-u-v1"
    assert TOCSIN_SIGNATURE_TO_RAW_CHANNEL == {
        TOCSIN_COMBINED_SIGNATURE: "combined_w",
        TOCSIN_COHESIVENESS_SIGNATURE: "cohesiveness_u",
    }


def _active_result(monkeypatch, detector, *, base_available=True):
    monkeypatch.setattr(tc, "_try_import_torch", lambda: object())
    monkeypatch.setattr(detector, "_ensure_bart", lambda: None)
    monkeypatch.setattr(detector, "_bart_score", lambda _src, _tgt: -1.0)
    monkeypatch.setattr(detector._base, "is_available", lambda: base_available)
    if base_available:
        monkeypatch.setattr(detector._base, "curvature", lambda _text: 1.2)
    return detector.analyze(" ".join(["word"] * 100), {})


def test_custom_model_tuple_gets_stable_identity_bound_signature(
    monkeypatch,
):
    monkeypatch.setenv("CAIRN_FAST_DETECTGPT_MODEL", "org/custom-fast")
    first = TokenCohesivenessDetector(model_name="org/custom-bart")
    second = TokenCohesivenessDetector(model_name="org/custom-bart")

    first_result = _active_result(monkeypatch, first)
    second_result = _active_result(monkeypatch, second)

    signature = first_result.signals["calibration_signature"]
    assert signature == second_result.signals["calibration_signature"]
    assert signature.startswith(TOCSIN_COMBINED_SIGNATURE + "-")
    assert signature != TOCSIN_COMBINED_SIGNATURE
    assert first_result.signals["calibration_identity"] == {
        "raw_channel": "combined_w",
        "tocsin_algorithm": "random-deletion-bartscore-v1",
        "bart_model": "org/custom-bart",
        "base_algorithm": "fast-detectgpt-curvature-v1",
        "base_model": "org/custom-fast",
    }


def test_fallback_signature_depends_on_bart_but_not_unused_base_model(monkeypatch):
    monkeypatch.setenv("CAIRN_FAST_DETECTGPT_MODEL", "org/custom-fast")
    default_bart = TokenCohesivenessDetector()
    custom_bart = TokenCohesivenessDetector(model_name="org/custom-bart")

    default_result = _active_result(monkeypatch, default_bart, base_available=False)
    custom_result = _active_result(monkeypatch, custom_bart, base_available=False)

    assert (
        default_result.signals["calibration_signature"]
        == TOCSIN_COHESIVENESS_SIGNATURE
    )
    assert custom_result.signals["calibration_signature"].startswith(
        TOCSIN_COHESIVENESS_SIGNATURE + "-"
    )
    assert "base_model" not in default_result.signals["calibration_identity"]
    assert "base_model" not in custom_result.signals["calibration_identity"]


def test_tocsin_moves_bart_tensors_and_fastdetectgpt_to_cuda(monkeypatch):
    tokenizer = FakeTokenizer()
    bart = FakeBart()
    monkeypatch.setattr(tc, "_try_import_torch", lambda: FakeCudaTorch)
    monkeypatch.setattr(
        tc,
        "_load_bart_components",
        lambda _name: (tokenizer, bart),
    )

    detector = TokenCohesivenessDetector()

    assert detector._device == "cuda"
    assert detector._base._device == "cuda"
    detector._ensure_bart()
    assert bart.to_calls == ["cuda"]
    assert bart.eval_called is True
    assert detector._bart_score("source", "target") == -1.25
    assert [tensor.to_calls for tensor in tokenizer.tensors] == [
        ["cuda"],
        ["cuda"],
    ]


def test_tocsin_defaults_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(tc, "_try_import_torch", lambda: FakeCpuTorch)

    detector = TokenCohesivenessDetector()

    assert detector._device == "cpu"
    assert detector._base._device == "cpu"


def test_tocsin_honors_explicit_device(monkeypatch):
    monkeypatch.setattr(tc, "_try_import_torch", lambda: FakeCudaTorch)

    detector = TokenCohesivenessDetector(device="cpu")

    assert detector._device == "cpu"
    assert detector._base._device == "cpu"


def test_lazy_tocsin_passes_explicit_device_from_config(monkeypatch):
    monkeypatch.setattr(tc._LazyTOCSIN, "_instance", None)
    monkeypatch.setattr(tc, "_try_import_torch", lambda: FakeCudaTorch)
    wrapper = tc._LazyTOCSIN()

    wrapper.analyze(
        "too short",
        {"device": "cpu", "model": "fake-bart"},
    )

    assert wrapper._instance._model_name == "fake-bart"
    assert wrapper._instance._device == "cpu"
    assert wrapper._instance._base._device == "cpu"

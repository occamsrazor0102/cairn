"""Behavior tests for the fast_detectgpt detector.

The no-torch paths (short-text SKIP, tier2-unavailable SKIP, heuristic
monotonicity) run everywhere; the real-model path is exercised via a
monkeypatched curvature so CI doesn't download GPT-2 to assert wiring.
"""
import cairn.benchmarks.fast_detectgpt as benchmark_fast_detectgpt
import cairn.detectors.fast_detectgpt as detector_fast_detectgpt
from cairn.benchmarks.fast_detectgpt import FastDetectGPTAdapter
from cairn.detectors.fast_detectgpt import FastDetectGPTDetector, DETECTOR


class FakeCudaTorch:
    class cuda:
        @staticmethod
        def is_available():
            return True


class FakeCpuTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False


def test_skips_short_text():
    d = FastDetectGPTDetector()
    result = d.analyze("only a few words here", {})
    assert result.determination == "SKIP"
    assert result.signals["skipped"] == "too_short"


def test_skips_when_tier2_unavailable(monkeypatch):
    d = FastDetectGPTDetector()
    monkeypatch.setattr(d._adapter, "is_available", lambda: False)
    monkeypatch.setattr(
        d._adapter,
        "available_dependencies",
        lambda: {"torch": False, "transformers": False},
    )
    long_text = " ".join(["word"] * 80)
    result = d.analyze(long_text, {})
    assert result.determination == "SKIP"
    assert result.signals["skipped"] == "tier2_unavailable"
    assert result.p_llm == 0.5
    assert result.confidence == 0.0


def test_curvature_maps_into_unit_range_and_surfaces_signal(monkeypatch):
    d = FastDetectGPTDetector()
    # Bypass model load: force availability and a fixed curvature.
    monkeypatch.setattr(d._adapter, "is_available", lambda: True)
    monkeypatch.setattr(d._adapter, "curvature", lambda text: 2.5)
    result = d.analyze(" ".join(["word"] * 100), {})
    assert result.layer_id == "fast_detectgpt"
    assert result.determination != "SKIP"
    assert result.signals["curvature"] == 2.5
    assert 0.0 <= result.p_llm <= 1.0
    # High positive curvature → LLM-leaning.
    assert result.p_llm > 0.5
    # raw_score carries the uncalibrated curvature for the detector calibrator.
    assert result.raw_score == 2.5


def test_negative_curvature_leans_human(monkeypatch):
    d = FastDetectGPTDetector()
    monkeypatch.setattr(d._adapter, "is_available", lambda: True)
    monkeypatch.setattr(d._adapter, "curvature", lambda text: -1.5)
    result = d.analyze(" ".join(["word"] * 100), {})
    assert result.p_llm < 0.5


def test_module_detector_id():
    assert DETECTOR.id == "fast_detectgpt"
    assert DETECTOR.domain == "universal"


def test_standalone_fastdetectgpt_uses_cuda_when_available(monkeypatch):
    monkeypatch.delenv("CAIRN_FAST_DETECTGPT_DEVICE", raising=False)
    monkeypatch.setattr(
        benchmark_fast_detectgpt,
        "_try_import_torch",
        lambda: (FakeCudaTorch, True),
    )

    detector = FastDetectGPTDetector()

    assert detector._adapter._device == "cuda"


def test_fastdetectgpt_adapter_defaults_to_cpu_without_cuda(monkeypatch):
    monkeypatch.delenv("CAIRN_FAST_DETECTGPT_DEVICE", raising=False)
    monkeypatch.setattr(
        benchmark_fast_detectgpt,
        "_try_import_torch",
        lambda: (FakeCpuTorch, True),
    )

    adapter = FastDetectGPTAdapter()

    assert adapter._device == "cpu"


def test_fastdetectgpt_adapter_honors_environment_before_cuda(monkeypatch):
    monkeypatch.setenv("CAIRN_FAST_DETECTGPT_DEVICE", "cpu")
    monkeypatch.setattr(
        benchmark_fast_detectgpt,
        "_try_import_torch",
        lambda: (FakeCudaTorch, True),
    )

    adapter = FastDetectGPTAdapter()

    assert adapter._device == "cpu"


def test_fastdetectgpt_adapter_honors_explicit_device_before_environment(
    monkeypatch,
):
    monkeypatch.setenv("CAIRN_FAST_DETECTGPT_DEVICE", "cuda")
    monkeypatch.setattr(
        benchmark_fast_detectgpt,
        "_try_import_torch",
        lambda: (FakeCudaTorch, True),
    )

    adapter = FastDetectGPTAdapter(device="cpu")

    assert adapter._device == "cpu"


def test_lazy_fastdetectgpt_passes_explicit_device_from_config(monkeypatch):
    monkeypatch.setattr(
        detector_fast_detectgpt._LazyFastDetectGPT,
        "_instance",
        None,
    )
    monkeypatch.delenv("CAIRN_FAST_DETECTGPT_DEVICE", raising=False)
    monkeypatch.setattr(
        benchmark_fast_detectgpt,
        "_try_import_torch",
        lambda: (FakeCpuTorch, True),
    )
    wrapper = detector_fast_detectgpt._LazyFastDetectGPT()

    wrapper.analyze("too short", {"device": "cuda:7"})

    assert wrapper._instance._adapter._device == "cuda:7"

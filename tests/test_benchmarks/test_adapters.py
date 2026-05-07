"""Structural tests for the competitor-adapter shelf.

These tests do NOT exercise any third-party network call or any
HuggingFace model download. Lazy-import gates make ``is_available()``
honest about missing deps, and we mock the LM forward pass for the
OSS adapters so ``score()`` returns a float in [0, 1] without ever
touching ``torch`` or ``transformers``.
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from cairn.benchmarks.base import (
    CompetitorAdapter,
    _p_llm_to_determination,
    to_pipeline,
    wrap_as_detector,
)
from cairn.benchmarks.binoculars import (
    BinocularsAdapter,
    _heuristic_ratio_to_p_llm,
)
from cairn.benchmarks.fast_detectgpt import (
    FastDetectGPTAdapter,
    _heuristic_curvature_to_p_llm,
)
from cairn.benchmarks.gptzero import GPTZeroAdapter
from cairn.benchmarks.originality import OriginalityAdapter
from cairn.contracts import Determination, LayerResult


# ---------------------------------------------------------------------------
# Adapter Protocol — all four adapters satisfy the structural contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter, expected_id",
    [
        (BinocularsAdapter(), "binoculars"),
        (FastDetectGPTAdapter(), "fast_detectgpt"),
        (GPTZeroAdapter(), "gptzero"),
        (OriginalityAdapter(), "originality"),
    ],
)
def test_adapter_satisfies_competitor_protocol(adapter, expected_id):
    """Every adapter exposes the (id, domain, methods) Protocol surface."""
    assert isinstance(adapter, CompetitorAdapter)
    assert adapter.id == expected_id
    assert adapter.domain == "universal"
    assert callable(adapter.is_available)
    assert callable(adapter.score)
    assert callable(adapter.available_dependencies)


@pytest.mark.parametrize(
    "adapter_cls",
    [BinocularsAdapter, FastDetectGPTAdapter, GPTZeroAdapter, OriginalityAdapter],
)
def test_available_dependencies_returns_dict_of_bools(adapter_cls):
    deps = adapter_cls().available_dependencies()
    assert isinstance(deps, dict)
    assert deps  # at least one dep listed
    for v in deps.values():
        assert isinstance(v, bool)


# ---------------------------------------------------------------------------
# Determination threshold mirrors decision.DecisionEngine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.0, "GREEN"),
        (0.10, "GREEN"),
        (0.25, "YELLOW"),
        (0.49, "YELLOW"),
        (0.50, "AMBER"),
        (0.74, "AMBER"),
        (0.75, "RED"),
        (0.99, "RED"),
    ],
)
def test_p_llm_thresholds_mirror_decision_engine(p, expected):
    assert _p_llm_to_determination(p) == expected


# ---------------------------------------------------------------------------
# OSS adapters — is_available() honestly reports missing deps
# ---------------------------------------------------------------------------


class _DropModules:
    """Context manager that hides the named modules from imports.

    We can't simply ``del sys.modules[name]`` because the lazy-import
    helpers use ``import torch`` which would *re-import* a real torch
    if it's installed. Instead we map the entry to None — Python
    treats that as "import has been attempted and failed".
    """

    def __init__(self, *names: str) -> None:
        self.names = names
        self._saved: dict[str, object | None] = {}

    def __enter__(self):
        for n in self.names:
            self._saved[n] = sys.modules.get(n, "__missing__")
            sys.modules[n] = None  # type: ignore[assignment]
        return self

    def __exit__(self, *exc):
        for n, prev in self._saved.items():
            if prev == "__missing__":
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = prev  # type: ignore[assignment]


def test_binoculars_unavailable_when_torch_missing():
    adapter = BinocularsAdapter()
    with _DropModules("torch"):
        assert adapter.is_available() is False
        deps = adapter.available_dependencies()
        assert deps["torch"] is False


def test_fast_detectgpt_unavailable_when_transformers_missing():
    adapter = FastDetectGPTAdapter()
    with _DropModules("transformers"):
        assert adapter.is_available() is False
        deps = adapter.available_dependencies()
        assert deps["transformers"] is False


def test_binoculars_score_raises_when_unavailable():
    adapter = BinocularsAdapter()
    with _DropModules("torch", "transformers"):
        with pytest.raises(RuntimeError, match="not available"):
            adapter.score("anything")


def test_fast_detectgpt_score_raises_when_unavailable():
    adapter = FastDetectGPTAdapter()
    with _DropModules("torch", "transformers"):
        with pytest.raises(RuntimeError, match="not available"):
            adapter.score("anything")


# ---------------------------------------------------------------------------
# OSS adapters — heuristic mappings produce values in [0, 1]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ratio",
    [-5.0, 0.0, 0.5, 0.9, 0.901, 1.0, 2.0, 100.0],
)
def test_binoculars_heuristic_in_unit_range(ratio):
    p = _heuristic_ratio_to_p_llm(ratio)
    assert 0.0 <= p <= 1.0


def test_binoculars_heuristic_is_monotone_decreasing():
    """Lower ratio → higher p_llm (LLM-like)."""
    low = _heuristic_ratio_to_p_llm(0.5)
    high = _heuristic_ratio_to_p_llm(1.5)
    assert low > high


@pytest.mark.parametrize(
    "curv",
    [-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 100.0],
)
def test_fast_detectgpt_heuristic_in_unit_range(curv):
    p = _heuristic_curvature_to_p_llm(curv)
    assert 0.0 <= p <= 1.0


def test_fast_detectgpt_heuristic_is_monotone_increasing():
    """Higher curvature → higher p_llm (LLM-like)."""
    low = _heuristic_curvature_to_p_llm(-1.0)
    high = _heuristic_curvature_to_p_llm(3.0)
    assert high > low


# ---------------------------------------------------------------------------
# OSS adapters — score() returns float in [0, 1] when LM is mocked
# ---------------------------------------------------------------------------


def test_binoculars_score_with_mocked_internals_returns_unit_float():
    """End-to-end ``score()`` returns a clamped float when the math is mocked.

    We do not attempt to fake torch's tensor algebra. We patch the
    adapter's two private numerical helpers so the unit-range mapping
    is exercised on a known ratio.
    """
    adapter = BinocularsAdapter()
    with patch.object(BinocularsAdapter, "is_available", return_value=True), patch.object(
        BinocularsAdapter, "_ensure_loaded", return_value=None
    ), patch.object(
        BinocularsAdapter, "_binoculars_ratio", return_value=0.85
    ):
        score = adapter.score("Some text the model would normally read.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_fast_detectgpt_score_with_mocked_internals_returns_unit_float():
    adapter = FastDetectGPTAdapter()
    with patch.object(
        FastDetectGPTAdapter, "is_available", return_value=True
    ), patch.object(
        FastDetectGPTAdapter, "_ensure_loaded", return_value=None
    ), patch.object(
        FastDetectGPTAdapter, "_curvature", return_value=2.0
    ):
        score = adapter.score("Some text the model would normally read.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Commercial adapters — gated, raise NotImplementedError
# ---------------------------------------------------------------------------


def test_gptzero_unavailable_without_env_var(monkeypatch):
    monkeypatch.delenv("GPTZERO_API_KEY", raising=False)
    adapter = GPTZeroAdapter()
    assert adapter.is_available() is False
    deps = adapter.available_dependencies()
    assert deps["GPTZERO_API_KEY"] is False


def test_gptzero_score_not_implemented_with_key(monkeypatch):
    monkeypatch.setenv("GPTZERO_API_KEY", "test-key-not-real")
    adapter = GPTZeroAdapter()
    assert adapter.is_available() is True
    with pytest.raises(NotImplementedError):
        adapter.score("anything")


def test_originality_unavailable_without_env_var(monkeypatch):
    monkeypatch.delenv("ORIGINALITY_API_KEY", raising=False)
    adapter = OriginalityAdapter()
    assert adapter.is_available() is False
    deps = adapter.available_dependencies()
    assert deps["ORIGINALITY_API_KEY"] is False


def test_originality_score_not_implemented_with_key(monkeypatch):
    monkeypatch.setenv("ORIGINALITY_API_KEY", "test-key-not-real")
    adapter = OriginalityAdapter()
    assert adapter.is_available() is True
    with pytest.raises(NotImplementedError):
        adapter.score("anything")


# ---------------------------------------------------------------------------
# wrap_as_detector — Detector-Protocol shape and LayerResult synthesis
# ---------------------------------------------------------------------------


def _stub_adapter(score_value: float | None = 0.7, available: bool = True):
    """Build a tiny adapter satisfying the Protocol for wrapper tests."""

    class _Stub:
        id = "stub_adapter"
        domain = "universal"

        def is_available(self) -> bool:
            return available

        def score(self, text: str) -> float:
            assert score_value is not None
            return score_value

        def available_dependencies(self) -> dict[str, bool]:
            return {"stub": available}

    return _Stub()


def test_wrap_as_detector_returns_detector_protocol_shape():
    detector = wrap_as_detector(_stub_adapter(score_value=0.8))
    assert detector.id == "stub_adapter"
    assert detector.domain == "universal"
    assert isinstance(detector.compute_cost, str)


def test_wrap_as_detector_synthesizes_layer_result():
    detector = wrap_as_detector(_stub_adapter(score_value=0.80))
    layer = detector.analyze("the quick brown fox", {})
    assert isinstance(layer, LayerResult)
    assert layer.layer_id == "stub_adapter"
    assert layer.p_llm == pytest.approx(0.80)
    assert layer.raw_score == pytest.approx(0.80)
    assert layer.confidence == pytest.approx(1.0)
    assert layer.determination == "RED"  # 0.80 ≥ 0.75
    assert layer.calibrator_applied is False


def test_wrap_as_detector_skip_when_adapter_unavailable():
    detector = wrap_as_detector(_stub_adapter(available=False))
    layer = detector.analyze("anything", {})
    assert layer.determination == "SKIP"
    assert layer.signals.get("skipped") == "adapter_unavailable"


def test_wrap_as_detector_clamps_score_into_unit_range():
    """An adapter that misbehaves and returns 1.5 must still produce a clamped LayerResult."""
    detector = wrap_as_detector(_stub_adapter(score_value=1.5))
    layer = detector.analyze("anything", {})
    assert 0.0 <= layer.p_llm <= 1.0


# ---------------------------------------------------------------------------
# to_pipeline — runner-shaped Determination so run_eval consumes it
# ---------------------------------------------------------------------------


def test_to_pipeline_returns_determination_for_runner():
    pipeline = to_pipeline(wrap_as_detector(_stub_adapter(score_value=0.6)))
    det = pipeline.analyze("hello world")
    assert isinstance(det, Determination)
    assert det.label == "AMBER"  # 0.6 ≥ 0.50
    assert det.p_llm == pytest.approx(0.6)
    assert det.detectors_run == ["stub_adapter"]


def test_to_pipeline_skip_when_adapter_unavailable():
    pipeline = to_pipeline(wrap_as_detector(_stub_adapter(available=False)))
    det = pipeline.analyze("hello world")
    assert det.label == "SKIP"
    assert det.warnings  # non-empty


def test_to_pipeline_round_trips_through_run_eval():
    """The runner consumes a wrapped adapter without modification."""
    from cairn.evaluation.dataset import EvalSample
    from cairn.evaluation.runner import run_eval

    samples = [
        EvalSample(id="s1", text="alpha beta gamma", label=1, tier="t1"),
        EvalSample(id="s2", text="delta epsilon zeta", label=0, tier="t1"),
    ]
    pipeline = to_pipeline(wrap_as_detector(_stub_adapter(score_value=0.6)))
    report = run_eval(pipeline, samples)
    assert report.n_samples == 2
    assert {p["id"] for p in report.predictions} == {"s1", "s2"}
    for p in report.predictions:
        assert p["determination"] == "AMBER"
        assert p["p_llm"] == pytest.approx(0.6)

"""Competitor-adapter Protocol + helpers for plugging adapters into the CAIRN runner.

The Protocol mirrors :class:`cairn.contracts.Detector` so that the same
:func:`cairn.evaluation.runner.run_eval` consumes a wrapped competitor
without any runner changes. The integration is intentionally thin:

* :class:`CompetitorAdapter` defines the smallest surface a third-party
  detector needs (id, domain, ``is_available()``, ``score()``,
  diagnostic deps).
* :func:`wrap_as_detector` lifts an adapter into a CAIRN
  :class:`~cairn.contracts.Detector`-shaped wrapper that synthesises a
  :class:`~cairn.contracts.LayerResult` from the adapter's score.
* :func:`to_pipeline` lifts a Detector-shaped wrapper into a
  :class:`~cairn.evaluation.runner.PipelineLike` so ``run_eval`` can
  iterate samples through it.

Honest-calibration note: ``LayerResult.calibrator_applied`` is left
``False`` for competitor scores. Their providers' calibration is
opaque, so we do not pretend their outputs match CAIRN's
``calibrated-router-conditional`` tier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cairn.contracts import Determination, LayerResult


# Thresholds mirror cairn.decision.DecisionEngine defaults so the
# wrapped adapter's "determination" agrees with what the decision
# engine would emit for the same p_llm.
_RED_THRESHOLD = 0.75
_AMBER_THRESHOLD = 0.50
_YELLOW_THRESHOLD = 0.25


def _p_llm_to_determination(p_llm: float) -> str:
    """Mirror cairn.decision.DecisionEngine._p_llm_to_label thresholds."""
    if p_llm >= _RED_THRESHOLD:
        return "RED"
    if p_llm >= _AMBER_THRESHOLD:
        return "AMBER"
    if p_llm >= _YELLOW_THRESHOLD:
        return "YELLOW"
    return "GREEN"


@runtime_checkable
class CompetitorAdapter(Protocol):
    """Structural contract every competitor adapter implements.

    Mirrors the shape of :class:`cairn.contracts.Detector` but with a
    simpler scoring surface — adapters return a single normalised
    p(LLM) per text rather than a populated :class:`LayerResult`.
    """

    id: str
    domain: str  # default "universal" for prose-and-prompts adapters

    def is_available(self) -> bool:
        """True when the adapter can run end-to-end (deps + creds + models present)."""
        ...

    def score(self, text: str) -> float:
        """Return p(LLM) in ``[0.0, 1.0]``. Raise if unavailable or input invalid."""
        ...

    def available_dependencies(self) -> dict[str, bool]:
        """Per-dep readiness map for diagnostic surfaces (settings UI, /health)."""
        ...


@dataclass
class _AdapterDetector:
    """Detector-Protocol-shaped wrapper around a :class:`CompetitorAdapter`.

    Provides the (id, domain, compute_cost, ``analyze``, ``calibrate``)
    surface that :class:`cairn.contracts.Detector` defines so any place
    that consumes a Detector accepts a wrapped adapter.

    The synthesised :class:`LayerResult` reports ``confidence=1.0`` and
    ``raw_score=p_llm`` because we cannot inspect the provider's
    internal uncertainty. Operators who care about calibration must
    rely on CAIRN's own conformal wrapper, not the wrapped adapter.
    """

    adapter: CompetitorAdapter
    compute_cost: str = "moderate"

    @property
    def id(self) -> str:
        return self.adapter.id

    @property
    def domain(self) -> str:
        return getattr(self.adapter, "domain", "universal")

    def analyze(self, text: str, config: dict | None = None) -> LayerResult:
        if not self.adapter.is_available():
            return LayerResult(
                layer_id=self.id,
                domain=self.domain,
                raw_score=0.0,
                p_llm=0.0,
                confidence=0.0,
                signals={
                    "skipped": "adapter_unavailable",
                    "dependencies": self.adapter.available_dependencies(),
                },
                determination="SKIP",
                attacker_tiers=[],
                compute_cost=self.compute_cost,
                min_text_length=0,
                spans=[],
                calibrator_applied=False,
            )
        score = float(self.adapter.score(text))
        score = max(0.0, min(1.0, score))
        return LayerResult(
            layer_id=self.id,
            domain=self.domain,
            raw_score=score,
            p_llm=score,
            confidence=1.0,
            signals={
                "adapter_id": self.adapter.id,
                "dependencies": self.adapter.available_dependencies(),
            },
            determination=_p_llm_to_determination(score),
            attacker_tiers=[],
            compute_cost=self.compute_cost,
            min_text_length=0,
            spans=[],
            calibrator_applied=False,
        )

    def calibrate(self, labeled_data: list) -> None:  # pragma: no cover
        """Competitor adapters expose no calibration hook — no-op."""
        return None


def wrap_as_detector(adapter: CompetitorAdapter) -> _AdapterDetector:
    """Wrap a :class:`CompetitorAdapter` so it satisfies the Detector Protocol.

    The wrapped object exposes ``id``, ``domain``, ``compute_cost``,
    ``analyze(text, config)``, and ``calibrate(...)`` — i.e. it walks
    and quacks like a CAIRN detector without modifying the runner.
    """
    return _AdapterDetector(adapter=adapter)


@dataclass
class _SingleDetectorPipeline:
    """Single-detector PipelineLike shim so ``run_eval`` consumes an adapter directly.

    The runner only needs ``analyze(text, task_metadata=None) -> Determination``
    (see :class:`cairn.evaluation.runner.PipelineLike`). We synthesise a
    one-detector :class:`Determination` that mirrors the wrapped
    adapter's verdict.
    """

    detector: _AdapterDetector

    @property
    def config(self) -> dict:
        # run_eval._config_hash reaches for ``.config``; an explicit
        # adapter-id-keyed dict makes hashes distinguishable per adapter.
        return {"adapter": self.detector.id}

    def analyze(self, text: str, task_metadata: dict | None = None) -> Determination:
        layer = self.detector.analyze(text, {})
        if layer.determination == "SKIP":
            return Determination(
                label="SKIP",
                p_llm=0.0,
                confidence_interval=(0.0, 0.0),
                prediction_set=["SKIP"],
                reason=f"Adapter '{self.detector.id}' unavailable.",
                top_features=[],
                override_applied=False,
                detectors_run=[self.detector.id],
                cascade_phases=[],
                mixed_report=None,
                layer_results=[layer],
                feature_contributions={},
                conformal_set=None,
                conformal_alpha=None,
                fusion_mode="naive",
                detector_errors=[],
                warnings=[
                    f"Competitor adapter '{self.detector.id}' is "
                    "unavailable; output is a stub."
                ],
                abstention_reason=None,
                calibration_status="heuristic",
            )
        return Determination(
            label=layer.determination,
            p_llm=layer.p_llm,
            confidence_interval=(layer.p_llm, layer.p_llm),
            prediction_set=[layer.determination],
            reason=(
                f"Competitor adapter '{self.detector.id}' "
                f"reported p_llm={layer.p_llm:.4f}."
            ),
            top_features=[(self.detector.id, layer.p_llm)],
            override_applied=False,
            detectors_run=[self.detector.id],
            cascade_phases=[],
            mixed_report=None,
            layer_results=[layer],
            feature_contributions={self.detector.id: layer.p_llm},
            conformal_set=None,
            conformal_alpha=None,
            fusion_mode="naive",
            detector_errors=[],
            warnings=[],
            abstention_reason=None,
            calibration_status="heuristic",
        )


def to_pipeline(detector_like: _AdapterDetector) -> _SingleDetectorPipeline:
    """Lift a wrapped adapter into a single-detector PipelineLike for ``run_eval``."""
    return _SingleDetectorPipeline(detector=detector_like)

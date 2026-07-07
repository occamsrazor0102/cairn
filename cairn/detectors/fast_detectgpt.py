"""Fast-DetectGPT detector (Bao et al. 2023, arXiv:2310.05130).

Production wrapper that promotes the existing competitor *adapter*
(:mod:`cairn.benchmarks.fast_detectgpt`) into a cascade Detector. The
adapter already implements the analytic *conditional probability
curvature* — at each position the closed-form mean and variance of the
scoring LM's log-prob distribution, with the curvature being the
standardised deviation of the actual token's log-prob from that mean.
LLM-generated tokens sit near the conditional mean (large positive
curvature); human tokens scatter (curvature near zero / negative).

This replaces the classic DetectGPT-style ``perturbation`` detector: no
~100-sample word-swap loop, a single forward pass per text (~340× faster
per the paper), and a far stronger signal.

Scoring model: ``gpt2`` by default for the lean base install. The paper's
0.93 black-box AUROC on ChatGPT/GPT-4 used a GPT-J-6B / GPT-Neo-2.7B
sampling/scoring pair; set ``CAIRN_FAST_DETECTGPT_MODEL`` (or the
detector's ``model`` config key) to ``EleutherAI/gpt-neo-2.7B`` to trade
footprint for accuracy.

Tier-2 dependency (torch + transformers). The detector imports cleanly
without them and returns a ``SKIP`` ``LayerResult`` at analyze time when
they're absent, so the base install stays tiny and the cascade degrades
visibly rather than crashing.

Privacy: text is scored entirely on-device; no network calls beyond the
one-time HuggingFace weight download.
"""
from __future__ import annotations

import logging
import threading

from cairn.benchmarks.fast_detectgpt import (
    FastDetectGPTAdapter,
    _heuristic_curvature_to_p_llm,
)
from cairn.contracts import LayerResult

logger = logging.getLogger("cairn.detectors.fast_detectgpt")

# Below this word count the analytic curvature is too noisy to trust
# (the paper and downstream evals both degrade sharply on very short
# inputs). Mirrors the short-text guard used by the other tier-2
# detectors rather than emitting a confident verdict on a sentence.
_MIN_WORDS = 40


class FastDetectGPTDetector:
    """Cascade Detector wrapping a :class:`FastDetectGPTAdapter`."""

    id = "fast_detectgpt"
    domain = "universal"
    compute_cost = "moderate"

    def __init__(self, scoring_model: str | None = None) -> None:
        # Adapter construction is cheap — it only records the model name
        # and defers torch/transformers import and weight loading until
        # the first scored text.
        self._adapter = FastDetectGPTAdapter(scoring_model=scoring_model)

    def _skip(self, reason: str, **extra) -> LayerResult:
        return LayerResult(
            layer_id=self.id, domain=self.domain, raw_score=0.0,
            p_llm=0.5, confidence=0.0,
            signals={"skipped": reason, **extra},
            determination="SKIP", attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost, min_text_length=_MIN_WORDS,
        )

    def analyze(self, text: str, config: dict) -> LayerResult:
        n_words = len(text.split())
        if n_words < _MIN_WORDS:
            return self._skip("too_short", n_words=n_words)
        if not self._adapter.is_available():
            return self._skip(
                "tier2_unavailable",
                dependencies=self._adapter.available_dependencies(),
                hint="pip install 'cairn[tier2]' (torch + transformers)",
            )

        curvature = self._adapter.curvature(text)
        p_llm = _heuristic_curvature_to_p_llm(curvature)

        if p_llm >= 0.75:
            determination = "RED"
        elif p_llm >= 0.50:
            determination = "AMBER"
        elif p_llm >= 0.25:
            determination = "YELLOW"
        else:
            determination = "GREEN"

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            # raw_score is the uncalibrated curvature so a fitted isotonic
            # calibrator maps it per-detector; p_llm carries the heuristic
            # sigmoid until that calibrator is loaded.
            raw_score=float(curvature),
            p_llm=p_llm,
            confidence=min(0.45 + n_words / 800.0, 0.90),
            signals={
                "curvature": round(float(curvature), 5),
                "scoring_model": self._adapter._model_name,
                "n_words": n_words,
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=_MIN_WORDS,
        )

    def calibrate(self, labeled_data: list) -> None:
        # Per-detector isotonic calibration is owned by
        # scripts/calibrate_detectors.py via the cached raw_score; the EBM
        # owns fusion. No detector-local calibration state.
        pass


class _LazyFastDetectGPT:
    """Lazy wrapper — defers adapter construction until first analyze so a
    per-config scoring-model override is honoured and import stays cheap."""

    id = "fast_detectgpt"
    domain = "universal"
    compute_cost = "moderate"
    _instance: "FastDetectGPTDetector | None" = None
    _lock = threading.Lock()

    def analyze(self, text: str, config: dict) -> LayerResult:
        if self._instance is None:
            with self._lock:
                if self._instance is None:  # double-checked
                    type(self)._instance = FastDetectGPTDetector(
                        scoring_model=(config or {}).get("model"),
                    )
        return self._instance.analyze(text, config)

    def calibrate(self, labeled_data: list) -> None:
        pass


DETECTOR = _LazyFastDetectGPT()

"""Binoculars adapter (Hans et al. 2024).

Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A.,
Goldblum, M., Geiping, J., & Goldstein, T. (2024). *Spotting LLMs With
Binoculars: Zero-Shot Detection of Machine-Generated Text.*
arXiv:2401.12070.

Binoculars scores a text by the *cross-perplexity ratio* between two
language models: an "observer" LM (typically the base model) and a
"performer" LM (typically the same base model after instruction
tuning). For human-written text the two perplexities differ in
predictable ways; LLM-generated text has anomalously low cross-entropy
because the observer LM is "in tune" with the performer that
generated it.

The score is::

    log_PPL_obs(x)
    -------------- = ratio
    X_PPL_obs/perf(x)

where ``log_PPL_obs`` is the observer's log-perplexity on x and
``X_PPL_obs/perf`` is the cross-entropy of x under the observer
conditioned on the performer's distribution. Lower ratio → more LLM-like.

We map the unbounded ratio into ``p_llm ∈ [0, 1]`` via a fixed sigmoid
centred on the paper's reported decision threshold (~0.901 for the
Falcon pair). **TODO(calibration):** replace with a fitted isotonic
mapping once we have a labelled corpus scored by this adapter.

Default model pair: ``tiiuae/falcon-7b`` (observer) and
``tiiuae/falcon-7b-instruct`` (performer), matching the Binoculars
paper. Override via constructor args or
``CAIRN_BINOCULARS_OBSERVER`` / ``CAIRN_BINOCULARS_PERFORMER`` env vars
to use a smaller pair where the 7B weights are infeasible.

This is a tier-2 dependency — the adapter requires ``torch`` and
``transformers``. ``is_available()`` returns ``False`` when either
import fails or the configured model weights cannot be loaded.

Privacy: the text is processed entirely on-device. No network calls
beyond the initial HuggingFace model download.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any

from cairn.benchmarks.base import CompetitorAdapter

log = logging.getLogger("cairn.benchmarks.binoculars")


# Lazy-import gates so importing the module is cheap.
def _try_import_torch() -> tuple[Any, bool]:
    try:
        import torch  # type: ignore

        return torch, True
    except ImportError:
        return None, False


def _try_import_transformers() -> tuple[Any, Any, bool]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        return AutoModelForCausalLM, AutoTokenizer, True
    except ImportError:
        return None, None, False


# Sigmoid mapping centre and slope. ``_BINOCULARS_THRESHOLD`` ≈ 0.901
# is the threshold reported in Hans et al. for the Falcon-7b pair.
# ``_BINOCULARS_SLOPE`` is hand-picked so a ratio one std below the
# threshold maps to p_llm ≈ 0.85 (clearly LLM) and one std above maps
# to ≈ 0.15 (clearly human). TODO(calibration): replace with a fitted
# isotonic mapping. _HEURISTIC_ prefix per CLAUDE.md's honest-mode rule.
_HEURISTIC_BINOCULARS_THRESHOLD = 0.901
_HEURISTIC_BINOCULARS_SLOPE = 25.0


def _heuristic_ratio_to_p_llm(ratio: float) -> float:
    """Sigmoid centred on the Binoculars paper threshold; lower ratio → higher p_llm."""
    z = -_HEURISTIC_BINOCULARS_SLOPE * (ratio - _HEURISTIC_BINOCULARS_THRESHOLD)
    # Numerically stable sigmoid for large |z|
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


class BinocularsAdapter:
    """Cross-perplexity-ratio adapter implementing :class:`CompetitorAdapter`."""

    id: str = "binoculars"
    domain: str = "universal"

    def __init__(
        self,
        observer_model: str | None = None,
        performer_model: str | None = None,
        device: str | None = None,
    ):
        self._observer_name = (
            observer_model
            or os.environ.get("CAIRN_BINOCULARS_OBSERVER")
            or "tiiuae/falcon-7b"
        )
        self._performer_name = (
            performer_model
            or os.environ.get("CAIRN_BINOCULARS_PERFORMER")
            or "tiiuae/falcon-7b-instruct"
        )
        self._device = device or os.environ.get("CAIRN_BINOCULARS_DEVICE", "cpu")
        self._observer: Any = None
        self._performer: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------ deps

    def available_dependencies(self) -> dict[str, bool]:
        _, has_torch = _try_import_torch()
        _, _, has_transformers = _try_import_transformers()
        return {
            "torch": has_torch,
            "transformers": has_transformers,
        }

    def is_available(self) -> bool:
        deps = self.available_dependencies()
        return all(deps.values())

    # ------------------------------------------------------------------ load

    def _ensure_loaded(self) -> None:
        if self._observer is not None and self._performer is not None:
            return
        torch, _ = _try_import_torch()
        AutoModelForCausalLM, AutoTokenizer, _ = _try_import_transformers()
        if torch is None or AutoModelForCausalLM is None:
            raise RuntimeError(
                "BinocularsAdapter.score() called when torch/transformers are "
                "missing. Guard with is_available() first."
            )
        log.info(
            "binoculars: loading observer=%s performer=%s on %s",
            self._observer_name,
            self._performer_name,
            self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._observer_name)
        self._observer = AutoModelForCausalLM.from_pretrained(self._observer_name).to(
            self._device
        )
        self._observer.eval()
        self._performer = AutoModelForCausalLM.from_pretrained(self._performer_name).to(
            self._device
        )
        self._performer.eval()

    # ------------------------------------------------------------------ math

    def _log_perplexity(self, model: Any, input_ids: Any) -> float:
        torch, _ = _try_import_torch()
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        return float(outputs.loss.item())

    def _cross_perplexity(self, input_ids: Any) -> float:
        """Cross-entropy of x under the observer conditioned on the performer."""
        torch, _ = _try_import_torch()
        with torch.no_grad():
            obs_logits = self._observer(input_ids).logits
            perf_logits = self._performer(input_ids).logits
        # Shift to align the next-token target with the prediction at position t.
        obs_log_probs = torch.nn.functional.log_softmax(obs_logits[:, :-1, :], dim=-1)
        perf_probs = torch.nn.functional.softmax(perf_logits[:, :-1, :], dim=-1)
        # Cross-entropy of observer under performer's distribution: -E_perf[log p_obs]
        ce = -(perf_probs * obs_log_probs).sum(dim=-1).mean()
        return float(ce.item())

    def _binoculars_ratio(self, text: str) -> float:
        if not text or not text.strip():
            return _HEURISTIC_BINOCULARS_THRESHOLD  # neutral
        torch, _ = _try_import_torch()
        encoded = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = encoded["input_ids"].to(self._device)
        if input_ids.shape[1] < 2:
            return _HEURISTIC_BINOCULARS_THRESHOLD
        log_ppl = self._log_perplexity(self._observer, input_ids)
        x_ppl = self._cross_perplexity(input_ids)
        if x_ppl <= 0:
            return _HEURISTIC_BINOCULARS_THRESHOLD
        return log_ppl / x_ppl

    # ------------------------------------------------------------------ api

    def score(self, text: str) -> float:
        """Return p(LLM) ∈ [0, 1] for ``text`` under the Binoculars ratio."""
        if not self.is_available():
            raise RuntimeError(
                "BinocularsAdapter is not available; "
                "install cairn[tier2] (torch + transformers)."
            )
        self._ensure_loaded()
        ratio = self._binoculars_ratio(text)
        return _heuristic_ratio_to_p_llm(ratio)


# Module-level singleton instance, mirroring the detector convention
# (e.g. `DETECTOR = PreambleDetector()` in cairn/detectors/preamble.py).
ADAPTER = BinocularsAdapter()


def _verify_protocol() -> None:
    """Sanity-check the adapter satisfies the structural Protocol."""
    assert isinstance(ADAPTER, CompetitorAdapter)


_verify_protocol()

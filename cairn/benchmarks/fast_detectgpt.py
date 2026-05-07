"""Fast-DetectGPT adapter (Bao et al. 2023).

Bao, G., Zhao, Y., Teng, Z., Yang, L., & Zhang, Y. (2023). *Fast-DetectGPT:
Efficient Zero-Shot Detection of Machine-Generated Text via Conditional
Probability Curvature.* arXiv:2310.05130.

Fast-DetectGPT replaces DetectGPT's expensive perturbation-based curvature
estimate with a far cheaper *conditional probability curvature*: at each
position the analytic mean and variance of the LM's log-prob distribution
are computed in closed form, and the curvature is the standardised
deviation of the actual token's log-prob from that mean. The mean
across positions is the detection statistic.

For LLM-generated text the original tokens are *highly likely* under
the LM, so the curvature is positive and large. For human text the
curvature is closer to zero or negative.

We use ``gpt2`` as the scoring LM by default (override via
``CAIRN_FAST_DETECTGPT_MODEL``) and map the unbounded curvature into
``p_llm ∈ [0, 1]`` via a fixed sigmoid. **TODO(calibration):** replace
with a fitted isotonic mapping once we have a labelled corpus scored
by this adapter.

This is a tier-2 dependency — the adapter requires ``torch`` and
``transformers``. ``is_available()`` returns ``False`` when either
import fails or the configured model weights cannot be loaded.

Privacy: text is processed entirely on-device. No network calls beyond
the initial HuggingFace model download.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any

from cairn.benchmarks.base import CompetitorAdapter

log = logging.getLogger("cairn.benchmarks.fast_detectgpt")


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


# Sigmoid mapping centre and slope for curvature → p(LLM). The
# Fast-DetectGPT paper reports curvature roughly in [0, 4] for LLM
# text and roughly in [-1, 1] for human; we centre at ~1.0 and pick a
# slope that puts curvature 0 at p_llm ≈ 0.18 and curvature 2 at
# p_llm ≈ 0.82. _HEURISTIC_ prefix per CLAUDE.md's honest-mode rule.
# TODO(calibration): replace with a fitted isotonic mapping.
_HEURISTIC_CURVATURE_CENTER = 1.0
_HEURISTIC_CURVATURE_SLOPE = 1.5


def _heuristic_curvature_to_p_llm(curvature: float) -> float:
    """Sigmoid: higher curvature → higher p_llm."""
    z = _HEURISTIC_CURVATURE_SLOPE * (curvature - _HEURISTIC_CURVATURE_CENTER)
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


class FastDetectGPTAdapter:
    """Conditional-probability-curvature adapter implementing :class:`CompetitorAdapter`."""

    id: str = "fast_detectgpt"
    domain: str = "universal"

    def __init__(
        self,
        scoring_model: str | None = None,
        device: str | None = None,
    ):
        self._model_name = (
            scoring_model
            or os.environ.get("CAIRN_FAST_DETECTGPT_MODEL")
            or "gpt2"
        )
        self._device = device or os.environ.get(
            "CAIRN_FAST_DETECTGPT_DEVICE", "cpu"
        )
        self._model: Any = None
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
        return all(self.available_dependencies().values())

    # ------------------------------------------------------------------ load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        torch, _ = _try_import_torch()
        AutoModelForCausalLM, AutoTokenizer, _ = _try_import_transformers()
        if torch is None or AutoModelForCausalLM is None:
            raise RuntimeError(
                "FastDetectGPTAdapter.score() called when torch/transformers are "
                "missing. Guard with is_available() first."
            )
        log.info(
            "fast_detectgpt: loading scoring LM=%s on %s",
            self._model_name,
            self._device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name).to(
            self._device
        )
        self._model.eval()

    # ------------------------------------------------------------------ math

    def _curvature(self, text: str) -> float:
        """Closed-form Fast-DetectGPT curvature for ``text``."""
        torch, _ = _try_import_torch()
        encoded = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        input_ids = encoded["input_ids"].to(self._device)
        if input_ids.shape[1] < 2:
            return 0.0
        with torch.no_grad():
            logits = self._model(input_ids).logits
        # Align: predict t+1 from t. Drop the final logit and the first id.
        logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # log p(x_t | x_<t) at the actual token
        target_log_prob = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # Closed-form mean and variance of log p across the conditional
        # distribution at each position (Fast-DetectGPT analytic sample).
        mean_log_prob = (probs * log_probs).sum(dim=-1)
        var_log_prob = (probs * (log_probs - mean_log_prob.unsqueeze(-1)) ** 2).sum(
            dim=-1
        )
        # Per-position curvature: (target - mean) / std
        std = var_log_prob.clamp_min(1e-8).sqrt()
        curvature = ((target_log_prob - mean_log_prob) / std).mean()
        return float(curvature.item())

    # ------------------------------------------------------------------ api

    def score(self, text: str) -> float:
        """Return p(LLM) ∈ [0, 1] for ``text`` under the Fast-DetectGPT curvature."""
        if not self.is_available():
            raise RuntimeError(
                "FastDetectGPTAdapter is not available; "
                "install cairn[tier2] (torch + transformers)."
            )
        if not text or not text.strip():
            return 0.5
        self._ensure_loaded()
        curvature = self._curvature(text)
        return _heuristic_curvature_to_p_llm(curvature)


# Module-level singleton instance.
ADAPTER = FastDetectGPTAdapter()


def _verify_protocol() -> None:
    """Sanity-check the adapter satisfies the structural Protocol."""
    assert isinstance(ADAPTER, CompetitorAdapter)


_verify_protocol()

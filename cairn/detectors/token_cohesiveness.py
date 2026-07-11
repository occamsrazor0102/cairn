# cairn/detectors/token_cohesiveness.py
"""TOCSIN — token cohesiveness via random deletion (Ma et al., EMNLP 2024 main,
arXiv:2409.16914).

Faithful re-implementation of the published method. LLM-generated text is
*more cohesive* under small random token deletions: regenerating the
original from a lightly-deleted copy is easier (higher BARTScore) than for
human text, whose meaning is more brittle to deletion.

Mechanism (per the paper):
  * make ``n=10`` copies of the text, each with ``ρ=1.5%`` of tokens
    randomly deleted;
  * measure semantic shift with **BARTScore** under ``facebook/bart-base``
    (139M) — the mean per-token log-likelihood of regenerating the
    original from each deleted copy;
  * the cohesiveness channel ``u(x)`` is the average of those scores
    (closer to 0 = more cohesive = more LLM-like);
  * combine with a base zero-shot logprob detector ``v(x)`` via the
    paper's exponential rule
    ``w(x) = e^{u(x)}·v(x)   if v(x) ≥ 0``,
    ``w(x) = e^{-u(x)}·v(x)  otherwise``.

The base channel ``v(x)`` reuses the in-tree Fast-DetectGPT *curvature*
(``cairn.benchmarks.fast_detectgpt``) so we don't load a third scoring LM.
The previous implementation diverged from the paper on every axis
(MiniLM cosine instead of BARTScore, single-token deletion, no base
channel) and scored at chance (AUROC ~0.50); this restores the method.

Raw channels (``cohesiveness_u``, ``base_v``, ``combined_w``) are surfaced
in ``signals`` so the EBM learns the mapping rather than relying on the
heuristic point estimate.

Tier-2 dependency (torch + transformers, via ``facebook/bart-base``). The
module imports cleanly without them and SKIPs at analyze time when absent.
Privacy: all scoring is on-device.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import threading

from cairn.benchmarks.fast_detectgpt import (
    FAST_DETECTGPT_ALGORITHM_ID,
    FAST_DETECTGPT_DEFAULT_MODEL,
    FastDetectGPTAdapter,
    _heuristic_curvature_to_p_llm,
    _select_device,
)
from cairn.contracts import LayerResult

logger = logging.getLogger("cairn.detectors.token_cohesiveness")

TOCSIN_COMBINED_SIGNATURE = "tocsin-bart-fastdetectgpt-combined-w-v1"
TOCSIN_COHESIVENESS_SIGNATURE = "tocsin-bart-cohesiveness-u-v1"
TOCSIN_SIGNATURE_TO_RAW_CHANNEL = {
    TOCSIN_COMBINED_SIGNATURE: "combined_w",
    TOCSIN_COHESIVENESS_SIGNATURE: "cohesiveness_u",
}

# Paper hyperparameters.
_DELETION_RATIO = 0.015          # ρ — fraction of tokens deleted per copy
_N_COPIES = 10                   # number of deleted copies per text
_BART_MODEL = "facebook/bart-base"
_TOCSIN_ALGORITHM_ID = "random-deletion-bartscore-v1"
# TOCSIN degrades on passages under ~90 tokens; ~60 words is a comparable
# floor. Below this we SKIP rather than emit a noisy verdict.
_MIN_WORDS = 60


def build_tocsin_calibration_signature(
    raw_channel: str,
    *,
    bart_model: str,
    base_model: str | None = None,
) -> tuple[str, dict[str, str]]:
    """Return a stable calibration signature and its auditable identity tuple."""
    if not isinstance(bart_model, str) or not bart_model.strip():
        raise ValueError("bart_model must be a non-empty string")
    if raw_channel == "combined_w":
        if not isinstance(base_model, str) or not base_model.strip():
            raise ValueError(
                "base_model must be a non-empty string for combined_w"
            )
        base_signature = TOCSIN_COMBINED_SIGNATURE
        identity = {
            "raw_channel": raw_channel,
            "tocsin_algorithm": _TOCSIN_ALGORITHM_ID,
            "bart_model": bart_model,
            "base_algorithm": FAST_DETECTGPT_ALGORITHM_ID,
            "base_model": base_model,
        }
        is_default = (
            bart_model == _BART_MODEL
            and base_model == FAST_DETECTGPT_DEFAULT_MODEL
        )
    elif raw_channel == "cohesiveness_u":
        base_signature = TOCSIN_COHESIVENESS_SIGNATURE
        identity = {
            "raw_channel": raw_channel,
            "tocsin_algorithm": _TOCSIN_ALGORITHM_ID,
            "bart_model": bart_model,
        }
        is_default = bart_model == _BART_MODEL
    else:  # defensive: signatures are calibration contracts, not free text
        raise ValueError(f"unsupported TOCSIN raw channel {raw_channel!r}")

    if is_default:
        return base_signature, identity
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    suffix = hashlib.sha256(canonical).hexdigest()[:16]
    return f"{base_signature}-{suffix}", identity


def _try_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except ImportError:
        return None


def _load_bart_components(model_name: str):
    from transformers import AutoTokenizer, BartForConditionalGeneration

    return (
        AutoTokenizer.from_pretrained(model_name),
        BartForConditionalGeneration.from_pretrained(model_name),
    )


class TokenCohesivenessDetector:
    id = "token_cohesiveness"
    domain = "prose"
    compute_cost = "moderate"

    def __init__(
        self,
        model_name: str = _BART_MODEL,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        torch = _try_import_torch()
        self._device = _select_device(torch, device)
        self._bart = None
        self._bart_tok = None
        # Base logprob channel v(x); shares the tier-2 deps with BART.
        self._base = FastDetectGPTAdapter(device=self._device)

    # -------------------------------------------------------------- loading

    def _ensure_bart(self) -> None:
        if self._bart is not None:
            return
        self._bart_tok, self._bart = _load_bart_components(self._model_name)
        self._bart = self._bart.to(self._device)
        self._bart.eval()

    def _bart_score(self, source: str, target: str) -> float:
        """Mean per-token log-likelihood of ``target`` given ``source``.

        BARTScore = -CE: ``BartForConditionalGeneration(..., labels=)``
        returns the mean cross-entropy over the target tokens, so the
        negated loss is the mean log-probability.
        """
        torch = _try_import_torch()
        enc = {
            key: value.to(self._device)
            for key, value in self._bart_tok(
                source, return_tensors="pt", truncation=True, max_length=512
            ).items()
        }
        labels = self._bart_tok(
            target, return_tensors="pt", truncation=True, max_length=512
        )["input_ids"].to(self._device)
        with torch.no_grad():
            out = self._bart(**enc, labels=labels)
        return float(-out.loss.item())

    # -------------------------------------------------------------- helpers

    def _deleted_copy(self, words: list[str], rng: random.Random) -> str:
        n_delete = max(1, int(len(words) * _DELETION_RATIO))
        drop = set(rng.sample(range(len(words)), min(n_delete, len(words))))
        return " ".join(w for i, w in enumerate(words) if i not in drop)

    def _skip(self, reason: str, **extra) -> LayerResult:
        return LayerResult(
            layer_id=self.id, domain=self.domain, raw_score=0.0,
            p_llm=0.5, confidence=0.0,
            signals={"skipped": reason, **extra},
            determination="SKIP", attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost, min_text_length=_MIN_WORDS,
        )

    # ----------------------------------------------------------------- main

    def analyze(self, text: str, config: dict) -> LayerResult:
        words = text.split()
        if len(words) < _MIN_WORDS:
            return self._skip("too_short", n_words=len(words))
        if _try_import_torch() is None:
            return self._skip(
                "tier2_unavailable",
                hint="pip install 'cairn[tier2]' (torch + transformers)",
            )

        try:
            self._ensure_bart()
        except Exception as exc:  # weights missing / load failure
            logger.warning("token_cohesiveness: BART load failed: %s", exc)
            return self._skip("bart_load_failed", error=str(exc))

        rng = random.Random(config.get("seed", 42))
        scores = []
        for _ in range(_N_COPIES):
            copy = self._deleted_copy(words, rng)
            scores.append(self._bart_score(copy, text))
        cohesiveness_u = sum(scores) / len(scores)

        # Base channel v(x): Fast-DetectGPT curvature. Standalone-degrade to
        # the cohesiveness channel alone if the base scorer is unavailable
        # (the paper reports cohesiveness-alone at ~0.94 AUROC, so this is a
        # graceful fallback, not a failure).
        base_v = None
        if self._base.is_available():
            try:
                base_v = self._base.curvature(text)
            except Exception as exc:
                logger.debug("token_cohesiveness: base channel failed: %s", exc)

        if base_v is None:
            combined_w = cohesiveness_u
            # cohesiveness alone: more cohesive (u→0) → more LLM-like.
            p_llm = _heuristic_cohesiveness_to_p_llm(cohesiveness_u)
            raw_score = cohesiveness_u
            calibration_raw_channel = "cohesiveness_u"
            calibration_signature, calibration_identity = (
                build_tocsin_calibration_signature(
                    calibration_raw_channel,
                    bart_model=self._model_name,
                )
            )
        else:
            # Paper's exponential combination, sign-aware on the base score.
            if base_v >= 0:
                combined_w = math.exp(cohesiveness_u) * base_v
            else:
                combined_w = math.exp(-cohesiveness_u) * base_v
            p_llm = _heuristic_curvature_to_p_llm(combined_w)
            raw_score = combined_w
            calibration_raw_channel = "combined_w"
            calibration_signature, calibration_identity = (
                build_tocsin_calibration_signature(
                    calibration_raw_channel,
                    bart_model=self._model_name,
                    base_model=self._base.model_name,
                )
            )

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
            raw_score=float(raw_score), p_llm=p_llm,
            confidence=min(0.40 + len(words) / 800.0, 0.85),
            signals={
                "cohesiveness_u": round(float(cohesiveness_u), 5),
                "base_v": round(float(base_v), 5) if base_v is not None else None,
                "combined_w": round(float(combined_w), 5),
                "calibration_signature": calibration_signature,
                "calibration_raw_channel": calibration_raw_channel,
                "calibration_identity": calibration_identity,
                "deletion_ratio": _DELETION_RATIO,
                "n_copies": _N_COPIES,
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=_MIN_WORDS,
        )

    def calibrate(self, labeled_data: list) -> None:
        pass


# Cohesiveness u(x) is a mean BART log-likelihood: roughly [-5, 0], closer
# to 0 = more cohesive = more LLM-like. Map with a heuristic sigmoid
# centred where human/LLM cohesiveness tends to separate. _HEURISTIC_ prefix
# per CLAUDE.md's honest-mode rule; the EBM relearns this from `cohesiveness_u`.
_HEURISTIC_COHESIVENESS_CENTER = -1.5
_HEURISTIC_COHESIVENESS_SLOPE = 1.0


def _heuristic_cohesiveness_to_p_llm(u: float) -> float:
    z = _HEURISTIC_COHESIVENESS_SLOPE * (u - _HEURISTIC_COHESIVENESS_CENTER)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


class _LazyTOCSIN:
    """Lazy wrapper — defers BART construction until first analyze."""

    id = "token_cohesiveness"
    domain = "prose"
    compute_cost = "moderate"
    _instance: "TokenCohesivenessDetector | None" = None
    _lock = threading.Lock()

    def analyze(self, text: str, config: dict) -> LayerResult:
        if self._instance is None:
            with self._lock:
                if self._instance is None:  # double-checked
                    type(self)._instance = TokenCohesivenessDetector(
                        model_name=(config or {}).get("model", _BART_MODEL),
                        device=(config or {}).get("device"),
                    )
        return self._instance.analyze(text, config)

    def calibrate(self, labeled_data: list) -> None:
        pass


DETECTOR = _LazyTOCSIN()

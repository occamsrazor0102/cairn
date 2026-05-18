# cairn/fusion/ebm.py
import logging
import math
from typing import Any
from cairn.contracts import LayerResult, FusionResult

logger = logging.getLogger("cairn.fusion")


# --- Feature Assembler ---

_NAN = float("nan")

class FeatureAssembler:
    """Assembles layer results into a flat feature dict for the EBM."""

    def __init__(self, drop_features: set[str] | None = None) -> None:
        # Optional feature-drop allowlist for ablation experiments. Keys
        # are popped from the returned dict after assembly. Used by
        # train_fusion --drop-features to test which specdetect (or other)
        # features actually carry signal vs. dilute the EBM's capacity.
        # Always-empty default; backward-compatible.
        self._drop_features: set[str] = set(drop_features or ())

    def assemble(self, results: list[LayerResult], word_count: int, domain: str) -> dict:
        by_id = {r.layer_id: r for r in results if r.determination != "SKIP"}

        def p(layer_id: str) -> float:
            return by_id[layer_id].p_llm if layer_id in by_id else _NAN

        def sig(layer_id: str, key: str) -> float:
            if layer_id not in by_id: return _NAN
            v = by_id[layer_id].signals.get(key, _NAN)
            return float(v) if v is not None else _NAN

        def sig_str(layer_id: str, key: str, mapping: dict) -> float:
            if layer_id not in by_id: return _NAN
            v = by_id[layer_id].signals.get(key)
            return float(mapping.get(v, _NAN))

        severity_map = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

        def missing(layer_id: str) -> float:
            return 0.0 if layer_id in by_id else 1.0

        def conf(layer_id: str) -> float:
            return float(by_id[layer_id].confidence) if layer_id in by_id else _NAN

        vec = {
            # Preamble
            "preamble_p_llm": p("preamble"),
            "preamble_severity": sig_str("preamble", "severity", severity_map),
            "preamble_n_matches": sig("preamble", "n_matches"),
            # Fingerprint vocab
            "fingerprint_p_llm": p("fingerprint_vocab"),
            "fingerprint_hits_per_1000": sig("fingerprint_vocab", "hits_per_1000"),
            "fingerprint_bigram_hits": sig("fingerprint_vocab", "bigram_hits"),
            # Prompt structure
            "ps_p_llm": p("prompt_structure"),
            "ps_cfd": sig("prompt_structure", "cfd"),
            "ps_distinct_frames": sig("prompt_structure", "distinct_frames"),
            "ps_mfsr": sig("prompt_structure", "mfsr"),
            "ps_framing_completeness": sig("prompt_structure", "framing_completeness"),
            "ps_meta_design_hits": sig("prompt_structure", "meta_design_hits"),
            # Voice-spec
            "vs_p_llm": p("voice_spec"),
            "vs_voice_score": sig("voice_spec", "voice_score"),
            "vs_spec_score": sig("voice_spec", "spec_score"),
            # IDI
            "idi_p_llm": p("instruction_density"),
            "idi_score": sig("instruction_density", "idi"),
            # NSSI
            "nssi_p_llm": p("nssi"),
            "nssi_n_signals_active": sig("nssi", "n_signals_active"),
            "nssi_formulaic_density": sig("nssi", "formulaic_density"),
            "nssi_discourse_scaffolding": sig("nssi", "discourse_scaffolding"),
            # Tier 2
            "surprisal_p_llm": p("surprisal_dynamics"),
            "surprisal_late_volatility": sig("surprisal_dynamics", "late_volatility_ratio"),
            "surprisal_diversity": sig("surprisal_dynamics", "surprisal_diversity"),
            "binoculars_p_llm": p("contrastive_lm"),
            "binoculars_ratio": sig("contrastive_lm", "binoculars_ratio"),
            "tocsin_p_llm": p("token_cohesiveness"),
            "tocsin_mean_deletion_impact": sig("token_cohesiveness", "mean_deletion_impact"),
            # Semantic kinematics (Phase 2, embedding-trajectory macro-feature).
            # Full detector ID is used for the {p_llm, missing, confidence}
            # triplet; signal-level keys live in their own `sk_*` namespace
            # (mean/std of velocity, acceleration, jerk along the sentence
            # trajectory) so the EBM can learn each moment independently.
            "semantic_kinematics_p_llm": p("semantic_kinematics"),
            "sk_velocity_mean": sig("semantic_kinematics", "sk_velocity_mean"),
            "sk_velocity_std": sig("semantic_kinematics", "sk_velocity_std"),
            "sk_acceleration_mean": sig("semantic_kinematics", "sk_acceleration_mean"),
            "sk_acceleration_std": sig("semantic_kinematics", "sk_acceleration_std"),
            "sk_jerk_mean": sig("semantic_kinematics", "sk_jerk_mean"),
            "sk_jerk_std": sig("semantic_kinematics", "sk_jerk_std"),
            # SpecDetect (Phase 2, frequency-domain log-prob features). p_llm
            # is derived from total energy alone, but the EBM gets the full
            # spectral breakdown so it can learn per-band weights. The total
            # energy itself is redundant with `surprisal_std` by Parseval;
            # the per-band fractions are the orthogonal signal.
            "specdetect_p_llm": p("specdetect"),
            "specdetect_total_energy": sig("specdetect", "dft_total_energy"),
            "specdetect_low_band": sig("specdetect", "dft_low_band_energy"),
            "specdetect_mid_band": sig("specdetect", "dft_mid_band_energy"),
            "specdetect_high_band": sig("specdetect", "dft_high_band_energy"),
            "specdetect_low_mid_ratio": sig("specdetect", "dft_low_mid_ratio"),
            "specdetect_spectral_entropy": sig("specdetect", "dft_spectral_entropy"),
            # Tier 3 / advanced
            "perturbation_p_llm": p("perturbation"),
            "contrastive_gen_p_llm": p("contrastive_gen"),
            "dna_gpt_p_llm": p("dna_gpt"),
            "lapd_p_llm": p("lapd"),
            "mixed_boundary_p_llm": p("mixed_boundary"),
            "cross_similarity_p_llm": p("cross_similarity"),
            # Per-detector missingness indicators (1.0 = absent / SKIP, 0.0 = present).
            # Same short-form prefix as the matching `<prefix>_p_llm` key.
            # Lets the EBM learn detector-specific "trust this signal less when
            # this other detector skipped" interactions instead of conflating
            # NaN imputation with mid-confidence predictions.
            "preamble_missing": missing("preamble"),
            "fingerprint_missing": missing("fingerprint_vocab"),
            "ps_missing": missing("prompt_structure"),
            "vs_missing": missing("voice_spec"),
            "idi_missing": missing("instruction_density"),
            "nssi_missing": missing("nssi"),
            "surprisal_missing": missing("surprisal_dynamics"),
            "binoculars_missing": missing("contrastive_lm"),
            "tocsin_missing": missing("token_cohesiveness"),
            "semantic_kinematics_missing": missing("semantic_kinematics"),
            "specdetect_missing": missing("specdetect"),
            "perturbation_missing": missing("perturbation"),
            "contrastive_gen_missing": missing("contrastive_gen"),
            "dna_gpt_missing": missing("dna_gpt"),
            "lapd_missing": missing("lapd"),
            "mixed_boundary_missing": missing("mixed_boundary"),
            "cross_similarity_missing": missing("cross_similarity"),
            # Per-detector confidence (NaN when detector absent, raw confidence
            # in [0, 1] when present). Acts as a soft gate so the EBM can
            # downweight a low-confidence detector's p_llm contribution
            # without us hard-coding the policy. NaNs imputed at training and
            # inference via the per-column means in `_cairn_column_means`.
            "preamble_confidence": conf("preamble"),
            "fingerprint_confidence": conf("fingerprint_vocab"),
            "ps_confidence": conf("prompt_structure"),
            "vs_confidence": conf("voice_spec"),
            "idi_confidence": conf("instruction_density"),
            "nssi_confidence": conf("nssi"),
            "surprisal_confidence": conf("surprisal_dynamics"),
            "binoculars_confidence": conf("contrastive_lm"),
            "tocsin_confidence": conf("token_cohesiveness"),
            "semantic_kinematics_confidence": conf("semantic_kinematics"),
            "specdetect_confidence": conf("specdetect"),
            "perturbation_confidence": conf("perturbation"),
            "contrastive_gen_confidence": conf("contrastive_gen"),
            "dna_gpt_confidence": conf("dna_gpt"),
            "lapd_confidence": conf("lapd"),
            "mixed_boundary_confidence": conf("mixed_boundary"),
            "cross_similarity_confidence": conf("cross_similarity"),
            # Metadata
            "word_count": float(word_count),
            "log_word_count": math.log1p(float(word_count)),
            # Length-bucket indicators give the EBM clean breakpoints to
            # learn length-conditional behaviour (e.g. "trust this detector
            # less on short text"). Buckets chosen from the empirical
            # distribution: <150 captures terse Q&A and brief replies,
            # 150-500 the bulk of long-form responses, 500+ the long tail.
            "is_short": 1.0 if word_count < 150 else 0.0,
            "is_medium": 1.0 if 150 <= word_count < 500 else 0.0,
            "is_long": 1.0 if word_count >= 500 else 0.0,
            "domain_prompt": 1.0 if domain == "prompt" else 0.0,
            "domain_prose": 1.0 if domain == "prose" else 0.0,
        }
        if self._drop_features:
            for k in self._drop_features:
                vec.pop(k, None)
        return vec


ASSEMBLER_FEATURE_NAMES: tuple[str, ...] = tuple(
    FeatureAssembler().assemble([], word_count=0, domain="prose").keys()
)


# --- EBM Fusion ---

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    _HAS_INTERPRET = True
except ImportError:
    _HAS_INTERPRET = False


class EBMFusion:
    """Fusion using a trained EBM. Falls back to naive fusion if untrained."""

    def __init__(self, model: Any = None, conformal: Any = None):
        self._model = model
        self._conformal = conformal
        self._assembler = FeatureAssembler()
        self._feature_names: list[str] | None = None
        self._column_means: dict[str, float] = {}
        if model is not None:
            names = getattr(model, "_cairn_feature_names", None) or getattr(model, "_beet_feature_names", None)
            if names:
                self._feature_names = list(names)
                artifact_set = set(names)
                assembler_set = set(ASSEMBLER_FEATURE_NAMES)
                extra_in_artifact = artifact_set - assembler_set
                extra_in_assembler = assembler_set - artifact_set
                if extra_in_artifact:
                    raise RuntimeError(
                        "EBM artifact requires features the current assembler "
                        f"cannot produce: {sorted(extra_in_artifact)}. Either "
                        "retrain the EBM or update the assembler to produce them."
                    )
                if extra_in_assembler:
                    logger.info(
                        "Assembler produces features not in EBM artifact "
                        "(ignored until retrain): %s", sorted(extra_in_assembler),
                    )
            cm = getattr(model, "_cairn_column_means", None)
            # Use `is not None` (not truthiness): an empty dict is a valid
            # "trained, but no columns required imputation" signal and must
            # propagate, not be coerced to "legacy artifact, fall back to 0.5".
            if cm is not None:
                self._column_means = dict(cm)

    def fuse(
        self,
        layer_results: list[LayerResult],
        word_count: int = 0,
        domain: str | None = "prose",
    ) -> FusionResult:
        if self._model is None:
            return self._naive_fuse(layer_results, domain=domain)
        return self._ebm_fuse(layer_results, word_count, domain)

    def _conformal_ci(
        self, p_llm: float, router_class: str | None = None
    ) -> tuple[float, float] | None:
        """Conformal-derived CI half-width when calibrated, else ``None``.

        Half-width is the calibrated nonconformity threshold for the
        predicted class (``threshold_for(cls, router_class)``). This is the
        smallest deviation between ``p_llm`` and the class boundary at which
        the calibrated wrapper would still cover the true label with
        probability ``1 - alpha`` — i.e. an honest, coverage-guaranteed
        uncertainty band rather than a hardcoded ±0.10.

        Returns ``None`` when the wrapper isn't loaded, isn't calibrated,
        or fails to produce a threshold (legacy wrappers without
        ``threshold_for``, or per-class data missing). Callers fall back
        to the path-appropriate default in that case.
        """
        if self._conformal is None or not getattr(self._conformal, "calibrated", False):
            return None
        cls = 1 if p_llm >= 0.5 else 0
        try:
            hw = self._conformal.threshold_for(cls, router_class=router_class)
        except (TypeError, AttributeError):
            try:
                hw = self._conformal.threshold_for(cls)
            except (TypeError, AttributeError):
                return None
        if hw is None:
            return None
        hw = float(hw)
        # Reject degenerate zero-width thresholds — return None so caller's fallback fires.
        if hw <= 0.0:
            return None
        return (max(0.0, p_llm - hw), min(1.0, p_llm + hw))

    def _maybe_conformal(
        self, p_llm: float, router_class: str | None = None
    ) -> tuple[list[str] | None, float | None]:
        """Return the honest binary conformal set (+alpha) when calibrated.

        Falls back to a router-unaware ``predict_set(p_llm)`` call when the
        wrapped conformal object is a legacy implementation that does not
        accept ``router_class`` (TypeError on the keyword).
        """
        if self._conformal is None or not getattr(self._conformal, "calibrated", False):
            return None, None
        try:
            cs = self._conformal.predict_set(p_llm, router_class=router_class)
        except TypeError:
            logger.debug(
                "Legacy ConformalWrapper detected — predict_set does not accept router_class; "
                "falling back to marginal call"
            )
            cs = self._conformal.predict_set(p_llm)
        return cs, float(getattr(self._conformal, "alpha", 0.0))

    def _ebm_fuse(self, results: list[LayerResult], word_count: int, domain: str | None) -> FusionResult:
        import numpy as np
        if domain is None:
            # _ebm_fuse called without a router decision — surface this rather than
            # silently emitting domain_prompt=0.0, domain_prose=0.0 to the model
            # (a feature combination the EBM was never trained on).
            logger.warning(
                "_ebm_fuse called with domain=None; defaulting to 'prose' (out-of-distribution "
                "for prompt inputs). Pass an explicit domain via fuse(..., domain=...)."
            )
            domain = "prose"
        vec = self._assembler.assemble(results, word_count, domain)
        features = self._feature_names or list(vec.keys())
        X = np.array([[vec.get(f, _NAN) for f in features]])
        # Replace NaN per-column using means stored on the artifact at training
        # time (`_cairn_column_means`). Legacy artifacts without that attribute
        # fall back to the historical 0.5 default.
        for j, name in enumerate(features):
            if np.isnan(X[0, j]):
                X[0, j] = self._column_means.get(name, 0.5)
        proba = self._model.predict_proba(X)[0]
        p_llm = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Per-feature contributions from EBM
        try:
            local = self._model.explain_local(X)
            contribs = dict(zip(features, local.data(0)["scores"]))
            top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        except Exception:
            logger.warning("EBM explain_local failed; feature attributions unavailable", exc_info=True)
            contribs = {}
            top = []

        ci = self._conformal_ci(p_llm, router_class=domain) or (
            max(0.0, p_llm - 0.10), min(1.0, p_llm + 0.10)
        )
        # prediction_set uses fixed 0.10 band; only confidence_interval comes from the conformal threshold.
        prediction_set = _p_llm_to_labels(p_llm, 0.10)
        conformal_set, conformal_alpha = self._maybe_conformal(p_llm, router_class=domain)
        return FusionResult(p_llm=p_llm, confidence_interval=ci,
                            prediction_set=prediction_set,
                            feature_contributions=contribs, top_contributors=top,
                            conformal_set=conformal_set, conformal_alpha=conformal_alpha,
                            fusion_mode="ebm")

    def _naive_fuse(self, results: list[LayerResult], domain: str | None = None) -> FusionResult:
        """Fallback when no EBM is trained."""
        active = [r for r in results if r.determination != "SKIP" and r.confidence > 0]
        if not active:
            return FusionResult(p_llm=0.5, confidence_interval=(0.25, 0.75),
                                prediction_set=["YELLOW", "AMBER"],
                                feature_contributions={}, top_contributors=[],
                                conformal_set=None, conformal_alpha=None,
                                fusion_mode="naive")
        w_sum = sum(r.confidence for r in active)
        p_llm = sum(r.p_llm * r.confidence for r in active) / w_sum
        spread = max(r.p_llm for r in active) - min(r.p_llm for r in active)
        hw = max(0.05, spread / 2)
        ci = self._conformal_ci(p_llm, router_class=domain) or (0.0, 1.0)
        contribs = {r.layer_id: (r.p_llm - 0.5) * r.confidence for r in active}
        top = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        prediction_set = _p_llm_to_labels(p_llm, hw)
        conformal_set, conformal_alpha = self._maybe_conformal(p_llm, router_class=domain)
        return FusionResult(p_llm=p_llm, confidence_interval=ci,
                            prediction_set=prediction_set,
                            feature_contributions=contribs, top_contributors=top,
                            conformal_set=conformal_set, conformal_alpha=conformal_alpha,
                            fusion_mode="naive")


def _p_llm_to_labels(p_llm: float, uncertainty: float) -> list[str]:
    bands = [("RED", 0.75, 1.01), ("AMBER", 0.50, 0.75),
             ("YELLOW", 0.25, 0.50), ("GREEN", 0.0, 0.25)]
    labels = []
    for label, lo, hi in bands:
        if not (p_llm + uncertainty < lo or p_llm - uncertainty >= hi):
            labels.append(label)
    return labels or ["UNCERTAIN"]


# NaiveFusion alias preserved for backwards compatibility
class NaiveFusion(EBMFusion):
    pass


DEFAULT_FUSION = EBMFusion()

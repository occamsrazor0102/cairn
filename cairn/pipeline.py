import importlib
import logging
import math
import os
from pathlib import Path
from typing import Callable
from cairn.config import load_config
from cairn.contracts import Determination, LayerResult, RouterDecision
from cairn.normalizer import normalize_text
from cairn.router import TextRouter
from cairn.cascade import CascadeScheduler
from cairn.fusion.ebm import EBMFusion, DEFAULT_FUSION
from cairn.fusion.conformal import ConformalWrapper
from cairn.decision import DecisionEngine
import cairn.detectors as detector_registry

logger = logging.getLogger("cairn")


class BatchCancelled(Exception):
    """Raised by ``CairnPipeline.analyze_batch`` when its ``cancel_check`` callable
    returns truthy mid-iteration. Lets the sidecar tell a worker thread to abort
    cleanly without leaving the JobRegistry in a half-completed state."""


# Detectors that can't do meaningful work in the single-submission analyze
# path — they need batch context (multiple submissions) or external
# dependencies (API keys, torch). Used by detector_availability() so the
# Settings tab can report which detectors are really active vs which are
# enabled-but-stub.
_BATCH_ONLY_DETECTORS = {"cross_similarity", "contributor_graph"}


def _has_module(name: str) -> bool:
    try:
        importlib.util.find_spec(name)
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detector_availability(config: dict) -> list[dict]:
    """Introspect which detectors will actually produce useful output.

    Returns one row per detector declared in config, with:
      - id
      - enabled: from config
      - available: will contribute non-SKIP results in the single-submission
        analyze path
      - reason: short explanation when unavailable
      - requires: tags describing what the detector depends on
    """
    detectors_cfg = (config or {}).get("detectors") or {}
    all_dets = detector_registry.get_all_detectors()
    missing = detector_registry.get_missing_detectors()

    has_transformers = _has_module("transformers") and _has_module("torch")
    has_anthropic = _has_module("anthropic")
    has_openai = _has_module("openai")
    anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    openai_key = bool(os.environ.get("OPENAI_API_KEY"))

    out: list[dict] = []
    for det_id, det_cfg in detectors_cfg.items():
        enabled = bool((det_cfg or {}).get("enabled", True))
        row = {
            "id": det_id,
            "enabled": enabled,
            "available": False,
            "reason": "",
            "requires": [],
        }
        if det_id in missing:
            row["reason"] = f"import failed: {missing[det_id]}"
            out.append(row)
            continue
        if det_id not in all_dets:
            row["reason"] = "not registered"
            out.append(row)
            continue
        if not enabled:
            row["reason"] = "disabled in config"
            out.append(row)
            continue
        # Detector-specific readiness gates
        if det_id in _BATCH_ONLY_DETECTORS:
            row["requires"] = ["batch"]
            row["reason"] = "batch-only (needs 2+ submissions)"
            out.append(row)
            continue
        if det_id in {"contrastive_lm", "surprisal_dynamics", "perturbation", "token_cohesiveness"}:
            row["requires"] = ["transformers", "torch"]
            if not has_transformers:
                row["reason"] = "requires cairn[tier2] (transformers + torch)"
                out.append(row)
                continue
        if det_id == "contrastive_gen":
            provider = (det_cfg or {}).get("provider", "anthropic")
            row["requires"] = [f"{provider} API key"]
            if provider == "anthropic" and not (has_anthropic and anthropic_key):
                row["reason"] = "requires cairn[tier3] + ANTHROPIC_API_KEY"
                out.append(row)
                continue
            if provider == "openai" and not (has_openai and openai_key):
                row["reason"] = "requires cairn[tier3] + OPENAI_API_KEY"
                out.append(row)
                continue
        if det_id == "dna_gpt":
            provider = (det_cfg or {}).get("provider", "anthropic")
            row["requires"] = [f"{provider} API key"]
            if provider == "anthropic" and not (has_anthropic and anthropic_key):
                row["reason"] = "requires cairn[tier3] + ANTHROPIC_API_KEY"
                out.append(row)
                continue
            if provider == "openai" and not (has_openai and openai_key):
                row["reason"] = "requires cairn[tier3] + OPENAI_API_KEY"
                out.append(row)
                continue
        row["available"] = True
        out.append(row)
    return out


class CairnPipeline:
    def __init__(self, config: dict):
        self._config = config
        self._router = TextRouter(config)
        self._cascade = CascadeScheduler(config)
        self._fusion = self._build_fusion(config)
        self._decision = DecisionEngine(config)
        self._detectors = detector_registry.get_all_detectors()
        self._detector_calibrator = self._build_detector_calibrator(config)
        # Per-call accumulator for detector failures — initialised here so
        # _run_phase can be invoked through any path without an
        # AttributeError, and reset at the top of analyze_detailed.
        self._phase_errors: list[dict] = []
        self._monitor = None
        if config.get("monitoring", {}).get("enabled"):
            from cairn.monitoring.drift import DriftMonitor
            self._monitor = DriftMonitor(
                Path(config["monitoring"].get("store_path", "data/monitoring")),
                config,
            )
        missing = detector_registry.get_missing_detectors()
        for name, err in missing.items():
            logger.info(f"Detector '{name}' unavailable: {err}")

    @staticmethod
    def _build_fusion(config: dict) -> EBMFusion:
        fusion_cfg = config.get("fusion", {}) or {}
        model = None
        conformal = None
        model_path = fusion_cfg.get("model_path")
        if model_path and Path(model_path).exists():
            try:
                from cairn.fusion.training import load_model
                model = load_model(Path(model_path))
            except Exception as e:
                logger.warning(f"Failed to load EBM model at {model_path}: {e}")
        elif model_path:
            logger.info(f"EBM model path '{model_path}' not found; falling back to naive fusion")
        conformal_path = fusion_cfg.get("conformal_path")
        if conformal_path and Path(conformal_path).exists():
            try:
                conformal = ConformalWrapper()
                conformal.load(Path(conformal_path))
            except Exception as e:
                logger.warning(f"Failed to load conformal calibration at {conformal_path}: {e}")
                conformal = None
        if model is None and conformal is None:
            return DEFAULT_FUSION
        return EBMFusion(model=model, conformal=conformal)

    @staticmethod
    def _build_detector_calibrator(config: dict):
        fusion_cfg = config.get("fusion", {}) or {}
        cal_path = fusion_cfg.get("detector_calibration_path")
        if not cal_path or not Path(cal_path).exists():
            if cal_path:
                logger.info(f"Detector calibration path '{cal_path}' not found; skipping")
            return None
        try:
            from cairn.calibration import DetectorCalibrator
            cal = DetectorCalibrator()
            cal.load(Path(cal_path))
            return cal
        except Exception as e:
            logger.warning(f"Failed to load detector calibrator at {cal_path}: {e}")
            return None

    @classmethod
    def from_config_file(cls, path: Path | str) -> "CairnPipeline":
        return cls(load_config(Path(path)))

    def analyze(self, text: str, task_metadata: dict | None = None) -> Determination:
        det, _results, _rd = self.analyze_detailed(text, task_metadata)
        return det

    def analyze_detailed(
        self, text: str, task_metadata: dict | None = None
    ) -> tuple[Determination, list[LayerResult], RouterDecision]:
        text = normalize_text(text)
        router_decision = self._router.route(text)
        cfg = self._config
        # Reset per-call detector-failure accumulator (initialised in __init__).
        self._phase_errors = []
        results: list[LayerResult] = []
        phases_run = [1]
        phase1_results = self._run_phase(1, text, cfg, router_decision.skip_detectors, task_metadata)
        results.extend(phase1_results)
        if self._cascade.should_run_phase2(phase1_results):
            phase2_results = self._run_phase(2, text, cfg, router_decision.skip_detectors, task_metadata)
            results.extend(phase2_results)
            phases_run.append(2)
            if self._cascade.should_run_phase3(results):
                phase3_results = self._run_phase(3, text, cfg, router_decision.skip_detectors, task_metadata)
                results.extend(phase3_results)
                phases_run.append(3)
        fusion_result = self._fusion.fuse(
            results,
            word_count=router_decision.word_count,
            domain=router_decision.domain,
        )
        determination = self._decision.decide(fusion_result, results)
        determination.cascade_phases = phases_run
        determination.detector_errors = list(self._phase_errors)
        if self._monitor is not None:
            try:
                vec = self._fusion._assembler.assemble(
                    results, word_count=router_decision.word_count, domain=router_decision.domain,
                )
                self._monitor.record(determination.p_llm, determination.label, vec)
            except Exception as e:
                logger.warning(f"drift monitor record failed: {e}")
        return determination, results, router_decision

    def analyze_batch(
        self,
        texts: dict[str, str],
        task_metadata: dict | None = None,
        *,
        progress_cb: Callable[[int, int, str | None], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Determination]:
        """Run per-submission analysis enriched with cross-submission signals.

        For each submission, runs the normal cascade, then appends a
        `cross_similarity` layer result computed across the whole batch,
        re-runs fusion + decision. Contributor-graph batch detector is
        invoked separately via `analyze_contributors` (periodic, not per-call).

        ``progress_cb(done, total, current_id)`` is invoked once before the
        first item (``done=0``), once after each item completes (``done=i+1``),
        and once before the cross-similarity post-pass (``current_id="__cross__"``).
        Total counts only the per-submission phase; the post-pass adds at most
        a constant amount of additional work.

        If ``cancel_check`` is supplied and returns truthy, ``BatchCancelled``
        is raised — partially-completed work is discarded.
        """
        total = len(texts)
        if progress_cb is not None:
            progress_cb(0, total, None)

        per_sub: dict[str, tuple[Determination, list[LayerResult], RouterDecision]] = {}
        for i, (sid, text) in enumerate(texts.items()):
            if cancel_check is not None and cancel_check():
                raise BatchCancelled()
            det, results, rd = self.analyze_detailed(text, task_metadata)
            per_sub[sid] = (det, list(results), rd)
            if progress_cb is not None:
                progress_cb(i + 1, total, sid)

        cross = self._detectors.get("cross_similarity")
        cross_cfg = self._config.get("detectors", {}).get("cross_similarity", {})
        if cross is not None and cross_cfg.get("enabled", False) and len(texts) > 1:
            if cancel_check is not None and cancel_check():
                raise BatchCancelled()
            if progress_cb is not None:
                progress_cb(total, total, "__cross__")
            batch_results = cross.analyze_batch(texts, cross_cfg)
            for sid, layer_result in batch_results.items():
                if sid not in per_sub:
                    continue
                det, results, rd = per_sub[sid]
                results.append(layer_result)
                fusion_result = self._fusion.fuse(
                    results, word_count=rd.word_count, domain=rd.domain
                )
                enriched = self._decision.decide(fusion_result, results)
                enriched.cascade_phases = det.cascade_phases + [4]
                per_sub[sid] = (enriched, results, rd)

        return {sid: triple[0] for sid, triple in per_sub.items()}

    def _run_phase(self, phase: int, text: str, config: dict, skip: list[str],
                   task_metadata: dict | None = None) -> list[LayerResult]:
        phase_detector_ids = self._cascade.detectors_for_phase(phase)
        results = []
        detector_cfg = config.get("detectors", {})
        for det_id in phase_detector_ids:
            if det_id in skip: continue
            det_config = dict(detector_cfg.get(det_id, {}))
            if not det_config.get("enabled", True): continue
            if task_metadata:
                for k, v in task_metadata.items():
                    det_config.setdefault(k, v)
            detector = self._detectors.get(det_id)
            if detector is None:
                # Enabled in config but not importable (missing extras). Record
                # so the verdict carries visible evidence of reduced coverage
                # rather than quietly dropping a layer.
                self._phase_errors.append({
                    "layer_id": det_id,
                    "phase": phase,
                    "error": "detector not registered (install the relevant extras?)",
                })
                continue
            try:
                result = detector.analyze(text, det_config)
                if (self._detector_calibrator is not None
                        and self._detector_calibrator.has(det_id)
                        and math.isfinite(result.raw_score)):
                    result.p_llm = self._detector_calibrator.transform(det_id, result.raw_score)
                    result.calibrator_applied = True
                results.append(result)
            except Exception as exc:
                logger.warning(
                    "detector %s raised in phase %d: %s", det_id, phase, exc,
                    exc_info=True,
                )
                self._phase_errors.append({
                    "layer_id": det_id,
                    "phase": phase,
                    "error": str(exc) or type(exc).__name__,
                })
        return results

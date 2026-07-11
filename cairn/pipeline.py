import datetime
import hashlib
import importlib
import logging
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
from cairn.history import mint_submission_id
from cairn.privacy.hashing import hash_text
from cairn.provenance.chain import AuditChain
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
        # An explicit config-disable is a deliberate operator choice and
        # takes precedence over import/registration state: a disabled
        # detector reports "disabled" even when its optional extras are
        # absent (so it never registered in this environment).
        if not enabled:
            row["reason"] = "disabled in config"
            out.append(row)
            continue
        if det_id in missing:
            row["reason"] = f"import failed: {missing[det_id]}"
            out.append(row)
            continue
        if det_id not in all_dets:
            row["reason"] = "not registered"
            out.append(row)
            continue
        # Detector-specific readiness gates
        if det_id in _BATCH_ONLY_DETECTORS:
            row["requires"] = ["batch"]
            row["reason"] = "batch-only (needs 2+ submissions)"
            out.append(row)
            continue
        if det_id in {"contrastive_lm", "surprisal_dynamics", "perturbation", "token_cohesiveness", "fast_detectgpt", "lapd", "specdetect"}:
            row["requires"] = ["transformers", "torch"]
            if not has_transformers:
                row["reason"] = "requires cairn[tier2] (transformers + torch)"
                out.append(row)
                continue
        if det_id in {"semantic_kinematics"}:
            row["requires"] = ["sentence_transformers"]
            if not _has_module("sentence_transformers"):
                row["reason"] = "requires cairn[retrieval] (sentence-transformers)"
                out.append(row)
                continue
        if det_id in ("contrastive_gen", "dna_gpt"):
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
            if provider == "openai-compat":
                env_name = (det_cfg or {}).get("api_key_env", "")
                if not (has_openai and env_name and os.environ.get(env_name)):
                    row["reason"] = f"requires cairn[tier3] + {env_name or 'api_key_env'} env var"
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
        # Tamper-evident audit chain. Off by default; opt-in via `audit.enabled`.
        # Computed once at init so analyze() doesn't re-stat the model file or
        # re-derive calibration_status on every call.
        self._audit_chain: AuditChain | None = None
        self._calibration_status = self._compute_calibration_status()
        self._model_artifact_hash = self._compute_model_artifact_hash(config)
        audit_cfg = config.get("audit") or {}
        if audit_cfg.get("enabled"):
            log_path = Path(audit_cfg.get("log_path") or "data/audit/chain.jsonl")
            try:
                self._audit_chain = AuditChain(log_path)
            except Exception as e:
                logger.warning(f"audit chain init failed at {log_path}: {e}")
                self._audit_chain = None
        self._retrieval_index = None
        rcfg = (config or {}).get("retrieval", {})
        if rcfg.get("enabled"):
            try:
                from cairn.index.store import PersistentRetrievalIndex
                self._retrieval_index = PersistentRetrievalIndex(
                    db_path=rcfg.get("db_path", "data/retrieval.db"),
                )
            except ImportError:
                logger.warning(
                    "retrieval extra not installed; cross_similarity falls back to batch."
                )
            except Exception:
                logger.warning("retrieval index init failed; continuing without", exc_info=True)
        missing = detector_registry.get_missing_detectors()
        for name, err in missing.items():
            logger.info(f"Detector '{name}' unavailable: {err}")

    def _compute_calibration_status(self) -> str:
        # Pinned at init so determinations and audit records reflect the
        # artifacts actually loaded, not later runtime mutation. Health APIs
        # use calibration_health(), which adds router-conditional detail and
        # publishes only a sanitized detector-calibration failure state.
        if self._detector_calibration_state["state"] == "load-failed":
            return "degraded"
        model = getattr(self._fusion, "_model", None)
        conformal = getattr(self._fusion, "_conformal", None)
        conformal_loaded = (
            conformal is not None
            and getattr(conformal, "calibrated", False)
        )
        if model is not None and conformal_loaded:
            return "calibrated"
        if model is not None:
            return "fusion-only"
        return "heuristic"

    def calibration_health(self) -> dict[str, object]:
        """Return the canonical, sanitized calibration state for health APIs."""
        model = getattr(self._fusion, "_model", None)
        conformal = getattr(self._fusion, "_conformal", None)
        model_loaded = model is not None
        conformal_loaded = (
            conformal is not None
            and getattr(conformal, "calibrated", False)
        )
        state = self._detector_calibration_state["state"]
        if state == "load-failed":
            calibration_status = "degraded"
        elif (
            model_loaded
            and conformal_loaded
            and getattr(conformal, "threshold_per_router", None)
        ):
            calibration_status = "calibrated-router-conditional"
        elif model_loaded and conformal_loaded:
            calibration_status = "calibrated"
        elif model_loaded:
            calibration_status = "fusion-only"
        else:
            calibration_status = "heuristic"
        return {
            "model_loaded": model_loaded,
            "conformal_loaded": conformal_loaded,
            "calibration_status": calibration_status,
            "detector_calibration": {
                "state": state,
                "error": (
                    "artifact-load-failed"
                    if state == "load-failed"
                    else None
                ),
            },
        }

    @staticmethod
    def _compute_model_artifact_hash(config: dict) -> str | None:
        model_path = (config.get("fusion") or {}).get("model_path")
        if not model_path:
            return None
        p = Path(model_path)
        if not p.exists():
            return None
        try:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return f"sha256:{h.hexdigest()}"
        except Exception as e:
            # Audit records distinguish "no model" (None when no path
            # configured) from "couldn't read it" — log so the latter
            # shows up in operator logs even though both serialize as
            # null in the manifest.
            logger.warning(f"model artifact hash failed for {p}: {e}")
            return None

    def _audit_append(
        self,
        text: str,
        determination: Determination,
        task_metadata: dict | None,
    ) -> None:
        # Best-effort: a failed write must never fail analyze. Same discipline
        # as history.py and monitoring/drift.py.
        if self._audit_chain is None:
            return
        try:
            sid = (task_metadata or {}).get("submission_id") or mint_submission_id()
            now_iso = (
                datetime.datetime.now(datetime.timezone.utc)
                .replace(tzinfo=None)
                .isoformat()
                + "Z"
            )
            manifest = {
                "submission_id": sid,
                "text_hash": hash_text(text),
                "profile": self._config.get("_profile", "default"),
                "p_llm": round(determination.p_llm, 4),
                "determination": determination.label,
                "calibration_status": self._calibration_status,
                "model_artifact_hash": self._model_artifact_hash,
                "timestamp": now_iso,
            }
            self._audit_chain.append(manifest)
        except Exception as e:
            logger.warning(f"audit chain append failed: {e}")

    def close(self) -> None:
        """Close the pipeline cleanly, flushing any persistent state."""
        if self._retrieval_index is not None:
            try:
                self._retrieval_index.close()
            except Exception:
                logger.warning("retrieval index close failed", exc_info=True)
            self._retrieval_index = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _build_fusion(config: dict) -> EBMFusion:
        fusion_cfg = config.get("fusion", {}) or {}
        model = None
        conformal = None
        model_path = fusion_cfg.get("model_path")
        if model_path and Path(model_path).exists():
            try:
                from cairn.fusion.serialization import load_model
                model, _meta = load_model(Path(model_path))
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
        # OOD-length guard (opt-in per profile). Caps word_count before the
        # length features so prompts far longer than the training distribution
        # don't inherit the EBM's saturated long-tail length contribution.
        clamp = fusion_cfg.get("word_count_clamp")
        return EBMFusion(
            model=model, conformal=conformal,
            word_count_clamp=int(clamp) if clamp is not None else None,
        )

    @property
    def detector_calibration_state(self) -> dict[str, str | None]:
        """Audit-safe state for the detector-calibration artifact declaration."""
        return dict(self._detector_calibration_state)

    def _build_detector_calibrator(self, config: dict):
        fusion_cfg = config.get("fusion", {}) or {}
        cal_path = fusion_cfg.get("detector_calibration_path")
        if not cal_path:
            self._detector_calibration_state = {
                "state": "not-configured",
                "path": None,
                "error": None,
            }
            return None
        path_text = str(cal_path)
        try:
            path = Path(cal_path)
        except (TypeError, ValueError) as exc:
            self._detector_calibration_state = {
                "state": "load-failed",
                "path": path_text,
                "error": str(exc),
            }
            logger.warning(
                f"Failed to resolve detector calibrator path {cal_path}: {exc}"
            )
            return None
        if not path.exists():
            error = "artifact path not found"
            self._detector_calibration_state = {
                "state": "load-failed",
                "path": path_text,
                "error": error,
            }
            logger.warning(
                f"Detector calibration path '{cal_path}' not found; "
                "marking calibration degraded"
            )
            return None
        try:
            from cairn.calibration import DetectorCalibrator
            cal = DetectorCalibrator()
            cal.load(path)
            self._detector_calibration_state = {
                "state": "loaded",
                "path": path_text,
                "error": None,
            }
            return cal
        except Exception as e:
            self._detector_calibration_state = {
                "state": "load-failed",
                "path": path_text,
                "error": str(e) or type(e).__name__,
            }
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
        # Auto-derive task_description when caller didn't supply one and the
        # config opts in. Default auto_mode is "off" (no behavior change for
        # existing callers). Warning is attached after decide() so the
        # decision engine can still own Determination.warnings normally.
        auto_warning: str | None = None
        td_cfg = cfg.get("task_description") or {}
        if td_cfg.get("auto_mode", "off") not in ("off", None, False):
            existing_td = (task_metadata or {}).get("task_description")
            if not existing_td:
                from cairn.task_description import auto_derive
                desc, _mode_used, auto_warning = auto_derive(text, td_cfg)
                if desc:
                    task_metadata = {**(task_metadata or {}), "task_description": desc}
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
        # decide() defaults calibration_status to "heuristic"; stamp the
        # pipeline's actual status so the determination (and report.py) agree
        # with /health instead of always claiming heuristic.
        determination.calibration_status = self._calibration_status
        if auto_warning:
            determination.warnings = list(determination.warnings or []) + [auto_warning]
        if self._monitor is not None:
            try:
                vec = self._fusion._assembler.assemble(
                    results, word_count=router_decision.word_count, domain=router_decision.domain,
                )
                self._monitor.record(determination.p_llm, determination.label, vec)
            except Exception as e:
                logger.warning(f"drift monitor record failed: {e}")
        if self._retrieval_index is not None:
            try:
                idx_sid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                self._retrieval_index.add(idx_sid, text)
            except Exception:
                logger.warning("retrieval index add failed", exc_info=True)
        self._audit_append(text, determination, task_metadata)
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
            # Thread the per-submission sid into task_metadata so the audit
            # chain stamps each batch entry with the correct submission_id
            # rather than minting a fresh uuid per row or smearing one
            # caller-supplied id across the whole batch.
            sub_meta = {**(task_metadata or {}), "submission_id": sid}
            det, results, rd = self.analyze_detailed(text, sub_meta)
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
                if self._detector_calibrator is not None:
                    self._detector_calibrator.apply(layer_result)
                results.append(layer_result)
                fusion_result = self._fusion.fuse(
                    results, word_count=rd.word_count, domain=rd.domain
                )
                enriched = self._decision.decide(fusion_result, results)
                enriched.cascade_phases = det.cascade_phases + [4]
                enriched.calibration_status = self._calibration_status
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
            if det_id == "cross_similarity" and self._retrieval_index is not None:
                det_config = {**det_config, "_index_handle": self._retrieval_index, "use_retrieval_index": True}
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
                if self._detector_calibrator is not None:
                    self._detector_calibrator.apply(result)
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

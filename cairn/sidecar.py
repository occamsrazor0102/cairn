"""CAIRN pipeline sidecar: line-delimited JSON-RPC over stdio.

Spawned by the Tauri shell as a long-running subprocess. Reads one JSON
object per line from stdin, writes one JSON object per line to stdout.
All diagnostic logging goes to stderr so stdout stays pure JSON-RPC.

Request envelope:  {"id": <any>, "method": <str>, "params": {...}}
Response (ok):     {"id": <same>, "result": {...}}
Response (err):    {"id": <same or null>, "error": {"code": <str>, "message": <str>}}

Methods:
  analyze        {text, submission_id?}                 -> report dict
  analyze_batch  {items: [{id, text}, ...]}             -> {results: [report, ...]}
  feedback       {text, confirmed_label,
                  submission_id?, reviewer_notes?}      -> {ok, submission_id, recorded_at}
  health         {}                                     -> {status, model_loaded, conformal_loaded}
  shutdown       {}                                     -> {ok: true}  (then exits)

On startup, before reading stdin, emits one line:
  {"event": "ready", "pid": <pid>, "protocol_version": 1}

Run:
  python -m cairn.sidecar --config configs/default.yaml
  python -m cairn.sidecar --profile default
  cairn sidecar
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from cairn.auto_routing import recommend_profile
from cairn.config import CONFIGS_DIR, list_profiles, load_config
from cairn.history import HistoryStore, mint_submission_id
from cairn.monitoring.drift import DriftMonitor
from cairn.monitoring.meta_detector import MetaDetector
from cairn.pipeline import BatchCancelled, CairnPipeline
from cairn.report import build_json_report
from cairn.router import TextRouter
from cairn.runtime import RuntimeContext


PROTOCOL_VERSION = 1
DEFAULT_FEEDBACK_PATH = Path("data") / "reviewer_feedback.jsonl"


def _curated_config_view(profile: str | None, config: dict) -> dict:
    """A UI-friendly projection of the config.

    Kept separate from the raw dict so the RPC shape stays stable even if
    the internal config layout changes.
    """
    from cairn.pipeline import detector_availability

    decision = (config or {}).get("decision") or {}
    abst = decision.get("abstention") or {}
    detectors_raw = (config or {}).get("detectors") or {}
    avail = {row["id"]: row for row in detector_availability(config or {})}
    detectors = [
        {
            "id": did,
            "enabled": bool((d or {}).get("enabled", True)),
            "weight": float((d or {}).get("weight", 1.0)),
            "available": bool(avail.get(did, {}).get("available", False)),
            "reason": avail.get(did, {}).get("reason", ""),
            "requires": avail.get(did, {}).get("requires", []),
        }
        for did, d in detectors_raw.items()
    ]
    history = ((config or {}).get("gui") or {}).get("history") or {}
    return {
        "profile": profile,
        "thresholds": {
            "red": float(decision.get("red_threshold", 0.75)),
            "amber": float(decision.get("amber_threshold", 0.50)),
            "yellow": float(decision.get("yellow_threshold", 0.25)),
            "abstention": {
                "enabled": bool(abst.get("enabled", True)),
                "max_prediction_set_size": int(abst.get("max_prediction_set_size", 3)),
            },
        },
        "detectors": detectors,
        "history": {
            "enabled": bool(history.get("enabled", True)),
            "retain_text": bool(history.get("retain_text", False)),
            "db_path": history.get("db_path", "data/cairn_history.sqlite3"),
        },
    }


def drift_from_config(config: dict | None) -> DriftMonitor | None:
    gui = (config or {}).get("gui") or {}
    drift_cfg = gui.get("drift") or {}
    if drift_cfg.get("enabled", True) is False:
        return None
    store_path = Path(drift_cfg.get("store_path", "data/drift_alerts"))
    # DriftMonitor reads tuning keys (window_size, kl_threshold, ece_threshold)
    # from config["drift_monitoring"]. Route any equivalent keys nested under
    # gui.drift through that top-level namespace so the obvious place to tune
    # drift (next to enabled/store_path) actually takes effect. Top-level
    # drift_monitoring wins if both are set.
    merged = dict(config or {})
    top_level = dict(merged.get("drift_monitoring") or {})
    tuning_keys = {"window_size", "kl_threshold", "ece_threshold"}
    for k in tuning_keys:
        if k not in top_level and k in drift_cfg:
            top_level[k] = drift_cfg[k]
    if top_level:
        merged["drift_monitoring"] = top_level
    # Optional sqlite-backed persistent observation window. When unset, the
    # DriftMonitor falls back to its in-memory list (preserving prior
    # behavior). Operators set `gui.drift.db_path` to enable persistence so
    # the in-flight window survives sidecar restart.
    db_path_raw = drift_cfg.get("db_path")
    db_path = Path(db_path_raw) if db_path_raw else None
    try:
        return DriftMonitor(store_path, merged, db_path=db_path)
    except Exception as e:
        log.warning("drift monitor disabled: %s", e)
        return None


def drift_baseline_from_history(
    drift: DriftMonitor, history: HistoryStore, *, limit: int = 500
) -> int:
    """Pull feature_contributions from recent history, seed drift baseline."""
    rows = history.list(limit=int(limit), offset=0)
    feature_vectors: list[dict] = []
    for item in rows.get("items", []):
        sid = item["submission_id"]
        got = history.get(sid)
        if not got:
            continue
        fc = (got.get("report") or {}).get("feature_contributions") or {}
        if fc:
            feature_vectors.append(fc)
    drift.set_baseline(feature_vectors)
    return len(feature_vectors)


def history_from_config(config: dict | None) -> HistoryStore | None:
    gui = (config or {}).get("gui") or {}
    hist = gui.get("history") or {}
    if hist.get("enabled", True) is False:
        return None
    db_path = Path(hist.get("db_path", "data/cairn_history.sqlite3"))
    retain = bool(hist.get("retain_text", False))
    try:
        return HistoryStore(db_path, retain_text=retain)
    except Exception as e:
        log.warning("history store disabled: %s", e)
        return None

log = logging.getLogger("cairn.sidecar")


class SidecarError(Exception):
    """Raised for method-level failures. `code` surfaces to the caller."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class JobState:
    """In-memory state for a long-running RPC job (e.g. async batch).

    Clients poll via ``get_job(job_id)`` and may abort via ``cancel_job``.
    ``result`` is populated only on successful completion; ``error`` only on
    a non-cancellation failure.
    """
    job_id: str
    kind: str
    total: int
    done: int = 0
    current: str | None = None
    status: str = "running"  # running | completed | cancelled | error
    error: str | None = None
    result: dict | None = None
    started_at: str = ""
    updated_at: str = ""

    def to_public(self) -> dict:
        return asdict(self)


class JobRegistry:
    """Thread-safe registry of running and recently-completed jobs.

    Worker threads update progress via ``update``; the main RPC loop reads
    snapshots via ``get`` and signals cancellation via ``cancel`` (which
    flips the threading.Event the worker is polling). Completed jobs are
    pruned best-effort on each new ``create`` so polling clients can still
    fetch the final result for ~max_age_seconds.
    """

    def __init__(self, *, max_age_seconds: float = 3600.0, max_jobs: int = 50):
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        self._cancels: dict[str, threading.Event] = {}
        self._max_age_seconds = max_age_seconds
        self._max_jobs = max_jobs

    @staticmethod
    def _now() -> str:
        return (
            datetime.datetime.now(datetime.timezone.utc)
            .replace(tzinfo=None)
            .isoformat()
            + "Z"
        )

    def create(self, kind: str, total: int) -> tuple[str, threading.Event]:
        job_id = uuid.uuid4().hex
        now = self._now()
        ev = threading.Event()
        with self._lock:
            self._prune_locked()
            self._jobs[job_id] = JobState(
                job_id=job_id,
                kind=kind,
                total=int(total),
                started_at=now,
                updated_at=now,
            )
            self._cancels[job_id] = ev
        return job_id, ev

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            ev = self._cancels.get(job_id)
            job = self._jobs.get(job_id)
        if ev is None:
            return False
        ev.set()
        # Flip status immediately so a poll between cancel and worker exit
        # surfaces the user's intent. Worker's ``finish`` respects this.
        with self._lock:
            if job and job.status == "running":
                job.status = "cancelled"
                job.updated_at = self._now()
        return True

    def update(
        self,
        job_id: str,
        *,
        done: int | None = None,
        current: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if done is not None:
                job.done = int(done)
            job.current = current
            job.updated_at = self._now()

    def finish(
        self,
        job_id: str,
        *,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            now = self._now()
            if error is not None:
                # Cancellation wins over error reporting — if the user already
                # asked to stop, don't surface the BatchCancelled traceback.
                if job.status != "cancelled":
                    job.status = "error"
                    job.error = error
            elif job.status != "cancelled":
                job.status = "completed"
                job.done = job.total
            if result is not None:
                job.result = result
            job.updated_at = now

    def _prune_locked(self) -> None:
        """Drop old terminal jobs and trim back to max_jobs."""
        terminal = {"completed", "cancelled", "error"}
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            seconds=self._max_age_seconds
        )
        for jid in list(self._jobs.keys()):
            j = self._jobs[jid]
            if j.status not in terminal:
                continue
            try:
                ts = datetime.datetime.fromisoformat(
                    j.updated_at.rstrip("Z")
                ).replace(tzinfo=datetime.timezone.utc)
            except Exception:
                continue
            if ts < cutoff:
                self._jobs.pop(jid, None)
                self._cancels.pop(jid, None)
        if len(self._jobs) > self._max_jobs:
            terminals = sorted(
                (
                    (jid, j)
                    for jid, j in self._jobs.items()
                    if j.status in terminal
                ),
                key=lambda kv: kv[1].updated_at,
            )
            overflow = len(self._jobs) - self._max_jobs
            for jid, _ in terminals[:overflow]:
                self._jobs.pop(jid, None)
                self._cancels.pop(jid, None)


def record_feedback(
    pipeline: CairnPipeline,
    meta: MetaDetector,
    feedback_path: Path,
    params: dict,
    history: HistoryStore | None = None,
) -> dict:
    """Shared feedback handler used by both the HTTP server and the sidecar.

    Re-runs the pipeline to get per-detector layer results, records them
    against a confirmed label via MetaDetector, and appends the raw submission
    to the JSONL. Pipeline failures are logged but do not block JSONL
    persistence — the ground-truth label is the authoritative output.
    """
    text = params.get("text")
    if not isinstance(text, str) or not text.strip():
        raise SidecarError("ERR_BAD_PARAMS", "text required (non-empty string)")
    label = params.get("confirmed_label")
    if label not in (0, 1):
        raise SidecarError("ERR_BAD_PARAMS", "confirmed_label must be 0 (human) or 1 (LLM)")
    submission_id = str(params.get("submission_id") or "")
    notes = params.get("reviewer_notes")
    if notes is not None and not isinstance(notes, str):
        raise SidecarError("ERR_BAD_PARAMS", "reviewer_notes must be a string")

    try:
        _det, layer_results, _rd = pipeline.analyze_detailed(text)
        for lr in layer_results:
            try:
                meta.record(lr, confirmed_label=int(label))
            except Exception as e:
                log.warning("meta record failed for %s: %s", lr.layer_id, e)
    except Exception as e:
        log.warning("feedback re-analysis failed; persisting label only: %s", e)

    recorded_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z"
    row: dict[str, Any] = {
        "submission_id": submission_id,
        "text": text,
        "confirmed_label": int(label),
        "recorded_at": recorded_at,
    }
    if notes:
        row["reviewer_notes"] = notes
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    if history is not None and submission_id:
        try:
            history.record_feedback(submission_id, int(label), notes)
        except Exception as e:
            log.warning("history feedback write failed: %s", e)

    return {"ok": True, "submission_id": submission_id, "recorded_at": recorded_at}


class Sidecar:
    def __init__(
        self,
        pipeline: CairnPipeline | None = None,
        feedback_path: Path | None = None,
        history: HistoryStore | None = None,
        profile: str | None = None,
        ctx: RuntimeContext | None = None,
        drift: DriftMonitor | None = None,
    ):
        if ctx is None:
            if pipeline is None:
                raise ValueError("Sidecar requires either ctx or pipeline")
            ctx = RuntimeContext(pipeline, profile, {})
        self._ctx = ctx
        self._feedback_path = feedback_path or DEFAULT_FEEDBACK_PATH
        self._history = history
        self._drift = drift
        self._meta = MetaDetector()
        self._jobs = JobRegistry()
        self._running = True

    @property
    def _pipeline(self) -> CairnPipeline:
        return self._ctx.pipeline

    @property
    def _profile(self) -> str | None:
        return self._ctx.profile

    def handle(self, method: str, params: dict) -> dict:
        if method == "analyze":
            return self._analyze(params)
        if method == "analyze_batch":
            return self._analyze_batch(params)
        if method == "analyze_batch_async":
            return self._analyze_batch_async(params)
        if method == "get_job":
            return self._get_job(params)
        if method == "cancel_job":
            return self._cancel_job(params)
        if method == "feedback":
            return self._feedback(params)
        if method == "health":
            return self._health()
        if method == "history_list":
            return self._history_list(params)
        if method == "history_get":
            return self._history_get(params)
        if method == "history_delete":
            return self._history_delete(params)
        if method == "history_export":
            return self._history_export(params)
        if method == "list_profiles":
            return self._list_profiles()
        if method == "get_config":
            return self._get_config()
        if method == "switch_profile":
            return self._switch_profile(params)
        if method == "monitoring_summary":
            return self._monitoring_summary(params)
        if method == "monitoring_timeline":
            return self._monitoring_timeline(params)
        if method == "monitoring_detectors":
            return self._monitoring_detectors(params)
        if method == "monitoring_cascade":
            return self._monitoring_cascade(params)
        if method == "run_eval":
            return self._run_eval(params)
        if method == "monitoring_drift":
            return self._monitoring_drift()
        if method == "monitoring_set_baseline":
            return self._monitoring_set_baseline(params)
        if method == "recommend_profile":
            return self._recommend_profile(params)
        if method == "shutdown":
            self._running = False
            return {"ok": True}
        raise SidecarError("ERR_METHOD_NOT_FOUND", f"unknown method '{method}'")

    @property
    def running(self) -> bool:
        return self._running

    def _analyze(self, params: dict) -> dict:
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SidecarError("ERR_BAD_PARAMS", "text required (non-empty string)")
        submission_id = str(params.get("submission_id") or "").strip() or mint_submission_id()
        try:
            det = self._pipeline.analyze(text)
        except Exception as e:
            raise SidecarError("ERR_PIPELINE", f"analyze failed: {e}") from e
        report = build_json_report(det, submission_id=submission_id)
        self._record(report, source="analyze", text=text)
        return report

    def _parse_batch_items(
        self, params: dict
    ) -> tuple[dict[str, str], list[tuple[int, str]], list[int], str]:
        """Validate and normalize batch RPC params.

        Returns (texts, order, skipped, batch_id). Raises SidecarError with
        ERR_BAD_PARAMS for any structural issue. Shared by the sync and async
        entry points so they enforce the same rules and produce the same
        skip/order semantics.
        """
        items = params.get("items") or []
        if not isinstance(items, list) or not items:
            raise SidecarError("ERR_BAD_PARAMS", "items required (non-empty list)")
        if len(items) > 500:
            raise SidecarError("ERR_BAD_PARAMS", "batch exceeds 500-item cap")
        batch_id = str(
            params.get("batch_id")
            or mint_submission_id().replace("sub_", "batch_")
        )
        texts: dict[str, str] = {}
        order: list[tuple[int, str]] = []
        skipped: list[int] = []
        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                skipped.append(idx)
                continue
            sid = str(it.get("id") or f"{batch_id}_{idx:04d}")
            txt = it.get("text")
            if not isinstance(txt, str) or not txt.strip():
                skipped.append(idx)
                continue
            if sid in texts:
                raise SidecarError(
                    "ERR_BAD_PARAMS",
                    f"duplicate submission_id '{sid}' at index {idx}",
                )
            texts[sid] = txt
            order.append((idx, sid))
        if not texts:
            raise SidecarError("ERR_BAD_PARAMS", "no valid items in batch")
        return texts, order, skipped, batch_id

    def _collect_batch_results(
        self,
        dets: dict,
        order: list[tuple[int, str]],
        texts: dict[str, str],
        skipped: list[int],
        batch_id: str,
    ) -> dict:
        """Build the response dict that both sync and async paths return."""
        results = []
        for _idx, sid in order:
            det = dets.get(sid)
            if det is None:
                continue
            report = build_json_report(det, submission_id=sid)
            try:
                self._record(report, source="batch", text=texts[sid], batch_id=batch_id)
            except Exception as e:
                log.warning("history record failed for %s: %s", sid, e)
            results.append(report)
        return {"results": results, "skipped": skipped, "batch_id": batch_id}

    def _analyze_batch(self, params: dict) -> dict:
        texts, order, skipped, batch_id = self._parse_batch_items(params)
        try:
            dets = self._pipeline.analyze_batch(texts)
        except Exception as e:
            raise SidecarError("ERR_PIPELINE", f"analyze_batch failed: {e}") from e
        return self._collect_batch_results(dets, order, texts, skipped, batch_id)

    def _analyze_batch_async(self, params: dict) -> dict:
        """Kick off a batch analysis in a worker thread, return a job_id.

        The worker reports progress via the JobRegistry; clients poll
        ``get_job`` and may abort via ``cancel_job``. Validation errors
        (bad shape, duplicate IDs, empty batch) still raise synchronously
        — only successful kick-off returns a job_id.
        """
        texts, order, skipped, batch_id = self._parse_batch_items(params)
        job_id, cancel_event = self._jobs.create(kind="batch", total=len(texts))

        # Capture refs so the worker doesn't depend on `self` mutation.
        pipeline = self._pipeline
        jobs = self._jobs
        collect = self._collect_batch_results

        def _progress(done: int, total: int, current: str | None) -> None:
            jobs.update(job_id, done=done, current=current)

        def _cancelled() -> bool:
            return cancel_event.is_set()

        def worker() -> None:
            try:
                dets = pipeline.analyze_batch(
                    texts, progress_cb=_progress, cancel_check=_cancelled
                )
                payload = collect(dets, order, texts, skipped, batch_id)
                jobs.finish(job_id, result=payload)
            except BatchCancelled:
                jobs.finish(job_id)  # cancel() already flipped status
            except Exception as e:
                log.warning(
                    "async batch %s failed: %s\n%s",
                    job_id,
                    e,
                    traceback.format_exc(),
                )
                jobs.finish(job_id, error=str(e) or type(e).__name__)

        threading.Thread(
            target=worker, name=f"cairn-batch-{job_id[:8]}", daemon=True
        ).start()
        return {
            "job_id": job_id,
            "kind": "batch",
            "total": len(texts),
            "batch_id": batch_id,
            "skipped": skipped,
        }

    def _get_job(self, params: dict) -> dict:
        job_id = str(params.get("job_id") or "")
        if not job_id:
            raise SidecarError("ERR_BAD_PARAMS", "job_id required")
        job = self._jobs.get(job_id)
        if job is None:
            raise SidecarError("ERR_NOT_FOUND", f"job '{job_id}' not found")
        return job.to_public()

    def _cancel_job(self, params: dict) -> dict:
        job_id = str(params.get("job_id") or "")
        if not job_id:
            raise SidecarError("ERR_BAD_PARAMS", "job_id required")
        ok = self._jobs.cancel(job_id)
        if not ok:
            raise SidecarError("ERR_NOT_FOUND", f"job '{job_id}' not found")
        return {"ok": True, "job_id": job_id}

    def _feedback(self, params: dict) -> dict:
        return record_feedback(
            self._pipeline, self._meta, self._feedback_path, params, history=self._history
        )

    def _health(self) -> dict:
        fusion = self._pipeline._fusion
        model_loaded = fusion._model is not None
        conformal_loaded = (
            fusion._conformal is not None
            and getattr(fusion._conformal, "calibrated", False)
        )
        router_conditional = (
            conformal_loaded
            and getattr(fusion._conformal, "threshold_per_router", None)
        )
        if model_loaded and router_conditional:
            cal_status = "calibrated-router-conditional"
        elif model_loaded and conformal_loaded:
            cal_status = "calibrated"
        elif model_loaded:
            cal_status = "fusion-only"  # trained fusion but no conformal
        else:
            cal_status = "heuristic"  # no trained model; naive fallback, hand-picked thresholds
        return {
            "status": "ok",
            "model_loaded": model_loaded,
            "conformal_loaded": conformal_loaded,
            "calibration_status": cal_status,
            "protocol_version": PROTOCOL_VERSION,
            "history_enabled": self._history is not None,
            "profile": self._profile,
        }

    def _history_list(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        return self._history.list(
            limit=int(params.get("limit", 25)),
            offset=int(params.get("offset", 0)),
            determination=params.get("determination") or None,
            since=params.get("since"),
            until=params.get("until"),
            batch_id=params.get("batch_id"),
            search=params.get("search"),
        )

    def _history_get(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        sid = str(params.get("submission_id") or "").strip()
        if not sid:
            raise SidecarError("ERR_BAD_PARAMS", "submission_id required")
        got = self._history.get(sid)
        if got is None:
            raise SidecarError("ERR_NOT_FOUND", f"no submission '{sid}'")
        return got

    def _history_delete(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        sid = str(params.get("submission_id") or "").strip()
        if not sid:
            raise SidecarError("ERR_BAD_PARAMS", "submission_id required")
        return {"ok": self._history.delete(sid)}

    def _history_export(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        fmt = str(params.get("format", "json")).lower()
        if fmt not in ("json", "csv"):
            raise SidecarError("ERR_BAD_PARAMS", "format must be 'json' or 'csv'")
        content, mime, filename = self._history.export(
            fmt=fmt,
            determination=params.get("determination") or None,
            since=params.get("since"),
            until=params.get("until"),
            batch_id=params.get("batch_id"),
            search=params.get("search"),
        )
        return {"content": content, "mime": mime, "filename": filename}

    def _list_profiles(self) -> dict:
        return {"profiles": list_profiles(), "current": self._ctx.profile}

    def _get_config(self) -> dict:
        return _curated_config_view(self._ctx.profile, self._ctx.config)

    def _switch_profile(self, params: dict) -> dict:
        name = str(params.get("name") or "").strip()
        if not name:
            raise SidecarError("ERR_BAD_PARAMS", "name required")
        try:
            self._ctx.switch_profile(name)
        except FileNotFoundError as e:
            raise SidecarError("ERR_BAD_PROFILE", str(e)) from e
        except Exception as e:
            raise SidecarError("ERR_BAD_PROFILE", f"failed to switch profile: {e}") from e
        view = _curated_config_view(self._ctx.profile, self._ctx.config)
        return {
            "ok": True,
            "profile": view["profile"],
            "detectors_enabled": [d["id"] for d in view["detectors"] if d["enabled"]],
        }

    def _monitoring_summary(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        return self._history.stats(since=params.get("since"))

    def _monitoring_timeline(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        return {"items": self._history.timeline(limit=int(params.get("limit", 200)))}

    def _monitoring_detectors(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        return {"detectors": self._history.detector_stats(limit=int(params.get("limit", 500)))}

    def _monitoring_cascade(self, params: dict) -> dict:
        if self._history is None:
            raise SidecarError("ERR_DISABLED", "history is disabled in this config")
        return self._history.cascade_distribution(limit=int(params.get("limit", 1000)))

    def _run_eval(self, params: dict) -> dict:
        import time
        from cairn.evaluation.dataset import EvalSample
        from cairn.evaluation.runner import eval_report_to_dict, run_eval

        items = params.get("items") or []
        if not isinstance(items, list) or not items:
            raise SidecarError("ERR_BAD_PARAMS", "items required (non-empty list)")
        cap = int(params.get("max_samples", 200))
        cap = max(1, min(cap, 1000))
        if len(items) > cap:
            raise SidecarError(
                "ERR_TOO_LARGE",
                f"dataset has {len(items)} samples, cap is {cap} (max 1000)",
            )
        samples: list[EvalSample] = []
        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                raise SidecarError("ERR_BAD_PARAMS", f"item {idx} is not an object")
            for required in ("id", "text", "label"):
                if required not in it:
                    raise SidecarError(
                        "ERR_BAD_PARAMS", f"item {idx} missing field '{required}'"
                    )
            if not isinstance(it["text"], str) or not it["text"].strip():
                raise SidecarError("ERR_BAD_PARAMS", f"item {idx} has empty text")
            try:
                samples.append(EvalSample(
                    id=str(it["id"]),
                    text=str(it["text"]),
                    label=int(it["label"]),
                    tier=it.get("tier"),
                    source=it.get("source"),
                    attack_name=it.get("attack_name"),
                    attack_category=it.get("attack_category"),
                    source_id=it.get("source_id"),
                ))
            except (TypeError, ValueError) as e:
                raise SidecarError("ERR_BAD_PARAMS", f"item {idx} invalid: {e}") from e

        t0 = time.monotonic()
        report = run_eval(self._ctx.pipeline, samples)
        duration_ms = int((time.monotonic() - t0) * 1000)
        result = eval_report_to_dict(report)
        result["duration_ms"] = duration_ms
        return result

    def _record(
        self,
        report: dict,
        *,
        source: str,
        text: str | None,
        batch_id: str | None = None,
    ) -> None:
        if self._history is not None:
            try:
                self._history.record(
                    report, source=source, text=text, profile=self._profile, batch_id=batch_id
                )
            except Exception as e:
                log.warning("history write failed: %s", e)
        if self._drift is not None:
            try:
                self._drift.record(
                    p_llm=float(report.get("p_llm", 0.0)),
                    determination=str(report.get("determination", "")),
                    feature_vector=dict(report.get("feature_contributions") or {}),
                )
            except Exception as e:
                log.warning("drift record failed: %s", e)

    def _monitoring_drift(self) -> dict:
        if self._drift is None:
            raise SidecarError("ERR_DISABLED", "drift monitor is disabled in this config")
        alerts = self._drift.check_drift()
        summary = self._drift.get_summary()
        has_baseline = bool(self._drift._baseline_hist)
        baseline_features = list(self._drift._baseline_hist.keys())
        return {
            "has_baseline": has_baseline,
            "baseline_features": baseline_features,
            "n_observations": len(self._drift._observations),
            "alerts": alerts,
            "summary": summary,
        }

    def _monitoring_set_baseline(self, params: dict) -> dict:
        if self._drift is None:
            raise SidecarError("ERR_DISABLED", "drift monitor is disabled in this config")
        if self._history is None:
            raise SidecarError(
                "ERR_DISABLED",
                "history is required to build a baseline; enable gui.history",
            )
        limit = int(params.get("limit", 500))
        n = drift_baseline_from_history(self._drift, self._history, limit=limit)
        return {
            "ok": n > 0,
            "n_samples": n,
            "baseline_features": list(self._drift._baseline_hist.keys()),
        }

    def _recommend_profile(self, params: dict) -> dict:
        """Return an auto-routing profile recommendation for ``text``.

        Pure compute — does not switch the runtime profile. Callers wire
        this into ``RuntimeContext.switch_profile`` themselves when
        ``runtime.auto_route`` is on.
        """
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SidecarError("ERR_BAD_PARAMS", "text required (non-empty string)")
        task_metadata = params.get("task_metadata")
        if task_metadata is not None and not isinstance(task_metadata, dict):
            raise SidecarError(
                "ERR_BAD_PARAMS", "task_metadata must be an object or null"
            )
        router = TextRouter(self._ctx.config or {})
        decision = router.route(text)
        available = [p["name"] for p in list_profiles()]
        recommendation = recommend_profile(decision, task_metadata, available)
        return asdict(recommendation)


def _write(obj: dict, stream=None) -> None:
    stream = stream or sys.stdout
    stream.write(json.dumps(obj, separators=(",", ":")) + "\n")
    stream.flush()


def run(
    pipeline: CairnPipeline | None = None,
    feedback_path: Path | None = None,
    stdin=None,
    stdout=None,
    history: HistoryStore | None = None,
    profile: str | None = None,
    ctx: RuntimeContext | None = None,
    drift: DriftMonitor | None = None,
) -> int:
    """Drive the JSON-RPC loop. Returns the process exit code."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    sidecar = Sidecar(
        pipeline=pipeline,
        feedback_path=feedback_path,
        history=history,
        profile=profile,
        ctx=ctx,
        drift=drift,
    )
    _write({"event": "ready", "pid": os.getpid(), "protocol_version": PROTOCOL_VERSION}, stream=stdout)

    for raw in stdin:
        line = raw.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _write({"id": None, "error": {"code": "ERR_BAD_JSON", "message": str(e)}}, stream=stdout)
            continue

        if not isinstance(req, dict):
            _write(
                {"id": None, "error": {"code": "ERR_BAD_REQUEST", "message": "request must be a JSON object"}},
                stream=stdout,
            )
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}
        if not isinstance(method, str):
            _write(
                {"id": req_id, "error": {"code": "ERR_BAD_REQUEST", "message": "method must be a string"}},
                stream=stdout,
            )
            continue
        if not isinstance(params, dict):
            _write(
                {"id": req_id, "error": {"code": "ERR_BAD_REQUEST", "message": "params must be an object"}},
                stream=stdout,
            )
            continue

        try:
            result = sidecar.handle(method, params)
            _write({"id": req_id, "result": result}, stream=stdout)
        except SidecarError as e:
            _write(
                {"id": req_id, "error": {"code": e.code, "message": e.message}}, stream=stdout
            )
        except Exception as e:
            log.exception("unhandled error in method %s", method)
            _write(
                {"id": req_id, "error": {"code": "ERR_INTERNAL", "message": str(e)}}, stream=stdout
            )

        if not sidecar.running:
            return 0

    return 0


def _resolve_config_path(config: str | None, profile: str) -> Path:
    if config:
        return Path(config)
    return CONFIGS_DIR / f"{profile}.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cairn-sidecar", description=__doc__.splitlines()[0])
    parser.add_argument("--config", "-c", default=None, help="Path to a config YAML")
    parser.add_argument("--profile", "-p", default="default", help="Profile name in configs/ (when --config is unset)")
    parser.add_argument(
        "--feedback-path",
        default=str(DEFAULT_FEEDBACK_PATH),
        help="JSONL path for reviewer feedback (default: data/reviewer_feedback.jsonl)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="[%(name)s] %(levelname)s %(message)s",
    )

    config_path = _resolve_config_path(args.config, args.profile)
    config = load_config(config_path)
    pipeline = CairnPipeline(config)
    feedback_path = Path(args.feedback_path)
    history = history_from_config(config)
    drift = drift_from_config(config)
    ctx = RuntimeContext(pipeline, args.profile, config)

    return run(feedback_path=feedback_path, history=history, ctx=ctx, drift=drift)


if __name__ == "__main__":
    sys.exit(main())

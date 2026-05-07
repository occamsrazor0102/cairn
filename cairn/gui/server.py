"""Minimal, self-contained HTTP server that serves the embedded SPA and
proxies /analyze, /batch, /feedback, and /history/* to the local BEET
pipeline. No external API dependency; calls the pipeline directly.

Optional security layer:

* Bearer-token auth via :mod:`cairn.security.auth` — gated on
  ``auth.tokens`` config or ``CAIRN_API_TOKEN`` env var. Default off.
* Sliding-window rate limiting via :mod:`cairn.security.rate_limit`
  — gated on ``rate_limit.enabled``. Default off.

Both run as a single per-request guard at the top of ``do_GET`` /
``do_POST``; when both configs are disabled the guard short-circuits
with zero overhead.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import traceback
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from cairn.auto_routing import recommend_profile
from cairn.config import list_profiles
from cairn.history import HistoryStore, mint_submission_id
from cairn.monitoring.drift import DriftMonitor
from cairn.monitoring.meta_detector import MetaDetector
from cairn.pipeline import BatchCancelled, CairnPipeline
from cairn.report import build_json_report
from cairn.router import TextRouter
from cairn.runtime import RuntimeContext
from cairn.security.auth import (
    AuthConfig,
    auth_config_from_dict,
    auth_status_for_health,
    check_bearer,
)
from cairn.security.rate_limit import (
    RateLimitConfig,
    RateLimiter,
    make_rate_limiter,
    rate_limit_config_from_dict,
    rate_limit_status_for_health,
)
from cairn.sidecar import (
    DEFAULT_FEEDBACK_PATH,
    JobRegistry,
    SidecarError,
    _curated_config_view,
    drift_baseline_from_history,
    drift_from_config,
    record_feedback,
)


STATIC_DIR = Path(__file__).parent / "static"


def _history_from_config(config: dict | None) -> HistoryStore | None:
    gui = (config or {}).get("gui") or {}
    hist = gui.get("history") or {}
    if hist.get("enabled", True) is False:
        return None
    db_path = Path(hist.get("db_path", "data/cairn_history.sqlite3"))
    retain = bool(hist.get("retain_text", False))
    try:
        return HistoryStore(db_path, retain_text=retain)
    except Exception as e:
        sys.stderr.write(f"[gui] history store disabled: {e}\n")
        return None


def _make_handler(
    pipeline: CairnPipeline | None = None,
    meta: MetaDetector | None = None,
    feedback_path: Path | None = None,
    history: HistoryStore | None = None,
    profile: str | None = None,
    ctx: RuntimeContext | None = None,
    drift: DriftMonitor | None = None,
    auth_cfg: AuthConfig | None = None,
    rate_limit_cfg: RateLimitConfig | None = None,
    rate_limiter: RateLimiter | None = None,
):
    if ctx is None:
        if pipeline is None:
            raise ValueError("_make_handler requires either ctx or pipeline")
        ctx = RuntimeContext(pipeline, profile, {})
    meta = meta or MetaDetector()
    feedback_path = feedback_path or DEFAULT_FEEDBACK_PATH
    jobs = JobRegistry()  # in-memory async-batch job tracking; mirrors sidecar
    # Resolve auth + rate-limit lazily from the live config when callers
    # didn't pass explicit values. Default-off behaviour is preserved
    # because both `auth_config_from_dict` and `rate_limit_config_from_dict`
    # return disabled configs when the keys are absent.
    if auth_cfg is None:
        auth_cfg = auth_config_from_dict(ctx.config)
    if rate_limit_cfg is None:
        rate_limit_cfg = rate_limit_config_from_dict(ctx.config)
    if rate_limiter is None and rate_limit_cfg.enabled:
        rate_limiter = make_rate_limiter(rate_limit_cfg)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            sys.stdout.write("[gui] " + (format % args) + "\n")

        def _security_guard(self) -> bool:
            """Per-request bearer + rate-limit check.

            Returns ``True`` when the request is allowed to proceed and
            the handler can read on. Returns ``False`` when the guard
            already wrote a 401 / 429 response — the handler must stop.

            The /health endpoint is always allowed (matches the
            FastAPI middleware) so liveness probes never get a 401.
            """
            path = urlparse(self.path).path
            if path == "/health":
                return True

            if auth_cfg.enabled:
                provided = self.headers.get(auth_cfg.header_name)
                if not check_bearer(provided, auth_cfg):
                    self._send_json(
                        401,
                        {"error": "unauthorized", "code": "ERR_AUTH_REQUIRED"},
                    )
                    return False

            if rate_limiter is not None:
                key = self._rate_limit_key()
                if not rate_limiter.check(key):
                    self._send_json(
                        429,
                        {
                            "error": "rate limit exceeded",
                            "code": "ERR_RATE_LIMIT",
                            "window_seconds": rate_limit_cfg.window_seconds,
                            "max_per_window": rate_limit_cfg.max_per_window,
                        },
                    )
                    return False

            return True

        def _rate_limit_key(self) -> str:
            """Pick a bucket key honouring ``auth_cfg.header_name``."""
            if auth_cfg.enabled:
                provided = self.headers.get(auth_cfg.header_name)
                if provided:
                    digest = hashlib.sha256(provided.encode("utf-8")).hexdigest()
                    return f"token:{digest[:16]}"
            ip = self.client_address[0] if self.client_address else "unknown"
            return f"ip:{ip}"

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_attachment(self, content: str, mime: str, filename: str) -> None:
            body = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._security_guard():
                return
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/index.html"):
                self._serve_static("index.html")
            elif path == "/health":
                pipe = ctx.pipeline
                model_loaded = pipe._fusion._model is not None
                conformal_loaded = (
                    pipe._fusion._conformal is not None
                    and getattr(pipe._fusion._conformal, "calibrated", False)
                )
                if model_loaded and conformal_loaded:
                    cal_status = "calibrated"
                elif model_loaded:
                    cal_status = "fusion-only"
                else:
                    cal_status = "heuristic"
                self._send_json(200, {
                    "status": "ok",
                    "model_loaded": model_loaded,
                    "conformal_loaded": conformal_loaded,
                    "calibration_status": cal_status,
                    "history_enabled": history is not None,
                    "profile": ctx.profile,
                    "auth": auth_status_for_health(auth_cfg),
                    "rate_limit": rate_limit_status_for_health(rate_limit_cfg),
                })
            elif path == "/config/profiles":
                self._send_json(200, {
                    "profiles": list_profiles(),
                    "current": ctx.profile,
                })
            elif path == "/config/current":
                self._send_json(200, _curated_config_view(ctx.profile, ctx.config))
            elif path == "/history/export":
                if history is None:
                    self._send_json(503, {"error": "history disabled"})
                    return
                q = parse_qs(parsed.query)
                fmt = (q.get("format", ["json"])[0] or "json").lower()
                if fmt not in ("json", "csv"):
                    self._send_json(400, {"error": "format must be 'json' or 'csv'"})
                    return
                content, mime, filename = history.export(
                    fmt=fmt,
                    determination=q.get("determination") or None,
                    since=q.get("since", [None])[0],
                    until=q.get("until", [None])[0],
                    batch_id=q.get("batch_id", [None])[0],
                    search=q.get("search", [None])[0],
                )
                self._send_attachment(content, mime, filename)
            else:
                self.send_error(404)

        def do_POST(self):
            if not self._security_guard():
                return
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return

            if path == "/analyze":
                self._handle_analyze(body)
            elif path == "/batch":
                self._handle_batch(body)
            elif path == "/batch/async":
                self._handle_batch_async(body)
            elif path == "/jobs/get":
                self._handle_job_get(body)
            elif path == "/jobs/cancel":
                self._handle_job_cancel(body)
            elif path == "/feedback":
                self._handle_feedback(body)
            elif path == "/history/list":
                self._handle_history_list(body)
            elif path == "/history/get":
                self._handle_history_get(body)
            elif path == "/history/delete":
                self._handle_history_delete(body)
            elif path == "/config/switch":
                self._handle_switch_profile(body)
            elif path == "/monitoring/summary":
                self._handle_monitoring_summary(body)
            elif path == "/monitoring/timeline":
                self._handle_monitoring_timeline(body)
            elif path == "/monitoring/detectors":
                self._handle_monitoring_detectors(body)
            elif path == "/monitoring/cascade":
                self._handle_monitoring_cascade(body)
            elif path == "/evaluation/run":
                self._handle_run_eval(body)
            elif path == "/monitoring/drift":
                self._handle_monitoring_drift(body)
            elif path == "/monitoring/set-baseline":
                self._handle_monitoring_set_baseline(body)
            elif path == "/recommend_profile":
                self._handle_recommend_profile(body)
            else:
                self.send_error(404)

        def _handle_analyze(self, body: dict) -> None:
            text = body.get("text", "")
            if not isinstance(text, str) or not text.strip():
                self._send_json(400, {"error": "text required"})
                return
            sid = str(body.get("submission_id") or "").strip() or mint_submission_id()
            try:
                det = ctx.pipeline.analyze(text)
                report = build_json_report(det, submission_id=sid)
                self._record_history(report, source="analyze", text=text)
                self._send_json(200, report)
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_batch(self, body: dict) -> None:
            items = body.get("items") or []
            if not isinstance(items, list) or not items:
                self._send_json(400, {"error": "items required"})
                return
            if len(items) > 500:
                self._send_json(400, {"error": "batch exceeds 500-item cap"})
                return
            batch_id = str(body.get("batch_id") or mint_submission_id().replace("sub_", "batch_"))
            results, skipped = [], []
            for idx, it in enumerate(items):
                if not isinstance(it, dict):
                    skipped.append(idx)
                    continue
                sid = str(it.get("id") or f"{batch_id}_{idx:04d}")
                txt = it.get("text", "")
                if not isinstance(txt, str) or not txt.strip():
                    skipped.append(idx)
                    continue
                try:
                    det = ctx.pipeline.analyze(txt)
                    report = build_json_report(det, submission_id=sid)
                    self._record_history(report, source="batch", text=txt, batch_id=batch_id)
                    results.append(report)
                except Exception as e:
                    skipped.append(idx)
                    sys.stderr.write(f"[gui] batch item {idx} failed: {e}\n")
            self._send_json(200, {"results": results, "skipped": skipped, "batch_id": batch_id})

        def _handle_batch_async(self, body: dict) -> None:
            items = body.get("items") or []
            if not isinstance(items, list) or not items:
                self._send_json(400, {"error": "items required"})
                return
            if len(items) > 500:
                self._send_json(400, {"error": "batch exceeds 500-item cap"})
                return
            batch_id = str(
                body.get("batch_id")
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
                txt = it.get("text", "")
                if not isinstance(txt, str) or not txt.strip():
                    skipped.append(idx)
                    continue
                if sid in texts:
                    self._send_json(
                        400,
                        {"error": f"duplicate submission_id '{sid}' at index {idx}"},
                    )
                    return
                texts[sid] = txt
                order.append((idx, sid))
            if not texts:
                self._send_json(400, {"error": "no valid items in batch"})
                return

            job_id, cancel_event = jobs.create(kind="batch", total=len(texts))
            pipeline = ctx.pipeline
            record_history = self._record_history

            def _progress(done: int, total: int, current: str | None) -> None:
                jobs.update(job_id, done=done, current=current)

            def _cancelled() -> bool:
                return cancel_event.is_set()

            def worker() -> None:
                try:
                    dets = pipeline.analyze_batch(
                        texts, progress_cb=_progress, cancel_check=_cancelled
                    )
                    results = []
                    for _idx, sid in order:
                        det = dets.get(sid)
                        if det is None:
                            continue
                        report = build_json_report(det, submission_id=sid)
                        try:
                            record_history(
                                report, source="batch", text=texts[sid], batch_id=batch_id
                            )
                        except Exception as e:
                            sys.stderr.write(
                                f"[gui] history record failed for {sid}: {e}\n"
                            )
                        results.append(report)
                    jobs.finish(
                        job_id,
                        result={
                            "results": results,
                            "skipped": skipped,
                            "batch_id": batch_id,
                        },
                    )
                except BatchCancelled:
                    jobs.finish(job_id)
                except Exception as e:
                    sys.stderr.write(f"[gui] async batch {job_id} failed: {e}\n")
                    jobs.finish(job_id, error=str(e) or type(e).__name__)

            threading.Thread(
                target=worker, name=f"cairn-batch-{job_id[:8]}", daemon=True
            ).start()
            self._send_json(
                200,
                {
                    "job_id": job_id,
                    "kind": "batch",
                    "total": len(texts),
                    "batch_id": batch_id,
                    "skipped": skipped,
                },
            )

        def _handle_job_get(self, body: dict) -> None:
            job_id = str(body.get("job_id") or "")
            if not job_id:
                self._send_json(400, {"error": "job_id required"})
                return
            job = jobs.get(job_id)
            if job is None:
                self._send_json(404, {"error": f"job '{job_id}' not found"})
                return
            self._send_json(200, job.to_public())

        def _handle_job_cancel(self, body: dict) -> None:
            job_id = str(body.get("job_id") or "")
            if not job_id:
                self._send_json(400, {"error": "job_id required"})
                return
            if not jobs.cancel(job_id):
                self._send_json(404, {"error": f"job '{job_id}' not found"})
                return
            self._send_json(200, {"ok": True, "job_id": job_id})

        def _handle_feedback(self, body: dict) -> None:
            try:
                result = record_feedback(ctx.pipeline, meta, feedback_path, body)
                if history is not None:
                    sid = str(body.get("submission_id") or result.get("submission_id") or "").strip()
                    if sid:
                        try:
                            history.record_feedback(
                                sid,
                                int(body.get("confirmed_label", 0)),
                                body.get("reviewer_notes"),
                            )
                        except Exception as e:
                            sys.stderr.write(f"[gui] history feedback write failed: {e}\n")
                self._send_json(200, result)
            except SidecarError as e:
                self._send_json(400, {"error": e.message, "code": e.code})
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_history_list(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            try:
                res = history.list(
                    limit=int(body.get("limit", 25)),
                    offset=int(body.get("offset", 0)),
                    determination=body.get("determination") or None,
                    since=body.get("since"),
                    until=body.get("until"),
                    batch_id=body.get("batch_id"),
                    search=body.get("search"),
                )
                self._send_json(200, res)
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_history_get(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            sid = str(body.get("submission_id") or "").strip()
            if not sid:
                self._send_json(400, {"error": "submission_id required"})
                return
            got = history.get(sid)
            if got is None:
                self._send_json(404, {"error": "not found"})
                return
            self._send_json(200, got)

        def _handle_history_delete(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            sid = str(body.get("submission_id") or "").strip()
            if not sid:
                self._send_json(400, {"error": "submission_id required"})
                return
            ok = history.delete(sid)
            self._send_json(200, {"ok": ok})

        def _handle_switch_profile(self, body: dict) -> None:
            name = str(body.get("name") or "").strip()
            if not name:
                self._send_json(400, {"error": "name required"})
                return
            try:
                ctx.switch_profile(name)
            except FileNotFoundError as e:
                self._send_json(400, {"error": str(e), "code": "ERR_BAD_PROFILE"})
                return
            except Exception as e:
                traceback.print_exc()
                self._send_json(400, {"error": str(e), "code": "ERR_BAD_PROFILE"})
                return
            view = _curated_config_view(ctx.profile, ctx.config)
            self._send_json(200, {
                "ok": True,
                "profile": view["profile"],
                "detectors_enabled": [d["id"] for d in view["detectors"] if d["enabled"]],
            })

        def _handle_monitoring_summary(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            try:
                self._send_json(200, history.stats(since=body.get("since")))
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_monitoring_timeline(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            try:
                limit = int(body.get("limit", 200))
                self._send_json(200, {"items": history.timeline(limit=limit)})
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_monitoring_detectors(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            try:
                limit = int(body.get("limit", 500))
                self._send_json(200, {"detectors": history.detector_stats(limit=limit)})
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_monitoring_cascade(self, body: dict) -> None:
            if history is None:
                self._send_json(503, {"error": "history disabled"})
                return
            try:
                limit = int(body.get("limit", 1000))
                self._send_json(200, history.cascade_distribution(limit=limit))
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_run_eval(self, body: dict) -> None:
            import time
            from cairn.evaluation.dataset import EvalSample
            from cairn.evaluation.runner import eval_report_to_dict, run_eval

            items = body.get("items") or []
            if not isinstance(items, list) or not items:
                self._send_json(400, {"error": "items required"})
                return
            cap = max(1, min(int(body.get("max_samples", 200)), 1000))
            if len(items) > cap:
                self._send_json(400, {
                    "error": f"dataset has {len(items)} samples, cap is {cap} (max 1000)",
                    "code": "ERR_TOO_LARGE",
                })
                return
            samples = []
            for idx, it in enumerate(items):
                if not isinstance(it, dict):
                    self._send_json(400, {"error": f"item {idx} is not an object"})
                    return
                for required in ("id", "text", "label"):
                    if required not in it:
                        self._send_json(400, {"error": f"item {idx} missing '{required}'"})
                        return
                if not isinstance(it["text"], str) or not it["text"].strip():
                    self._send_json(400, {"error": f"item {idx} has empty text"})
                    return
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
                    self._send_json(400, {"error": f"item {idx}: {e}"})
                    return
            t0 = time.monotonic()
            try:
                report = run_eval(ctx.pipeline, samples)
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})
                return
            result = eval_report_to_dict(report)
            result["duration_ms"] = int((time.monotonic() - t0) * 1000)
            self._send_json(200, result)

        def _handle_monitoring_drift(self, body: dict) -> None:
            if drift is None:
                self._send_json(503, {"error": "drift monitor disabled"})
                return
            try:
                alerts = drift.check_drift()
                summary = drift.get_summary()
                self._send_json(200, {
                    "has_baseline": bool(drift._baseline_hist),
                    "baseline_features": list(drift._baseline_hist.keys()),
                    "n_observations": len(drift._observations),
                    "alerts": alerts,
                    "summary": summary,
                })
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_monitoring_set_baseline(self, body: dict) -> None:
            if drift is None:
                self._send_json(503, {"error": "drift monitor disabled"})
                return
            if history is None:
                self._send_json(503, {"error": "history required for baseline"})
                return
            limit = int(body.get("limit", 500))
            try:
                n = drift_baseline_from_history(drift, history, limit=limit)
                self._send_json(200, {
                    "ok": n > 0,
                    "n_samples": n,
                    "baseline_features": list(drift._baseline_hist.keys()),
                })
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _handle_recommend_profile(self, body: dict) -> None:
            text = body.get("text", "")
            if not isinstance(text, str) or not text.strip():
                self._send_json(400, {"error": "text required"})
                return
            task_metadata = body.get("task_metadata")
            if task_metadata is not None and not isinstance(task_metadata, dict):
                self._send_json(
                    400, {"error": "task_metadata must be an object or null"}
                )
                return
            try:
                router = TextRouter(ctx.config or {})
                decision = router.route(text)
                available = [p["name"] for p in list_profiles()]
                rec = recommend_profile(decision, task_metadata, available)
                self._send_json(200, asdict(rec))
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})

        def _record_history(
            self,
            report: dict,
            *,
            source: str,
            text: str | None,
            batch_id: str | None = None,
        ) -> None:
            if history is not None:
                try:
                    history.record(
                        report, source=source, text=text, profile=ctx.profile, batch_id=batch_id
                    )
                except Exception as e:
                    sys.stderr.write(f"[gui] history write failed: {e}\n")
            if drift is not None:
                try:
                    drift.record(
                        p_llm=float(report.get("p_llm", 0.0)),
                        determination=str(report.get("determination", "")),
                        feature_vector=dict(report.get("feature_contributions") or {}),
                    )
                except Exception as e:
                    sys.stderr.write(f"[gui] drift record failed: {e}\n")

        def _serve_static(self, name: str) -> None:
            path = STATIC_DIR / name
            if not path.exists():
                self.send_error(404)
                return
            data = path.read_bytes()
            ctype = "text/html" if name.endswith(".html") else "text/plain"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(
    pipeline: CairnPipeline,
    host: str = "127.0.0.1",
    port: int = 8877,
    open_browser: bool = True,
    feedback_path: Path | None = None,
    config: dict | None = None,
    profile: str | None = None,
) -> None:
    meta = MetaDetector()
    fp = feedback_path or DEFAULT_FEEDBACK_PATH
    history = _history_from_config(config)
    drift = drift_from_config(config)
    ctx = RuntimeContext(pipeline, profile, config or {})
    auth_cfg = auth_config_from_dict(config or {})
    rl_cfg = rate_limit_config_from_dict(config or {})
    rate_limiter = make_rate_limiter(rl_cfg) if rl_cfg.enabled else None
    handler = _make_handler(
        meta=meta,
        feedback_path=fp,
        history=history,
        ctx=ctx,
        drift=drift,
        auth_cfg=auth_cfg,
        rate_limit_cfg=rl_cfg,
        rate_limiter=rate_limiter,
    )
    httpd = HTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    hist_status = "enabled" if history is not None else "disabled"
    auth_status = "enabled" if auth_cfg.enabled else "disabled"
    rl_status = "enabled" if rl_cfg.enabled else "disabled"
    print(
        f"[gui] serving on {url}  "
        f"(history {hist_status}, auth {auth_status}, rate_limit {rl_status}, Ctrl-C to stop)"
    )
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[gui] stopping")
        httpd.server_close()

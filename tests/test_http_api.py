"""HTTP API tests: live socket against the stdlib GUI server handler."""
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

from cairn.config import load_config
from cairn.gui.server import _make_handler
from cairn.history import HistoryStore
from cairn.monitoring.drift import DriftMonitor
from cairn.monitoring.meta_detector import MetaDetector
from cairn.pipeline import CairnPipeline
from cairn.runtime import RuntimeContext
from cairn.security.auth import AuthConfig
from cairn.security.rate_limit import (
    RateLimitConfig,
    make_rate_limiter,
)


@pytest.fixture
def server(tmp_path):
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    cfg = load_config(config_path)
    ctx = RuntimeContext(CairnPipeline(cfg), "screening", cfg)
    meta = MetaDetector()
    feedback_path = tmp_path / "feedback.jsonl"
    history = HistoryStore(tmp_path / "history.sqlite3", retain_text=True)
    drift = DriftMonitor(tmp_path / "drift", cfg)
    handler = _make_handler(
        meta=meta, feedback_path=feedback_path,
        history=history, ctx=ctx, drift=drift,
    )
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", history
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _make_secure_server(tmp_path, *, auth_cfg=None, rate_limit_cfg=None, rate_limiter=None):
    """Spin up a live HTTP server with optional auth + rate-limit guards.

    Returns ``(url, httpd, thread, history)`` so callers can clean up.
    Used by the auth / rate-limit fixtures below so tests can exercise
    each path independently of the default-off ``server`` fixture.
    """
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    cfg = load_config(config_path)
    ctx = RuntimeContext(CairnPipeline(cfg), "screening", cfg)
    meta = MetaDetector()
    feedback_path = tmp_path / "feedback.jsonl"
    history = HistoryStore(tmp_path / "history.sqlite3", retain_text=True)
    drift = DriftMonitor(tmp_path / "drift", cfg)
    handler = _make_handler(
        meta=meta,
        feedback_path=feedback_path,
        history=history,
        ctx=ctx,
        drift=drift,
        auth_cfg=auth_cfg,
        rate_limit_cfg=rate_limit_cfg,
        rate_limiter=rate_limiter,
    )
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{httpd.server_address[1]}", httpd, thread, history


@pytest.fixture
def auth_server(tmp_path):
    """Server fixture with bearer-token auth enabled."""
    auth_cfg = AuthConfig(
        enabled=True,
        tokens=("test-token-12345-abcdef-67890",),
        header_name="Authorization",
    )
    url, httpd, thread, history = _make_secure_server(tmp_path, auth_cfg=auth_cfg)
    try:
        yield url, "test-token-12345-abcdef-67890", history
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@pytest.fixture
def rate_limit_server(tmp_path):
    """Server fixture with a 2-per-window rate limit, no auth."""
    rl_cfg = RateLimitConfig(
        enabled=True,
        window_seconds=60,
        max_per_window=2,
        store="memory",
        hash_keys=False,
    )
    rl = make_rate_limiter(rl_cfg)
    url, httpd, thread, history = _make_secure_server(
        tmp_path, rate_limit_cfg=rl_cfg, rate_limiter=rl
    )
    try:
        yield url, history
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _get(url, path):
    with urllib.request.urlopen(url + path, timeout=10) as resp:
        return resp.status, resp.read(), dict(resp.headers)


def _get_json(url, path):
    status, data, _ = _get(url, path)
    return status, json.loads(data)


def _post(url, path, body):
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_reports_history_enabled(server):
    url, _ = server
    status, body = _get_json(url, "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["history_enabled"] is True
    assert body["profile"] == "screening"
    # The UI relies on this field to show the right calibration banner.
    assert body["calibration_status"] in (
        "heuristic", "fusion-only", "calibrated", "calibrated-router-conditional",
    )


def test_index_served(server):
    url, _ = server
    status, data, _ = _get(url, "/")
    assert status == 200
    text = data.decode("utf-8").lower()
    assert "<html" in text and "cairn" in text


def test_analyze_writes_history(server):
    url, hist = server
    status, body = _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    assert status == 200
    assert "determination" in body
    listed = hist.list()
    assert listed["total"] == 1
    assert listed["items"][0]["submission_id"] == body["submission_id"]


def test_analyze_rejects_empty(server):
    url, _ = server
    status, _ = _post(url, "/analyze", {"text": ""})
    assert status == 400


def test_batch_writes_history_with_batch_id(server):
    url, hist = server
    status, body = _post(url, "/batch", {"items": [
        {"id": "a", "text": "first submission text"},
        {"id": "b", "text": "second submission text"},
    ]})
    assert status == 200
    assert len(body["results"]) == 2
    assert body["batch_id"]
    listed = hist.list(batch_id=body["batch_id"])
    assert listed["total"] == 2


def test_batch_cap_enforced(server):
    url, _ = server
    status, body = _post(url, "/batch", {"items": [{"id": f"i{i}", "text": "x"} for i in range(501)]})
    assert status == 400
    assert "cap" in body["error"].lower()


def test_history_list_filter_by_determination(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    _post(url, "/analyze", {"text": "quick human scratch note about nothing"})
    status, all_items = _post(url, "/history/list", {})
    assert status == 200
    assert all_items["total"] == 2
    dets = {i["determination"] for i in all_items["items"]}
    pick = next(iter(dets))
    status, filtered = _post(url, "/history/list", {"determination": [pick]})
    assert status == 200
    assert all(i["determination"] == pick for i in filtered["items"])


def test_history_get_returns_full_report(server):
    url, _ = server
    _, analyzed = _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    sid = analyzed["submission_id"]
    status, body = _post(url, "/history/get", {"submission_id": sid})
    assert status == 200
    assert body["report"]["determination"] == analyzed["determination"]
    assert body["text"] is not None


def test_history_delete(server):
    url, hist = server
    _, analyzed = _post(url, "/analyze", {"text": "quick note"})
    sid = analyzed["submission_id"]
    status, body = _post(url, "/history/delete", {"submission_id": sid})
    assert status == 200 and body["ok"] is True
    assert hist.get(sid) is None


def test_history_export_json_download(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, data, headers = _get(url, "/history/export?format=json")
    assert status == 200
    assert "application/json" in headers["Content-Type"]
    assert "attachment" in headers["Content-Disposition"]
    parsed = json.loads(data)
    assert isinstance(parsed, list) and len(parsed) == 1


def test_history_export_csv_download(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, data, headers = _get(url, "/history/export?format=csv")
    assert status == 200
    assert "text/csv" in headers["Content-Type"]
    text = data.decode("utf-8")
    assert text.splitlines()[0].startswith("submission_id,")


def test_config_profiles_lists_repo_profiles(server):
    url, _ = server
    status, body = _get_json(url, "/config/profiles")
    assert status == 200
    names = {p["name"] for p in body["profiles"]}
    assert {"default", "screening", "strict"} <= names
    assert body["current"] == "screening"


def test_config_current_curated_view(server):
    url, _ = server
    status, body = _get_json(url, "/config/current")
    assert status == 200
    assert body["profile"] == "screening"
    assert "red" in body["thresholds"]
    assert isinstance(body["detectors"], list) and body["detectors"]
    # Availability introspection: each detector reports available/reason/requires
    for d in body["detectors"]:
        assert "available" in d and isinstance(d["available"], bool)
        assert "reason" in d
        assert "requires" in d and isinstance(d["requires"], list)


def test_config_switch_updates_active_profile(server):
    url, _ = server
    status, body = _post(url, "/config/switch", {"name": "strict"})
    assert status == 200 and body["ok"] is True and body["profile"] == "strict"
    # health now reports strict
    _, health = _get_json(url, "/health")
    assert health["profile"] == "strict"


def test_config_switch_bad_name_returns_400(server):
    url, _ = server
    status, body = _post(url, "/config/switch", {"name": "no-such-xyz"})
    assert status == 400
    assert body.get("code") == "ERR_BAD_PROFILE"


def test_monitoring_summary_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, body = _post(url, "/monitoring/summary", {})
    assert status == 200
    assert body["total"] == 1
    assert "mean_p_llm" in body


def test_monitoring_timeline_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, body = _post(url, "/monitoring/timeline", {"limit": 50})
    assert status == 200
    assert len(body["items"]) == 1
    assert "p_llm" in body["items"][0]


def test_monitoring_detectors_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, body = _post(url, "/monitoring/detectors", {"limit": 100})
    assert status == 200
    assert isinstance(body["detectors"], list)


def test_monitoring_cascade_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, body = _post(url, "/monitoring/cascade", {})
    assert status == 200
    assert body["n_samples"] >= 1
    assert "phase_counts" in body
    assert "p_llm_histogram" in body


def test_evaluation_run_happy_path(server):
    url, _ = server
    items = [
        {"id": "a", "text": "Certainly! Here is a comprehensive overview.", "label": 1, "tier": "A0"},
        {"id": "b", "text": "quick human scratch note about nothing", "label": 0, "tier": "A0"},
    ]
    status, body = _post(url, "/evaluation/run", {"items": items})
    assert status == 200
    assert body["n_samples"] == 2
    assert "metrics" in body
    assert "confusion" in body
    assert "duration_ms" in body


def test_evaluation_run_cap_enforced(server):
    url, _ = server
    items = [{"id": f"x{i}", "text": "t" * 5, "label": 0} for i in range(5)]
    status, body = _post(url, "/evaluation/run", {"items": items, "max_samples": 3})
    assert status == 400
    assert body.get("code") == "ERR_TOO_LARGE"


def test_monitoring_drift_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    status, body = _post(url, "/monitoring/drift", {})
    assert status == 200
    assert body["n_observations"] >= 1
    assert isinstance(body["alerts"], list)


def test_monitoring_set_baseline_endpoint(server):
    url, _ = server
    _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    _post(url, "/analyze", {"text": "another sample text for baseline seeding"})
    status, body = _post(url, "/monitoring/set-baseline", {"limit": 10})
    assert status == 200
    assert body["ok"] is True and body["n_samples"] == 2
    assert body["baseline_features"]


def test_feedback_records_in_history(server):
    url, hist = server
    _, analyzed = _post(url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."})
    sid = analyzed["submission_id"]
    status, _ = _post(url, "/feedback", {
        "text": "Certainly! Here is a comprehensive overview.",
        "confirmed_label": 1,
        "submission_id": sid,
        "reviewer_notes": "obvious",
    })
    assert status == 200
    got = hist.get(sid)
    assert len(got["feedback"]) == 1
    assert got["feedback"][0]["confirmed_label"] == 1


# --- async batch over HTTP -------------------------------------------------

def test_batch_async_completes(server):
    """Async kick-off + polling produces results equivalent to the sync /batch path."""
    import time

    url, hist = server
    status, started = _post(url, "/batch/async", {"items": [
        {"id": "a", "text": "first submission text"},
        {"id": "b", "text": "second submission text"},
    ]})
    assert status == 200, f"unexpected start: {started}"
    assert started["job_id"]
    assert started["total"] == 2

    deadline = time.time() + 10.0
    job = None
    while time.time() < deadline:
        s, job = _post(url, "/jobs/get", {"job_id": started["job_id"]})
        assert s == 200
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job is not None and job["status"] == "completed", f"unexpected: {job}"
    assert len(job["result"]["results"]) == 2
    assert hist.list(batch_id=job["result"]["batch_id"])["total"] == 2


def test_jobs_get_unknown_returns_404(server):
    url, _ = server
    status, body = _post(url, "/jobs/get", {"job_id": "no-such-job"})
    assert status == 404
    assert "not found" in body["error"].lower()


def test_jobs_cancel_unknown_returns_404(server):
    url, _ = server
    status, _ = _post(url, "/jobs/cancel", {"job_id": "no-such-job"})
    assert status == 404


# ---------------------------------------------------------------------------
# _redact() coverage - provider-prefix keys + structural-secret-suffix rule.
# ---------------------------------------------------------------------------


def test_redact_provider_prefix_keys():
    from cairn.api import _redact
    cfg = {
        "anthropic_api_key": "sk-ant-aaaaaaaaaaaaaaaaaaaa",
        "openai_endpoint_secret": "abc123",
        "gemini_token": "xyz789",
    }
    out = _redact(cfg)
    for k in cfg:
        assert out[k] != cfg[k], f"{k} not redacted"


def test_redact_structural_long_value_under_key_suffix():
    from cairn.api import _redact
    long_val = "x" * 64
    cfg = {"my_custom_endpoint_key": long_val}
    out = _redact(cfg)
    assert out["my_custom_endpoint_key"] != long_val


def test_redact_does_not_redact_short_value_under_key_suffix():
    """Structural rule requires len(value) > 32 to avoid false positives
    on short identifiers like `prefix_key: "k1"`."""
    from cairn.api import _redact
    cfg = {"my_custom_endpoint_key": "short"}
    out = _redact(cfg)
    assert out["my_custom_endpoint_key"] == "short"


def test_redact_does_not_redact_unrelated_long_strings():
    """Long string under a non-sensitive key suffix passes through."""
    from cairn.api import _redact
    long_description = "This is a long description with no sensitive content " * 5
    cfg = {"description": long_description, "notes": long_description}
    out = _redact(cfg)
    assert out["description"] == long_description
    assert out["notes"] == long_description


def test_redact_existing_sensitive_tokens_still_redacted():
    """Regression on existing _SENSITIVE_TOKENS coverage."""
    from cairn.api import _redact
    cfg = {
        "secret": "topsecret",
        "password": "hunter2",
        "auth_header": "Bearer xyz",
        "api_key": "sk-test",
    }
    out = _redact(cfg)
    for k in cfg:
        assert out[k] != cfg[k], f"{k} regression"


def test_redact_recursive_walk_preserved():
    """The recursive walk over nested dicts/lists must still work."""
    from cairn.api import _redact
    cfg = {
        "providers": {
            "anthropic_api_key": "sk-ant-xxxxxxxxxxxxxxxxxxxx",
            "models": ["claude-sonnet", "claude-opus"],
        }
    }
    out = _redact(cfg)
    assert out["providers"]["anthropic_api_key"] != "sk-ant-xxxxxxxxxxxxxxxxxxxx"
    assert out["providers"]["models"] == ["claude-sonnet", "claude-opus"]  # untouched


# ---------------------------------------------------------------------------
# Bearer-token auth + rate-limit integration tests.
# These use the dedicated `auth_server` / `rate_limit_server` fixtures so
# the existing tests on the open `server` fixture continue to pass.
# ---------------------------------------------------------------------------


def _post_with_header(url, path, body, header):
    """POST helper that lets us add an Authorization header."""
    headers = {"Content-Type": "application/json"}
    if header is not None:
        headers["Authorization"] = header
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get_with_header(url, path, header):
    """GET helper with an optional Authorization header."""
    headers = {}
    if header is not None:
        headers["Authorization"] = header
    req = urllib.request.Request(url + path, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_default_off_reports_auth_disabled(server):
    """The default-off fixture must honestly report `auth: disabled`."""
    url, _ = server
    status, body = _get_json(url, "/health")
    assert status == 200
    assert body["auth"] == "disabled"
    assert body["rate_limit"] == "disabled"


def test_auth_rejects_missing_header(auth_server):
    """No Authorization header on a guarded endpoint must 401."""
    url, _token, _ = auth_server
    status, body = _post_with_header(
        url, "/analyze", {"text": "Certainly! Here is a comprehensive overview."}, None
    )
    assert status == 401
    assert body.get("code") == "ERR_AUTH_REQUIRED"


def test_auth_rejects_wrong_token(auth_server):
    """A bearer header with a non-matching token must 401."""
    url, _token, _ = auth_server
    status, body = _post_with_header(
        url, "/analyze", {"text": "anything goes here"}, "Bearer wrong-token"
    )
    assert status == 401
    assert body.get("code") == "ERR_AUTH_REQUIRED"


def test_auth_accepts_correct_token(auth_server):
    """A bearer header carrying the configured token must reach the handler."""
    url, token, _ = auth_server
    status, body = _post_with_header(
        url, "/analyze",
        {"text": "Certainly! Here is a comprehensive overview."},
        f"Bearer {token}",
    )
    assert status == 200
    assert "determination" in body


def test_auth_health_open_without_token(auth_server):
    """/health must remain unauthenticated so probes keep working."""
    url, _token, _ = auth_server
    status, body = _get_with_header(url, "/health", None)
    assert status == 200
    assert body["auth"] == "bearer-token"


def test_auth_health_reports_bearer_token(auth_server):
    """When auth is on, /health reports `auth: bearer-token`."""
    url, _token, _ = auth_server
    status, body = _get_with_header(url, "/health", None)
    assert status == 200
    assert body["auth"] == "bearer-token"


def test_rate_limit_returns_429_after_threshold(rate_limit_server):
    """Hammer 5 requests; first 2 succeed, last 3 must 429."""
    url, _ = rate_limit_server
    text_body = {"text": "Certainly! Here is a comprehensive overview."}
    statuses = []
    for _ in range(5):
        status, _body = _post_with_header(url, "/analyze", text_body, None)
        statuses.append(status)
    # First 2 under the cap; the next 3 must each return 429.
    assert statuses[:2] == [200, 200], f"unexpected pre-cap statuses: {statuses}"
    assert statuses[2:] == [429, 429, 429], f"unexpected post-cap statuses: {statuses}"


def test_rate_limit_health_reports_window(rate_limit_server):
    """/health must surface the live rate-limit window honestly."""
    url, _ = rate_limit_server
    status, body = _get_with_header(url, "/health", None)
    assert status == 200
    assert body["rate_limit"]["max_per_window"] == 2
    assert body["rate_limit"]["window_seconds"] == 60
    assert body["rate_limit"]["store"] == "memory"


# --- recommend_profile (B3) -------------------------------------------------

def test_recommend_profile_round_trip(server):
    """POST /recommend_profile returns the dict-form ProfileRecommendation."""
    url, _ = server
    long_prose = (
        "The afternoon sun filtered through the leaded glass windows, "
        "casting fractured rainbows across the worn parquet floor. "
        "She paused at the threshold, listening to the muted ticking of "
        "the grandfather clock in the foyer. Memories crowded in: the "
        "smell of beeswax polish, the sound of her grandmother humming "
        "in the kitchen, the cold brass of the doorknob under her palm. "
        "Years had passed since she last stood here, and the house felt "
        "both intimately familiar and strangely diminished."
    )
    status, body = _post(url, "/recommend_profile", {"text": long_prose})
    assert status == 200
    assert {"profile", "confidence", "reason", "caveats"} <= set(body.keys())
    assert isinstance(body["profile"], str)
    assert 0.0 <= float(body["confidence"]) <= 1.0
    assert isinstance(body["caveats"], list)


def test_recommend_profile_rejects_empty_text(server):
    """Empty text returns 400."""
    url, _ = server
    status, body = _post(url, "/recommend_profile", {"text": "   "})
    assert status == 400
    assert "error" in body


def test_recommend_profile_passes_language_metadata(server):
    """task_metadata.language='es' surfaces a non-english caveat."""
    url, _ = server
    text = (
        "Esta es una prueba de un texto en espanol con suficiente "
        "longitud para evitar el caveat de short_text."
    )
    status, body = _post(
        url,
        "/recommend_profile",
        {"text": text, "task_metadata": {"language": "es"}},
    )
    assert status == 200
    assert "non-english" in body["caveats"]


def test_recommend_profile_rejects_non_dict_metadata(server):
    """Non-object task_metadata returns 400."""
    url, _ = server
    status, body = _post(
        url,
        "/recommend_profile",
        {"text": "some sample text", "task_metadata": "not-a-dict"},
    )
    assert status == 400
    assert "error" in body

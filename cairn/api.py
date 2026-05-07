"""FastAPI REST server for BEET.

Endpoints:
    POST /analyze     Single-text analysis
    POST /batch       Batch analysis with optional cross-similarity
    GET  /health      Liveness + pipeline model info
    GET  /config      Current config (with secrets redacted)

The pipeline is a singleton initialised on startup. FastAPI is an optional
dependency; import errors surface at server-construction time so unit tests
that don't hit the API do not pay the cost.

Optional security middleware:
    * Bearer-token auth via :mod:`cairn.security.auth` — enabled when
      ``CAIRN_API_TOKEN`` is set or ``auth.tokens`` is non-empty in
      config.
    * Sliding-window rate limiting via :mod:`cairn.security.rate_limit`
      — enabled when ``rate_limit.enabled: true`` in config.

Both default to disabled so the existing loopback-friendly local
experience does not regress.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("cairn.api")

from cairn.pipeline import CairnPipeline
from cairn.report import build_json_report
from cairn.security.auth import (
    AuthConfig,
    auth_config_from_dict,
    auth_status_for_health,
    check_bearer,
)
from cairn.security.rate_limit import (
    RateLimitConfig,
    make_rate_limiter,
    rate_limit_config_from_dict,
    rate_limit_status_for_health,
)

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    _HAS_FASTAPI = True

    class AnalyzeRequest(BaseModel):
        text: str = Field(..., min_length=1)
        task_metadata: dict | None = None
        submission_id: str | None = None

    class BatchItem(BaseModel):
        id: str
        text: str
        task_metadata: dict | None = None

    class BatchRequest(BaseModel):
        items: list[BatchItem]
        cross_similarity: bool = False

except ImportError:
    _HAS_FASTAPI = False


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


_SENSITIVE_TOKENS = (
    "secret", "api_key", "apikey", "token", "password", "passwd",
    "credential", "auth",
)

_PROVIDER_PREFIXES = (
    "anthropic_", "openai_", "google_", "gemini_", "mistral_",
    "deepseek_", "moonshot_", "azure_", "aws_", "cohere_",
    "huggingface_", "replicate_",
)

_STRUCTURAL_SECRET_SUFFIXES = ("_key", "_token", "_secret", "_credential")


def _redact(config: dict) -> dict:
    """Strip API keys, credentials, and auth tokens from the config dict.

    Walks the config recursively (not just a fixed set of sections) and
    replaces any value whose key contains a sensitive token with the
    literal string "[redacted]".

    A key is considered sensitive when any of:
      * its lowered form contains an entry from ``_SENSITIVE_TOKENS``
      * its lowered form starts with any provider prefix in
        ``_PROVIDER_PREFIXES`` (catches e.g. ``anthropic_endpoint_token``
        even when the trailing token alone might be missed)
      * its lowered form ends with a structural suffix from
        ``_STRUCTURAL_SECRET_SUFFIXES`` AND the value is a string longer
        than 32 chars (catches secret-shaped values under custom keys
        without false-positiving on short identifiers).
    """
    import copy

    def _is_sensitive_key(key: str, value) -> bool:
        lk = str(key).lower()
        # Whole-key match for the bare identifier "key" — a top-level
        # `key: "sk-..."` config entry would not be caught by substring
        # matching ("key" is intentionally NOT in _SENSITIVE_TOKENS
        # because it would over-match common words like "monkey" or
        # "keyword"), so we special-case the exact equality.
        if lk == "key":
            return True
        if any(t in lk for t in _SENSITIVE_TOKENS):
            return True
        if any(lk.startswith(p) for p in _PROVIDER_PREFIXES):
            return True
        if any(lk.endswith(s) for s in _STRUCTURAL_SECRET_SUFFIXES):
            if isinstance(value, str) and len(value) > 32:
                return True
        return False

    def _walk(node):
        if isinstance(node, dict):
            return {
                k: ("[redacted]" if _is_sensitive_key(k, v) else _walk(v))
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [_walk(x) for x in node]
        return node

    return _walk(copy.deepcopy(config))


def _rate_limit_key(request: "Request", auth_cfg: AuthConfig) -> str:
    """Pick a rate-limit bucket key for the request.

    When the caller authenticated with a bearer token, key by an SHA-256
    prefix of the token so identical-token replays bucket together
    without storing the token plaintext. Otherwise key by client IP.
    """
    auth_header = request.headers.get(auth_cfg.header_name) if auth_cfg.enabled else None
    if auth_header:
        # SHA-256 prefix; sufficient to bucket repeat callers without
        # writing the bearer plaintext to disk if the sqlite store is on.
        digest = hashlib.sha256(auth_header.encode("utf-8")).hexdigest()
        return f"token:{digest[:16]}"
    client = request.client
    ip = client.host if client else "unknown"
    return f"ip:{ip}"


def create_app(config_path: Path | str | None = None) -> Any:
    """Construct a FastAPI app bound to a pipeline instance.

    Called lazily so the import-time cost of FastAPI/Pydantic is only paid
    when the REST server is actually being started.
    """
    if not _HAS_FASTAPI:
        raise ImportError(
            "FastAPI and Pydantic required for the REST API. "
            "Install with: pip install 'cairn[api]'"
        )

    cfg_path = Path(config_path or os.environ.get("CAIRN_CONFIG") or os.environ.get("BEET_CONFIG") or DEFAULT_CONFIG_PATH)
    pipeline = CairnPipeline.from_config_file(cfg_path)

    auth_cfg = auth_config_from_dict(pipeline._config)
    rl_cfg = rate_limit_config_from_dict(pipeline._config)
    rate_limiter = make_rate_limiter(rl_cfg) if rl_cfg.enabled else None

    app = FastAPI(title="CAIRN API", version="2.0.0")

    if auth_cfg.enabled:
        @app.middleware("http")
        async def bearer_auth_middleware(request: "Request", call_next):
            # /health stays open so liveness probes work without a token.
            if request.url.path == "/health":
                return await call_next(request)
            header = request.headers.get(auth_cfg.header_name)
            if not check_bearer(header, auth_cfg):
                # No token material is logged; only the path + verdict.
                logger.info("auth: rejected %s %s", request.method, request.url.path)
                return JSONResponse(
                    status_code=401,
                    content={"error": "unauthorized", "code": "ERR_AUTH_REQUIRED"},
                )
            return await call_next(request)

    if rate_limiter is not None:
        @app.middleware("http")
        async def rate_limit_middleware(request: "Request", call_next):
            if request.url.path == "/health":
                return await call_next(request)
            key = _rate_limit_key(request, auth_cfg)
            if not rate_limiter.check(key):
                logger.info(
                    "rate_limit: 429 %s %s", request.method, request.url.path
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate limit exceeded",
                        "code": "ERR_RATE_LIMIT",
                        "window_seconds": rl_cfg.window_seconds,
                        "max_per_window": rl_cfg.max_per_window,
                    },
                )
            return await call_next(request)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": "2.0.0",
            "model_loaded": pipeline._fusion._model is not None,
            "conformal_loaded": pipeline._fusion._conformal is not None,
            "detectors": list(pipeline._detectors.keys()),
            "auth": auth_status_for_health(auth_cfg),
            "rate_limit": rate_limit_status_for_health(rl_cfg),
        }

    @app.get("/config")
    def get_config() -> dict:
        return _redact(pipeline._config)

    @app.post("/analyze")
    def analyze(req: AnalyzeRequest) -> dict:
        try:
            det = pipeline.analyze(req.text, task_metadata=req.task_metadata)
        except Exception:
            logger.exception("analyze failed for submission %s", req.submission_id)
            raise HTTPException(status_code=500, detail="Internal analysis error")
        return build_json_report(det, submission_id=req.submission_id or "")

    @app.post("/batch")
    def batch(req: BatchRequest) -> dict:
        if not req.items:
            raise HTTPException(status_code=400, detail="items list is empty")
        if req.cross_similarity:
            texts = {it.id: it.text for it in req.items}
            batch_metadata = next((it.task_metadata for it in req.items if it.task_metadata), None)
            dets = pipeline.analyze_batch(texts, task_metadata=batch_metadata)
            return {
                "results": [
                    build_json_report(det, submission_id=sid)
                    for sid, det in dets.items()
                ],
            }
        out = []
        for it in req.items:
            det = pipeline.analyze(it.text, task_metadata=it.task_metadata)
            out.append(build_json_report(det, submission_id=it.id))
        return {"results": out}

    return app

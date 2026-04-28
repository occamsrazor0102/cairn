"""FastAPI REST server for BEET.

Endpoints:
    POST /analyze     Single-text analysis
    POST /batch       Batch analysis with optional cross-similarity
    GET  /health      Liveness + pipeline model info
    GET  /config      Current config (with secrets redacted)

The pipeline is a singleton initialised on startup. FastAPI is an optional
dependency; import errors surface at server-construction time so unit tests
that don't hit the API do not pay the cost.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("cairn.api")

from cairn.pipeline import CairnPipeline
from cairn.report import build_json_report

try:
    from fastapi import FastAPI, HTTPException
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


def _redact(config: dict) -> dict:
    """Strip API keys, credentials, and auth tokens from the config dict.

    Walks the config recursively (not just a fixed set of sections) and
    replaces any value whose key contains a sensitive token with the
    literal string "[redacted]".
    """
    import copy

    def _walk(node):
        if isinstance(node, dict):
            return {
                k: ("[redacted]" if any(t in str(k).lower() for t in _SENSITIVE_TOKENS)
                    else _walk(v))
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [_walk(x) for x in node]
        return node

    return _walk(copy.deepcopy(config))


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

    app = FastAPI(title="CAIRN API", version="2.0.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": "2.0.0",
            "model_loaded": pipeline._fusion._model is not None,
            "conformal_loaded": pipeline._fusion._conformal is not None,
            "detectors": list(pipeline._detectors.keys()),
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

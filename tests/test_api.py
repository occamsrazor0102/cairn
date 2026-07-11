"""FastAPI REST endpoints tests.

Skips gracefully when FastAPI isn't installed; doesn't require uvicorn at
test time since TestClient drives the app directly.
"""
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not installed")
from fastapi.testclient import TestClient  # noqa: E402

from cairn.api import create_app  # noqa: E402
from cairn.config import load_config  # noqa: E402
from cairn.pipeline import CairnPipeline  # noqa: E402


@pytest.fixture
def client():
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    app = create_app(config_path)
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "detectors" in data
    assert data["calibration_status"] in (
        "heuristic", "fusion-only", "calibrated", "calibrated-router-conditional",
        "degraded",
    )


def test_health_uses_pipeline_degraded_state_and_sanitizes_load_error(
    tmp_path,
    monkeypatch,
):
    secret_path = tmp_path / "private" / "bad-calibration.json"
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    pipeline = CairnPipeline(load_config(config_path))
    pipeline._detector_calibration_state = {
        "state": "load-failed",
        "path": str(secret_path),
        "error": f"invalid JSON at {secret_path}",
    }
    pipeline._calibration_status = "degraded"
    monkeypatch.setattr(
        CairnPipeline,
        "from_config_file",
        lambda _path: pipeline,
    )

    data = TestClient(create_app(config_path)).get("/health").json()

    assert data["calibration_status"] == "degraded"
    assert data["detector_calibration"] == {
        "state": "load-failed",
        "error": "artifact-load-failed",
    }
    assert str(secret_path) not in str(data)


def test_health_sanitizes_loaded_detector_identity_incompatibility(monkeypatch):
    config_path = Path(__file__).parent.parent / "configs" / "screening.yaml"
    pipeline = CairnPipeline(load_config(config_path))
    pipeline._detector_calibration_state = {
        "state": "loaded-incompatible",
        "path": "private/detector-calibration.json",
        "error": "token_cohesiveness expected secret-model-signature",
    }
    pipeline._calibration_status = "degraded"
    monkeypatch.setattr(
        CairnPipeline,
        "from_config_file",
        lambda _path: pipeline,
    )

    data = TestClient(create_app(config_path)).get("/health").json()

    assert data["calibration_status"] == "degraded"
    assert data["detector_calibration"] == {
        "state": "loaded-incompatible",
        "error": "artifact-incompatible",
    }
    assert "secret-model-signature" not in str(data)


def test_config_is_redacted(client):
    r = client.get("/config")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_analyze_returns_determination(client):
    r = client.post("/analyze", json={"text": "Certainly! Here is a comprehensive overview."})
    assert r.status_code == 200
    data = r.json()
    assert "determination" in data
    assert "p_llm" in data
    assert 0.0 <= data["p_llm"] <= 1.0


def test_analyze_rejects_empty(client):
    r = client.post("/analyze", json={"text": ""})
    assert r.status_code == 422


def test_batch_returns_per_item_results(client):
    r = client.post("/batch", json={
        "items": [
            {"id": "a", "text": "first submission text here for analysis."},
            {"id": "b", "text": "second submission text, different content."},
        ],
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 2
    ids = {x["submission_id"] for x in data["results"]}
    assert ids == {"a", "b"}


def test_batch_rejects_empty(client):
    r = client.post("/batch", json={"items": []})
    assert r.status_code == 400

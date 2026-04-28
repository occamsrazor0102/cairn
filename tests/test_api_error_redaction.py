import pytest

fastapi = pytest.importorskip("fastapi", reason="FastAPI not installed")
from fastapi.testclient import TestClient  # noqa: E402
from pathlib import Path  # noqa: E402
from beet.api import create_app  # noqa: E402
from beet.pipeline import BeetPipeline  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    def _boom(self, text, **kwargs):
        raise RuntimeError("secret token=sk-proj-abcdef1234567890 in traceback")

    monkeypatch.setattr(BeetPipeline, "analyze", _boom)
    app = create_app(Path(__file__).parent.parent / "configs" / "screening.yaml")
    return TestClient(app, raise_server_exceptions=False)


def test_analyze_500_does_not_leak_exception_text(client):
    r = client.post("/analyze", json={"text": "some text"})
    assert r.status_code == 500
    body = r.json()
    assert "sk-proj-abcdef1234567890" not in str(body), (
        "Raw exception text must not appear in the HTTP response"
    )


def test_analyze_500_returns_opaque_detail(client):
    r = client.post("/analyze", json={"text": "some text"})
    assert r.status_code == 500
    detail = r.json().get("detail", "")
    assert len(detail) < 80, "Error detail should be a short opaque message"
    assert "secret" not in detail.lower()

import pytest
from pathlib import Path
from cairn.pipeline import CairnPipeline


@pytest.fixture
def pipeline():
    return CairnPipeline.from_config_file(Path(__file__).parent.parent / "configs" / "screening.yaml")


def test_task_metadata_reaches_detector_config(pipeline, monkeypatch):
    captured = {}
    det = pipeline._detectors.get("preamble")
    if det is None:
        pytest.skip("preamble not registered")

    original = det.analyze

    def spy(text, config):
        captured["config"] = dict(config)
        return original(text, config)

    monkeypatch.setattr(det, "analyze", spy)
    pipeline.analyze("word " * 60, task_metadata={"task_description": "summarise a chart"})

    assert "task_description" in captured.get("config", {}), (
        "task_metadata.task_description was not threaded into detector config"
    )


def test_yaml_config_wins_over_task_metadata(pipeline, monkeypatch):
    captured = {}
    det = pipeline._detectors.get("preamble")
    if det is None:
        pytest.skip("preamble not registered")

    original = det.analyze

    def spy(text, config):
        captured["config"] = dict(config)
        return original(text, config)

    monkeypatch.setattr(det, "analyze", spy)
    preamble_cfg = pipeline._config.get("detectors", {}).get("preamble", {})
    enabled_val = preamble_cfg.get("enabled", True)

    pipeline.analyze("word " * 60, task_metadata={"enabled": False})
    assert captured["config"].get("enabled", True) == enabled_val, (
        "task_metadata should not override YAML config keys"
    )

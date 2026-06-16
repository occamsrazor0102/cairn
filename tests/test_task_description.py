"""Tests for cairn.task_description (auto-derive task_description)."""
from __future__ import annotations

from cairn import task_description as td


def test_derive_local_extracts_trailing_question():
    text = (
        "The patient presents with multifactorial symptoms. "
        "After a long preamble describing the situation in detail, "
        "we ask the following: what is the probability of recovery?"
    )
    result = td.derive_local(text)
    assert result.startswith("Task: ")
    assert result.endswith("?")
    assert "probability of recovery" in result


def test_derive_local_falls_back_to_first_sentence_when_no_question():
    text = (
        "Summarise the chart of monthly sales. "
        "Include trend lines and seasonal markers."
    )
    result = td.derive_local(text)
    assert result.startswith("Task: Summarise the chart")
    assert result.endswith(".")


def test_derive_local_truncates_long_text():
    text = "x" * 5000
    result = td.derive_local(text)
    assert len(result) <= 240
    assert result.endswith("…")
    assert result.startswith("Task: ")


def test_derive_local_handles_empty():
    assert td.derive_local("") == ""
    assert td.derive_local("   \n  ") == ""


def test_derive_local_not_substring_of_input():
    """contrastive_gen rejects task_descriptions that are substrings of
    the submission. Auto-derive must produce something that escapes
    that guard — the prefix is the minimum perturbation."""
    text = (
        "Long preamble about the situation in elaborate detail. "
        "What is the probability of recovery in the worst case?"
    )
    result = td.derive_local(text)
    assert result not in text  # the verbatim guard must not trip


def test_auto_derive_off_returns_empty():
    desc, mode, warning = td.auto_derive("anything", {"auto_mode": "off"})
    assert desc == ""
    assert mode == "off"
    assert warning is None


def test_auto_derive_missing_key_treats_as_off():
    desc, mode, warning = td.auto_derive("anything", {})
    assert mode == "off"
    assert desc == ""
    assert warning is None


def test_auto_derive_local_yields_description_no_warning():
    text = "Compute the median household income. Reply with a single number."
    desc, mode, warning = td.auto_derive(text, {"auto_mode": "local"})
    assert desc
    assert mode == "local"
    assert warning is None


def test_auto_derive_unknown_mode_returns_off_with_note():
    desc, mode, warning = td.auto_derive("text", {"auto_mode": "wat"})
    assert desc == ""
    assert mode == "off"
    assert warning is not None
    assert "unknown" in warning.lower()


def test_auto_derive_llm_falls_back_to_local_on_provider_error(monkeypatch):
    def fake_llm(text, **kwargs):
        return "", "ConnectionError: simulated outage"

    monkeypatch.setattr(td, "derive_llm", fake_llm)
    text = "What is the airspeed velocity of an unladen swallow?"
    desc, mode, warning = td.auto_derive(text, {"auto_mode": "llm"})
    assert desc
    assert mode == "llm_fallback_local"
    assert warning is not None
    assert "failed" in warning.lower() or "fallback" in warning.lower()


def test_auto_derive_llm_success_attaches_leak_warning(monkeypatch):
    def fake_llm(text, **kwargs):
        return "Summarise the situation in one sentence.", None

    monkeypatch.setattr(td, "derive_llm", fake_llm)
    desc, mode, warning = td.auto_derive(
        "long submission text " * 50, {"auto_mode": "llm"}
    )
    assert desc == "Summarise the situation in one sentence."
    assert mode == "llm"
    assert warning is not None
    assert "submission text was sent" in warning


def test_pipeline_local_mode_attaches_no_leak_warning():
    """Smoke: pipeline.analyze_detailed in local auto-mode should NOT
    append a 'submission text was sent' warning, since local derivation
    never touches a provider."""
    from pathlib import Path
    from cairn.pipeline import CairnPipeline

    pipeline = CairnPipeline.from_config_file(
        Path(__file__).parent.parent / "configs" / "screening.yaml"
    )
    pipeline._config["task_description"] = {"auto_mode": "local"}
    text = "Summarise the data clearly. " * 30
    det = pipeline.analyze(text)
    leak_warnings = [w for w in (det.warnings or []) if "submission text was sent" in w]
    assert leak_warnings == []

# tests/test_config.py
import pytest
from cairn.config import load_config, get_pattern_list, ConfigError

def test_load_default_config(default_config_path):
    cfg = load_config(default_config_path)
    assert "detectors" in cfg
    assert "cascade" in cfg
    assert "decision" in cfg

def test_load_strict_profile(tmp_path):
    from pathlib import Path
    strict = Path(__file__).parent.parent / "configs" / "strict.yaml"
    cfg = load_config(strict)
    assert cfg["decision"]["red_threshold"] < 0.75  # strict = lower threshold

def test_missing_required_key_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("detectors: {}\n")
    with pytest.raises(ConfigError):
        load_config(bad)

def test_get_pattern_list_returns_strings():
    patterns = get_pattern_list("fingerprint_words")
    assert isinstance(patterns, list)
    assert "delve" in patterns
    assert "utilize" in patterns

def test_get_pattern_list_preamble():
    patterns = get_pattern_list("preamble_patterns")
    assert isinstance(patterns, dict)
    assert "critical" in patterns


# ---------------------------------------------------------------------------
# D1 — `_deep_merge` preserves `_profile` / `_description` from the override
# ---------------------------------------------------------------------------
# Pre-fix every extending profile silently reported its base's `_profile`
# after the merge. The audit chain reads `_profile` and was logging
# "default" for every analyze call regardless of which profile loaded.

def test_extending_profile_preserves_overlay_profile_name():
    from pathlib import Path
    profile_path = (
        Path(__file__).parent.parent / "configs" / ".profile_master_minus_arxiv.yaml"
    )
    cfg = load_config(profile_path)
    assert cfg["_profile"] == "profile_master_minus_arxiv"


def _write_yaml(path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_deep_merge_preserves_overlay_description(tmp_path, monkeypatch):
    import cairn.config as cfg_mod
    base = tmp_path / "base.yaml"
    overlay = tmp_path / "overlay.yaml"
    _write_yaml(
        base,
        "_profile: base\n"
        "_description: base description\n"
        "detectors: {a: {enabled: true}}\n"
        "cascade: {phase1_short_circuit_high: 0.85}\n"
        "decision: {red_threshold: 0.7}\n",
    )
    _write_yaml(
        overlay,
        "_profile: overlay\n"
        "_description: overlay description\n"
        f"_extends: {base.stem}\n",
    )
    monkeypatch.setattr(cfg_mod, "CONFIGS_DIR", tmp_path)
    cfg = load_config(overlay)
    assert cfg["_profile"] == "overlay"
    assert cfg["_description"] == "overlay description"


def test_deep_merge_strips_extends_from_merged_config(tmp_path, monkeypatch):
    """`_extends` is a one-shot directive — must not survive into the
    merged dict or a reload would re-trigger expansion."""
    import cairn.config as cfg_mod
    base = tmp_path / "base.yaml"
    overlay = tmp_path / "overlay.yaml"
    _write_yaml(
        base,
        "_profile: base\n"
        "detectors: {a: {enabled: true}}\n"
        "cascade: {phase1_short_circuit_high: 0.85}\n"
        "decision: {red_threshold: 0.7}\n",
    )
    _write_yaml(
        overlay,
        "_profile: overlay\n"
        f"_extends: {base.stem}\n",
    )
    monkeypatch.setattr(cfg_mod, "CONFIGS_DIR", tmp_path)
    cfg = load_config(overlay)
    assert "_extends" not in cfg
    assert cfg["_profile"] == "overlay"


def test_deep_merge_unit_preserves_meta_strips_extends():
    from cairn.config import _deep_merge
    base = {
        "_profile": "base",
        "_description": "base text",
        "detectors": {"a": {"enabled": True}, "b": {"enabled": False}},
    }
    overlay = {
        "_profile": "overlay",
        "_description": "overlay text",
        "_extends": "base",
        "_internal_note": "scratch",
        "detectors": {"a": {"enabled": False}},
    }
    merged = _deep_merge(base, overlay)
    assert merged["_profile"] == "overlay"
    assert merged["_description"] == "overlay text"
    assert "_extends" not in merged
    assert "_internal_note" not in merged
    assert merged["detectors"]["a"] == {"enabled": False}
    assert merged["detectors"]["b"] == {"enabled": False}

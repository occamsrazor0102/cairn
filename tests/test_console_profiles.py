from pathlib import Path

from beet.config import CONFIGS_DIR


def test_profile_choices_covers_all_config_files():
    from beet.console import _PROFILE_CHOICES

    disk_profiles = {p.stem for p in CONFIGS_DIR.glob("*.yaml") if not p.name.startswith("_")}
    missing = disk_profiles - set(_PROFILE_CHOICES)
    assert not missing, f"Profiles on disk but absent from _PROFILE_CHOICES: {sorted(missing)}"


def test_profile_choices_has_no_stale_entries():
    from beet.console import _PROFILE_CHOICES

    disk_profiles = {p.stem for p in CONFIGS_DIR.glob("*.yaml") if not p.name.startswith("_")}
    stale = set(_PROFILE_CHOICES) - disk_profiles
    assert not stale, f"Entries in _PROFILE_CHOICES with no matching YAML: {sorted(stale)}"

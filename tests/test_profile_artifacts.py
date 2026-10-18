"""Regression tests for the calibrated-profile artifacts tracked in git.

``cairn.fusion.serialization.load_model`` refuses a skops archive that has no
``.hmac`` sidecar, and ``CairnPipeline._build_fusion`` downgrades that refusal
to a log warning and silently falls back to naive fusion. A profile whose
``ebm.pkl`` is committed without its sidecar therefore *looks* calibrated on
the machine that trained it (where the gitignored sidecar still exists) but
runs heuristic fusion on every fresh checkout -- including the RAID CI gate.

These tests pin the three links in that chain so the failure shows up in the
suite instead of as a quiet ``calibration_status`` downgrade:

* the ``.gitignore`` allowlist re-includes a sidecar for every model it
  re-includes,
* every skops model on disk has a sidecar that verifies under the published
  default key (the key a fresh checkout will use),
* every git-tracked skops model also tracks its sidecar,
* and the CI profile really does load its trained EBM end-to-end.
"""
from __future__ import annotations

import hmac
import logging
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from cairn.fusion.serialization import _HMAC_KEY_ENV, _digest, _looks_like_skops

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_PREFIX = "models/calib2000/profiles"
PROFILES_DIR = REPO_ROOT / PROFILES_PREFIX
MODEL_FILENAMES = ("ebm.pkl", "ebm.skops")

_MODEL_REINCLUDE_RE = re.compile(
    r"^!/models/calib2000/profiles/[^/]+/ebm\.(?:pkl|skops)$"
)


def _hmac_sidecar(model_path: Path) -> Path:
    # Same derivation save_model / load_model use: ``ebm.pkl`` -> ``ebm.pkl.hmac``.
    return model_path.with_suffix(model_path.suffix + ".hmac")


def _gitignore_reincludes() -> set[str]:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip().startswith("!")}


def _skops_models_on_disk() -> list[Path]:
    if not PROFILES_DIR.is_dir():
        return []
    found: list[Path] = []
    for profile_dir in sorted(p for p in PROFILES_DIR.iterdir() if p.is_dir()):
        for name in MODEL_FILENAMES:
            candidate = profile_dir / name
            if candidate.is_file() and _looks_like_skops(candidate):
                found.append(candidate)
    return found


def _git_tracked_profile_files() -> set[str] | None:
    """Repo-relative paths git tracks under the profiles dir, or None if unknowable."""
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            [git, "ls-files", "--", PROFILES_PREFIX],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def test_gitignore_reincludes_hmac_sidecar_for_every_reincluded_model():
    """The allowlist must ship the sidecar alongside every model it ships.

    The models directory is ignored wholesale and individual artifacts are
    re-included by name, so forgetting one ``!...ebm.pkl.hmac`` line is all it
    takes to commit an unloadable model.
    """
    reincludes = _gitignore_reincludes()
    models = sorted(line for line in reincludes if _MODEL_REINCLUDE_RE.match(line))
    assert models, "expected at least one re-included skops model in .gitignore"
    missing = [line for line in models if f"{line}.hmac" not in reincludes]
    assert not missing, (
        "these .gitignore re-includes ship a skops model without its .hmac "
        f"sidecar, so load_model will refuse it on a fresh checkout: {missing}"
    )


def test_every_skops_model_on_disk_has_verifying_hmac_sidecar(monkeypatch):
    """Each profile's skops model must carry a sidecar a fresh checkout can verify.

    The sidecar is checked with the published default key -- the key in effect
    when CAIRN_MODEL_HMAC_KEY is unset, as it is in CI -- because that is the
    condition under which a committed artifact has to load.
    """
    monkeypatch.delenv(_HMAC_KEY_ENV, raising=False)
    models = _skops_models_on_disk()
    assert models, f"expected at least one skops model under {PROFILES_PREFIX}"
    problems: list[str] = []
    for model_path in models:
        rel = model_path.relative_to(REPO_ROOT).as_posix()
        sidecar = _hmac_sidecar(model_path)
        if not sidecar.is_file():
            problems.append(f"{rel}: .hmac sidecar missing")
            continue
        expected = sidecar.read_text(encoding="ascii").strip()
        if not hmac.compare_digest(expected, _digest(model_path.read_bytes())):
            problems.append(f"{rel}: .hmac sidecar does not verify under the default key")
    assert not problems, (
        "load_model will refuse these artifacts and the pipeline will fall back "
        "to naive fusion:\n  " + "\n  ".join(problems)
    )


def test_every_tracked_skops_model_tracks_its_hmac_sidecar():
    """What git ships must be self-consistent: tracked model => tracked sidecar.

    Guards the case the allowlist test cannot -- the re-include line exists but
    the sidecar was never ``git add``-ed -- and does so from the committer's own
    checkout rather than waiting for CI to discover it.
    """
    tracked = _git_tracked_profile_files()
    if tracked is None:
        pytest.skip("git is unavailable or this is not a git checkout")
    models = sorted(
        path for path in tracked
        if Path(path).name in MODEL_FILENAMES
        and (not (REPO_ROOT / path).is_file() or _looks_like_skops(REPO_ROOT / path))
    )
    assert models, f"expected at least one tracked skops model under {PROFILES_PREFIX}"
    missing = [path for path in models if f"{path}.hmac" not in tracked]
    assert not missing, (
        "these tracked skops models have no tracked .hmac sidecar, so a fresh "
        f"checkout cannot load them: {missing}"
    )


def test_master_minus_arxiv_profile_loads_trained_ebm(monkeypatch, caplog):
    """The RAID CI profile must actually run its trained EBM, not naive fusion.

    This is the end-to-end symptom the sidecar checks above exist to prevent.
    The profile may still report ``degraded`` for unrelated detector-calibration
    reasons; the assertion is only that the fusion model itself loaded.
    """
    pytest.importorskip("skops")
    pytest.importorskip("interpret")
    from cairn.config import load_config
    from cairn.pipeline import CairnPipeline

    monkeypatch.delenv(_HMAC_KEY_ENV, raising=False)
    # The profile's artifact paths are repo-relative.
    monkeypatch.chdir(REPO_ROOT)
    cfg = load_config(REPO_ROOT / "configs" / ".profile_master_minus_arxiv.yaml")
    caplog.set_level(logging.WARNING)
    pipeline = CairnPipeline(cfg)
    health = pipeline.calibration_health()
    assert health["model_loaded"] is True, (
        "trained EBM did not load, so the profile is running naive fusion; "
        f"health={health!r}; warnings:\n{caplog.text}"
    )
    assert health["calibration_status"] != "heuristic"

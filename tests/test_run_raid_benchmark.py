"""Structural smoke test for scripts/run_raid_benchmark.py.

This test does NOT execute the runner against a real pipeline — that lives in
the GitHub Actions workflow `.github/workflows/raid.yml`, which has the
calibrated profile artifact provisioned. Here we just confirm the script:

  1. parses cleanly (no SyntaxError, no missing imports)
  2. exposes a `main` callable
  3. accepts the argparse flags the workflow + plan promise (--dataset,
     --config, --fpr-ceiling, --min-tpr, --out)

If any of those drift, the workflow will fail with an inscrutable shell
error; this test fails fast with a useful message.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_raid_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_run_raid_benchmark", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, "could not locate runner script"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_raid_benchmark_imports_cleanly():
    mod = _load_module()
    assert callable(mod.main), "scripts/run_raid_benchmark.py must expose main()"


def test_run_raid_benchmark_argparse_signature(monkeypatch):
    """Confirm the runner's argparse flags match what the workflow + plan promise."""
    mod = _load_module()
    # We cannot call main() with --help because it would call SystemExit AND
    # try to import a calibrated artifact that isn't in the worktree.  So we
    # short-circuit at parse_args time and inspect the configured actions.
    import argparse
    captured = {}

    real_parser = argparse.ArgumentParser

    class CapturingParser(real_parser):  # type: ignore[misc, valid-type]
        def parse_args(self, *a, **kw):
            captured["actions"] = {action.dest for action in self._actions}
            raise SystemExit(0)  # short-circuit before pipeline construction

    monkeypatch.setattr(argparse, "ArgumentParser", CapturingParser)
    # Stub out load_dotenv so we don't hit the filesystem unnecessarily.
    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **kw: None)

    with pytest.raises(SystemExit):
        mod.main()

    expected = {"dataset", "config", "fpr_ceiling", "min_tpr", "out"}
    assert expected.issubset(captured.get("actions", set())), (
        f"missing argparse flags; got {captured.get('actions')}, need {expected}"
    )

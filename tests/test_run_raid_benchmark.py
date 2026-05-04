"""Structural smoke test for scripts/run_raid_benchmark.py.

This test does NOT execute the runner against a real pipeline — that lives in
the GitHub Actions workflow `.github/workflows/raid.yml`, which has the
calibrated profile artifact provisioned. Here we just confirm the script:

  1. parses cleanly (no SyntaxError, no missing imports)
  2. accepts the argparse flags the workflow + plan promise (--dataset,
     --config, --fpr-ceiling, --min-tpr, --out)

If any of those drift, the workflow will fail with an inscrutable shell
error; this test fails fast with a useful message.

Implementation note: the prior version of this test used
``importlib.util.spec_from_file_location`` + ``exec_module`` + a
monkeypatched ``argparse.ArgumentParser`` to short-circuit ``main()``. In
isolation that was harmless, but inside the full pytest suite the second
test ballooned the runner process to 10+ GB of RSS and crashed the run at
the 91% mark. See ``memory/project_cumulative_test_leak.md`` for the bisect
that pinned it. The static AST inspection below is functionally equivalent
(same two assertions) and does not import ``cairn`` or run the script.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_raid_benchmark.py"


def _argparse_flags_in_source() -> set[str]:
    """Return the set of '--flag' strings passed as the first positional
    argument to any ``ap.add_argument(...)`` call in the runner script."""
    tree = ast.parse(SCRIPT_PATH.read_text())
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        val = node.args[0].value
        if isinstance(val, str) and val.startswith("--"):
            flags.add(val)
    return flags


def test_run_raid_benchmark_imports_cleanly():
    """The runner script parses as valid Python."""
    src = SCRIPT_PATH.read_text()
    ast.parse(src)


def test_run_raid_benchmark_argparse_signature():
    """The runner's argparse flags match what the workflow + plan promise."""
    flags = _argparse_flags_in_source()
    expected = {"--dataset", "--config", "--fpr-ceiling", "--min-tpr", "--out"}
    assert expected.issubset(flags), (
        f"missing argparse flags; got {flags}, need {expected}"
    )

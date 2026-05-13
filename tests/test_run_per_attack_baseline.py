"""Structural smoke test for scripts/run_per_attack_baseline.py.

Mirrors tests/test_run_raid_benchmark.py — static AST inspection only,
no exec_module or argparse monkeypatching, to avoid the cumulative-test
memory leak documented in memory/project_cumulative_test_leak.md.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_per_attack_baseline.py"


def _argparse_flags_in_source() -> set[str]:
    tree = ast.parse(SCRIPT_PATH.read_text())
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        val = node.args[0].value
        if isinstance(val, str) and val.startswith("--"):
            flags.add(val)
    return flags


def test_run_per_attack_baseline_imports_cleanly():
    """The runner script parses as valid Python."""
    src = SCRIPT_PATH.read_text()
    ast.parse(src)


def test_run_per_attack_baseline_argparse_signature():
    """The runner's argparse flags match what Wave 2 / A2 step 1 promises."""
    flags = _argparse_flags_in_source()
    expected = {
        "--dataset", "--config", "--attacks", "--limit", "--seed",
        "--out-csv", "--include-prompt-attacks", "--provider", "--out-json",
    }
    assert expected.issubset(flags), (
        f"missing argparse flags; got {flags}, need {expected}"
    )


def test_run_per_attack_baseline_default_attacks_match_plan():
    """The default attack list must include the plan's 5 transforms.

    Plan names them (paraphrase, synonym_swap, typo, casualize,
    mix_human); the registered attack ids are paraphrase_launder,
    synonym_swap, inject_typos, casualize, mix_human.
    """
    tree = ast.parse(SCRIPT_PATH.read_text())
    defaults: tuple[str, ...] | None = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_DEFAULT_ATTACKS"):
            continue
        if not isinstance(node.value, ast.Tuple):
            continue
        defaults = tuple(
            elt.value for elt in node.value.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        )
        break
    assert defaults is not None, "_DEFAULT_ATTACKS tuple not found"
    expected = (
        "paraphrase_launder", "synonym_swap", "inject_typos",
        "casualize", "mix_human",
    )
    assert defaults == expected, (
        f"_DEFAULT_ATTACKS drifted from plan; got {defaults}, want {expected}"
    )

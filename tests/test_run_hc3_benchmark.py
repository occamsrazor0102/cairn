"""Pure-helper tests for scripts/run_hc3_benchmark.py.

The orchestrator's heavy lifting (subprocess, downloads, full pipeline)
is exercised manually via the CLI. These tests cover the pure plumbing:
path resolution, split-presence detection, report parsing, and summary
artefact rendering.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_hc3_benchmark.py"


def _load_script_module() -> types.ModuleType:
    """Load run_hc3_benchmark.py without triggering its argparse main().

    The file is a CLI entry point — importing it via spec_from_file_location
    runs only top-level definitions because main() lives behind
    `if __name__ == "__main__":`.
    """
    name = "scripts_run_hc3_benchmark"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def hb():
    return _load_script_module()


def test_module_loads_without_main_side_effects(hb):
    assert hb.HC3_SUBSETS
    assert "finance" in hb.HC3_SUBSETS
    assert "all" in hb.HC3_SUBSETS
    assert "configs/default.yaml" in hb.DEFAULT_CONFIGS
    assert any("general_hc3" in c for c in hb.DEFAULT_CONFIGS)


def test_main_is_gated_by_name_main():
    """The CLI entry point must not run on import."""
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    has_guard = False
    for node in tree.body:
        if isinstance(node, ast.If):
            t = node.test
            if (isinstance(t, ast.Compare)
                and isinstance(t.left, ast.Name)
                and t.left.id == "__name__"
                and any(
                    isinstance(c, ast.Constant) and c.value == "__main__"
                    for c in t.comparators
                )):
                has_guard = True
                break
    assert has_guard


def test_hc3_prefix_and_cell_paths(hb, tmp_path):
    assert hb.hc3_prefix("finance") == "hc3_finance"
    cell = hb.cell_paths(
        "finance",
        Path("configs/.profile_general_hc3.yaml"),
        data_dir=tmp_path,
        out_dir=tmp_path / "out",
    )
    assert cell.subset == "finance"
    assert cell.dataset == tmp_path / "hc3_finance_test.jsonl"
    # Hidden-config dotfiles produce a clean tag (no leading dot).
    assert cell.out == tmp_path / "out" / "finance" / "profile_general_hc3.json"


def test_cell_paths_for_visible_config(hb, tmp_path):
    cell = hb.cell_paths(
        "medicine",
        Path("configs/default.yaml"),
        data_dir=tmp_path,
        out_dir=tmp_path / "out",
    )
    assert cell.out.name == "default.json"


def test_splits_present_requires_all_three(hb, tmp_path):
    assert not hb.splits_present(tmp_path, "hc3_x")
    (tmp_path / "hc3_x_train.jsonl").write_text("")
    (tmp_path / "hc3_x_calib.jsonl").write_text("")
    assert not hb.splits_present(tmp_path, "hc3_x")
    (tmp_path / "hc3_x_test.jsonl").write_text("")
    assert hb.splits_present(tmp_path, "hc3_x")


def test_fmt_handles_none_and_nan(hb):
    assert hb._fmt(None) == ""
    assert hb._fmt(float("nan")) == ""
    assert hb._fmt(0.123456789) == "0.1235"
    assert hb._fmt(1.0) == "1.0000"


def test_parse_report_round_trips_through_disk(hb, tmp_path):
    raw = {
        "n_samples": 150,
        "n_failed": 0,
        "duration_seconds": 12.34,
        "metrics": {"auroc": 0.9165, "ece": 0.0498, "brier": 0.119,
                    "tpr_at_fpr_01": 0.4267},
        "confusion": {"accuracy": 0.8467, "precision": 0.84,
                      "recall": 0.83, "f1": 0.8369},
    }
    p = tmp_path / "report.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    res = hb.parse_report(p, subset="finance", config="configs/default.yaml")
    assert res.subset == "finance"
    assert res.n_samples == 150
    assert res.metrics["auroc"] == pytest.approx(0.9165)
    assert res.confusion["f1"] == pytest.approx(0.8369)
    assert res.error is None


def test_parse_report_handles_missing_optionals(hb, tmp_path):
    p = tmp_path / "thin.json"
    p.write_text(json.dumps({"metrics": {}, "confusion": {}}), encoding="utf-8")
    res = hb.parse_report(p, subset="finance", config="x")
    assert res.n_samples == 0
    assert res.duration_seconds == 0.0
    assert res.metrics == {}
    assert res.confusion == {}


def _mk(hb, **kw):
    return hb.CellResult(
        subset=kw.get("subset", "finance"),
        config=kw.get("config", "configs/default.yaml"),
        n_samples=kw.get("n_samples", 100),
        n_failed=kw.get("n_failed", 0),
        duration_seconds=kw.get("duration_seconds", 1.0),
        metrics=kw.get("metrics", {}),
        confusion=kw.get("confusion", {}),
        error=kw.get("error"),
    )


def test_write_summary_csv_columns_and_rows(hb, tmp_path):
    results = [
        _mk(hb, subset="finance",
            metrics={"auroc": 0.91, "ece": 0.05, "brier": 0.12, "tpr_at_fpr_01": 0.42},
            confusion={"accuracy": 0.85, "precision": 0.84, "recall": 0.83, "f1": 0.83}),
        _mk(hb, subset="medicine", config="configs/.profile_general_hc3.yaml",
            metrics={"auroc": 0.95}, error="boom"),
    ]
    out = tmp_path / "out" / "summary.csv"
    hb.write_summary_csv(results, out)
    text = out.read_text(encoding="utf-8")
    header, *rows = text.strip().splitlines()
    assert header.split(",")[0:5] == [
        "subset", "config", "n_samples", "n_failed", "duration_s",
    ]
    assert len(rows) == 2
    assert "finance" in rows[0]
    assert "0.9100" in rows[0]
    assert "boom" in rows[1]


def test_render_markdown_groups_by_config(hb):
    results = [
        _mk(hb, subset="finance", config="A",
            metrics={"auroc": 0.91, "ece": 0.05, "brier": 0.12, "tpr_at_fpr_01": 0.42},
            confusion={"accuracy": 0.85, "f1": 0.83}),
        _mk(hb, subset="medicine", config="A", metrics={"auroc": 0.88}),
        _mk(hb, subset="finance", config="B", metrics={"auroc": 0.74}),
    ]
    md = hb.render_markdown(results)
    assert md.startswith("# HC3 benchmark grid")
    assert "## `A`" in md
    assert "## `B`" in md
    assert md.count("| finance |") == 2
    assert md.count("| medicine |") == 1
    # Missing metrics fall back to "n/a".
    assert "n/a" in md


def test_render_markdown_handles_empty_results(hb):
    md = hb.render_markdown([])
    assert md.strip() == "# HC3 benchmark grid"


def test_concat_test_splits_writes_union_skipping_blanks(hb, tmp_path):
    """The cache input must be the line-by-line union of every requested
    subset's test split, ignoring trailing blanks so we don't poison
    cache_pipeline_features with empty rows."""
    (tmp_path / "hc3_finance_test.jsonl").write_text(
        '{"id": "a", "text": "x", "label": 0}\n'
        '{"id": "b", "text": "y", "label": 1}\n\n',  # trailing blank
        encoding="utf-8",
    )
    (tmp_path / "hc3_medicine_test.jsonl").write_text(
        '{"id": "c", "text": "z", "label": 0}\n',
        encoding="utf-8",
    )
    out = tmp_path / "concat.jsonl"
    n = hb._concat_test_splits(["finance", "medicine"], data_dir=tmp_path, out=out)
    assert n == 3
    rows = [
        line for line in out.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(rows) == 3
    assert '"id": "a"' in rows[0]
    assert '"id": "c"' in rows[2]


def test_concat_test_splits_raises_when_split_missing(hb, tmp_path):
    with pytest.raises(FileNotFoundError, match="missing test split"):
        hb._concat_test_splits(["finance"], data_dir=tmp_path,
                               out=tmp_path / "concat.jsonl")


def test_ensure_features_cache_is_noop_when_file_exists(hb, tmp_path, monkeypatch):
    """If the cache already exists we must not re-run the heavy build step."""
    cache = tmp_path / "cache.jsonl"
    cache.write_text("{}\n")
    calls: list[list[str]] = []
    monkeypatch.setattr(hb, "_run", lambda cmd: calls.append(cmd))
    hb.ensure_features_cache(cache, subsets=["finance"], data_dir=tmp_path,
                             python="python")
    assert calls == []


def test_ensure_features_cache_invokes_cache_script_with_cache_builder_config(
    hb, tmp_path, monkeypatch
):
    """When the cache is missing, ensure_features_cache must:
       1. concat the test splits, and
       2. shell out to cache_pipeline_features.py with the cache-builder
          config (widened cascade thresholds) so every enabled phase
          runs on every sample — otherwise the calibrated reader feeds
          NaN→0.5 imputations into the EBM and accuracy collapses."""
    (tmp_path / "hc3_finance_test.jsonl").write_text(
        '{"id": "a", "text": "x", "label": 0}\n', encoding="utf-8",
    )
    cache = tmp_path / "cache.jsonl"
    captured: list[list[str]] = []
    monkeypatch.setattr(hb, "_run", lambda cmd: captured.append(cmd))
    hb.ensure_features_cache(cache, subsets=["finance"], data_dir=tmp_path,
                             python="python")
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:3] == ["python", "-u", "scripts/cache_pipeline_features.py"]
    assert "--config" in cmd
    cfg_arg = cmd[cmd.index("--config") + 1]
    assert cfg_arg == hb.CACHE_BUILDER_CONFIG
    assert cfg_arg.endswith("_cache_builder.yaml")
    # The constant must point at a real file on disk, otherwise
    # cache_pipeline_features.py will SystemExit on load.
    assert (REPO_ROOT / cfg_arg).exists(), (
        f"cache-builder config not found at {REPO_ROOT / cfg_arg}"
    )
    assert cmd[cmd.index("--out") + 1] == str(cache)
    # Concat input was written.
    assert (cache.parent / "_hc3_cache_input.jsonl").exists()


def test_cache_builder_config_widens_cascade_thresholds(hb):
    """The cache-builder config must keep enabled-detector inheritance from
    default.yaml AND widen cascade thresholds so phase-2 always runs.
    Otherwise we silently regress to the same NaN-imputation problem."""
    import yaml
    cfg_path = REPO_ROOT / hb.CACHE_BUILDER_CONFIG
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw.get("_extends") == "default", (
        "cache-builder must extend default.yaml so detector enables stay aligned"
    )
    cascade = raw.get("cascade") or {}
    # Phase 1 / 2 short-circuits: any p_llm in [0,1] must satisfy
    # `low < agg < high`, which means low < 0 and high > 1.
    assert cascade.get("phase1_short_circuit_low", 0) < 0
    assert cascade.get("phase1_short_circuit_high", 1) > 1
    assert cascade.get("phase2_short_circuit_low", 0) < 0
    assert cascade.get("phase2_short_circuit_high", 1) > 1


def test_default_configs_are_paths_when_no_flag_supplied(hb):
    """Regression: argparse defaults bypass `type=Path`, so `--configs` default
    must already be a list[Path] — otherwise `cfg.exists()` blows up the moment
    the harness is invoked without `--configs` (e.g. by the Modal driver)."""
    args = hb.build_parser().parse_args([])
    assert args.configs, "default configs must be non-empty"
    for cfg in args.configs:
        assert isinstance(cfg, Path), (
            f"default --configs entry is {type(cfg).__name__}, expected Path"
        )


def test_user_supplied_configs_become_paths(hb):
    args = hb.build_parser().parse_args(["--configs", "configs/default.yaml"])
    assert args.configs == [Path("configs/default.yaml")]
    assert all(isinstance(c, Path) for c in args.configs)


def test_normalize_args_coerces_strings(hb):
    """Belt-and-suspenders: even a Namespace with str configs comes out clean."""
    import argparse
    ns = argparse.Namespace(configs=["a.yaml", Path("b.yaml")])
    out = hb._normalize_args(ns)
    assert out.configs == [Path("a.yaml"), Path("b.yaml")]
    assert all(isinstance(c, Path) for c in out.configs)


def test_cellresult_row_serialisation(hb):
    r = _mk(hb,
            metrics={"auroc": 0.5, "ece": 0.1, "brier": 0.2, "tpr_at_fpr_01": 0.3},
            confusion={"accuracy": 0.6, "precision": 0.7, "recall": 0.8, "f1": 0.9})
    row = r.row()
    assert row["subset"] == "finance"
    assert row["auroc"] == "0.5000"
    assert row["f1"] == "0.9000"
    assert row["error"] == ""

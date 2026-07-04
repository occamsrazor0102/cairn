"""Run the HC3 benchmark grid end-to-end.

Orchestrates the existing per-step scripts so a single command can:

1. Ensure each requested HC3 subset is downloaded (`scripts/download_hc3.py`).
2. Ensure stratified train/calib/test splits exist (`scripts/split_dataset.py`).
3. Run `scripts/run_benchmark.py` for every (subset x config) cell.
4. Emit a CSV + Markdown summary.

Defaults benchmark the canonical six HC3 subsets against two configs:
  * `configs/default.yaml` — heuristic baseline
  * `configs/.profile_general_hc3.yaml` — calibrated profile (general_hc3)

Usage:
    # Full grid (download missing subsets, split, benchmark heuristic + calibrated)
    python scripts/run_hc3_benchmark.py

    # Just the test split that is already on disk, only the calibrated profile
    python scripts/run_hc3_benchmark.py \\
        --subsets finance medicine \\
        --configs configs/.profile_general_hc3.yaml \\
        --skip-existing

    # Quick smoke run on a sub-sample
    python scripts/run_hc3_benchmark.py --subsets finance --limit 50
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Canonical HC3 subsets exposed by `scripts/download_hc3.py`.
HC3_SUBSETS: tuple[str, ...] = (
    "finance",
    "medicine",
    "open_qa",
    "reddit_eli5",
    "wiki_csai",
    "all",
)

DEFAULT_CONFIGS: tuple[str, ...] = (
    "configs/default.yaml",
    "configs/.profile_general_hc3.yaml",
)

METRIC_COLS: tuple[str, ...] = (
    "auroc",
    "ece",
    "brier",
    "tpr_at_fpr_01",
)
CONFUSION_COLS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
)


@dataclass(frozen=True)
class Cell:
    """One (subset, config) benchmark cell."""

    subset: str
    config: Path
    dataset: Path
    out: Path


@dataclass
class CellResult:
    """Metrics extracted from a `run_benchmark.py` JSON report."""

    subset: str
    config: str
    n_samples: int = 0
    n_failed: int = 0
    duration_seconds: float = 0.0
    metrics: dict[str, float | None] = field(default_factory=dict)
    confusion: dict[str, float | None] = field(default_factory=dict)
    error: str | None = None

    def row(self) -> dict[str, str]:
        out: dict[str, str] = {
            "subset": self.subset,
            "config": self.config,
            "n_samples": str(self.n_samples),
            "n_failed": str(self.n_failed),
            "duration_s": f"{self.duration_seconds:.1f}",
        }
        for k in METRIC_COLS:
            out[k] = _fmt(self.metrics.get(k))
        for k in CONFUSION_COLS:
            out[k] = _fmt(self.confusion.get(k))
        out["error"] = self.error or ""
        return out


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.4f}"


def splits_present(data_dir: Path, prefix: str) -> bool:
    """True iff the three stratified split files all exist on disk."""
    return all(
        (data_dir / f"{prefix}_{name}.jsonl").exists()
        for name in ("train", "calib", "test")
    )


def hc3_prefix(subset: str) -> str:
    return f"hc3_{subset}"


def cell_paths(
    subset: str,
    config: Path,
    *,
    data_dir: Path,
    out_dir: Path,
) -> Cell:
    """Resolve the dataset (test split) and output JSON path for one cell."""
    prefix = hc3_prefix(subset)
    dataset = data_dir / f"{prefix}_test.jsonl"
    config_tag = config.stem.lstrip(".")
    out = out_dir / subset / f"{config_tag}.json"
    return Cell(subset=subset, config=config, dataset=dataset, out=out)


def parse_report(path: Path, *, subset: str, config: str) -> CellResult:
    """Read a `run_benchmark.py` report JSON into a CellResult."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CellResult(
        subset=subset,
        config=config,
        n_samples=int(raw.get("n_samples") or 0),
        n_failed=int(raw.get("n_failed") or 0),
        duration_seconds=float(raw.get("duration_seconds") or 0.0),
        metrics=dict(raw.get("metrics") or {}),
        confusion=dict(raw.get("confusion") or {}),
    )


def write_summary_csv(results: list[CellResult], path: Path) -> None:
    cols = ["subset", "config", "n_samples", "n_failed", "duration_s",
            *METRIC_COLS, *CONFUSION_COLS, "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow(r.row())


def render_markdown(results: list[CellResult]) -> str:
    """Pretty Markdown table grouped by config."""
    by_cfg: dict[str, list[CellResult]] = {}
    for r in results:
        by_cfg.setdefault(r.config, []).append(r)

    parts: list[str] = ["# HC3 benchmark grid", ""]
    for cfg, rows in by_cfg.items():
        parts.append(f"## `{cfg}`")
        parts.append("")
        parts.append("| subset | n | AUROC | ECE | Brier | TPR@1%FPR | acc | F1 |")
        parts.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            parts.append(
                "| {sub} | {n} | {auroc} | {ece} | {brier} | {tpr} | {acc} | {f1} |".format(
                    sub=r.subset,
                    n=r.n_samples,
                    auroc=_fmt(r.metrics.get("auroc")) or "n/a",
                    ece=_fmt(r.metrics.get("ece")) or "n/a",
                    brier=_fmt(r.metrics.get("brier")) or "n/a",
                    tpr=_fmt(r.metrics.get("tpr_at_fpr_01")) or "n/a",
                    acc=_fmt(r.confusion.get("accuracy")) or "n/a",
                    f1=_fmt(r.confusion.get("f1")) or "n/a",
                )
            )
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def _run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"command failed (rc={res.returncode}): {' '.join(cmd)}")


def ensure_subset(
    subset: str,
    *,
    data_dir: Path,
    max_per_class: int,
    python: str,
) -> None:
    """Download the raw HC3 subset and create stratified splits if missing."""
    prefix = hc3_prefix(subset)
    raw = data_dir / f"{prefix}.jsonl"
    if not raw.exists():
        print(f"[fetch] {subset} -> {raw}", flush=True)
        _run([python, "-u", "scripts/download_hc3.py",
              "--subset", subset, "--out", str(raw),
              "--max-per-class", str(max_per_class)])

    if not splits_present(data_dir, prefix):
        print(f"[split] {subset}", flush=True)
        _run([python, "-u", "scripts/split_dataset.py",
              "--in", str(raw), "--out-dir", str(data_dir),
              "--prefix", prefix])


def _concat_test_splits(
    subsets: list[str],
    *,
    data_dir: Path,
    out: Path,
) -> int:
    """Concatenate every requested HC3 test split into a single JSONL.

    The feature cache is calibrator-independent (see cache_pipeline_features.py),
    so caching once over the union of test splits lets every (subset, config)
    cell read from the same cache.

    Returns the number of rows written.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fout:
        for subset in subsets:
            test = data_dir / f"{hc3_prefix(subset)}_test.jsonl"
            if not test.exists():
                raise FileNotFoundError(
                    f"missing test split for {subset}: {test}"
                )
            for line in test.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                fout.write(line + "\n")
                n += 1
    return n


CACHE_BUILDER_CONFIG = "configs/_cache_builder.yaml"


def ensure_features_cache(
    cache_path: Path,
    *,
    subsets: list[str],
    data_dir: Path,
    python: str,
) -> None:
    """Build `cache_path` from the union of test splits if it does not exist.

    Uses configs/_cache_builder.yaml — same enabled detectors as
    configs/default.yaml but with the cascade short-circuit thresholds
    widened so every enabled phase runs on every sample. That's what makes
    the cache truly calibrator-independent: a default-cascade build leaves
    ~80% of phase-2 features missing (heuristic short-circuit fires
    aggressively on uncalibrated p_llm), and the calibrated reader then
    feeds NaN→0.5 imputations into the EBM and accuracy collapses.
    """
    if cache_path.exists():
        return
    print(f"[cache] building feature cache at {cache_path}", flush=True)
    concat = cache_path.parent / "_hc3_cache_input.jsonl"
    n = _concat_test_splits(subsets, data_dir=data_dir, out=concat)
    print(f"[cache] {n} rows across {len(subsets)} test splits", flush=True)
    _run([python, "-u", "scripts/cache_pipeline_features.py",
          "--dataset", str(concat),
          "--config", CACHE_BUILDER_CONFIG,
          "--out", str(cache_path)])


def run_cell(
    cell: Cell,
    *,
    python: str,
    limit: int | None,
    seed: int,
    progress: bool,
    features_cache: Path | None,
) -> CellResult:
    cell.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-u", "scripts/run_benchmark.py",
           "--dataset", str(cell.dataset),
           "--config", str(cell.config),
           "--out", str(cell.out),
           "--seed", str(seed)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if progress:
        cmd += ["--progress"]
    if features_cache is not None:
        cmd += ["--features-cache", str(features_cache)]

    t0 = time.monotonic()
    try:
        _run(cmd)
    except Exception as e:
        return CellResult(
            subset=cell.subset, config=str(cell.config),
            duration_seconds=time.monotonic() - t0,
            error=str(e),
        )
    return parse_report(cell.out, subset=cell.subset, config=str(cell.config))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--subsets", nargs="+", default=list(HC3_SUBSETS),
                    choices=list(HC3_SUBSETS),
                    help="HC3 subsets to evaluate.")
    # argparse's `type=` only fires for user-supplied values, so the default
    # must already be a list[Path] — bare strings would crash on cfg.exists().
    ap.add_argument("--configs", nargs="+",
                    default=[Path(c) for c in DEFAULT_CONFIGS],
                    type=Path,
                    help="One or more cairn config YAMLs to benchmark.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/benchmarks/hc3_grid"))
    ap.add_argument("--summary-csv", type=Path,
                    default=Path("docs/benchmarks/hc3_grid/summary.csv"))
    ap.add_argument("--summary-md", type=Path,
                    default=Path("docs/benchmarks/hc3_grid/summary.md"))
    ap.add_argument("--max-per-class", type=int, default=500,
                    help="Cap per (human/llm) class when fetching missing HC3 subsets.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Sub-sample each test split to N items (passed to run_benchmark).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--features-cache", type=Path, default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip cells whose output JSON already exists.")
    ap.add_argument("--python", default=sys.executable)
    return ap


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Coerce path-typed args to Path even when a stale default leaks a str."""
    args.configs = [Path(c) for c in args.configs]
    return args


def main() -> None:
    args = _normalize_args(build_parser().parse_args())

    for cfg in args.configs:
        if not cfg.exists():
            raise SystemExit(f"config not found: {cfg}")

    # Phase 1: ensure data is on disk for every requested subset.
    for subset in args.subsets:
        ensure_subset(subset, data_dir=args.data_dir,
                      max_per_class=args.max_per_class, python=args.python)
        prefix = hc3_prefix(subset)
        if not splits_present(args.data_dir, prefix):
            raise SystemExit(
                f"split files for {prefix} are still missing after split step"
            )

    # Phase 1b: build feature cache once over the union of test splits if
    # the user asked for one and it doesn't exist yet. CachedPipeline opens
    # the cache file at construct time, so we must populate it before any
    # cell runs.
    if args.features_cache is not None:
        ensure_features_cache(
            args.features_cache,
            subsets=args.subsets,
            data_dir=args.data_dir,
            python=args.python,
        )

    # Phase 2: run each (subset, config) cell.
    results: list[CellResult] = []
    for subset in args.subsets:
        for cfg in args.configs:
            cell = cell_paths(subset, cfg, data_dir=args.data_dir, out_dir=args.out_dir)
            if args.skip_existing and cell.out.exists():
                print(f"[skip] {subset} x {cfg} (already at {cell.out})", flush=True)
                results.append(parse_report(cell.out, subset=subset, config=str(cfg)))
                continue

            print(f"\n=== {subset} x {cfg} ===", flush=True)
            res = run_cell(cell, python=args.python, limit=args.limit,
                           seed=args.seed, progress=args.progress,
                           features_cache=args.features_cache)
            results.append(res)

    # Phase 3: emit summary artifacts.
    write_summary_csv(results, args.summary_csv)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(render_markdown(results), encoding="utf-8")

    print(f"\nwrote {args.summary_csv}")
    print(f"wrote {args.summary_md}")
    print(render_markdown(results))


if __name__ == "__main__":
    main()

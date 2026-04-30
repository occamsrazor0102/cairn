"""Run a calibration-ablation grid across provider combinations.

For each ablation, this script:
  1. Filters the train + calib splits to only the ablation's providers
  2. Writes a temp config wiring the artifacts the run will produce
  3. Runs calibrate_detectors -> train_fusion -> calibrate_conformal
  4. Benchmarks against the SHARED full test set (all providers)
  5. Appends metrics to a single summary CSV

All four sub-runs use --features-cache so detector inference happens
exactly zero times during the grid; the cache must be built first via
scripts/cache_pipeline_features.py.

Default grid: full + 6 single-provider + 6 leave-one-out = 13 ablations.

Usage:
    python scripts/run_ablation_grid.py \\
        --features-cache data/calib2000_features.jsonl \\
        --train data/calib2000_full_train.jsonl \\
        --calib data/calib2000_full_calib.jsonl \\
        --test  data/calib2000_full_test.jsonl \\
        --out-dir models/calib2000/ablation \\
        --benchmarks-dir docs/benchmarks/calib2000/ablation \\
        --summary-csv docs/benchmarks/calib2000/ablation_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# Provider name -> substring(s) that appear in the row's `source` field
# (e.g. "hc3_finance_regen_claude-sonnet-4-6" matches "claude-sonnet-4-6").
PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": ["claude-sonnet-4-6"],
    "openai":    ["gpt-5.4-mini"],
    "gemini":    ["gemini-3-flash-preview"],
    "mistral":   ["mistral-medium-latest"],
    "deepseek":  ["deepseek-chat"],
    "moonshot":  ["kimi-k2.6"],
}

ALL_PROVIDERS = list(PROVIDER_MODELS.keys())


def default_grid() -> list[dict]:
    """13 ablations: full + 6 solos + 6 leave-one-out."""
    grid: list[dict] = [{"name": "full", "providers": list(ALL_PROVIDERS)}]
    for p in ALL_PROVIDERS:
        grid.append({"name": f"solo_{p}", "providers": [p]})
    for p in ALL_PROVIDERS:
        others = [q for q in ALL_PROVIDERS if q != p]
        grid.append({"name": f"loo_{p}", "providers": others})
    return grid


def filter_split(src: Path, dst: Path, providers: list[str]) -> int:
    """Write rows whose `source` matches any of the providers' model strings."""
    patterns: list[str] = []
    for p in providers:
        patterns.extend(PROVIDER_MODELS[p])
    n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line_s = line.strip()
            if not line_s:
                continue
            row = json.loads(line_s)
            source = row.get("source", "")
            if any(pat in source for pat in patterns):
                fout.write(line)
                n += 1
    return n


def write_config(path: Path, name: str, det_cal: Path, ebm: Path, conformal: Path) -> None:
    body = (
        f"_profile: ablation_{name}\n"
        f"_extends: default\n"
        f"fusion:\n"
        f"  model_path: {ebm.as_posix()}\n"
        f"  conformal_path: {conformal.as_posix()}\n"
        f"  detector_calibration_path: {det_cal.as_posix()}\n"
    )
    path.write_text(body, encoding="utf-8")


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"command failed (rc={res.returncode}): {cmd}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--features-cache", required=True, type=Path)
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--calib", required=True, type=Path)
    ap.add_argument("--test",  required=True, type=Path)
    ap.add_argument("--out-dir",        required=True, type=Path,
                    help="Per-ablation models go to <out-dir>/<name>/")
    ap.add_argument("--benchmarks-dir", required=True, type=Path,
                    help="Per-ablation benchmark JSONs go to <benchmarks-dir>/<name>.json")
    ap.add_argument("--summary-csv",    required=True, type=Path)
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to spawn sub-runs with (default: current)")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Conformal alpha (default 0.05)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Run only ablations with these names (default: full grid)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip ablations whose summary row already exists")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.benchmarks_dir.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    grid = default_grid()
    if args.only:
        grid = [a for a in grid if a["name"] in set(args.only)]
        if not grid:
            raise SystemExit(f"no ablations matched --only {args.only}")

    csv_exists = args.summary_csv.exists()
    done_names: set[str] = set()
    if csv_exists and args.skip_existing:
        with open(args.summary_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            done_names = {row["ablation"] for row in reader}

    csv_handle = open(args.summary_csv, "a", encoding="utf-8", newline="")
    writer = csv.writer(csv_handle)
    if not csv_exists:
        writer.writerow([
            "ablation", "providers", "n_train", "n_calib", "n_test",
            "auroc", "ece", "brier", "tpr_at_fpr_01",
            "f1", "accuracy", "duration_s",
        ])
        csv_handle.flush()

    for ab in grid:
        name = ab["name"]
        providers = ab["providers"]
        if name in done_names:
            print(f"[skip] {name} already in summary CSV", flush=True)
            continue
        ab_dir = args.out_dir / name
        ab_dir.mkdir(parents=True, exist_ok=True)
        train_f = ab_dir / "train.jsonl"
        calib_f = ab_dir / "calib.jsonl"
        n_train = filter_split(args.train, train_f, providers)
        n_calib = filter_split(args.calib, calib_f, providers)

        det_cal = ab_dir / "detector_calibration.json"
        ebm     = ab_dir / "ebm.pkl"
        conf    = ab_dir / "conformal.json"
        cfg     = Path("configs") / f".ablation_{name}.yaml"
        write_config(cfg, name, det_cal, ebm, conf)

        bench_path = args.benchmarks_dir / f"{name}.json"

        print(f"\n=== ablation: {name} (providers={providers}, "
              f"n_train={n_train}, n_calib={n_calib}) ===", flush=True)
        t0 = time.monotonic()

        run([args.python, "-u", "scripts/calibrate_detectors.py",
             "--dataset", str(train_f),
             "--config",  str(cfg),
             "--out",     str(det_cal),
             "--features-cache", str(args.features_cache)])
        run([args.python, "-u", "scripts/train_fusion.py",
             "--dataset", str(train_f),
             "--config",  str(cfg),
             "--out",     str(ebm),
             "--features-cache", str(args.features_cache)])
        run([args.python, "-u", "scripts/calibrate_conformal.py",
             "--dataset", str(calib_f),
             "--config",  str(cfg),
             "--alpha",   str(args.alpha),
             "--out",     str(conf),
             "--features-cache", str(args.features_cache)])
        run([args.python, "-u", "scripts/run_benchmark.py",
             "--dataset", str(args.test),
             "--config",  str(cfg),
             "--out",     str(bench_path),
             "--features-cache", str(args.features_cache)])

        duration = time.monotonic() - t0
        report = json.loads(bench_path.read_text(encoding="utf-8"))
        m = report.get("metrics", {})
        c = report.get("confusion", {})
        n_test = report.get("n_samples", 0)

        def _f(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else ""

        writer.writerow([
            name, "+".join(providers), n_train, n_calib, n_test,
            m.get("auroc"), m.get("ece"), m.get("brier"), m.get("tpr_at_fpr_01"),
            c.get("f1"), c.get("accuracy"), round(duration, 1),
        ])
        csv_handle.flush()
        print(f"[done] {name}: AUROC={_f(m.get('auroc'))}  "
              f"F1={_f(c.get('f1'))}  duration={duration:.1f}s", flush=True)

    csv_handle.close()
    print(f"\nablation grid complete -> {args.summary_csv}", flush=True)


if __name__ == "__main__":
    main()

"""Approach 2: per-source leave-one-out matrix.

For each source-tag prefix in the master corpus, train a profile that
EXCLUDES that source. Benchmark on the standard 9 test sets. Compute
the AUROC delta vs the "all sources" baseline.

Sources whose removal HURTS test AUROC are critical to keep. Sources
whose removal HELPS are dragging the model down and should be dropped.

This is finer than the source-mix profile grid: instead of mixing
broad categories, we test each individual source-tag prefix in
isolation.

Usage:
    python scripts/run_per_source_loo.py
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


TEST_SETS: list[tuple[str, str]] = [
    ("in_domain", "data/calib2000_full_test.jsonl"),
    ("ood_m4", "data/ood_m4_test.jsonl"),
    ("ood_raid", "data/ood_raid_test.jsonl"),
    ("ood_mage", "data/ood_mage_test.jsonl"),
    ("verb_low", "data/sweep_verb_low_test.jsonl"),
    ("ctrl_anthropic_t1", "data/sweep_anthropic_temp_1.jsonl"),
    ("ctrl_openai_med_high", "data/sweep_openai_effort_medium_verb_high.jsonl"),
    ("prompts_heldout", "data/prompts_v5_test.jsonl"),
    ("prompts_fresh", "data/prompts_test_fresh.jsonl"),
]


def source_prefix(src: str) -> str:
    """First two underscore-separated segments — e.g. hc3_finance, prompt_alpaca, domain_gsm8k."""
    parts = src.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return src


def write_loo_train(master_path: Path, out_path: Path, exclude_prefix: str) -> int:
    n = 0
    with open(master_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = row.get("source", "")
            if source_prefix(src) == exclude_prefix:
                continue
            fout.write(line + "\n")
            n += 1
    return n


def write_config(path: Path, name: str, det: Path, ebm: Path, conf: Path) -> None:
    body = (
        f"_profile: loo_{name}\n"
        f"_extends: default\n"
        f"fusion:\n"
        f"  model_path: {ebm.as_posix()}\n"
        f"  conformal_path: {conf.as_posix()}\n"
        f"  detector_calibration_path: {det.as_posix()}\n"
    )
    path.write_text(body, encoding="utf-8")


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"command failed (rc={res.returncode}): {cmd}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-train", default="data/calib2000_v6_train.jsonl")
    ap.add_argument("--master-calib", default="data/calib2000_v2_calib.jsonl")
    ap.add_argument("--features-cache", default="data/all_features_v6.jsonl")
    ap.add_argument("--out-models-dir", default="models/calib2000/loo")
    ap.add_argument("--out-bench-dir", default="docs/benchmarks/calib2000/loo")
    ap.add_argument("--summary-csv", default="docs/benchmarks/calib2000/loo_summary.csv")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--min-source-rows", type=int, default=200)
    args = ap.parse_args()

    Path(args.out_models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_bench_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    print("inventorying source-tag prefixes...", flush=True)
    counts: Counter[str] = Counter()
    with open(args.master_train, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts[source_prefix(row.get("source", ""))] += 1
    sources = [s for s, n in counts.most_common() if n >= args.min_source_rows]
    print(f"  {len(sources)} sources >= {args.min_source_rows} rows", flush=True)
    for s in sources:
        print(f"    {s:<30} {counts[s]:>6}")

    csv_exists = Path(args.summary_csv).exists()
    csv_handle = open(args.summary_csv, "a", encoding="utf-8", newline="")
    writer = csv.writer(csv_handle)
    if not csv_exists:
        cols = ["excluded_source", "n_excluded", "n_train", "duration_s"]
        for tn, _ in TEST_SETS:
            cols.extend([f"{tn}_auroc", f"{tn}_ece", f"{tn}_tpr01", f"{tn}_f1"])
        writer.writerow(cols)
        csv_handle.flush()

    for src in sources:
        sname = src.replace("/", "_")
        sdir = Path(args.out_models_dir) / sname
        sdir.mkdir(parents=True, exist_ok=True)
        bdir = Path(args.out_bench_dir) / sname
        bdir.mkdir(parents=True, exist_ok=True)

        train_f = sdir / "train.jsonl"
        n_train = write_loo_train(Path(args.master_train), train_f, src)
        n_excluded = counts[src]

        det_p = sdir / "detector_calibration.json"
        ebm_p = sdir / "ebm.pkl"
        conf_p = sdir / "conformal.json"
        cfg_p = Path("configs") / f".loo_{sname}.yaml"
        write_config(cfg_p, sname, det_p, ebm_p, conf_p)

        print(f"\n=== LOO-{src}  (n_train={n_train}, excluded={n_excluded}) ===", flush=True)
        t0 = time.monotonic()
        try:
            run([args.python, "-u", "scripts/calibrate_detectors.py",
                 "--dataset", str(train_f), "--config", str(cfg_p),
                 "--out", str(det_p), "--features-cache", args.features_cache])
            run([args.python, "-u", "scripts/train_fusion.py",
                 "--dataset", str(train_f), "--config", str(cfg_p),
                 "--out", str(ebm_p), "--features-cache", args.features_cache])
            run([args.python, "-u", "scripts/calibrate_conformal.py",
                 "--dataset", str(args.master_calib), "--config", str(cfg_p),
                 "--alpha", "0.05", "--out", str(conf_p),
                 "--features-cache", args.features_cache])
        except Exception as e:
            print(f"[FAIL train] {e}", flush=True)
            continue

        bench: dict[str, dict] = {}
        for tn, tp in TEST_SETS:
            outj = bdir / f"{tn}.json"
            try:
                run([args.python, "-u", "scripts/run_benchmark.py",
                     "--dataset", tp, "--config", str(cfg_p),
                     "--out", str(outj), "--features-cache", args.features_cache])
                r = json.loads(outj.read_text())
                bench[tn] = {
                    "auroc": r.get("metrics", {}).get("auroc"),
                    "ece": r.get("metrics", {}).get("ece"),
                    "tpr01": r.get("metrics", {}).get("tpr_at_fpr_01"),
                    "f1": r.get("confusion", {}).get("f1"),
                }
            except Exception as e:
                print(f"[FAIL bench {tn}] {e}", flush=True)
                bench[tn] = {}

        dt = time.monotonic() - t0
        row = [src, n_excluded, n_train, round(dt, 1)]
        for tn, _ in TEST_SETS:
            r = bench.get(tn, {})
            row.extend([r.get("auroc"), r.get("ece"), r.get("tpr01"), r.get("f1")])
        writer.writerow(row)
        csv_handle.flush()
        print(f"[done] LOO-{src}: in_domain_auroc={bench.get('in_domain', {}).get('auroc')}, dt={dt:.1f}s",
              flush=True)

    csv_handle.close()
    print(f"\nLOO analysis complete -> {args.summary_csv}", flush=True)


if __name__ == "__main__":
    main()

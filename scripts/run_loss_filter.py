"""Approach 1: loss-based training-point filtering.

Score every row of the master training corpus with an existing trained
profile (default: general_hc3). For each row, compute the model's loss
on that row's label. Then build several filtered training subsets:

  - high_loss_top30  : keep top 30% rows by per-row loss
  - high_loss_top50  : top 50%
  - uncertain        : keep rows where calibrated p_llm in [0.3, 0.7]
  - exclude_easy     : drop rows where the model is very confident +
                       correct (loss < 0.1)

Train a new EBM/calibrator/conformal on each filtered subset; benchmark
on the standard 9 test sets; tabulate.

Hypothesis: smaller corpora of "harder" examples often outperform the
full corpus because the EBM's interaction terms are dominated by
redundant easy cases.

Usage:
    python scripts/run_loss_filter.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from cairn.calibration import DetectorCalibrator
from cairn.dotenv import load_dotenv
from cairn.feature_cache import load_cache, text_hash
from cairn.fusion.ebm import FeatureAssembler
from cairn.fusion.serialization import load_model
from cairn.pipeline import normalize_text


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


def score_corpus(rows: list[dict], cache: dict, calibrator: DetectorCalibrator,
                 ebm) -> list[float]:
    assembler = FeatureAssembler()
    feature_names = getattr(ebm, "_cairn_feature_names", None)
    if not feature_names:
        from cairn.fusion.ebm import ASSEMBLER_FEATURE_NAMES
        feature_names = list(ASSEMBLER_FEATURE_NAMES)

    import numpy as np
    from cairn.contracts import LayerResult, RouterDecision
    preds: list[float] = []
    for r in rows:
        h = text_hash(normalize_text(r["text"]))
        rec = cache.get(h)
        if rec is None:
            preds.append(0.5)
            continue
        results = []
        for d in rec["results"]:
            lr = LayerResult(**d)
            if calibrator.has(lr.layer_id) and math.isfinite(lr.raw_score):
                lr.p_llm = calibrator.transform(lr.layer_id, lr.raw_score)
                lr.calibrator_applied = True
            results.append(lr)
        rd = RouterDecision(**rec["router"])
        vec = assembler.assemble(results, word_count=rd.word_count, domain=rd.domain)
        x_row = []
        for f in feature_names:
            v = vec.get(f, 0.5)
            if isinstance(v, float) and math.isnan(v):
                v = 0.5
            x_row.append(v)
        x = np.array([x_row])
        try:
            p = float(ebm.predict_proba(x)[0, 1])
        except Exception:
            p = 0.5
        preds.append(p)
    return preds


def write_config(path: Path, name: str, det: Path, ebm: Path, conf: Path) -> None:
    body = (
        f"_profile: lossfilter_{name}\n"
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
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-train", default="data/calib2000_v6_train.jsonl")
    ap.add_argument("--master-calib", default="data/calib2000_v2_calib.jsonl")
    ap.add_argument("--features-cache", default="data/all_features_v6.jsonl")
    ap.add_argument("--scorer-profile", default="general_hc3")
    ap.add_argument("--out-models-dir", default="models/calib2000/loss_filter")
    ap.add_argument("--out-bench-dir", default="docs/benchmarks/calib2000/loss_filter")
    ap.add_argument("--summary-csv", default="docs/benchmarks/calib2000/loss_filter_summary.csv")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    Path(args.out_models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_bench_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    print("loading scorer profile + cache...", flush=True)
    scorer_dir = Path("models/calib2000/profiles") / args.scorer_profile
    cal = DetectorCalibrator()
    cal.load(scorer_dir / "detector_calibration.json")
    ebm, _meta = load_model(scorer_dir / "ebm.pkl")
    cache = load_cache(Path(args.features_cache))
    print(f"  cache: {len(cache)} records", flush=True)

    print("loading master corpus...", flush=True)
    rows: list[dict] = []
    with open(args.master_train, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"  {len(rows)} rows", flush=True)

    print("scoring corpus...", flush=True)
    t0 = time.monotonic()
    preds = score_corpus(rows, cache, cal, ebm)
    print(f"  scored in {time.monotonic() - t0:.1f}s", flush=True)

    losses: list[float] = []
    eps = 1e-9
    for r, p in zip(rows, preds):
        y = int(r["label"])
        p_clipped = max(eps, min(1 - eps, p))
        loss = -math.log(p_clipped) if y == 1 else -math.log(1 - p_clipped)
        losses.append(loss)
    sorted_losses = sorted(losses)
    print(f"  loss percentiles: p25={sorted_losses[len(losses)//4]:.3f}  "
          f"p50={sorted_losses[len(losses)//2]:.3f}  "
          f"p75={sorted_losses[3*len(losses)//4]:.3f}  "
          f"max={sorted_losses[-1]:.3f}", flush=True)

    indexed = list(zip(rows, preds, losses))
    by_loss_desc = sorted(indexed, key=lambda x: -x[2])
    strategies = {
        "high_loss_top30": [t[0] for t in by_loss_desc[:int(0.30 * len(rows))]],
        "high_loss_top50": [t[0] for t in by_loss_desc[:int(0.50 * len(rows))]],
        "uncertain":       [r for r, p, l in indexed if 0.3 <= p <= 0.7],
        "exclude_easy":    [r for r, p, l in indexed if l >= 0.1],
    }
    for name, sub in strategies.items():
        print(f"  {name}: {len(sub)} rows", flush=True)

    csv_exists = Path(args.summary_csv).exists()
    csv_handle = open(args.summary_csv, "a", encoding="utf-8", newline="")
    writer = csv.writer(csv_handle)
    if not csv_exists:
        cols = ["strategy", "n_train", "duration_s"]
        for tn, _ in TEST_SETS:
            cols.extend([f"{tn}_auroc", f"{tn}_ece", f"{tn}_tpr01", f"{tn}_f1"])
        writer.writerow(cols)
        csv_handle.flush()

    for sname, subset in strategies.items():
        if not subset:
            continue
        sdir = Path(args.out_models_dir) / sname
        sdir.mkdir(parents=True, exist_ok=True)
        bdir = Path(args.out_bench_dir) / sname
        bdir.mkdir(parents=True, exist_ok=True)

        train_f = sdir / "train.jsonl"
        with open(train_f, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")

        det_p = sdir / "detector_calibration.json"
        ebm_p = sdir / "ebm.pkl"
        conf_p = sdir / "conformal.json"
        cfg_p = Path("configs") / f".lossfilter_{sname}.yaml"
        write_config(cfg_p, sname, det_p, ebm_p, conf_p)

        print(f"\n=== {sname}  (n_train={len(subset)}) ===", flush=True)
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
        row = [sname, len(subset), round(dt, 1)]
        for tn, _ in TEST_SETS:
            r = bench.get(tn, {})
            row.extend([r.get("auroc"), r.get("ece"), r.get("tpr01"), r.get("f1")])
        writer.writerow(row)
        csv_handle.flush()
        print(f"[done] {sname}: in_domain_auroc={bench.get('in_domain', {}).get('auroc')}, dt={dt:.1f}s",
              flush=True)

    csv_handle.close()
    print(f"\nloss-filter analysis complete -> {args.summary_csv}", flush=True)


if __name__ == "__main__":
    main()

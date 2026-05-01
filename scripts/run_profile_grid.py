"""Train + benchmark specialized calibration profiles by source-tag filter.

Builds N calibration profiles, each from a different subset of the
master training corpus. Each profile gets its own EBM/calibrator/
conformal artifact and a row in a summary CSV.

The intent is to find which subsets dominate which test distributions
— surfacing de-facto routing rules between profiles.

Usage:
    python scripts/run_profile_grid.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path


PROFILES: list[dict] = [
    {"name": "general_hc3", "patterns": [r"^hc3_"], "note": "HC3 only (calib2000 baseline)"},
    {"name": "general_full", "patterns": [r"^hc3_", r"^raid_", r"^m4_", r"^mage$"],
     "note": "HC3 + OOD (raid/m4/mage)"},
    {"name": "stem", "patterns": [r"^domain_arxiv", r"^domain_gsm8k", r"^domain_humaneval"],
     "note": "v6 STEM domain corpus (math/code/sci)"},
    {"name": "stem_with_prompts", "patterns": [r"^domain_arxiv", r"^domain_gsm8k", r"^domain_humaneval",
                                                r"^prompt_gsm8k", r"^prompt_humaneval", r"^prompt_mmlu",
                                                r"^prompt_bbh", r"^prompt_gpqa"],
     "note": "STEM + math/code/reasoning prompts"},
    {"name": "medical", "patterns": [r"^hc3_medicine", r"^prompt_medqa"],
     "note": "Medical-domain (HC3 medicine + MedQA prompts)"},
    {"name": "finance", "patterns": [r"^hc3_finance"],
     "note": "Finance-domain (HC3 finance only)"},
    {"name": "short_form", "patterns": [r"_concise$", r"_verb_low$"],
     "note": "Concise regens + verb_low cells"},
    {"name": "prompts_only", "patterns": [r"^prompt_"],
     "note": "Prompts-only profile"},
    {"name": "chat_prompts", "patterns": [r"^prompt_alpaca", r"^prompt_magpie",
                                           r"^prompt_slimorca", r"^prompt_wizardlm",
                                           r"^prompt_hh", r"^prompt_hc3"],
     "note": "Conversational prompts only"},
    {"name": "general_plus_stem", "patterns": [r"^hc3_", r"^raid_", r"^m4_", r"^mage$",
                                                r"^domain_arxiv", r"^domain_gsm8k", r"^domain_humaneval"],
     "note": "HC3 + OOD + STEM"},
    {"name": "general_plus_short", "patterns": [r"^hc3_", r"^raid_", r"^m4_", r"^mage$",
                                                 r"_concise$", r"_verb_low$"],
     "note": "HC3 + OOD + verb_low + concise"},
    {"name": "general_plus_medical", "patterns": [r"^hc3_", r"^raid_", r"^m4_", r"^mage$",
                                                   r"^prompt_medqa"],
     "note": "HC3 + OOD + MedQA prompts"},
    {"name": "everything", "patterns": [r".*"],
     "note": "All training data (full v6 — for reference)"},
]


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


def filter_train(src: Path, dst: Path, patterns: list[str]) -> int:
    compiled = [re.compile(p) for p in patterns]
    n = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = row.get("source", "")
            if any(c.search(source) for c in compiled):
                fout.write(line + "\n")
                n += 1
    return n


def write_config(path: Path, name: str, det: Path, ebm: Path, conf: Path) -> None:
    body = (
        f"_profile: profile_{name}\n"
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
    ap.add_argument("--out-models-dir", default="models/calib2000/profiles")
    ap.add_argument("--out-bench-dir", default="docs/benchmarks/calib2000/profiles")
    ap.add_argument("--summary-csv", default="docs/benchmarks/calib2000/profile_grid_summary.csv")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    Path(args.out_models_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_bench_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)

    profiles = PROFILES
    if args.only:
        profiles = [p for p in PROFILES if p["name"] in set(args.only)]

    csv_exists = Path(args.summary_csv).exists()
    done_names: set[str] = set()
    if csv_exists and args.skip_existing:
        with open(args.summary_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done_names.add(r["profile"])

    csv_handle = open(args.summary_csv, "a", encoding="utf-8", newline="")
    writer = csv.writer(csv_handle)
    if not csv_exists:
        cols = ["profile", "patterns", "n_train", "duration_s"]
        for name, _ in TEST_SETS:
            cols.extend([f"{name}_auroc", f"{name}_ece", f"{name}_tpr01", f"{name}_f1"])
        writer.writerow(cols)
        csv_handle.flush()

    for prof in profiles:
        name = prof["name"]
        if name in done_names:
            print(f"[skip] {name} already in summary", flush=True)
            continue

        prof_dir = Path(args.out_models_dir) / name
        prof_dir.mkdir(parents=True, exist_ok=True)
        bench_dir = Path(args.out_bench_dir) / name
        bench_dir.mkdir(parents=True, exist_ok=True)

        train_f = prof_dir / "train.jsonl"
        n_train = filter_train(Path(args.master_train), train_f, prof["patterns"])

        det_path = prof_dir / "detector_calibration.json"
        ebm_path = prof_dir / "ebm.pkl"
        conf_path = prof_dir / "conformal.json"
        cfg_path = Path("configs") / f".profile_{name}.yaml"
        write_config(cfg_path, name, det_path, ebm_path, conf_path)

        print(f"\n=== {name}  (n_train={n_train})  patterns={prof['patterns']} ===", flush=True)
        t0 = time.monotonic()

        try:
            run([args.python, "-u", "scripts/calibrate_detectors.py",
                 "--dataset", str(train_f), "--config", str(cfg_path),
                 "--out", str(det_path), "--features-cache", args.features_cache])
            run([args.python, "-u", "scripts/train_fusion.py",
                 "--dataset", str(train_f), "--config", str(cfg_path),
                 "--out", str(ebm_path), "--features-cache", args.features_cache])
            run([args.python, "-u", "scripts/calibrate_conformal.py",
                 "--dataset", str(args.master_calib), "--config", str(cfg_path),
                 "--alpha", str(args.alpha), "--out", str(conf_path),
                 "--features-cache", args.features_cache])
        except Exception as e:
            print(f"[FAIL train] {name}: {e}", flush=True)
            continue

        bench_results: dict[str, dict] = {}
        for tname, tpath in TEST_SETS:
            out_json = bench_dir / f"{tname}.json"
            try:
                run([args.python, "-u", "scripts/run_benchmark.py",
                     "--dataset", tpath, "--config", str(cfg_path),
                     "--out", str(out_json), "--features-cache", args.features_cache])
                r = json.loads(out_json.read_text())
                bench_results[tname] = {
                    "auroc": r.get("metrics", {}).get("auroc"),
                    "ece": r.get("metrics", {}).get("ece"),
                    "tpr01": r.get("metrics", {}).get("tpr_at_fpr_01"),
                    "f1": r.get("confusion", {}).get("f1"),
                }
            except Exception as e:
                print(f"[FAIL bench {tname}] {e}", flush=True)
                bench_results[tname] = {}

        duration = time.monotonic() - t0
        row = [name, "+".join(prof["patterns"]), n_train, round(duration, 1)]
        for tname, _ in TEST_SETS:
            r = bench_results.get(tname, {})
            row.extend([r.get("auroc"), r.get("ece"), r.get("tpr01"), r.get("f1")])
        writer.writerow(row)
        csv_handle.flush()
        print(f"[done] {name}: n_train={n_train}, "
              f"in_domain_auroc={bench_results.get('in_domain', {}).get('auroc')}, "
              f"duration={duration:.1f}s", flush=True)

    csv_handle.close()
    print(f"\nprofile grid complete -> {args.summary_csv}", flush=True)


if __name__ == "__main__":
    main()

# scripts/run_raid_benchmark.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import load_dataset
from cairn.evaluation.metrics import tpr_at_fpr_ceiling
from cairn.evaluation.runner import run_eval
from cairn.pipeline import CairnPipeline


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--fpr-ceiling", type=float, default=0.01)
    ap.add_argument("--min-tpr", type=float, default=None,
                    help="Exit non-zero if TPR is below this floor (CI gate).")
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()

    pipeline = CairnPipeline.from_config_file(args.config)
    samples = load_dataset(args.dataset)
    report = run_eval(pipeline, samples, progress=True)

    scores = [p["p_llm"] for p in report.predictions]
    labels = [p["label"] for p in report.predictions]
    tpr, threshold, achieved_fpr = tpr_at_fpr_ceiling(scores, labels, ceiling=args.fpr_ceiling)
    summary = {
        "headline": {"metric": f"TPR@FPR<={args.fpr_ceiling}", "value": tpr},
        "n_samples": report.n_samples,
        "fpr_ceiling": args.fpr_ceiling,
        "threshold": threshold,
        "achieved_fpr": achieved_fpr,
        "auroc": report.metrics.get("auroc"),
        "per_source": report.per_source,
        "profile_used": str(args.config),
    }
    print(f"RAID TPR@FPR<={args.fpr_ceiling}: {tpr:.4f}")
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
    if args.min_tpr is not None and tpr < args.min_tpr:
        print(f"ERROR: TPR {tpr:.3f} below floor {args.min_tpr:.3f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

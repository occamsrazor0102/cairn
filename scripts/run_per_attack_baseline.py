# scripts/run_per_attack_baseline.py
"""Per-attack baseline runner — produces the regression gate reference.

Wave 2 / A2 step 1: write
``docs/benchmarks/calib2000/per_attack_auroc.csv`` under the *current*
``master_minus_arxiv`` profile so the post-retrain gate has a pre-A2
reference. Existing 5 transforms in
``cairn/adversarial/transforms.py`` (paraphrase_launder, synonym_swap,
inject_typos, casualize, mix_human) are run via
``cairn.evaluation.robustness.run_robustness_eval``; per-attack AUROC
(plus deltas vs. the no-attack baseline) is written one row per attack
to the CSV.

Provider-based attacks (paraphrase_launder, coached_casual) require an
LLM provider callable. They are SKIPPED by default to keep this baseline
zero-API-spend; pass ``--include-prompt-attacks`` (with ``--provider``
configured) to include them.

Calibrated EBM artifacts under ``models/calib2000/profiles/`` are still
in the legacy pickle format pending the skops migration. To load them
during this run, set ``CAIRN_ALLOW_LEGACY_PICKLE=1`` in the environment
before invoking this script — otherwise the pipeline silently degrades
to naive fusion and the produced "baseline" is not the calibrated
reference the regression gate needs. See ``cairn/fusion/serialization.py``
for the loader behaviour.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from cairn.dotenv import load_dotenv
# `cairn.evaluation.robustness` must be imported BEFORE
# `cairn.adversarial.registry` to bootstrap the `cairn.evaluation`
# package completely. Reversing the order triggers a circular import
# (adversarial.__init__ → adversarial.generator → evaluation.dataset →
# evaluation.__init__ → evaluation.robustness → adversarial.generator
# (partial)). Loading evaluation first makes evaluation.dataset
# available to generator.py before the adversarial package is touched.
from cairn.evaluation.dataset import load_dataset
from cairn.evaluation.robustness import run_robustness_eval
from cairn.adversarial.registry import get_attack
from cairn.pipeline import CairnPipeline


# Plan-mandated transform set: the five attacks named in
# `docs/superpowers/specs/2026-05-07-post-honest-calibration-roadmap.md`
# Wave 2 step 1. Names map to registered attack ids in
# `cairn/adversarial/transforms.py`.
_DEFAULT_ATTACKS = (
    "paraphrase_launder",  # plan: "paraphrase"
    "synonym_swap",
    "inject_typos",        # plan: "typo"
    "casualize",
    "mix_human",
)


def _make_provider(name: str | None):
    """Build a provider callable for prompt-category attacks.

    Only ``anthropic`` is wired today — the existing roadmap names that
    provider as the default LLM gateway. Returning ``None`` here means
    "no provider": prompt-category attacks must be skipped.
    """
    if name is None:
        return None
    if name == "anthropic":
        # Lazy import — keeps the base install slim.
        from cairn.providers.anthropic import provider_callable
        return provider_callable()
    raise ValueError(
        f"unknown provider {name!r}; supported: anthropic"
    )


def _filter_attacks(attacks: list[str], *, include_prompt: bool, provider) -> list[str]:
    """Drop unknown attacks; drop prompt-category attacks when no
    provider is wired. Print one line per drop so the operator sees
    why the CSV is missing rows."""
    out: list[str] = []
    for name in attacks:
        attack = get_attack(name)
        if attack is None:
            print(f"[skip] unknown attack: {name}", file=sys.stderr)
            continue
        if attack.category == "prompt":
            if not include_prompt or provider is None:
                print(
                    f"[skip] prompt-category attack {name!r} "
                    f"(needs --include-prompt-attacks + --provider)",
                    file=sys.stderr,
                )
                continue
        out.append(name)
    return out


def _row(attack_name: str, metrics: dict, n_samples: int,
         baseline_auroc: float | None) -> dict:
    auroc = metrics.get("auroc")
    delta = (
        None
        if (auroc is None or baseline_auroc is None)
        else float(auroc) - float(baseline_auroc)
    )
    return {
        "attack": attack_name,
        "n_samples": n_samples,
        "auroc": auroc,
        "ece": metrics.get("ece"),
        "brier": metrics.get("brier"),
        "tpr_at_fpr_01": metrics.get("tpr_at_fpr_01"),
        "baseline_auroc": baseline_auroc,
        "delta_auroc": delta,
    }


def _format(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, type=Path,
                    help="JSONL eval dataset (id, text, label, ...)")
    ap.add_argument("--config", required=True, type=Path,
                    help="Pipeline config YAML — typically the master_minus_arxiv profile")
    ap.add_argument(
        "--attacks", default=",".join(_DEFAULT_ATTACKS),
        help="Comma-separated attack names. Default: the plan's 5 transforms.",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap dataset to first N rows. Default: full dataset.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default=Path("docs/benchmarks/calib2000/per_attack_auroc.csv"),
                    type=Path,
                    help="Output CSV path. Parent directory is created if missing.")
    ap.add_argument("--include-prompt-attacks", action="store_true",
                    help="Run prompt-category attacks (paraphrase_launder, "
                         "coached_casual). Requires --provider.")
    ap.add_argument("--provider", default=None,
                    help="Provider for prompt-category attacks. "
                         "Currently supports: anthropic")
    ap.add_argument("--out-json", default=None, type=Path,
                    help="Optional sidecar JSON with full RobustnessReport.")
    args = ap.parse_args()

    samples = load_dataset(args.dataset)
    if args.limit is not None and args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        print("ERROR: empty dataset (after --limit)", file=sys.stderr)
        return 2

    provider = _make_provider(args.provider)
    requested_attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    attacks = _filter_attacks(
        requested_attacks,
        include_prompt=args.include_prompt_attacks,
        provider=provider,
    )
    if not attacks:
        print("ERROR: no runnable attacks after filtering", file=sys.stderr)
        return 2

    pipeline = CairnPipeline.from_config_file(args.config)
    print(f"Loaded pipeline from {args.config}; running on {len(samples)} samples "
          f"with {len(attacks)} attacks: {attacks}")

    report = run_robustness_eval(
        pipeline,
        samples,
        attacks,
        provider=provider,
        seed=args.seed,
        progress=True,
    )

    baseline_auroc = report.baseline.metrics.get("auroc")
    rows = [_row("(none)", report.baseline.metrics,
                 report.baseline.n_samples, baseline_auroc)]
    for name in attacks:
        sub = report.per_attack.get(name)
        if sub is None:
            continue
        rows.append(_row(name, sub.metrics, sub.n_samples, baseline_auroc))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "attack", "n_samples", "auroc", "ece", "brier",
        "tpr_at_fpr_01", "baseline_auroc", "delta_auroc",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for row in rows:
            w.writerow([_format(row[k]) for k in fields])

    print(f"Wrote {len(rows)} rows to {args.out_csv}")
    print(f"Baseline AUROC (no attack): {_format(baseline_auroc)}")
    for row in rows[1:]:
        print(
            f"  {row['attack']:<20s} "
            f"AUROC={_format(row['auroc'])}  "
            f"delta={_format(row['delta_auroc'])}  "
            f"n={row['n_samples']}"
        )

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "baseline": {
                "n_samples": report.baseline.n_samples,
                "metrics": report.baseline.metrics,
            },
            "per_attack": {
                name: {
                    "n_samples": rep.n_samples,
                    "metrics": rep.metrics,
                }
                for name, rep in report.per_attack.items()
            },
            "vulnerability_ranking": [
                {"attack": n, "delta_auroc": d}
                for n, d in report.vulnerability_ranking
            ],
            "config_path": str(args.config),
            "dataset_path": str(args.dataset),
            "n_samples": len(samples),
            "seed": args.seed,
        }, indent=2))
        print(f"Wrote sidecar JSON to {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

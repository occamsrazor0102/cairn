"""Run deterministic TOCSIN candidate/comparator promotion evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from cairn.config import load_config
from cairn.evaluation.tocsin_retrain import (
    EvaluatedRow,
    ProfileEvaluation,
    build_humanized_hard_slice,
    decide_promotion,
    evaluate_detector_calibration,
    evaluate_profile,
    load_evaluation_slice,
    preflight_profiles,
    profile_name,
)


_OOD_DATASETS = ("ood_m4_test", "ood_raid_test", "ood_mage_test")
_FULL_REPORTS = (
    "detector_calibration.json",
    "benchmarks.json",
    "benchmarks_long.csv",
    "july10_per_task.csv",
    "promotion_decision.json",
    "PROMOTION_REPORT.md",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument(
        "--baseline-config", type=Path, action="append", required=True
    )
    parser.add_argument("--prompt-test", type=Path, nargs=2, required=True)
    parser.add_argument("--humanized", type=Path, nargs=2)
    parser.add_argument("--hard-human", type=Path, nargs=2)
    parser.add_argument("--ood-m4", type=Path, nargs=2)
    parser.add_argument("--ood-raid", type=Path, nargs=2)
    parser.add_argument("--ood-mage", type=Path, nargs=2)
    parser.add_argument("--challenge", type=Path, nargs=2)
    parser.add_argument("--stale-calibration", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--diagnostic-on-invalid-comparator", action="store_true"
    )
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _profile_report(evaluation: ProfileEvaluation) -> dict[str, object]:
    return {
        "prompt_test": evaluation.metrics["prompt_test"],
        "humanized_hard": evaluation.metrics["humanized_hard"],
        "ood": {
            corpus: evaluation.metrics[corpus] for corpus in _OOD_DATASETS
        },
    }


def _scalar_metric_rows(
    role: str,
    profile: str,
    metrics: Mapping[str, object],
):
    for dataset, block in metrics.items():
        if dataset == "ood":
            for corpus, corpus_metrics in block.items():
                for metric, value in corpus_metrics.items():
                    if isinstance(value, (bool, int, float)) and math.isfinite(
                        float(value)
                    ):
                        yield [role, profile, corpus, metric, value]
            continue
        if not isinstance(block, Mapping):
            continue
        for metric, value in block.items():
            if isinstance(value, (bool, int, float)) and math.isfinite(float(value)):
                yield [role, profile, dataset, metric, value]


def _write_benchmarks_long(
    path: Path,
    candidate_name: str,
    candidate: Mapping[str, object],
    comparators: Mapping[str, Mapping[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["role", "profile", "dataset", "metric", "value"])
        writer.writerows(
            _scalar_metric_rows("candidate", candidate_name, candidate)
        )
        for name in sorted(comparators):
            writer.writerows(
                _scalar_metric_rows("comparator", name, comparators[name])
            )


def _task_id(row: EvaluatedRow) -> str:
    source_id = row.source_id or row.id
    return source_id.split("|", 1)[0].removeprefix("id=july10_")


def _challenge_rows(
    candidate: ProfileEvaluation,
    comparators: Mapping[str, ProfileEvaluation],
) -> list[dict[str, object]]:
    candidate_rows = candidate.rows["challenge"]
    current_name = (
        "prompts_only"
        if "prompts_only" in comparators
        else (sorted(comparators)[0] if comparators else None)
    )
    current_by_id = (
        {row.id: row for row in comparators[current_name].rows["challenge"]}
        if current_name is not None
        else {}
    )
    rows: list[dict[str, object]] = []
    for candidate_row in candidate_rows:
        current = current_by_id.get(candidate_row.id)
        rows.append(
            {
                "task_id": _task_id(candidate_row),
                "label": candidate_row.label,
                "current_profile": current_name,
                "current_p_llm": current.p_llm if current is not None else None,
                "candidate_p_llm": candidate_row.p_llm,
                "delta_candidate_minus_current": (
                    candidate_row.p_llm - current.p_llm
                    if current is not None
                    else None
                ),
            }
        )
    return rows


def _write_challenge_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "task_id",
        "label",
        "current_profile",
        "current_p_llm",
        "candidate_p_llm",
        "delta_candidate_minus_current",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(
    decision: Mapping[str, object],
    report: Mapping[str, object],
) -> str:
    lines = [f"# {decision['status']}", "", "## Preflight", ""]
    preflight = report.get("preflight") or {}
    candidate = preflight.get("candidate") or {}
    lines.append(
        f"- Candidate `{candidate.get('name', 'unknown')}`: "
        f"{'VALID' if candidate.get('valid') else 'INVALID'}"
    )
    for name, record in sorted((preflight.get("comparators") or {}).items()):
        lines.append(
            f"- Comparator `{name}`: {'VALID' if record.get('valid') else 'INVALID'}"
        )
    lines.extend(["", "## Automatic promotion gates", ""])
    failures = decision.get("failures") or []
    if failures:
        lines.extend(f"- FAIL: {failure.replace(chr(10), '; ')}" for failure in failures)
    else:
        lines.append("- All locked automatic gates passed.")

    detector = report.get("detector_calibration")
    if detector:
        lines.extend(
            [
                "",
                "## Detector calibration",
                "",
                f"- Eligible rows: {detector['eligible_rows']}",
                f"- Detector SKIP rows: {detector['detector_skip_rows']}",
                f"- Support coverage: {detector['support_coverage']:.6f}",
                f"- Raw combined_w range: {detector['raw']['min']:.6g} to {detector['raw']['max']:.6g}",
                f"- Candidate Brier: {detector['candidate']['brier']:.6f}",
                f"- Legacy unsafe clipped diagnostic Brier: {detector['legacy_unsafe_clip_diagnostic']['brier']:.6f}",
            ]
        )

    lines.extend(["", "## Adaptive July 10 challenge (secondary)", ""])
    challenge = report.get("challenge")
    if challenge:
        cm = challenge["candidate"]
        lines.extend(
            [
                "These results are serialized for review and never enter automatic promotion logic.",
                "",
                f"- Brier: {cm['brier']:.6f}",
                f"- Balanced accuracy: {cm['balanced_accuracy']:.6f}",
                f"- Human specificity: {cm['specificity']:.6f}",
                f"- Conformal truth coverage: {cm['conformal_truth_coverage']:.6f}",
            ]
        )
    else:
        reason = report.get("challenge_not_run_reason") or "challenge_not_provided"
        lines.append(f"Not run: `{reason}`.")
    return "\n".join(lines) + "\n"


def _write_decision_reports(
    out_dir: Path,
    decision: Mapping[str, object],
    report: Mapping[str, object],
) -> None:
    _write_json(out_dir / "promotion_decision.json", decision)
    (out_dir / "PROMOTION_REPORT.md").write_text(
        _markdown_report(decision, report), encoding="utf-8"
    )


def _complete_empty_reports(out_dir: Path, report: Mapping[str, object]) -> None:
    _write_json(out_dir / "detector_calibration.json", {})
    _write_json(out_dir / "benchmarks.json", report)
    with (out_dir / "benchmarks_long.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        handle.write("role,profile,dataset,metric,value\n")
    _write_challenge_csv(out_dir / "july10_per_task.csv", [])


def _required_full_arguments(args: argparse.Namespace) -> list[str]:
    missing: list[str] = []
    for attribute in (
        "humanized",
        "hard_human",
        "ood_m4",
        "ood_raid",
        "ood_mage",
        "stale_calibration",
    ):
        if getattr(args, attribute) is None:
            missing.append("--" + attribute.replace("_", "-"))
    return missing


def _valid_baseline_paths(
    baseline_paths: Sequence[Path], preflight: Mapping[str, object]
) -> list[Path]:
    records = preflight.get("comparators") or {}
    return [
        path
        for path in baseline_paths
        if (records.get(profile_name(path)) or {}).get("valid")
    ]


def _candidate_calibration_path(config_path: Path) -> Path:
    value = (load_config(config_path).get("fusion") or {}).get(
        "detector_calibration_path"
    )
    if not value:
        raise ValueError("candidate profile lacks detector_calibration_path")
    return Path(value)


def _run_full(args: argparse.Namespace, preflight: dict[str, object]) -> tuple[dict, dict]:
    prompt_samples = load_evaluation_slice(args.prompt_test[0])
    humanized_samples = load_evaluation_slice(args.humanized[0])
    hard_human_samples = load_evaluation_slice(args.hard_human[0])
    combined = build_humanized_hard_slice(humanized_samples, hard_human_samples)
    positives = [sample for sample in combined if sample.label == 1]
    negatives = [sample for sample in combined if sample.label == 0]
    ood_samples = {
        "ood_m4_test": load_evaluation_slice(args.ood_m4[0]),
        "ood_raid_test": load_evaluation_slice(args.ood_raid[0]),
        "ood_mage_test": load_evaluation_slice(args.ood_mage[0]),
    }
    datasets = {
        "prompt_test": [(prompt_samples, args.prompt_test[1])],
        "humanized_hard": [
            (positives, args.humanized[1]),
            (negatives, args.hard_human[1]),
        ],
        "ood_m4_test": [(ood_samples["ood_m4_test"], args.ood_m4[1])],
        "ood_raid_test": [(ood_samples["ood_raid_test"], args.ood_raid[1])],
        "ood_mage_test": [(ood_samples["ood_mage_test"], args.ood_mage[1])],
    }
    challenge_samples = None
    if args.challenge is not None:
        challenge_samples = load_evaluation_slice(args.challenge[0])
        datasets["challenge"] = [(challenge_samples, args.challenge[1])]

    candidate_eval = evaluate_profile(args.candidate_config, datasets)
    comparator_evals = {
        profile_name(path): evaluate_profile(path, datasets)
        for path in _valid_baseline_paths(args.baseline_config, preflight)
    }
    detector = evaluate_detector_calibration(
        prompt_samples,
        args.prompt_test[1],
        _candidate_calibration_path(args.candidate_config),
        args.stale_calibration,
    )
    candidate_report = _profile_report(candidate_eval)
    comparator_reports = {
        name: _profile_report(evaluation)
        for name, evaluation in comparator_evals.items()
    }
    report: dict[str, object] = {
        "preflight": preflight,
        "detector_calibration": detector,
        "candidate_name": candidate_eval.name,
        "candidate": candidate_report,
        "comparators": comparator_reports,
        "challenge": None,
    }
    challenge_rows: list[dict[str, object]] = []
    if challenge_samples is not None:
        challenge_rows = _challenge_rows(candidate_eval, comparator_evals)
        report["challenge"] = {
            "adaptive_secondary": True,
            "candidate": candidate_eval.metrics["challenge"],
            "comparators": {
                name: evaluation.metrics["challenge"]
                for name, evaluation in comparator_evals.items()
            },
            "per_task": challenge_rows,
        }
    elif preflight.get("failures"):
        report["challenge_not_run_reason"] = (
            "invalid_comparator_avoided_additional_compute"
        )
    else:
        report["challenge_not_run_reason"] = "challenge_not_provided"

    outputs = {
        "candidate_eval": candidate_eval,
        "comparator_evals": comparator_evals,
        "challenge_rows": challenge_rows,
    }
    return report, outputs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preflight = preflight_profiles(
        args.candidate_config,
        args.baseline_config,
        args.prompt_test[0],
        args.prompt_test[1],
    )
    report: dict[str, object] = {"preflight": preflight}

    if args.preflight_only:
        decision = decide_promotion(report)
        _write_decision_reports(args.out_dir, decision, report)
        return 1 if preflight["failures"] else 0

    if args.diagnostic_on_invalid_comparator and not preflight["failures"]:
        preflight["failures"].append(
            "diagnostic-on-invalid-comparator requires a failed preflight"
        )
        decision = decide_promotion(report)
        _complete_empty_reports(args.out_dir, report)
        _write_decision_reports(args.out_dir, decision, report)
        return 1
    if preflight["failures"] and not args.diagnostic_on_invalid_comparator:
        decision = decide_promotion(report)
        _write_decision_reports(args.out_dir, decision, report)
        return 1

    missing_arguments = _required_full_arguments(args)
    if missing_arguments:
        preflight["failures"].append(
            "missing required benchmark arguments: " + ", ".join(missing_arguments)
        )
        decision = decide_promotion(report)
        _complete_empty_reports(args.out_dir, report)
        _write_decision_reports(args.out_dir, decision, report)
        return 1

    try:
        report, outputs = _run_full(args, preflight)
    except Exception as exc:
        preflight["failures"].append(
            f"evaluation infrastructure failure: {type(exc).__name__}: {exc}"
        )
        report = {"preflight": preflight}
        decision = decide_promotion(report)
        _complete_empty_reports(args.out_dir, report)
        _write_decision_reports(args.out_dir, decision, report)
        return 1

    decision = decide_promotion(report)
    _write_json(args.out_dir / "detector_calibration.json", report["detector_calibration"])
    _write_json(args.out_dir / "benchmarks.json", report)
    _write_benchmarks_long(
        args.out_dir / "benchmarks_long.csv",
        report["candidate_name"],
        report["candidate"],
        report["comparators"],
    )
    _write_challenge_csv(
        args.out_dir / "july10_per_task.csv", outputs["challenge_rows"]
    )
    _write_decision_reports(args.out_dir, decision, report)
    if preflight["failures"]:
        return 1
    return 0 if decision["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

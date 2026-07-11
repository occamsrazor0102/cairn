"""Run deterministic TOCSIN candidate/comparator promotion evaluation."""
from __future__ import annotations

import argparse
import csv
import io
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
_LOCKED_SLICE_ORDER = (
    "prompt_test",
    "humanized_hard",
    "ood_m4_test",
    "ood_raid_test",
    "ood_mage_test",
    "challenge",
)
_PREDICTION_CSV_FIELDS = (
    "profile",
    "slice",
    "id",
    "label",
    "p_llm",
    "predicted_label",
    "conformal_set",
    "fusion_mode",
    "reconstruction_error",
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


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _render_json(payload: object) -> str:
    return (
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _atomic_write_set(out_dir: Path, rendered: Mapping[str, str]) -> None:
    """Render first, then replace each report from a fully written temp file."""

    out_dir.mkdir(parents=True, exist_ok=True)
    temporary: dict[str, Path] = {}
    try:
        for name, content in rendered.items():
            temp_path = out_dir / f".{name}.tmp"
            temp_path.write_text(content, encoding="utf-8", newline="")
            temporary[name] = temp_path
        for name in rendered:
            temporary[name].replace(out_dir / name)
    finally:
        for temp_path in temporary.values():
            if temp_path.exists():
                temp_path.unlink()


def _profile_report(evaluation: ProfileEvaluation) -> dict[str, object]:
    return {
        "prompt_test": evaluation.metrics["prompt_test"],
        "humanized_hard": evaluation.metrics["humanized_hard"],
        "ood": {
            corpus: evaluation.metrics[corpus] for corpus in _OOD_DATASETS
        },
    }


def _valid_probability(value: object) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None
    return probability


def _finite_csv_number(value: object) -> float | str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    return number if math.isfinite(number) else ""


def _serialize_conformal_set(values: Sequence[str] | None) -> str:
    if values is None:
        return ""
    members = {str(value) for value in values}
    canonical = [name for name in ("human", "LLM") if name in members]
    canonical.extend(sorted(members - {"human", "LLM"}))
    return json.dumps(canonical, separators=(",", ":"), ensure_ascii=False)


def _ordered_slices(evaluation: ProfileEvaluation) -> list[str]:
    known = [name for name in _LOCKED_SLICE_ORDER if name in evaluation.rows]
    return known + sorted(set(evaluation.rows) - set(_LOCKED_SLICE_ORDER))


def _render_prediction_rows(evaluations: Sequence[ProfileEvaluation]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
        fieldnames=_PREDICTION_CSV_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for evaluation in evaluations:
        for slice_name in _ordered_slices(evaluation):
            for row in evaluation.rows[slice_name]:
                probability = _valid_probability(row.p_llm)
                writer.writerow(
                    {
                        "profile": evaluation.name,
                        "slice": slice_name,
                        "id": row.id,
                        "label": row.label,
                        "p_llm": probability if probability is not None else "",
                        "predicted_label": (
                            int(probability >= 0.5)
                            if probability is not None
                            else ""
                        ),
                        "conformal_set": _serialize_conformal_set(
                            row.conformal_set
                        ),
                        "fusion_mode": row.fusion_mode,
                        "reconstruction_error": _finite_csv_number(
                            row.reconstruction_error
                        ),
                    }
                )
    return handle.getvalue()


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


def _csv_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return "" if value is None else value


def _render_challenge_csv(rows: Sequence[Mapping[str, object]]) -> str:
    fields = [
        "task_id",
        "label",
        "current_profile",
        "current_p_llm",
        "candidate_p_llm",
        "delta_candidate_minus_current",
    ]
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {key: _csv_safe(value) for key, value in row.items()} for row in rows
    )
    return handle.getvalue()


def _format_metric(value: object, precision: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{precision}f}"


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
        raw = detector.get("raw") or {}
        candidate_detector = detector.get("candidate") or {}
        stale_detector = detector.get("legacy_unsafe_clip_diagnostic") or {}
        lines.extend(
            [
                "",
                "## Detector calibration",
                "",
                f"- Eligible rows: {detector.get('eligible_rows', 'NA')}",
                f"- Detector SKIP rows: {detector.get('detector_skip_rows', 'NA')}",
                f"- Support coverage: {_format_metric(detector.get('support_coverage'))}",
                "- Raw combined_w range: "
                f"{_format_metric(raw.get('min'))} to {_format_metric(raw.get('max'))}",
                f"- Candidate Brier: {_format_metric(candidate_detector.get('brier'))}",
                "- Legacy unsafe clipped diagnostic Brier: "
                f"{_format_metric(stale_detector.get('brier'))}",
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
                f"- Brier: {_format_metric(cm.get('brier'))}",
                f"- Balanced accuracy: {_format_metric(cm.get('balanced_accuracy'))}",
                f"- Human specificity: {_format_metric(cm.get('specificity'))}",
                "- Conformal truth coverage: "
                f"{_format_metric(cm.get('conformal_truth_coverage'))}",
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
    rendered = {
        "promotion_decision.json": _render_json(decision),
        "PROMOTION_REPORT.md": _markdown_report(decision, report),
    }
    _atomic_write_set(out_dir, rendered)


def _publish_full_reports(
    out_dir: Path,
    decision: Mapping[str, object],
    report: Mapping[str, object],
    *,
    evaluations: Sequence[ProfileEvaluation],
    challenge_rows: Sequence[Mapping[str, object]],
    detector: Mapping[str, object],
) -> None:
    # Every renderer runs before the first destination path is replaced.
    rendered = {
        "detector_calibration.json": _render_json(detector),
        "benchmarks.json": _render_json(report),
        "benchmarks_long.csv": _render_prediction_rows(evaluations),
        "july10_per_task.csv": _render_challenge_csv(challenge_rows),
        "promotion_decision.json": _render_json(decision),
        "PROMOTION_REPORT.md": _markdown_report(decision, report),
    }
    _atomic_write_set(out_dir, rendered)


def _publish_empty_full_reports(
    out_dir: Path,
    decision: Mapping[str, object],
    report: Mapping[str, object],
) -> None:
    _publish_full_reports(
        out_dir,
        decision,
        report,
        evaluations=[],
        challenge_rows=[],
        detector=report.get("detector_calibration") or {},
    )


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


def _diagnostic_block_reason(preflight: Mapping[str, object]) -> str | None:
    candidate = preflight.get("candidate") or {}
    if not candidate.get("valid"):
        return "diagnostic-on-invalid-comparator requires a fully valid candidate"
    if not preflight.get("comparator_identity_valid"):
        return (
            "diagnostic-on-invalid-comparator requires exactly the comparators "
            "prompts_only and prompts_specdetect"
        )
    comparators = preflight.get("comparators") or {}
    if not any(not record.get("valid") for record in comparators.values()):
        return (
            "diagnostic-on-invalid-comparator requires at least one invalid comparator"
        )
    return None


def _evaluations_from_outputs(outputs: Mapping[str, object]) -> list[ProfileEvaluation]:
    evaluations: list[ProfileEvaluation] = []
    candidate = outputs.get("candidate_eval")
    if isinstance(candidate, ProfileEvaluation):
        evaluations.append(candidate)
    comparators = outputs.get("comparator_evals") or {}
    for name in sorted(comparators):
        evaluation = comparators[name]
        if isinstance(evaluation, ProfileEvaluation):
            evaluations.append(evaluation)
    return evaluations


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

    if args.diagnostic_on_invalid_comparator:
        diagnostic_block = _diagnostic_block_reason(preflight)
        if diagnostic_block is not None:
            preflight["failures"].append(diagnostic_block)
            decision = decide_promotion(report)
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
        _publish_empty_full_reports(args.out_dir, decision, report)
        return 1

    try:
        report, outputs = _run_full(args, preflight)
    except Exception as exc:
        preflight["failures"].append(
            f"evaluation infrastructure failure: {type(exc).__name__}: {exc}"
        )
        report = {"preflight": preflight}
        decision = decide_promotion(report)
        _publish_empty_full_reports(args.out_dir, decision, report)
        return 1

    decision = decide_promotion(report)
    _publish_full_reports(
        args.out_dir,
        decision,
        report,
        evaluations=_evaluations_from_outputs(outputs),
        challenge_rows=outputs["challenge_rows"],
        detector=report["detector_calibration"],
    )
    if preflight["failures"]:
        return 1
    return 0 if decision["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic evaluation helpers for the TOCSIN retrain candidate.

The promotion runner deliberately keeps evaluation identity, probability
metrics, detector-only diagnostics, and automatic gates in one small module.
None of the helpers mutate source datasets or production profile artifacts.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.special import expit
from scipy.stats import binom
from sklearn.metrics import log_loss as sklearn_log_loss

from cairn.calibration import (
    CALIBRATION_SUPPORT_TOLERANCE_FLOOR,
    CALIBRATION_SUPPORT_TOLERANCE_FRACTION,
    DetectorCalibrator,
)
from cairn.config import load_config
from cairn.contracts import LayerResult
from cairn.detectors.token_cohesiveness import (
    TOCSIN_COMBINED_SIGNATURE,
    TOCSIN_SIGNATURE_TO_RAW_CHANNEL,
)
from cairn.evaluation.dataset import EvalSample
from cairn.evaluation.metrics import auroc, brier, ece, tpr_at_fpr
from cairn.feature_cache import CachedPipeline, load_cache, text_hash
from cairn.normalizer import normalize_text


_TOCSIN_ID = "token_cohesiveness"
_RECONSTRUCTION_TOLERANCE = 1e-9
_LOCKED_EVALUATION_SLICES = (
    "prompt_test",
    "humanized_hard",
    "ood_m4_test",
    "ood_raid_test",
    "ood_mage_test",
)
_LOCKED_COMPARATOR_PROFILES = ("prompts_only", "prompts_specdetect")


@dataclass(frozen=True)
class EvaluatedRow:
    """Privacy-safe result for one locked evaluation sample."""

    id: str
    source_id: str | None
    label: int
    p_llm: float
    conformal_set: tuple[str, ...] | None
    fusion_mode: str
    reconstruction_error: float


@dataclass(frozen=True)
class ProfileEvaluation:
    name: str
    metrics: dict[str, dict[str, object]]
    rows: dict[str, tuple[EvaluatedRow, ...]]


def load_evaluation_slice(path: Path) -> list[EvalSample]:
    """Load a locked slice with hash-derived, collision-safe row identity."""

    path = Path(path)
    samples: list[EvalSample] = []
    seen_hashes: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "text" not in row or "label" not in row:
                raise ValueError(
                    f"{path}:{line_number}: text and label are required"
                )
            digest = text_hash(normalize_text(str(row["text"])))
            label = int(row["label"])
            sample_id = f"{path.stem}:{digest}"
            original_id = row.get("id")
            original_source_id = row.get("source_id")
            source_id_parts = []
            if original_source_id is not None:
                source_id_parts.append(str(original_source_id))
            if original_id is not None:
                source_id_parts.append(f"id={original_id}")
            evaluation_source_id = (
                "|".join(source_id_parts) if source_id_parts else None
            )
            if digest in seen_hashes:
                if seen_hashes[digest] != label:
                    raise ValueError(f"{path}: cross-label duplicate {digest}")
                raise ValueError(f"{path}: duplicate normalized text {digest}")
            seen_hashes[digest] = label
            samples.append(
                EvalSample(
                    id=sample_id,
                    text=str(row["text"]),
                    label=label,
                    tier=row.get("tier"),
                    source=row.get("source"),
                    attack_name=row.get("attack_name"),
                    attack_category=row.get("attack_category"),
                    source_id=evaluation_source_id,
                )
            )
    return samples


def reliability_curve(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int,
) -> list[dict[str, float | int | None]]:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    truth = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if truth.shape != scores.shape:
        raise ValueError("labels and probabilities must have the same shape")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int | None]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (
            (scores >= lower) & (scores <= upper)
            if index == n_bins - 1
            else (scores >= lower) & (scores < upper)
        )
        rows.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": int(mask.sum()),
                "mean_probability": (
                    float(scores[mask].mean()) if mask.any() else None
                ),
                "positive_rate": (
                    float(truth[mask].mean()) if mask.any() else None
                ),
            }
        )
    return rows


def _finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: object) -> int | None:
    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def probability_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, object]:
    scores = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int)
    if truth.shape != scores.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if truth.ndim != 1 or len(truth) == 0:
        raise ValueError("probability metrics require a non-empty 1-D sample")
    finite = bool(np.isfinite(scores).all())
    in_unit_interval = bool(((scores >= 0.0) & (scores <= 1.0)).all())
    safe_for_probability_metrics = finite and in_unit_interval
    human = scores[truth == 0]
    positive = scores[truth == 1]
    brier_value = (
        _finite_number(brier(truth, scores)) if safe_for_probability_metrics else None
    )
    log_loss_value = (
        _finite_number(sklearn_log_loss(truth, scores, labels=[0, 1]))
        if safe_for_probability_metrics
        else None
    )
    auroc_value = (
        _finite_number(auroc(truth, scores)) if safe_for_probability_metrics else None
    )
    ece_value = (
        _finite_number(ece(truth, scores)) if safe_for_probability_metrics else None
    )
    return {
        "n": int(len(truth)),
        "n_unique": int(len(np.unique(scores))),
        "finite": finite,
        "in_unit_interval": in_unit_interval,
        "brier": brier_value,
        "log_loss": log_loss_value,
        "auroc": auroc_value,
        "ece": ece_value,
        "human_mean": (
            _finite_number(human.mean())
            if safe_for_probability_metrics and len(human)
            else None
        ),
        "positive_mean": (
            _finite_number(positive.mean())
            if safe_for_probability_metrics and len(positive)
            else None
        ),
        "endpoint_fraction": (
            float(np.mean((scores == 0.0) | (scores == 1.0)))
            if safe_for_probability_metrics
            else None
        ),
        "reliability_curve": (
            reliability_curve(truth, scores, n_bins=10)
            if safe_for_probability_metrics
            else None
        ),
    }


def detector_comparison(
    labels: Sequence[int],
    raw_scores: Sequence[float],
    stale_probabilities: Sequence[float],
    candidate_probabilities: Sequence[float],
    candidate_applied: Sequence[bool],
    *,
    support: tuple[float, float],
    tolerance: float,
    detector_skip_rows: int,
) -> dict[str, object]:
    truth = np.asarray(labels, dtype=int)
    raw = np.asarray(raw_scores, dtype=float)
    stale = np.asarray(stale_probabilities, dtype=float)
    candidate = np.asarray(candidate_probabilities, dtype=float)
    applied = np.asarray(candidate_applied, dtype=bool)
    if not (
        truth.shape == raw.shape == stale.shape == candidate.shape == applied.shape
    ):
        raise ValueError("detector comparison inputs must have the same shape")
    if len(truth) == 0:
        raise ValueError("detector comparison requires eligible rows")
    lower, upper = support
    in_support = (raw >= lower - tolerance) & (raw <= upper + tolerance)
    if not np.array_equal(in_support, applied):
        raise ValueError("candidate apply mask disagrees with support guard")
    common = applied
    if not common.any():
        raise ValueError("detector comparison has no common in-support rows")
    prevalence = float(truth[common].mean())
    constant = np.full(int(common.sum()), prevalence, dtype=float)
    return {
        "eligible_rows": int(len(truth)),
        "detector_skip_rows": int(detector_skip_rows),
        "in_support_rows": int(common.sum()),
        "support_coverage": float(common.mean()),
        "support": [float(lower), float(upper)],
        "tolerance": float(tolerance),
        "raw": {
            "min": float(raw.min()),
            "max": float(raw.max()),
            "auroc": auroc(truth, raw),
            "human_mean": float(raw[truth == 0].mean()),
            "positive_mean": float(raw[truth == 1].mean()),
        },
        "candidate": probability_metrics(truth[common], candidate[common]),
        "legacy_unsafe_clip_diagnostic": probability_metrics(
            truth[common], stale[common]
        ),
        "constant_prevalence": probability_metrics(truth[common], constant),
    }


def _class_name(label: int) -> str:
    return "LLM" if int(label) == 1 else "human"


def profile_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    conformal_sets: Sequence[Sequence[str] | None] | None = None,
    reconstruction_errors: Sequence[float] | None = None,
    *,
    fusion_modes: Sequence[str] | None = None,
) -> dict[str, object]:
    truth = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if truth.shape != scores.shape or truth.ndim != 1 or len(truth) == 0:
        raise ValueError("profile metrics require equal non-empty 1-D inputs")
    n = len(truth)
    conformal = list(conformal_sets) if conformal_sets is not None else [None] * n
    errors = (
        np.asarray(reconstruction_errors, dtype=float)
        if reconstruction_errors is not None
        else np.full(n, np.inf, dtype=float)
    )
    modes = list(fusion_modes) if fusion_modes is not None else [""] * n
    if len(conformal) != n or errors.shape != truth.shape or len(modes) != n:
        raise ValueError("profile metadata must match labels")

    probability_values_valid = bool(
        np.isfinite(scores).all()
        and ((scores >= 0.0) & (scores <= 1.0)).all()
    )
    predicted = scores >= 0.5 if probability_values_valid else None
    negatives = truth == 0
    positives = truth == 1
    fpr = (
        float(predicted[negatives].mean())
        if probability_values_valid and negatives.any()
        else None
    )
    specificity = (
        float((~predicted[negatives]).mean())
        if probability_values_valid and negatives.any()
        else None
    )
    sensitivity = (
        float(predicted[positives].mean())
        if probability_values_valid and positives.any()
        else None
    )
    balanced = (
        float((specificity + sensitivity) / 2.0)
        if specificity is not None and sensitivity is not None
        else None
    )

    covered: list[bool] = []
    singleton_correct: list[bool] = []
    singleton_count = 0
    for label, prediction_set in zip(truth, conformal):
        members = set(prediction_set or [])
        expected = _class_name(int(label))
        covered.append(expected in members)
        if len(members) == 1:
            singleton_count += 1
            singleton_correct.append(expected in members)

    metrics = probability_metrics(truth, scores)
    metrics.update(
        {
            "balanced_accuracy": balanced,
            "human_fpr": fpr,
            "specificity": specificity,
            "tpr_at_fpr_05": (
                _finite_number(tpr_at_fpr(truth, scores, target_fpr=0.05))
                if probability_values_valid
                else None
            ),
            "tpr_at_fpr_10": (
                _finite_number(tpr_at_fpr(truth, scores, target_fpr=0.10))
                if probability_values_valid
                else None
            ),
            "conformal_truth_coverage_count": int(sum(covered)),
            "conformal_truth_coverage_n": int(n),
            "conformal_truth_coverage": float(np.mean(covered)),
            "singleton_rate": float(singleton_count / n),
            "selective_accuracy": (
                float(np.mean(singleton_correct)) if singleton_correct else None
            ),
            "all_ebm": all(mode == "ebm" for mode in modes),
            "max_ebm_reconstruction_error": float(np.max(errors)),
        }
    )
    return metrics


def binomial_target_interval(n: int, target: float) -> tuple[int, int]:
    lower, upper = binom.interval(0.95, n, target)
    return int(lower), int(upper)


def build_humanized_hard_slice(
    humanized_samples: Sequence[EvalSample],
    hard_human_samples: Sequence[EvalSample],
) -> list[EvalSample]:
    positives = [sample for sample in humanized_samples if sample.label == 1]
    negatives = [sample for sample in hard_human_samples if sample.label == 0]
    if len(positives) != 102 or len(negatives) != 296:
        raise ValueError("expected 102 humanized positives and 296 clean hard Humans")
    positive_hashes = {
        text_hash(normalize_text(sample.text)) for sample in positives
    }
    negative_hashes = {
        text_hash(normalize_text(sample.text)) for sample in negatives
    }
    if len(positive_hashes) != len(positives):
        raise ValueError("humanized-positive slice contains duplicate text")
    if len(negative_hashes) != len(negatives):
        raise ValueError("clean hard-Human slice contains duplicate text")
    conflicts = positive_hashes & negative_hashes
    if conflicts:
        raise ValueError("humanized/hard-Human slice has cross-label hashes")
    return positives + negatives


def legacy_unsafe_transform_for_diagnostic(
    calibrator: DetectorCalibrator,
    detector_id: str,
    raw_score: float,
) -> float:
    return calibrator.transform(  # AUDITED_UNSAFE_STALE_DIAGNOSTIC
        detector_id, raw_score
    )


def profile_name(config_path: Path) -> str:
    raw_name = str(
        load_config(Path(config_path)).get("_profile") or Path(config_path).stem
    )
    return raw_name.removeprefix("profile_")


def profile_artifact_failures(config_path: Path) -> list[str]:
    """Return missing hard requirements without instantiating the profile."""

    config = load_config(Path(config_path))
    fusion = config.get("fusion") or {}
    model_value = fusion.get("model_path")
    conformal_value = fusion.get("conformal_path")
    calibration_value = fusion.get("detector_calibration_path")
    required: list[Path] = []
    if model_value:
        model_path = Path(model_value)
        required.extend(
            [model_path, model_path.with_suffix(model_path.suffix + ".hmac")]
        )
    else:
        required.extend([Path("<model_path>"), Path("<model_path>.hmac")])
    required.append(Path(conformal_value) if conformal_value else Path("<conformal_path>"))
    required.append(
        Path(calibration_value)
        if calibration_value
        else Path("<detector_calibration_path>")
    )
    return [
        f"missing {path.as_posix()}"
        for path in required
        if not path.is_file()
    ]


def _first_smoke_text(path: Path) -> str:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "text" not in row:
                raise ValueError(f"{path}:{line_number}: text is required")
            return str(row["text"])
    raise ValueError(f"{path}: no smoke row")


def _reconstruction_error(pipeline: CachedPipeline, determination) -> float:
    if determination.fusion_mode != "ebm":
        return float("inf")
    model = pipeline._inner._fusion._model
    contributions = determination.feature_contributions
    if model is None or not contributions:
        return float("inf")
    intercept_values = np.ravel(getattr(model, "intercept_", []))
    if len(intercept_values) != 1:
        return float("inf")
    logit = float(intercept_values[0]) + math.fsum(
        float(value) for value in contributions.values()
    )
    reconstructed = float(expit(logit))
    return abs(reconstructed - float(determination.p_llm))


def preflight_profiles(
    candidate_config: Path,
    baseline_configs: Sequence[Path],
    smoke_dataset: Path,
    smoke_cache: Path,
) -> dict[str, object]:
    """Validate all paths first, then load valid profiles and one cache row."""

    specs = [("candidate", Path(candidate_config))] + [
        ("comparator", Path(path)) for path in baseline_configs
    ]
    records: list[dict[str, object]] = []
    for role, path in specs:
        try:
            name = profile_name(path)
            failures = profile_artifact_failures(path)
        except Exception as exc:
            name = path.stem
            failures = [f"profile config error: {exc}"]
        records.append(
            {
                "role": role,
                "name": name,
                "config_path": path,
                "failures": failures,
            }
        )

    expected_comparators = {"prompts_only", "prompts_specdetect"}
    comparator_names = [
        str(record["name"])
        for record in records
        if record["role"] == "comparator"
    ]
    comparator_counts = {
        name: comparator_names.count(name) for name in sorted(set(comparator_names))
    }
    identity_issues = [
        f"missing {name}"
        for name in sorted(expected_comparators - set(comparator_names))
    ]
    identity_issues.extend(
        f"duplicate {name} ({count})"
        for name, count in comparator_counts.items()
        if count > 1
    )
    identity_issues.extend(
        f"unexpected {name}"
        for name in sorted(set(comparator_names) - expected_comparators)
    )
    comparator_identity_valid = not identity_issues

    # Dataset/cache reads happen only after every candidate/comparator artifact
    # has been inventoried, preventing fallback fusion from entering history.
    if any(not record["failures"] for record in records):
        try:
            smoke_text = _first_smoke_text(smoke_dataset)
        except Exception as exc:
            smoke_text = None
            for record in records:
                if not record["failures"]:
                    record["failures"].append(f"smoke dataset error: {exc}")
        if smoke_text is not None:
            for record in records:
                if record["failures"]:
                    continue
                try:
                    cached = CachedPipeline(smoke_cache, record["config_path"])
                    determination = cached.analyze(smoke_text)
                    status = cached._inner._calibration_status
                    if determination.fusion_mode != "ebm":
                        record["failures"].append(
                            f"smoke fusion_mode={determination.fusion_mode}, expected ebm"
                        )
                    if not str(status).startswith("calibrated"):
                        record["failures"].append(
                            f"smoke calibration_status={status}, expected calibrated*"
                        )
                    if cached._inner._detector_calibrator is None:
                        record["failures"].append(
                            "smoke detector calibrator did not load"
                        )
                    if not math.isfinite(float(determination.p_llm)):
                        record["failures"].append("smoke p_llm is non-finite")
                    error = _reconstruction_error(cached, determination)
                    if error > _RECONSTRUCTION_TOLERANCE:
                        record["failures"].append(
                            f"smoke EBM reconstruction error {error:.12g} exceeds 1e-9"
                        )
                except Exception as exc:
                    record["failures"].append(f"smoke profile error: {exc}")

    failures: list[str] = []
    if identity_issues:
        failures.append(
            "INVALID_COMPARATOR_SET:\n" + "\n".join(identity_issues)
        )
    candidate_record: dict[str, object] | None = None
    comparator_records: dict[str, dict[str, object]] = {}
    comparator_key_counts: dict[str, int] = {}
    for record in records:
        public = {
            "name": record["name"],
            "valid": not record["failures"],
            "failures": list(record["failures"]),
        }
        if record["role"] == "candidate":
            candidate_record = public
            prefix = "INVALID_CANDIDATE"
        else:
            name = str(record["name"])
            occurrence = comparator_key_counts.get(name, 0) + 1
            comparator_key_counts[name] = occurrence
            key = name if occurrence == 1 else f"{name}#{occurrence}"
            comparator_records[key] = public
            prefix = "INVALID_COMPARATOR"
        if record["failures"]:
            failures.append(
                f"{prefix} {record['name']}:\n" + "\n".join(record["failures"])
            )
    return {
        "candidate": candidate_record,
        "comparators": comparator_records,
        "comparator_identity_valid": comparator_identity_valid,
        "failures": failures,
    }


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _audit_row(
    *,
    gate_id: str,
    label: str,
    scope: str,
    profile: str | None,
    slice_name: str | None,
    evaluated: bool,
    passed: bool,
    observed: object,
    operator: str,
    reference: object,
    failure_detail: str,
    not_evaluated_reason: str | None = None,
) -> dict[str, object]:
    status = "NOT_EVALUATED" if not evaluated else ("PASS" if passed else "FAIL")
    return {
        "gate_id": gate_id,
        "label": label,
        "scope": scope,
        "profile": profile,
        "slice": slice_name,
        "status": status,
        "observed": observed,
        "operator": operator,
        "reference": reference,
        "failure_detail": failure_detail if status == "FAIL" else None,
        "not_evaluated_reason": (
            not_evaluated_reason if status == "NOT_EVALUATED" else None
        ),
    }


def _take_preflight_failure(
    failures: Sequence[str], consumed: set[int], prefix: str
) -> str | None:
    for index, failure in enumerate(failures):
        if index not in consumed and failure.startswith(prefix):
            consumed.add(index)
            return failure
    return None


def _profile_slice(
    profile: Mapping[str, object], slice_name: str
) -> Mapping[str, object] | None:
    if slice_name == "prompt_test":
        value = profile.get("prompt_test")
    elif slice_name == "humanized_hard":
        value = profile.get("humanized_hard")
    else:
        value = _as_mapping(profile.get("ood")).get(slice_name)
    return value if isinstance(value, Mapping) else None


def _append_integrity_audit(
    audit: list[dict[str, object]],
    *,
    gate_prefix: str,
    label_prefix: str,
    scope: str,
    profile: str,
    slice_name: str,
    metrics: Mapping[str, object] | None,
    evaluated: bool,
    not_evaluated_reason: str | None,
) -> None:
    metrics_missing = metrics is None
    values = metrics or {}
    missing_detail = f"{label_prefix}: metrics missing"
    flag_specs = (
        (
            "all_ebm",
            "All rows use fusion_mode=ebm",
            "not every row used fusion_mode=ebm",
        ),
        ("finite", "All p_llm values are finite", "non-finite p_llm"),
        (
            "in_unit_interval",
            "All p_llm values are in [0, 1]",
            "p_llm outside [0, 1]",
        ),
    )
    for metric, label, failure_suffix in flag_specs:
        raw = values.get(metric)
        observed = bool(raw) if raw is not None else None
        audit.append(
            _audit_row(
                gate_id=f"{gate_prefix}.{metric}",
                label=f"{label_prefix}: {label}",
                scope=scope,
                profile=profile,
                slice_name=slice_name,
                evaluated=evaluated,
                passed=not metrics_missing and bool(raw),
                observed=observed,
                operator="is",
                reference=True,
                failure_detail=(
                    missing_detail
                    if metrics_missing
                    else f"{label_prefix}: {failure_suffix}"
                ),
                not_evaluated_reason=not_evaluated_reason,
            )
        )

    error = _finite_number(values.get("max_ebm_reconstruction_error"))
    audit.append(
        _audit_row(
            gate_id=f"{gate_prefix}.max_ebm_reconstruction_error",
            label=f"{label_prefix}: maximum EBM reconstruction error",
            scope=scope,
            profile=profile,
            slice_name=slice_name,
            evaluated=evaluated,
            passed=(
                not metrics_missing
                and error is not None
                and error <= _RECONSTRUCTION_TOLERANCE
            ),
            observed=error,
            operator="<=",
            reference=_RECONSTRUCTION_TOLERANCE,
            failure_detail=(
                missing_detail
                if metrics_missing
                else f"{label_prefix}: EBM reconstruction error exceeds 1e-9"
            ),
            not_evaluated_reason=not_evaluated_reason,
        )
    )


def _append_comparison_audit(
    audit: list[dict[str, object]],
    *,
    gate_id: str,
    label: str,
    profile: str,
    slice_name: str,
    candidate: object,
    baseline: object,
    operator: str,
    allowance: float,
    evaluated: bool,
    not_evaluated_reason: str | None,
) -> None:
    candidate_value = _finite_number(candidate)
    baseline_value = _finite_number(baseline)
    if operator == "<=":
        reference = baseline_value
        passed = (
            candidate_value is not None
            and reference is not None
            and candidate_value <= reference
        )
        regression_detail = f"{label}: candidate regressed"
    else:
        reference = (
            baseline_value - allowance if baseline_value is not None else None
        )
        passed = (
            candidate_value is not None
            and reference is not None
            and candidate_value >= reference
        )
        regression_detail = f"{label}: regression exceeds {allowance:.2f}"
    failure_detail = (
        f"{label}: metric missing or non-finite"
        if candidate_value is None or baseline_value is None
        else regression_detail
    )
    audit.append(
        _audit_row(
            gate_id=gate_id,
            label=label,
            scope="comparator",
            profile=profile,
            slice_name=slice_name,
            evaluated=evaluated,
            passed=passed,
            observed=candidate_value,
            operator=operator,
            reference=reference,
            failure_detail=failure_detail,
            not_evaluated_reason=not_evaluated_reason,
        )
    )


def promotion_gate_audit(report: Mapping[str, object]) -> list[dict[str, object]]:
    """Return every locked automatic gate in deterministic decision order.

    The adaptive challenge is deliberately not inspected.  Preflight-only runs
    retain the full inventory with evaluation gates marked ``NOT_EVALUATED``.
    """

    audit: list[dict[str, object]] = []
    preflight = _as_mapping(report.get("preflight"))
    preflight_failures = [str(item) for item in (preflight.get("failures") or [])]
    consumed_failures: set[int] = set()
    candidate_raw = report.get("candidate")
    candidate = _as_mapping(candidate_raw)
    comparator_raw = report.get("comparators")
    comparators = _as_mapping(comparator_raw)
    full_evaluation = (
        candidate_raw is not None or report.get("detector_calibration") is not None
    )

    preflight_comparators = _as_mapping(preflight.get("comparators"))
    identity_declared = "comparator_identity_valid" in preflight
    if identity_declared:
        identity_observed = bool(preflight.get("comparator_identity_valid"))
        identity_evaluated = True
    else:
        inferred_names = set(comparators)
        inferred_names.update(
            str(_as_mapping(record).get("name") or key)
            for key, record in preflight_comparators.items()
        )
        identity_evaluated = bool(inferred_names)
        identity_observed = (
            inferred_names == set(_LOCKED_COMPARATOR_PROFILES)
            if identity_evaluated
            else None
        )
    identity_failure = None
    if identity_evaluated and identity_observed is not True:
        identity_failure = _take_preflight_failure(
            preflight_failures, consumed_failures, "INVALID_COMPARATOR_SET:"
        )
    if identity_failure is None:
        identity_failure = (
            "INVALID_COMPARATOR_SET:\nexpected exactly prompts_only and "
            "prompts_specdetect"
        )
    audit.append(
        _audit_row(
            gate_id="preflight.comparator_identity",
            label="Locked comparator identity set",
            scope="preflight",
            profile=None,
            slice_name=None,
            evaluated=identity_evaluated,
            passed=identity_observed is True,
            observed=identity_observed,
            operator="is",
            reference=True,
            failure_detail=identity_failure,
            not_evaluated_reason="comparator identities were not inspected",
        )
    )

    candidate_record_value = preflight.get("candidate")
    candidate_record_present = isinstance(candidate_record_value, Mapping)
    candidate_record = _as_mapping(candidate_record_value)
    candidate_name = str(
        report.get("candidate_name")
        or candidate_record.get("name")
        or "candidate"
    )
    if candidate_record_present:
        candidate_valid = bool(candidate_record.get("valid"))
        candidate_validity_evaluated = True
    elif candidate_raw is not None:
        candidate_valid = True
        candidate_validity_evaluated = True
    else:
        candidate_valid = False
        candidate_validity_evaluated = False
    candidate_failure = None
    if candidate_validity_evaluated and not candidate_valid:
        candidate_failure = _take_preflight_failure(
            preflight_failures,
            consumed_failures,
            f"INVALID_CANDIDATE {candidate_name}:",
        )
    if candidate_failure is None:
        record_failures = [str(item) for item in (candidate_record.get("failures") or [])]
        candidate_failure = f"INVALID_CANDIDATE {candidate_name}:"
        if record_failures:
            candidate_failure += "\n" + "\n".join(record_failures)
        else:
            candidate_failure += " artifact validation failed"
    audit.append(
        _audit_row(
            gate_id="preflight.candidate_artifacts",
            label="Candidate artifact validity",
            scope="preflight",
            profile=candidate_name,
            slice_name=None,
            evaluated=candidate_validity_evaluated,
            passed=candidate_valid,
            observed=candidate_valid if candidate_validity_evaluated else None,
            operator="is",
            reference=True,
            failure_detail=candidate_failure,
            not_evaluated_reason="candidate artifacts were not inspected",
        )
    )

    comparator_names = list(_LOCKED_COMPARATOR_PROFILES)
    comparator_names.extend(
        sorted(
            (set(preflight_comparators) | set(comparators))
            - set(_LOCKED_COMPARATOR_PROFILES)
        )
    )
    comparator_status: dict[str, str] = {}
    for comparator_name in comparator_names:
        record_value = preflight_comparators.get(comparator_name)
        record_present = isinstance(record_value, Mapping)
        record = _as_mapping(record_value)
        evaluated_report_present = comparator_name in comparators
        if record_present:
            comparator_valid = bool(record.get("valid"))
            validity_evaluated = True
        elif evaluated_report_present:
            comparator_valid = True
            validity_evaluated = True
        else:
            comparator_valid = False
            validity_evaluated = False
        record_name = str(record.get("name") or comparator_name)
        comparator_failure = None
        if validity_evaluated and not comparator_valid:
            comparator_failure = _take_preflight_failure(
                preflight_failures,
                consumed_failures,
                f"INVALID_COMPARATOR {record_name}:",
            )
        if comparator_failure is None:
            record_failures = [str(item) for item in (record.get("failures") or [])]
            comparator_failure = f"INVALID_COMPARATOR {record_name}:"
            if record_failures:
                comparator_failure += "\n" + "\n".join(record_failures)
            else:
                comparator_failure += " artifact validation failed"
        row = _audit_row(
            gate_id=f"preflight.comparator.{comparator_name}.artifacts",
            label=f"Comparator {comparator_name} artifact validity",
            scope="preflight",
            profile=comparator_name,
            slice_name=None,
            evaluated=validity_evaluated,
            passed=comparator_valid,
            observed=comparator_valid if validity_evaluated else None,
            operator="is",
            reference=True,
            failure_detail=comparator_failure,
            not_evaluated_reason="comparator artifacts were not inspected",
        )
        comparator_status[comparator_name] = str(row["status"])
        audit.append(row)

    for index, failure in enumerate(preflight_failures, start=1):
        if index - 1 in consumed_failures:
            continue
        audit.append(
            _audit_row(
                gate_id=f"preflight.reported_failure.{index:03d}",
                label="Preflight reported failure",
                scope="preflight",
                profile=None,
                slice_name=None,
                evaluated=True,
                passed=False,
                observed=failure,
                operator="must_not_exist",
                reference=None,
                failure_detail=failure,
            )
        )

    detector = _as_mapping(report.get("detector_calibration"))
    candidate_detector = _as_mapping(detector.get("candidate"))
    stale = _as_mapping(detector.get("legacy_unsafe_clip_diagnostic"))
    constant = _as_mapping(detector.get("constant_prevalence"))
    detector_reason = "full benchmark was not evaluated"
    eligible_rows = _integer(detector.get("eligible_rows"))
    audit.append(
        _audit_row(
            gate_id="detector.eligible_rows",
            label="Detector eligible row count",
            scope="detector",
            profile=_TOCSIN_ID,
            slice_name="prompt_test",
            evaluated=full_evaluation,
            passed=eligible_rows == 1822,
            observed=eligible_rows,
            operator="==",
            reference=1822,
            failure_detail="detector: expected exactly 1,822 eligible rows",
            not_evaluated_reason=detector_reason,
        )
    )
    support_coverage = _finite_number(detector.get("support_coverage"))
    audit.append(
        _audit_row(
            gate_id="detector.support_coverage",
            label="Detector support coverage",
            scope="detector",
            profile=_TOCSIN_ID,
            slice_name="prompt_test",
            evaluated=full_evaluation,
            passed=support_coverage == 1.0,
            observed=support_coverage,
            operator="==",
            reference=1.0,
            failure_detail="detector: support coverage must equal 1.0",
            not_evaluated_reason=detector_reason,
        )
    )
    detector_flag_specs = (
        (
            "finite",
            "detector.candidate.finite",
            "Detector candidate probabilities are finite",
            "detector: candidate probabilities are non-finite",
        ),
        (
            "in_unit_interval",
            "detector.candidate.in_unit_interval",
            "Detector candidate probabilities are in [0, 1]",
            "detector: candidate probabilities fall outside [0, 1]",
        ),
    )
    for metric, gate_id, label, failure_detail in detector_flag_specs:
        raw = candidate_detector.get(metric)
        audit.append(
            _audit_row(
                gate_id=gate_id,
                label=label,
                scope="detector",
                profile=_TOCSIN_ID,
                slice_name="prompt_test",
                evaluated=full_evaluation,
                passed=bool(raw),
                observed=bool(raw) if raw is not None else None,
                operator="is",
                reference=True,
                failure_detail=failure_detail,
                not_evaluated_reason=detector_reason,
            )
        )
    n_unique = _integer(candidate_detector.get("n_unique"))
    audit.append(
        _audit_row(
            gate_id="detector.candidate.n_unique",
            label="Detector candidate probability uniqueness",
            scope="detector",
            profile=_TOCSIN_ID,
            slice_name="prompt_test",
            evaluated=full_evaluation,
            passed=n_unique is not None and n_unique > 3,
            observed=n_unique,
            operator=">",
            reference=3,
            failure_detail=(
                "detector: candidate must have more than 3 unique probabilities"
            ),
            not_evaluated_reason=detector_reason,
        )
    )
    candidate_brier = _finite_number(candidate_detector.get("brier"))
    stale_brier = _finite_number(stale.get("brier"))
    candidate_log_loss = _finite_number(candidate_detector.get("log_loss"))
    stale_log_loss = _finite_number(stale.get("log_loss"))
    constant_brier = _finite_number(constant.get("brier"))
    detector_comparisons = (
        (
            "detector.candidate.brier_vs_legacy",
            "Detector candidate Brier versus legacy diagnostic",
            candidate_brier,
            stale_brier,
            "detector: candidate must strictly improve stale Brier",
        ),
        (
            "detector.candidate.log_loss_vs_legacy",
            "Detector candidate log loss versus legacy diagnostic",
            candidate_log_loss,
            stale_log_loss,
            "detector: candidate must strictly improve stale log loss",
        ),
        (
            "detector.candidate.brier_vs_constant",
            "Detector candidate Brier versus constant prevalence",
            candidate_brier,
            constant_brier,
            "detector: candidate Brier must beat constant prevalence",
        ),
    )
    for gate_id, label, observed, reference, failure_detail in detector_comparisons:
        audit.append(
            _audit_row(
                gate_id=gate_id,
                label=label,
                scope="detector",
                profile=_TOCSIN_ID,
                slice_name="prompt_test",
                evaluated=full_evaluation,
                passed=(
                    observed is not None
                    and reference is not None
                    and observed < reference
                ),
                observed=observed,
                operator="<",
                reference=reference,
                failure_detail=failure_detail,
                not_evaluated_reason=detector_reason,
            )
        )
    endpoint_fraction = _finite_number(candidate_detector.get("endpoint_fraction"))
    audit.append(
        _audit_row(
            gate_id="detector.candidate.endpoint_fraction",
            label="Detector candidate endpoint fraction",
            scope="detector",
            profile=_TOCSIN_ID,
            slice_name="prompt_test",
            evaluated=full_evaluation,
            passed=endpoint_fraction is not None and endpoint_fraction < 0.95,
            observed=endpoint_fraction,
            operator="<",
            reference=0.95,
            failure_detail=(
                "detector: candidate endpoint fraction must be below 0.95"
            ),
            not_evaluated_reason=detector_reason,
        )
    )

    candidate_evaluated = full_evaluation and candidate_valid
    candidate_reason = (
        "candidate artifacts are invalid"
        if candidate_validity_evaluated and not candidate_valid
        else "full benchmark was not evaluated"
    )
    for slice_name in _LOCKED_EVALUATION_SLICES:
        _append_integrity_audit(
            audit,
            gate_prefix=f"candidate.{slice_name}",
            label_prefix=f"candidate {slice_name}",
            scope="candidate",
            profile=candidate_name,
            slice_name=slice_name,
            metrics=_profile_slice(candidate, slice_name),
            evaluated=candidate_evaluated,
            not_evaluated_reason=candidate_reason,
        )

    prompt = _profile_slice(candidate, "prompt_test") or {}
    coverage_n = _integer(prompt.get("conformal_truth_coverage_n", prompt.get("n")))
    coverage_count = _integer(prompt.get("conformal_truth_coverage_count"))
    coverage_reference: list[int] | None = None
    if coverage_n is None or coverage_n <= 0:
        coverage_passed = False
        coverage_failure = (
            "candidate prompt_test: conformal coverage denominator missing"
        )
    else:
        lower, upper = binomial_target_interval(coverage_n, 0.95)
        coverage_reference = [lower, upper]
        if coverage_count is None:
            coverage_passed = False
            coverage_failure = "candidate prompt_test: conformal coverage count missing"
        else:
            coverage_passed = lower <= coverage_count <= upper
            coverage_failure = (
                "candidate prompt_test: conformal coverage count "
                f"{coverage_count} outside [{lower}, {upper}]"
            )
    audit.append(
        _audit_row(
            gate_id="candidate.prompt_test.conformal_truth_coverage_count",
            label="Candidate prompt_test conformal truth coverage count",
            scope="candidate",
            profile=candidate_name,
            slice_name="prompt_test",
            evaluated=candidate_evaluated,
            passed=coverage_passed,
            observed=coverage_count,
            operator="between_inclusive",
            reference=coverage_reference,
            failure_detail=coverage_failure,
            not_evaluated_reason=candidate_reason,
        )
    )

    for comparator_name in comparator_names:
        comparator = _as_mapping(comparators.get(comparator_name))
        validity = comparator_status[comparator_name]
        comparator_evaluated = full_evaluation and validity == "PASS"
        if validity == "FAIL":
            comparator_reason = "comparator artifacts are invalid"
        elif validity == "NOT_EVALUATED":
            comparator_reason = "comparator artifact validity was not evaluated"
        else:
            comparator_reason = "full benchmark was not evaluated"
        for slice_name in _LOCKED_EVALUATION_SLICES:
            _append_integrity_audit(
                audit,
                gate_prefix=f"comparator.{comparator_name}.{slice_name}",
                label_prefix=f"comparator {comparator_name} {slice_name}",
                scope="comparator",
                profile=comparator_name,
                slice_name=slice_name,
                metrics=_profile_slice(comparator, slice_name),
                evaluated=comparator_evaluated,
                not_evaluated_reason=comparator_reason,
            )

        performance_evaluated = comparator_evaluated and candidate_valid
        performance_reason = (
            "candidate artifacts are invalid"
            if comparator_evaluated and not candidate_valid
            else comparator_reason
        )
        baseline_prompt = _profile_slice(comparator, "prompt_test") or {}
        for metric in ("brier", "log_loss"):
            _append_comparison_audit(
                audit,
                gate_id=f"comparator.{comparator_name}.prompt_test.{metric}",
                label=f"{comparator_name} prompt_test {metric}",
                profile=comparator_name,
                slice_name="prompt_test",
                candidate=prompt.get(metric),
                baseline=baseline_prompt.get(metric),
                operator="<=",
                allowance=0.0,
                evaluated=performance_evaluated,
                not_evaluated_reason=performance_reason,
            )
        for metric in ("balanced_accuracy", "auroc"):
            label_metric = (
                "balanced accuracy" if metric == "balanced_accuracy" else metric
            )
            _append_comparison_audit(
                audit,
                gate_id=f"comparator.{comparator_name}.prompt_test.{metric}",
                label=f"{comparator_name} prompt_test {label_metric}",
                profile=comparator_name,
                slice_name="prompt_test",
                candidate=prompt.get(metric),
                baseline=baseline_prompt.get(metric),
                operator=">=",
                allowance=0.01,
                evaluated=performance_evaluated,
                not_evaluated_reason=performance_reason,
            )
        _append_comparison_audit(
            audit,
            gate_id=f"comparator.{comparator_name}.prompt_test.human_fpr",
            label=f"{comparator_name} prompt_test Human FPR",
            profile=comparator_name,
            slice_name="prompt_test",
            candidate=prompt.get("human_fpr"),
            baseline=baseline_prompt.get("human_fpr"),
            operator="<=",
            allowance=0.0,
            evaluated=performance_evaluated,
            not_evaluated_reason=performance_reason,
        )

        candidate_humanized = _profile_slice(candidate, "humanized_hard") or {}
        baseline_humanized = _profile_slice(comparator, "humanized_hard") or {}
        for metric in ("auroc", "tpr_at_fpr_05", "tpr_at_fpr_10"):
            _append_comparison_audit(
                audit,
                gate_id=f"comparator.{comparator_name}.humanized_hard.{metric}",
                label=f"{comparator_name} humanized_hard {metric}",
                profile=comparator_name,
                slice_name="humanized_hard",
                candidate=candidate_humanized.get(metric),
                baseline=baseline_humanized.get(metric),
                operator=">=",
                allowance=0.02,
                evaluated=performance_evaluated,
                not_evaluated_reason=performance_reason,
            )

        for corpus in ("ood_m4_test", "ood_raid_test", "ood_mage_test"):
            candidate_ood = _profile_slice(candidate, corpus) or {}
            baseline_ood = _profile_slice(comparator, corpus) or {}
            _append_comparison_audit(
                audit,
                gate_id=f"comparator.{comparator_name}.{corpus}.auroc",
                label=f"{comparator_name} {corpus} auroc",
                profile=comparator_name,
                slice_name=corpus,
                candidate=candidate_ood.get("auroc"),
                baseline=baseline_ood.get("auroc"),
                operator=">=",
                allowance=0.02,
                evaluated=performance_evaluated,
                not_evaluated_reason=performance_reason,
            )

    return audit


def decide_promotion(report: Mapping[str, object]) -> dict[str, object]:
    """Apply only locked automatic gates; the adaptive challenge is ignored."""

    failures: list[str] = []
    seen: set[str] = set()
    for row in promotion_gate_audit(report):
        if row["status"] != "FAIL":
            continue
        detail = str(row["failure_detail"])
        if detail not in seen:
            failures.append(detail)
            seen.add(detail)
    return {
        "status": "PASS" if not failures else "DO_NOT_PROMOTE",
        "failures": failures,
    }


def evaluate_detector_calibration(
    samples: Sequence[EvalSample],
    cache_path: Path,
    candidate_calibration_path: Path,
    stale_calibration_path: Path,
) -> dict[str, object]:
    cache = load_cache(Path(cache_path))
    candidate_calibrator = DetectorCalibrator()
    candidate_calibrator.load(Path(candidate_calibration_path))
    stale_calibrator = DetectorCalibrator()
    stale_calibrator.load(Path(stale_calibration_path))
    if not candidate_calibrator.has(_TOCSIN_ID):
        raise ValueError("candidate calibration lacks token_cohesiveness")

    expected_channel = TOCSIN_SIGNATURE_TO_RAW_CHANNEL[TOCSIN_COMBINED_SIGNATURE]
    labels: list[int] = []
    raw_scores: list[float] = []
    stale_probabilities: list[float] = []
    candidate_probabilities: list[float] = []
    candidate_applied: list[bool] = []
    for sample in samples:
        digest = text_hash(normalize_text(sample.text))
        cached = cache.get(digest)
        if cached is None:
            raise KeyError(f"evaluation cache missing {sample.id}")
        matches = [
            row
            for row in cached["results"]
            if row.get("layer_id") == _TOCSIN_ID
        ]
        if len(matches) != 1:
            continue
        raw_row = matches[0]
        signals = raw_row.get("signals") or {}
        if (
            raw_row.get("determination") == "SKIP"
            or signals.get("calibration_signature") != TOCSIN_COMBINED_SIGNATURE
            or signals.get("calibration_raw_channel") != expected_channel
        ):
            continue
        result = LayerResult(**raw_row)
        raw = float(result.raw_score)
        applied = candidate_calibrator.apply(result)
        labels.append(int(sample.label))
        raw_scores.append(raw)
        candidate_probabilities.append(float(result.p_llm))
        candidate_applied.append(bool(applied))
        stale_probabilities.append(
            legacy_unsafe_transform_for_diagnostic(
                stale_calibrator, _TOCSIN_ID, raw
            )
        )

    support = candidate_calibrator.support(_TOCSIN_ID)
    tolerance = max(
        CALIBRATION_SUPPORT_TOLERANCE_FRACTION * (support[1] - support[0]),
        CALIBRATION_SUPPORT_TOLERANCE_FLOOR,
    )
    return detector_comparison(
        labels,
        raw_scores,
        stale_probabilities,
        candidate_probabilities,
        candidate_applied,
        support=support,
        tolerance=tolerance,
        detector_skip_rows=len(samples) - len(labels),
    )


def evaluate_profile(
    config_path: Path,
    datasets: Mapping[
        str, Sequence[tuple[Sequence[EvalSample], Path]]
    ],
) -> ProfileEvaluation:
    """Replay one profile over cache-backed, already-locked evaluation rows."""

    first_segment = next(
        (
            segment
            for segments in datasets.values()
            for segment in segments
            if len(segment[0])
        ),
        None,
    )
    if first_segment is None:
        raise ValueError("evaluate_profile requires at least one sample")
    pipeline = CachedPipeline(Path(first_segment[1]), Path(config_path))
    metrics_by_dataset: dict[str, dict[str, object]] = {}
    rows_by_dataset: dict[str, tuple[EvaluatedRow, ...]] = {}
    for dataset_name, segments in datasets.items():
        evaluated: list[EvaluatedRow] = []
        for samples, cache_path in segments:
            pipeline._cache = load_cache(Path(cache_path))
            for sample in samples:
                determination = pipeline.analyze(sample.text)
                evaluated.append(
                    EvaluatedRow(
                        id=sample.id,
                        source_id=sample.source_id,
                        label=int(sample.label),
                        p_llm=float(determination.p_llm),
                        conformal_set=(
                            tuple(determination.conformal_set)
                            if determination.conformal_set is not None
                            else None
                        ),
                        fusion_mode=str(determination.fusion_mode),
                        reconstruction_error=_reconstruction_error(
                            pipeline, determination
                        ),
                    )
                )
        if not evaluated:
            raise ValueError(f"{dataset_name}: no evaluation rows")
        rows_by_dataset[dataset_name] = tuple(evaluated)
        metrics_by_dataset[dataset_name] = profile_metrics(
            [row.label for row in evaluated],
            [row.p_llm for row in evaluated],
            [row.conformal_set for row in evaluated],
            [row.reconstruction_error for row in evaluated],
            fusion_modes=[row.fusion_mode for row in evaluated],
        )
    return ProfileEvaluation(
        name=profile_name(Path(config_path)),
        metrics=metrics_by_dataset,
        rows=rows_by_dataset,
    )

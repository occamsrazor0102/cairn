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
_COMPARISON_EPSILON = 1e-12


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
    return {
        "n": int(len(truth)),
        "n_unique": int(len(np.unique(scores))),
        "finite": finite,
        "in_unit_interval": in_unit_interval,
        "brier": brier(truth, scores) if finite else None,
        "log_loss": (
            float(sklearn_log_loss(truth, scores, labels=[0, 1]))
            if safe_for_probability_metrics
            else None
        ),
        "auroc": auroc(truth, scores) if finite else None,
        "ece": ece(truth, scores) if finite else None,
        "human_mean": float(human.mean()) if len(human) else None,
        "positive_mean": float(positive.mean()) if len(positive) else None,
        "endpoint_fraction": float(
            np.mean((scores == 0.0) | (scores == 1.0))
        ),
        "reliability_curve": reliability_curve(truth, scores, n_bins=10),
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

    predicted = scores >= 0.5
    negatives = truth == 0
    positives = truth == 1
    fpr = float(predicted[negatives].mean()) if negatives.any() else None
    specificity = float((~predicted[negatives]).mean()) if negatives.any() else None
    sensitivity = float(predicted[positives].mean()) if positives.any() else None
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
            "tpr_at_fpr_05": tpr_at_fpr(truth, scores, target_fpr=0.05),
            "tpr_at_fpr_10": tpr_at_fpr(truth, scores, target_fpr=0.10),
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
    return calibrator.transform(
        detector_id, raw_score
    )  # AUDITED_UNSAFE_STALE_DIAGNOSTIC


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
    candidate_record: dict[str, object] | None = None
    comparator_records: dict[str, dict[str, object]] = {}
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
            comparator_records[str(record["name"])] = public
            prefix = "INVALID_COMPARATOR"
        if record["failures"]:
            failures.append(
                f"{prefix} {record['name']}:\n" + "\n".join(record["failures"])
            )
    return {
        "candidate": candidate_record,
        "comparators": comparator_records,
        "failures": failures,
    }


def _comparison_passes(candidate: float, baseline: float, delta: float = 0.0) -> bool:
    return float(candidate) + _COMPARISON_EPSILON >= float(baseline) - delta


def _append_integrity_failures(
    failures: list[str], label: str, metrics: Mapping[str, object]
) -> None:
    if not bool(metrics.get("all_ebm")):
        failures.append(f"{label}: not every row used fusion_mode=ebm")
    if not bool(metrics.get("finite")):
        failures.append(f"{label}: non-finite p_llm")
    if not bool(metrics.get("in_unit_interval")):
        failures.append(f"{label}: p_llm outside [0, 1]")
    error = metrics.get("max_ebm_reconstruction_error")
    if error is None or float(error) > _RECONSTRUCTION_TOLERANCE + 1e-15:
        failures.append(f"{label}: EBM reconstruction error exceeds 1e-9")


def decide_promotion(report: Mapping[str, object]) -> dict[str, object]:
    """Apply only locked automatic gates; the adaptive challenge is ignored."""

    preflight = report.get("preflight") or {}
    failures = list(preflight.get("failures") or [])

    detector = report.get("detector_calibration")
    if detector is not None:
        candidate_detector = detector.get("candidate") or {}
        stale = detector.get("legacy_unsafe_clip_diagnostic") or {}
        constant = detector.get("constant_prevalence") or {}
        if int(detector.get("eligible_rows", -1)) != 1822:
            failures.append("detector: expected exactly 1,822 eligible rows")
        if float(detector.get("support_coverage", -1.0)) != 1.0:
            failures.append("detector: support coverage must equal 1.0")
        if not bool(candidate_detector.get("finite")):
            failures.append("detector: candidate probabilities are non-finite")
        if not bool(candidate_detector.get("in_unit_interval")):
            failures.append("detector: candidate probabilities fall outside [0, 1]")
        if int(candidate_detector.get("n_unique", 0)) <= 3:
            failures.append("detector: candidate must have more than 3 unique probabilities")
        if not (
            candidate_detector.get("brier") is not None
            and stale.get("brier") is not None
            and float(candidate_detector["brier"]) < float(stale["brier"])
        ):
            failures.append("detector: candidate must strictly improve stale Brier")
        if not (
            candidate_detector.get("log_loss") is not None
            and stale.get("log_loss") is not None
            and float(candidate_detector["log_loss"]) < float(stale["log_loss"])
        ):
            failures.append("detector: candidate must strictly improve stale log loss")
        if not (
            candidate_detector.get("brier") is not None
            and constant.get("brier") is not None
            and float(candidate_detector["brier"]) < float(constant["brier"])
        ):
            failures.append("detector: candidate Brier must beat constant prevalence")
        if float(candidate_detector.get("endpoint_fraction", 1.0)) >= 0.95:
            failures.append("detector: candidate endpoint fraction must be below 0.95")

    candidate = report.get("candidate")
    comparators = report.get("comparators") or {}
    if candidate is not None:
        candidate_blocks = {
            "prompt_test": candidate.get("prompt_test"),
            "humanized_hard": candidate.get("humanized_hard"),
            **(candidate.get("ood") or {}),
        }
        for dataset_name, metrics in candidate_blocks.items():
            if metrics is None:
                failures.append(f"candidate {dataset_name}: metrics missing")
            else:
                _append_integrity_failures(
                    failures, f"candidate {dataset_name}", metrics
                )

        prompt = candidate.get("prompt_test") or {}
        coverage_n = int(prompt.get("conformal_truth_coverage_n", prompt.get("n", 0)))
        coverage_count = int(prompt.get("conformal_truth_coverage_count", -1))
        if coverage_n <= 0:
            failures.append("candidate prompt_test: conformal coverage denominator missing")
        else:
            lower, upper = binomial_target_interval(coverage_n, 0.95)
            if not lower <= coverage_count <= upper:
                failures.append(
                    "candidate prompt_test: conformal coverage count "
                    f"{coverage_count} outside [{lower}, {upper}]"
                )

        for comparator_name, comparator in comparators.items():
            comparator_blocks = {
                "prompt_test": comparator.get("prompt_test"),
                "humanized_hard": comparator.get("humanized_hard"),
                **(comparator.get("ood") or {}),
            }
            for dataset_name, metrics in comparator_blocks.items():
                if metrics is None:
                    failures.append(
                        f"comparator {comparator_name} {dataset_name}: metrics missing"
                    )
                else:
                    _append_integrity_failures(
                        failures,
                        f"comparator {comparator_name} {dataset_name}",
                        metrics,
                    )

            baseline_prompt = comparator.get("prompt_test") or {}
            for metric in ("brier", "log_loss"):
                if float(prompt.get(metric, math.inf)) > float(
                    baseline_prompt.get(metric, -math.inf)
                ) + _COMPARISON_EPSILON:
                    failures.append(
                        f"{comparator_name} prompt_test {metric}: candidate regressed"
                    )
            for metric in ("balanced_accuracy", "auroc"):
                if not _comparison_passes(
                    prompt.get(metric, -math.inf),
                    baseline_prompt.get(metric, math.inf),
                    0.01,
                ):
                    label = "balanced accuracy" if metric == "balanced_accuracy" else metric
                    failures.append(
                        f"{comparator_name} prompt_test {label}: regression exceeds 0.01"
                    )
            if float(prompt.get("human_fpr", math.inf)) > float(
                baseline_prompt.get("human_fpr", -math.inf)
            ) + _COMPARISON_EPSILON:
                failures.append(
                    f"{comparator_name} prompt_test Human FPR: candidate regressed"
                )

            candidate_humanized = candidate.get("humanized_hard") or {}
            baseline_humanized = comparator.get("humanized_hard") or {}
            for metric in ("auroc", "tpr_at_fpr_05", "tpr_at_fpr_10"):
                if not _comparison_passes(
                    candidate_humanized.get(metric, -math.inf),
                    baseline_humanized.get(metric, math.inf),
                    0.02,
                ):
                    failures.append(
                        f"{comparator_name} humanized_hard {metric}: regression exceeds 0.02"
                    )

            for corpus in ("ood_m4_test", "ood_raid_test", "ood_mage_test"):
                candidate_ood = (candidate.get("ood") or {}).get(corpus) or {}
                baseline_ood = (comparator.get("ood") or {}).get(corpus) or {}
                if not _comparison_passes(
                    candidate_ood.get("auroc", -math.inf),
                    baseline_ood.get("auroc", math.inf),
                    0.02,
                ):
                    failures.append(
                        f"{comparator_name} {corpus} auroc: regression exceeds 0.02"
                    )

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

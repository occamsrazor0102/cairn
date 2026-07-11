"""Fit per-detector isotonic calibration from a labeled JSONL dataset.

Usage:
    python scripts/calibrate_detectors.py \
        --dataset data/seed_dataset.jsonl \
        --config configs/default.yaml \
        --out models/detector_calibration.json

Runs the pipeline on each sample, collects (raw_score, label) per detector,
and fits a DetectorCalibrator. Execution of this script depends on a labeled
dataset being available (Phase 7).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

from cairn.calibration import CalibrationMetadata, DetectorCalibrator
from cairn.calibration_training import (
    SourceLengthWeights,
    compute_source_length_weights,
)
from cairn.contracts import LayerResult, RouterDecision
from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import EvalSample, load_dataset
from cairn.feature_cache import load_cache, text_hash
from cairn.pipeline import CairnPipeline, normalize_text

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(it, **_kw):
        return it


_CALIBRATOR_REJECTION_SIGNAL_KEYS = frozenset(
    {
        "calibrator_missing_raw",
        "calibrator_nonfinite_raw",
        "calibrator_out_of_domain",
        "calibrator_signature_mismatch",
        "calibrator_invalid_artifact",
    }
)


@dataclass(frozen=True)
class _ObservationSelection:
    accepted: tuple[tuple[LayerResult, EvalSample], ...]
    observed_rows: int
    rejected_reasons: dict[str, int]


def _select_matching_observations(
    observations: Iterable[tuple[LayerResult, EvalSample]],
    detector_id: str,
    detector_signature: str,
    raw_channel: str,
) -> _ObservationSelection:
    accepted: list[tuple[LayerResult, EvalSample]] = []
    rejected: Counter[str] = Counter()
    observed_rows = 0
    for result, sample in observations:
        observed_rows += 1
        if result.layer_id != detector_id:
            rejected["detector_id_mismatch"] += 1
            continue
        if result.determination == "SKIP":
            rejected["inactive"] += 1
            continue
        if result.signals.get("calibration_signature") != detector_signature:
            rejected["signature_mismatch"] += 1
            continue
        if result.signals.get("calibration_raw_channel") != raw_channel:
            rejected["raw_channel_mismatch"] += 1
            continue
        if result.raw_score is None:
            rejected["missing_raw_score"] += 1
            continue
        try:
            raw_score = float(result.raw_score)
        except (TypeError, ValueError):
            rejected["nonfinite_raw_score"] += 1
            continue
        if not math.isfinite(raw_score):
            rejected["nonfinite_raw_score"] += 1
            continue
        accepted.append((result, sample))
    return _ObservationSelection(
        accepted=tuple(accepted),
        observed_rows=observed_rows,
        rejected_reasons=dict(sorted(rejected.items())),
    )


def _load_merge_base(merge_from: Path) -> DetectorCalibrator:
    payload = json.loads(merge_from.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("merge source must be a JSON object")
    if any(not isinstance(key, str) or not isinstance(value, dict)
           for key, value in payload.items()):
        raise ValueError("merge source entries must be JSON objects")

    calibrator = DetectorCalibrator()
    calibrator.load(merge_from)
    # Preserve every parsed entry exactly until a fit deliberately replaces
    # that detector. DetectorCalibrator.fit() removes the target from this
    # map, while save() emits untouched entries verbatim from it.
    calibrator._serialized_entries = {
        key: dict(value) for key, value in payload.items()
    }
    return calibrator


def _build_training_manifest(
    selection: _ObservationSelection,
    weighted: SourceLengthWeights,
) -> dict[str, object]:
    accepted = selection.accepted
    scores = [float(result.raw_score) for result, _ in accepted]
    class_counts = Counter(int(sample.label) for _, sample in accepted)
    return {
        "observed_rows": selection.observed_rows,
        "accepted_rows": len(accepted),
        "rejected_rows": selection.observed_rows - len(accepted),
        "accepted_reasons": {"matched": len(accepted)},
        "rejected_reasons": dict(sorted(selection.rejected_reasons.items())),
        "class_counts": {
            str(label): int(class_counts.get(label, 0)) for label in (0, 1)
        },
        "support": {"min": min(scores), "max": max(scores)},
        "weighting_policy": weighted.manifest,
    }


def fit_isolated_detector_calibration(
    observations: Iterable[tuple[LayerResult, EvalSample]],
    *,
    detector_id: str,
    merge_from: Path,
    detector_signature: str,
    raw_channel: str,
    training_dataset_hash: str,
    min_class_rows: int = 500,
) -> tuple[DetectorCalibrator, dict[str, object]]:
    selection = _select_matching_observations(
        observations,
        detector_id,
        detector_signature,
        raw_channel,
    )
    accepted = list(selection.accepted)
    weighted = compute_source_length_weights(
        [sample for _, sample in accepted]
    )
    class_counts = Counter(int(sample.label) for _, sample in accepted)
    if min(class_counts.get(0, 0), class_counts.get(1, 0)) < min_class_rows:
        raise ValueError(
            f"accepted class counts {dict(class_counts)} are below "
            f"minimum {min_class_rows}"
        )
    calibrator = _load_merge_base(merge_from)
    calibrator.fit(
        detector_id,
        [float(result.raw_score) for result, _ in accepted],
        [int(sample.label) for _, sample in accepted],
        sample_weight=weighted.weights,
        metadata=CalibrationMetadata(
            detector_signature=detector_signature,
            raw_channel=raw_channel,
            training_dataset_hash=training_dataset_hash,
            weighting_policy=weighted.manifest,
        ),
    )
    return calibrator, _build_training_manifest(selection, weighted)


def _matching_result(
    results: Iterable[LayerResult],
    detector_id: str,
    sample: EvalSample,
) -> LayerResult:
    matching = [result for result in results if result.layer_id == detector_id]
    if len(matching) != 1:
        raise ValueError(
            f"sample {sample.id!r} has {len(matching)} results for "
            f"detector {detector_id!r}; expected exactly one"
        )
    return matching[0]


def _require_native_cached_result(
    result: LayerResult,
    detector_id: str,
    sample: EvalSample,
) -> None:
    mutation_signals = sorted(
        _CALIBRATOR_REJECTION_SIGNAL_KEYS.intersection(result.signals)
    )
    if not result.calibrator_applied and not mutation_signals:
        return
    raise ValueError(
        f"cached target result for sample {sample.id!r} detector "
        f"{detector_id!r} is not native: "
        f"calibrator_applied={result.calibrator_applied!r}, "
        f"guard_mutation_signals={mutation_signals}"
    )


def _collect_isolated_observations(
    samples: Sequence[EvalSample],
    *,
    detector_id: str,
    config_path: Path,
    features_cache: Path | None,
) -> list[tuple[LayerResult, EvalSample]]:
    observations: list[tuple[LayerResult, EvalSample]] = []
    if features_cache is not None:
        cache = load_cache(features_cache)
        for sample in tqdm(samples, desc="analyzing", unit="sample"):
            normalized = normalize_text(sample.text)
            key = text_hash(normalized)
            cached = cache.get(key)
            if cached is None:
                raise KeyError(
                    f"text not present in feature cache (hash={key[:12]}...)"
                )
            router = RouterDecision(**cached["router"])
            normalized_word_count = len(normalized.split())
            if router.word_count != normalized_word_count:
                raise ValueError(
                    f"cached router word count {router.word_count} does not match "
                    f"normalized text count {normalized_word_count} for "
                    f"sample {sample.id!r}"
                )
            results = [LayerResult(**row) for row in cached["results"]]
            target_result = _matching_result(results, detector_id, sample)
            _require_native_cached_result(target_result, detector_id, sample)
            observations.append((target_result, sample))
        return observations

    pipeline = CairnPipeline.from_config_file(config_path)
    # Isolated fitting consumes native raw results. A configured calibrator
    # would otherwise mutate determination/p_llm inside analyze_detailed.
    pipeline._detector_calibrator = None
    for sample in tqdm(samples, desc="analyzing", unit="sample"):
        _, results, _ = pipeline.analyze_detailed(sample.text)
        observations.append((_matching_result(results, detector_id, sample), sample))
    return observations


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _paths_alias(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    if left.exists() and right.exists():
        try:
            return os.path.samefile(left, right)
        except OSError:
            return False
    return False


def _save_isolated_outputs(
    calibrator: DetectorCalibrator,
    *,
    merge_from: Path,
    out: Path,
    detector_id: str,
    manifest_out: Path,
    manifest: dict[str, object],
) -> None:
    if _paths_alias(out, merge_from):
        raise ValueError("output path aliases merge source")
    if _paths_alias(manifest_out, merge_from):
        raise ValueError("manifest path aliases merge source")
    if _paths_alias(manifest_out, out):
        raise ValueError("manifest path aliases calibration output")
    source_bytes = merge_from.read_bytes()
    base_payload = json.loads(source_bytes)
    out.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(out)
    output_payload = json.loads(out.read_text(encoding="utf-8"))
    base_non_target = {
        key: value for key, value in base_payload.items()
        if key != detector_id
    }
    output_non_target = {
        key: value for key, value in output_payload.items()
        if key != detector_id
    }
    if output_non_target != base_non_target:
        raise RuntimeError("isolated calibration changed non-target entries")
    if merge_from.read_bytes() != source_bytes:
        raise RuntimeError("isolated calibration changed merge source bytes")
    _write_json(manifest_out, manifest)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--features-cache", type=Path, default=None,
                    help="Path to a precomputed feature cache (built by "
                         "scripts/cache_pipeline_features.py). When set, "
                         "skip detector inference and read raw_scores from "
                         "the cache. Massively faster for ablation runs.")
    ap.add_argument("--only-detector")
    ap.add_argument("--merge-from", type=Path)
    ap.add_argument("--detector-signature")
    ap.add_argument("--raw-channel")
    ap.add_argument("--source-length-weighting", action="store_true")
    ap.add_argument("--manifest-out", type=Path)
    return ap


def _validate_mode_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    isolated_values = (
        args.merge_from,
        args.detector_signature,
        args.raw_channel,
        args.source_length_weighting,
        args.manifest_out,
    )
    if args.only_detector is None:
        if any(value is not None and value is not False for value in isolated_values):
            parser.error("isolated calibration flags require --only-detector")
        return

    required = (
        ("--merge-from", args.merge_from),
        ("--detector-signature", args.detector_signature),
        ("--raw-channel", args.raw_channel),
        ("--source-length-weighting", args.source_length_weighting),
        ("--manifest-out", args.manifest_out),
    )
    for flag, value in required:
        if value is None or value is False:
            parser.error(f"--only-detector requires {flag}")
    if _paths_alias(args.out, args.merge_from):
        parser.error("--out must differ from --merge-from")
    if (
        _paths_alias(args.manifest_out, args.merge_from)
        or _paths_alias(args.manifest_out, args.out)
    ):
        parser.error("--manifest-out must differ from --merge-from and --out")


def _run_legacy_mode(args: argparse.Namespace, samples: Sequence[EvalSample]) -> None:
    cache = load_cache(args.features_cache) if args.features_cache else None
    pipeline = (
        None
        if cache is not None
        else CairnPipeline.from_config_file(args.config)
    )
    # Calibration fitting must consume the detector's native raw result.
    if pipeline is not None:
        pipeline._detector_calibrator = None
    scores: dict[str, list[float]] = defaultdict(list)
    labels: dict[str, list[int]] = defaultdict(list)
    for sample in tqdm(samples, desc="analyzing", unit="sample"):
        if cache is not None:
            key = text_hash(normalize_text(sample.text))
            cached = cache.get(key)
            if cached is None:
                raise KeyError(
                    f"text not present in feature cache (hash={key[:12]}...)"
                )
            results = [LayerResult(**row) for row in cached["results"]]
            for result in results:
                _require_native_cached_result(result, result.layer_id, sample)
        else:
            assert pipeline is not None
            _, results, _ = pipeline.analyze_detailed(sample.text)
        for result in results:
            signature = result.signals.get("calibration_signature")
            raw_channel = result.signals.get("calibration_raw_channel")
            if signature is not None or raw_channel is not None:
                raise ValueError(
                    "legacy all-detector calibration encountered a signed "
                    f"calibration observation for detector {result.layer_id!r}; "
                    "signed observations require isolated v2 training. Re-run "
                    f"with --only-detector {result.layer_id} "
                    f"--detector-signature {signature} "
                    f"--raw-channel {raw_channel} and the remaining isolated "
                    "calibration flags."
                )
            if result.determination == "SKIP":
                continue
            scores[result.layer_id].append(float(result.raw_score))
            labels[result.layer_id].append(int(sample.label))

    calibrator = DetectorCalibrator()
    for detector_id in scores:
        calibrator.fit(detector_id, scores[detector_id], labels[detector_id])
        print(
            f"fit {detector_id}: n={len(scores[detector_id])} "
            f"has={calibrator.has(detector_id)}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(args.out)
    print(f"wrote {args.out}")


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_mode_args(parser, args)
    samples = load_dataset(args.dataset)
    if args.only_detector is None:
        _run_legacy_mode(args, samples)
        return

    dataset_hash = _sha256_file(args.dataset)
    observations = _collect_isolated_observations(
        samples,
        detector_id=args.only_detector,
        config_path=args.config,
        features_cache=args.features_cache,
    )
    calibrator, training_manifest = fit_isolated_detector_calibration(
        observations,
        detector_id=args.only_detector,
        merge_from=args.merge_from,
        detector_signature=args.detector_signature,
        raw_channel=args.raw_channel,
        training_dataset_hash=dataset_hash,
    )
    manifest = {
        "dataset_path": str(args.dataset),
        "dataset_sha256": dataset_hash,
        "features_cache_path": (
            str(args.features_cache) if args.features_cache is not None else None
        ),
        "features_cache_sha256": (
            _sha256_file(args.features_cache)
            if args.features_cache is not None
            else None
        ),
        "detector_id": args.only_detector,
        "detector_signature": args.detector_signature,
        "raw_channel": args.raw_channel,
        **training_manifest,
    }
    _save_isolated_outputs(
        calibrator,
        merge_from=args.merge_from,
        out=args.out,
        detector_id=args.only_detector,
        manifest_out=args.manifest_out,
        manifest=manifest,
    )
    print(
        f"fit {args.only_detector}: n={training_manifest['accepted_rows']} "
        f"has={calibrator.has(args.only_detector)}"
    )
    print(f"wrote {args.out}")
    print(f"wrote {args.manifest_out}")


if __name__ == "__main__":
    main()

"""Build and validate current-schema TOCSIN feature-cache artifacts."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

from cairn.detectors.token_cohesiveness import (
    build_tocsin_calibration_signature,
    TOCSIN_SIGNATURE_TO_RAW_CHANNEL,
    _DELETION_RATIO,
    _MIN_WORDS,
    _N_COPIES,
)
from cairn.normalizer import normalize_text


_TOCSIN_LAYER_ID = "token_cohesiveness"
_API_LAYER_IDS = frozenset({"contrastive_gen", "dna_gpt"})
_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RAW_CHANNEL_ABS_TOLERANCE = 1e-5


def _validated_calibration_identity(
    identity: object,
) -> tuple[str, str]:
    if not isinstance(identity, dict):
        raise ValueError("calibration_identity must be an object")
    channel = identity.get("raw_channel")
    if channel == "combined_w":
        required = {
            "raw_channel",
            "tocsin_algorithm",
            "bart_model",
            "base_algorithm",
            "base_model",
        }
    elif channel == "cohesiveness_u":
        required = {"raw_channel", "tocsin_algorithm", "bart_model"}
    else:
        raise ValueError("calibration_identity has an unsupported raw_channel")
    if set(identity) != required or not all(
        isinstance(identity[field], str) and bool(identity[field].strip())
        for field in required
    ):
        raise ValueError("calibration_identity fields are malformed")
    try:
        expected_signature, expected_identity = (
            build_tocsin_calibration_signature(
                channel,
                bart_model=identity["bart_model"],
                base_model=identity.get("base_model"),
            )
        )
    except ValueError as exc:
        raise ValueError(f"calibration_identity is malformed: {exc}") from exc
    if identity != expected_identity:
        raise ValueError(
            "calibration signature identity uses unsupported algorithm metadata"
        )
    return expected_signature, channel


@dataclass(frozen=True)
class TocsinCacheSummary:
    dataset_sha256: str
    cache_sha256: str
    expected_unique_hashes: int
    cached_unique_hashes: int
    active_rows: int
    skip_rows: int
    calibration_signature: str | None
    calibration_raw_channel: str | None
    raw_min: float | None
    raw_max: float | None
    identical_duplicate_hashes: int
    api_layer_ids: tuple[str, ...]
    layer_counts: dict[str, dict[str, int]]
    unexpected_layer_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalized_path(path: Path) -> str:
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_file(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"artifact path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"artifact path is not a file: {path}")


def _jsonl_objects(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    _require_file(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: line {line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}: line {line_number}: JSONL row must be an object"
                )
            yield line_number, value


def _dataset_inventory(
    dataset_path: Path,
) -> tuple[list[str], dict[str, int]]:
    hash_order: list[str] = []
    word_counts: dict[str, int] = {}
    for line_number, row in _jsonl_objects(dataset_path):
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(
                f"{dataset_path}: line {line_number}: missing string field 'text'"
            )
        normalized = normalize_text(text)
        key = _text_hash(normalized)
        if key not in word_counts:
            hash_order.append(key)
            word_counts[key] = len(normalized.split())
    return hash_order, word_counts


def _cache_records(
    cache_path: Path,
) -> tuple[dict[str, dict[str, object]], set[str]]:
    records: dict[str, dict[str, object]] = {}
    identical_duplicate_hashes: set[str] = set()
    for line_number, record in _jsonl_objects(cache_path):
        key = record.get("hash")
        if not isinstance(key, str) or _FULL_SHA256.fullmatch(key) is None:
            raise ValueError(
                f"{cache_path}: line {line_number}: hash={key!r}: "
                "layer=<record>: cache hash must be a full lowercase SHA-256"
            )
        results = record.get("results")
        if not isinstance(results, list):
            raise ValueError(
                f"{cache_path}: hash={key}: layer=<record>: results must be a list"
            )
        for result in results:
            if not isinstance(result, dict):
                raise ValueError(
                    f"{cache_path}: hash={key}: layer=<unknown>: result must be an object"
                )
        router = record.get("router")
        if not isinstance(router, dict):
            raise ValueError(
                f"{cache_path}: hash={key}: layer=<router>: router must be an object"
            )
        skip_detectors = router.get("skip_detectors")
        if not isinstance(skip_detectors, list) or not all(
            isinstance(layer_id, str) for layer_id in skip_detectors
        ):
            raise ValueError(
                f"{cache_path}: hash={key}: layer=<router>: "
                "skip_detectors must be a list of layer IDs"
            )

        previous = records.get(key)
        if previous is None:
            records[key] = record
        elif previous == record:
            identical_duplicate_hashes.add(key)
        else:
            raise ValueError(
                f"{cache_path}: hash={key}: layer=<record>: "
                "conflicting duplicate hash"
            )
    return records, identical_duplicate_hashes


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _record_layer_counts(
    records: dict[str, dict[str, object]],
    expected_layer_ids: tuple[str, ...],
) -> tuple[
    dict[str, dict[str, int]], tuple[str, ...], tuple[str, str] | None
]:
    observed_layer_ids = {
        str(result.get("layer_id"))
        for record in records.values()
        for result in record["results"]  # type: ignore[index]
        if isinstance(result.get("layer_id"), str)
    }
    unexpected = (
        tuple(sorted(observed_layer_ids - set(expected_layer_ids)))
        if expected_layer_ids
        else ()
    )
    report_ids = list(dict.fromkeys(expected_layer_ids))
    report_ids.extend(sorted(observed_layer_ids - set(report_ids)))
    counts = {
        layer_id: {
            "present": 0,
            "active": 0,
            "skip": 0,
            "router_skipped": 0,
            "unexpected_absent": 0,
        }
        for layer_id in report_ids
    }
    first_unexpected_absent: tuple[str, str] | None = None

    for key, record in records.items():
        results = record["results"]
        router = record["router"]
        skip_detectors = set(router["skip_detectors"])
        by_layer: dict[str, list[dict[str, object]]] = {}
        for result in results:  # type: ignore[assignment]
            layer_id = result.get("layer_id")
            if isinstance(layer_id, str):
                by_layer.setdefault(layer_id, []).append(result)

        for layer_id in report_ids:
            layer_results = by_layer.get(layer_id, [])
            if layer_id in skip_detectors:
                counts[layer_id]["router_skipped"] += 1
            if layer_results:
                counts[layer_id]["present"] += 1
                if any(
                    result.get("determination") != "SKIP"
                    for result in layer_results
                ):
                    counts[layer_id]["active"] += 1
                else:
                    counts[layer_id]["skip"] += 1
            elif layer_id in expected_layer_ids and layer_id not in skip_detectors:
                counts[layer_id]["unexpected_absent"] += 1
                if first_unexpected_absent is None:
                    first_unexpected_absent = (key, layer_id)

    return counts, unexpected, first_unexpected_absent


def _validate_tocsin_result(
    cache_path: Path,
    key: str,
    result: dict[str, object],
) -> tuple[str, str, float]:
    layer = _TOCSIN_LAYER_ID
    signals = result.get("signals")
    if not isinstance(signals, dict):
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: signals must be an object"
        )
    if "mean_deletion_impact" in signals:
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: retired "
            "mean_deletion_impact evidence is forbidden"
        )
    if result.get("min_text_length") != _MIN_WORDS:
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: TOCSIN "
            f"min_text_length must equal {_MIN_WORDS}"
        )

    signature = signals.get("calibration_signature")
    if signature is None:
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: missing TOCSIN "
            "calibration_signature"
        )
    if not isinstance(signature, str) or not signature:
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: unsupported TOCSIN "
            f"calibration_signature={signature!r}"
        )
    channel = signals.get("calibration_raw_channel")
    identity = signals.get("calibration_identity")
    if signature in TOCSIN_SIGNATURE_TO_RAW_CHANNEL and identity is None:
        expected_channel = TOCSIN_SIGNATURE_TO_RAW_CHANNEL[signature]
    else:
        if identity is None:
            raise ValueError(
                f"{cache_path}: hash={key}: layer={layer}: custom TOCSIN "
                "signature requires calibration_identity"
            )
        try:
            expected_signature, expected_channel = (
                _validated_calibration_identity(identity)
            )
        except ValueError as exc:
            raise ValueError(
                f"{cache_path}: hash={key}: layer={layer}: {exc}"
            ) from exc
        if signature != expected_signature:
            raise ValueError(
                f"{cache_path}: hash={key}: layer={layer}: calibration "
                "signature disagrees with calibration_identity"
            )
    if channel != expected_channel:
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: raw channel {channel!r} "
            f"disagrees with signature {signature!r}; expected {expected_channel!r}"
        )

    raw_score = result.get("raw_score")
    if not _is_finite_number(raw_score):
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: nonfinite TOCSIN "
            f"raw_score={raw_score!r}"
        )
    channel_value = signals.get(expected_channel)
    if not _is_finite_number(channel_value):
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: nonfinite TOCSIN "
            f"signal {expected_channel}={channel_value!r}"
        )
    for signal_name in ("cohesiveness_u", "combined_w"):
        signal_value = signals.get(signal_name)
        if not _is_finite_number(signal_value):
            raise ValueError(
                f"{cache_path}: hash={key}: layer={layer}: nonfinite TOCSIN "
                f"signal {signal_name}={signal_value!r}"
            )
    base_value = signals.get("base_v")
    if base_value is not None and not _is_finite_number(base_value):
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: nonfinite TOCSIN "
            f"signal base_v={base_value!r}"
        )
    if expected_channel == "combined_w" and not _is_finite_number(base_value):
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: combined TOCSIN "
            "signature requires a finite base_v"
        )
    if expected_channel == "cohesiveness_u" and base_value is not None:
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: cohesiveness-only "
            "TOCSIN signature requires base_v=None"
        )
    if not math.isclose(
        float(raw_score),
        float(channel_value),
        rel_tol=0.0,
        abs_tol=_RAW_CHANNEL_ABS_TOLERANCE,
    ):
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: raw_score={raw_score!r} "
            f"disagrees with {expected_channel}={channel_value!r} beyond "
            f"{_RAW_CHANNEL_ABS_TOLERANCE}"
        )
    if signals.get("deletion_ratio") != _DELETION_RATIO:
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: TOCSIN deletion_ratio "
            f"must equal {_DELETION_RATIO}"
        )
    if signals.get("n_copies") != _N_COPIES:
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer}: TOCSIN n_copies "
            f"must equal {_N_COPIES}"
        )
    return signature, expected_channel, float(raw_score)


def validate_tocsin_cache(
    dataset_path: Path,
    cache_path: Path,
    *,
    expected_signature: str | None = None,
    expected_layer_ids: Sequence[str] = (),
    require_all_layers_seen: bool = False,
) -> TocsinCacheSummary:
    """Validate exact dataset coverage and the current TOCSIN cache contract."""
    dataset_path = Path(dataset_path)
    cache_path = Path(cache_path)
    expected_hash_order, word_counts = _dataset_inventory(dataset_path)
    records, identical_duplicates = _cache_records(cache_path)

    expected_hashes = set(expected_hash_order)
    cached_hashes = set(records)
    if expected_hashes != cached_hashes:
        missing = next(
            (key for key in expected_hash_order if key not in cached_hashes), None
        )
        unexpected = next(
            (key for key in records if key not in expected_hashes), None
        )
        raise ValueError(
            f"{cache_path}: layer=<record>: hash coverage mismatch for "
            f"dataset={dataset_path}; "
            f"missing_hash={missing}; unexpected_hash={unexpected}"
        )

    api_layer_ids = tuple(
        sorted(
            {
                str(result.get("layer_id"))
                for record in records.values()
                for result in record["results"]  # type: ignore[index]
                if result.get("layer_id") in _API_LAYER_IDS
            }
        )
    )
    if api_layer_ids:
        layer_id = api_layer_ids[0]
        key = next(
            key
            for key, record in records.items()
            if any(
                result.get("layer_id") == layer_id
                for result in record["results"]  # type: ignore[index]
            )
        )
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer_id}: API-backed layer "
            f"result is forbidden ({layer_id})"
        )

    expected_layers = tuple(dict.fromkeys(str(value) for value in expected_layer_ids))
    layer_counts, unexpected_layer_ids, first_absent = _record_layer_counts(
        records, expected_layers
    )
    if first_absent is not None:
        key, layer_id = first_absent
        raise ValueError(
            f"{cache_path}: hash={key}: layer={layer_id}: unexpected_absent "
            "for router-eligible cache record"
        )
    if require_all_layers_seen:
        never_seen = next(
            (
                layer_id
                for layer_id in expected_layers
                if layer_counts[layer_id]["present"] == 0
            ),
            None,
        )
        if never_seen is not None:
            raise ValueError(
                f"{cache_path}: hash=<all>: layer={never_seen}: enabled layer "
                "was never present"
            )

    signatures: set[str] = set()
    channels: set[str] = set()
    raw_scores: list[float] = []
    active_rows = 0
    skip_rows = 0
    for key, record in records.items():
        tocsin_results = [
            result
            for result in record["results"]  # type: ignore[index]
            if result.get("layer_id") == _TOCSIN_LAYER_ID
        ]
        if len(tocsin_results) > 1:
            raise ValueError(
                f"{cache_path}: hash={key}: layer={_TOCSIN_LAYER_ID}: "
                "more than one TOCSIN result"
            )
        eligible = word_counts[key] >= _MIN_WORDS
        if not tocsin_results:
            if eligible:
                raise ValueError(
                    f"{cache_path}: hash={key}: layer={_TOCSIN_LAYER_ID}: "
                    "eligible row is missing an active TOCSIN result"
                )
            raise ValueError(
                f"{cache_path}: hash={key}: layer={_TOCSIN_LAYER_ID}: short row "
                "must be represented as SKIP with skipped='too_short'"
            )

        result = tocsin_results[0]
        if result.get("min_text_length") != _MIN_WORDS:
            raise ValueError(
                f"{cache_path}: hash={key}: layer={_TOCSIN_LAYER_ID}: TOCSIN "
                f"min_text_length must equal {_MIN_WORDS}"
            )
        if eligible:
            if result.get("determination") == "SKIP":
                raise ValueError(
                    f"{cache_path}: hash={key}: layer={_TOCSIN_LAYER_ID}: "
                    "eligible row is missing an active TOCSIN result"
                )
            signature, channel, raw_score = _validate_tocsin_result(
                cache_path, key, result
            )
            signatures.add(signature)
            channels.add(channel)
            raw_scores.append(raw_score)
            active_rows += 1
        else:
            signals = result.get("signals")
            if (
                result.get("determination") != "SKIP"
                or not isinstance(signals, dict)
                or signals.get("skipped") != "too_short"
            ):
                raise ValueError(
                    f"{cache_path}: hash={key}: layer={_TOCSIN_LAYER_ID}: short "
                    "row must be represented as SKIP with skipped='too_short'"
                )
            if "mean_deletion_impact" in signals:
                raise ValueError(
                    f"{cache_path}: hash={key}: layer={_TOCSIN_LAYER_ID}: "
                    "retired mean_deletion_impact evidence is forbidden"
                )
            raw_score = result.get("raw_score")
            if not _is_finite_number(raw_score):
                raise ValueError(
                    f"{cache_path}: hash={key}: layer={_TOCSIN_LAYER_ID}: "
                    f"nonfinite TOCSIN raw_score={raw_score!r}"
                )
            if signals.get("calibration_signature") is not None or signals.get(
                "calibration_raw_channel"
            ) is not None:
                raise ValueError(
                    f"{cache_path}: hash={key}: layer={_TOCSIN_LAYER_ID}: "
                    "short-row TOCSIN SKIP must not declare calibration identity"
                )
            skip_rows += 1

    if len(signatures) > 1:
        raise ValueError(
            f"{cache_path}: hash=<multiple>: layer={_TOCSIN_LAYER_ID}: mixed "
            f"TOCSIN signatures: {sorted(signatures)}"
        )
    signature = next(iter(signatures), None)
    channel = next(iter(channels), None)
    if expected_signature is not None:
        if signature is None:
            raise ValueError(
                f"{cache_path}: hash=<all>: layer={_TOCSIN_LAYER_ID}: expected "
                f"TOCSIN signature {expected_signature!r}, observed none"
            )
        if signature != expected_signature:
            raise ValueError(
                f"{cache_path}: hash=<all>: layer={_TOCSIN_LAYER_ID}: expected "
                f"TOCSIN signature {expected_signature!r}, observed {signature!r}"
            )

    return TocsinCacheSummary(
        dataset_sha256=_sha256_file(dataset_path),
        cache_sha256=_sha256_file(cache_path),
        expected_unique_hashes=len(expected_hashes),
        cached_unique_hashes=len(cached_hashes),
        active_rows=active_rows,
        skip_rows=skip_rows,
        calibration_signature=signature,
        calibration_raw_channel=channel,
        raw_min=min(raw_scores) if raw_scores else None,
        raw_max=max(raw_scores) if raw_scores else None,
        identical_duplicate_hashes=len(identical_duplicates),
        api_layer_ids=api_layer_ids,
        layer_counts=layer_counts,
        unexpected_layer_ids=unexpected_layer_ids,
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json_atomic(path: Path, payload: object) -> None:
    """Write an indented JSON artifact by replacing a complete temp file."""
    _atomic_write_text(Path(path), json.dumps(payload, indent=2) + "\n")


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, object]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
    )
    _atomic_write_text(path, text)


def build_union_dataset(
    dataset_paths: Sequence[Path], output_path: Path
) -> dict[str, object]:
    """Build a deterministic normalized-text union while preserving full rows."""
    paths = [Path(path) for path in dataset_paths]
    if not paths:
        raise ValueError("at least one dataset path is required")
    output_path = Path(output_path)
    seen: set[str] = set()
    union_rows: list[dict[str, object]] = []
    dataset_entries: list[dict[str, object]] = []
    total_rows = 0
    eligible_rows = 0

    for path in paths:
        input_total = 0
        input_unique = 0
        input_eligible = 0
        input_seen: set[str] = set()
        for line_number, row in _jsonl_objects(path):
            text = row.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"{path}: line {line_number}: missing string field 'text'"
                )
            input_total += 1
            total_rows += 1
            normalized = normalize_text(text)
            key = _text_hash(normalized)
            is_eligible = len(normalized.split()) >= _MIN_WORDS
            if key not in input_seen:
                input_seen.add(key)
                input_unique += 1
                if is_eligible:
                    input_eligible += 1
            if key in seen:
                continue
            seen.add(key)
            union_rows.append(row)
            if is_eligible:
                eligible_rows += 1
        dataset_entries.append(
            {
                "path": _normalized_path(path),
                "sha256": _sha256_file(path),
                "total_rows": input_total,
                "unique_rows": input_unique,
                "duplicate_rows": input_total - input_unique,
                "eligible_rows": input_eligible,
            }
        )

    _write_jsonl_atomic(output_path, union_rows)
    return {
        "datasets": dataset_entries,
        "union": {
            "path": _normalized_path(output_path),
            "sha256": _sha256_file(output_path),
            "total_rows": total_rows,
            "unique_rows": len(union_rows),
            "duplicate_rows": total_rows - len(union_rows),
            "eligible_rows": eligible_rows,
        },
    }


def extract_cache_for_dataset(
    source_cache: Path, dataset_path: Path, output_path: Path
) -> int:
    """Slice a cache in unique dataset-hash order without partial writes."""
    source_cache = Path(source_cache)
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    records, _identical_duplicates = _cache_records(source_cache)
    hash_order, _word_counts = _dataset_inventory(dataset_path)
    for key in hash_order:
        if key not in records:
            raise ValueError(
                f"{source_cache}: hash={key}: layer=<record>: absent from source "
                f"cache while slicing dataset={dataset_path}"
            )
    rows = [records[key] for key in hash_order]
    _write_jsonl_atomic(output_path, rows)
    return len(rows)


def hash_artifacts(paths: Sequence[Path]) -> dict[str, object]:
    """Return deterministic full SHA-256 and byte-size metadata for files."""
    entries: dict[str, object] = {}
    for supplied_path in paths:
        path = Path(supplied_path)
        _require_file(path)
        entries[_normalized_path(path)] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {key: entries[key] for key in sorted(entries)}

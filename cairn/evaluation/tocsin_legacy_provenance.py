"""Truthful retrospective provenance for the July 2026 legacy TOCSIN cache."""
from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Sequence

from cairn.cascade import PHASE1_DETECTORS, PHASE2_DETECTORS, PHASE3_DETECTORS
from cairn.config import load_config
from cairn.detectors.token_cohesiveness import TOCSIN_COMBINED_SIGNATURE
from cairn.tocsin_cache import validate_tocsin_cache


LEGACY_PROVENANCE_STATUS = "VERIFIED_BYTES_LEGACY_IDENTITY_UNBOUND"
PROVENANCE_SCHEMA = "cairn.tocsin-cache-retrospective.v1"
COST_SCHEMA = "cairn.tocsin-cost-retrospective.v1"
SLICE_STEMS = (
    "prompts_v2_train",
    "prompts_only_calib",
    "prompts_v5_test",
    "_humanized_eval",
    "_hard_human_clean",
    "ood_m4_test",
    "ood_raid_test",
    "ood_mage_test",
)
LEGACY_UNBOUND_FIELDS = (
    "modal_output_lineage",
    "modal_output_sha256",
    "actual_base_identity",
    "huggingface_model_revisions",
    "modal_image_digest",
)
BASE_MATCH_TOLERANCE = Decimal("0.00001")


def require_exact_legacy_provenance_contract(
    payload: object,
) -> Mapping[str, object]:
    """Return a legacy provenance object only when every admission field matches."""

    if not isinstance(payload, Mapping):
        raise ValueError("legacy provenance must be an object")
    expected = {
        "schema": PROVENANCE_SCHEMA,
        "status": LEGACY_PROVENANCE_STATUS,
        "diagnostic_allowed": True,
        "promotion_allowed": False,
        "calibration_identity": None,
        "unbound_fields": list(LEGACY_UNBOUND_FIELDS),
    }
    mismatched: list[str] = []
    for field, value in expected.items():
        if field not in payload:
            mismatched.append(field)
        elif field == "diagnostic_allowed":
            if payload[field] is not True:
                mismatched.append(field)
        elif field == "promotion_allowed":
            if payload[field] is not False:
                mismatched.append(field)
        elif payload[field] != value:
            mismatched.append(field)
    if mismatched:
        raise ValueError(
            "legacy provenance contract mismatch: " + ", ".join(mismatched)
        )
    return payload


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"required retrospective source is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(repo_root: Path, path: Path) -> str:
    try:
        root = Path(os.path.abspath(repo_root))
        candidate = Path(os.path.abspath(path))
        return candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"retrospective source is outside repository: {path}") from exc


def _artifact(repo_root: Path, path: Path) -> dict[str, object]:
    path = Path(path)
    return {
        "path": _relative_path(repo_root, path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid {label} {path}: expected JSON object")
    return payload


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite nonnegative decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative decimal") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} must be a finite nonnegative decimal")
    return result


def _signed_decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{label} must be a finite decimal")
    return result


def _enabled_and_disabled_layers(config_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    configured = load_config(Path(config_path)).get("detectors", {})
    if not isinstance(configured, Mapping):
        raise ValueError("validator config is missing detector settings")
    enabled: list[str] = []
    disabled: list[str] = []
    for layer_id in PHASE1_DETECTORS + PHASE2_DETECTORS + PHASE3_DETECTORS:
        settings = configured.get(layer_id)
        if isinstance(settings, Mapping) and settings.get("enabled") is True:
            enabled.append(layer_id)
        elif isinstance(settings, Mapping) and settings.get("enabled") is False:
            disabled.append(layer_id)
    if "token_cohesiveness" not in enabled or "fast_detectgpt" not in enabled:
        raise ValueError("validator config must enable token_cohesiveness and fast_detectgpt")
    return tuple(enabled), tuple(disabled)


def _assert_validation_manifest(
    path: Path,
    summary: Mapping[str, object],
    disabled_layers: Sequence[str],
) -> None:
    declared = _json_object(path, "cache validation manifest")
    expected = _json_value(summary)
    assert isinstance(expected, dict)
    expected["disabled_layer_ids"] = list(disabled_layers)
    if dict(declared) != expected:
        differing = sorted(
            key
            for key in set(declared) | set(expected)
            if declared.get(key) != expected.get(key)
        )
        raise ValueError(
            f"cache validation manifest mismatch: {path}; fields={differing}"
        )


def _input_inventory_by_stem(
    path: Path,
) -> tuple[dict[str, Mapping[str, object]], Mapping[str, object]]:
    payload = _json_object(path, "input corpus manifest")
    datasets = payload.get("datasets")
    union = payload.get("union")
    if not isinstance(datasets, list) or not isinstance(union, Mapping):
        raise ValueError("input corpus manifest has malformed datasets/union")
    by_stem: dict[str, Mapping[str, object]] = {}
    for record in datasets:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise ValueError("input corpus manifest has malformed dataset record")
        stem = Path(str(record["path"])).stem
        if stem in by_stem:
            raise ValueError(f"input corpus manifest has duplicate dataset stem: {stem}")
        by_stem[stem] = record
    if set(by_stem) != set(SLICE_STEMS):
        raise ValueError(
            "input corpus manifest slice inventory mismatch: "
            f"observed={sorted(by_stem)}"
        )
    return by_stem, union


def _validate_inventory_record(
    record: Mapping[str, object],
    *,
    repo_root: Path,
    path: Path,
    summary: Mapping[str, object],
) -> None:
    expected = {
        "path": _relative_path(repo_root, path),
        "sha256": summary["dataset_sha256"],
        "unique_rows": summary["expected_unique_hashes"],
        "eligible_rows": summary["active_rows"],
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"input corpus manifest {path.name} {key} mismatch")
    total = record.get("total_rows")
    duplicates = record.get("duplicate_rows")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or isinstance(duplicates, bool)
        or not isinstance(duplicates, int)
        or total - int(summary["expected_unique_hashes"]) != duplicates
    ):
        raise ValueError(f"input corpus manifest {path.name} row counts mismatch")


def _base_identity_corroboration(cache_path: Path) -> dict[str, object]:
    active = 0
    matched = 0
    max_delta = Decimal("0")
    models: set[str] = set()
    with Path(cache_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            results = row.get("results") if isinstance(row, Mapping) else None
            if not isinstance(results, list):
                raise ValueError(f"{cache_path}:{line_number}: malformed results")
            tocsin = [
                item
                for item in results
                if isinstance(item, Mapping)
                and item.get("layer_id") == "token_cohesiveness"
            ]
            if len(tocsin) != 1 or tocsin[0].get("determination") == "SKIP":
                continue
            active += 1
            signals = tocsin[0].get("signals")
            if not isinstance(signals, Mapping):
                raise ValueError(f"{cache_path}:{line_number}: malformed TOCSIN signals")
            if signals.get("calibration_identity") is not None:
                raise ValueError(
                    f"{cache_path}:{line_number}: legacy cache unexpectedly "
                    "binds calibration_identity"
                )
            if (
                signals.get("calibration_signature") != TOCSIN_COMBINED_SIGNATURE
                or signals.get("calibration_raw_channel") != "combined_w"
            ):
                raise ValueError(f"{cache_path}:{line_number}: static signature mismatch")
            base_rows = [
                item
                for item in results
                if isinstance(item, Mapping)
                and item.get("layer_id") == "fast_detectgpt"
                and item.get("determination") != "SKIP"
            ]
            if len(base_rows) != 1:
                raise ValueError(
                    f"{cache_path}:{line_number}: active TOCSIN lacks one "
                    "active fast_detectgpt result"
                )
            base_signals = base_rows[0].get("signals")
            model = base_signals.get("scoring_model") if isinstance(base_signals, Mapping) else None
            if not isinstance(model, str) or not model:
                raise ValueError(f"{cache_path}:{line_number}: missing standalone scoring_model")
            models.add(model)
            left = _signed_decimal(signals.get("base_v"), "TOCSIN base_v")
            right = _signed_decimal(
                base_rows[0].get("raw_score"), "fast_detectgpt raw_score"
            )
            delta = abs(left - right)
            max_delta = max(max_delta, delta)
            if delta > BASE_MATCH_TOLERANCE:
                raise ValueError(
                    f"{cache_path}:{line_number}: base_v/fast_detectgpt mismatch {delta}"
                )
            matched += 1
    if active == 0:
        raise ValueError("legacy union cache has no active TOCSIN rows")
    if models != {"gpt2"}:
        raise ValueError(
            "standalone Fast-DetectGPT model evidence is not exactly gpt2: "
            f"{sorted(models)}"
        )
    return {
        "matched_rows": matched,
        "mismatched_rows": active - matched,
        "absolute_tolerance": format(BASE_MATCH_TOLERANCE, "f"),
        "max_absolute_delta": format(max_delta, "f"),
        "standalone_scoring_models": sorted(models),
        "interpretation": "corroborating value match only; actual base identity remains unbound",
    }


def _cost_retrospective(
    *,
    repo_root: Path,
    pilot_path: Path,
    full_path: Path,
    pilot: Mapping[str, object],
    full: Mapping[str, object],
    original_cost_cap_usd: Decimal,
) -> dict[str, object]:
    n_eligible = int(pilot["n_timed_eligible"])
    n_ineligible = int(pilot["n_timed_ineligible"])
    n_warmups = int(pilot.get("n_warmups", 1))
    if n_eligible <= 0 or n_ineligible <= 0 or n_warmups <= 0:
        raise ValueError("pilot timing strata and warmups must be positive")
    warmup = _decimal(pilot["warmup_seconds"], "pilot warmup_seconds")
    eligible = _decimal(pilot["eligible_scoring_seconds"], "pilot eligible_scoring_seconds")
    ineligible = _decimal(pilot["ineligible_scoring_seconds"], "pilot ineligible_scoring_seconds")
    elapsed = _decimal(pilot["worker_elapsed_seconds"], "pilot worker_elapsed_seconds")
    unattributed = elapsed - warmup - eligible - ineligible
    if unattributed < 0:
        raise ValueError("pilot unattributed worker seconds is negative")
    timed_rows = n_eligible + n_ineligible
    model = {
        "warmup_seconds": format(warmup / Decimal(n_warmups), "f"),
        "eligible_seconds_per_row": format(eligible / Decimal(n_eligible), "f"),
        "ineligible_seconds_per_row": format(ineligible / Decimal(n_ineligible), "f"),
        "unattributed_seconds_per_row": format(unattributed / Decimal(timed_rows), "f"),
    }
    total_rows = int(full["n_total"])
    eligible_rows = int(full["n_eligible"])
    ineligible_rows = int(full["n_ineligible"])
    if eligible_rows + ineligible_rows != total_rows:
        raise ValueError("full Modal row strata do not sum to n_total")
    projected_seconds = (
        Decimal(model["warmup_seconds"])
        + Decimal(eligible_rows) * Decimal(model["eligible_seconds_per_row"])
        + Decimal(ineligible_rows) * Decimal(model["ineligible_seconds_per_row"])
        + Decimal(total_rows) * Decimal(model["unattributed_seconds_per_row"])
    )
    rate = _decimal(pilot["compute_usd_per_second"], "compute_usd_per_second")
    pilot_actual = _decimal(pilot["run_compute_usd"], "pilot run_compute_usd")
    full_actual = _decimal(full["run_compute_usd"], "full run_compute_usd")
    cumulative_actual = _decimal(full["cumulative_compute_usd"], "full cumulative_compute_usd")
    if pilot_actual + full_actual != cumulative_actual:
        raise ValueError("actual pilot plus full cost does not equal cumulative cost")
    cap = _decimal(pilot["cost_cap_usd"], "effective cost cap")
    if _decimal(full["cost_cap_usd"], "full effective cost cap") != cap:
        raise ValueError("pilot/full cost caps disagree")
    projected_full_cost = projected_seconds * rate
    projected_cumulative = pilot_actual + projected_full_cost
    original_seconds = _decimal(pilot["projected_full_seconds"], "original projected_full_seconds")
    original_cumulative = _decimal(
        pilot["projected_cumulative_usd"],
        "original projected_cumulative_usd",
    )
    return {
        "schema": COST_SCHEMA,
        "source_manifests": {
            "pilot": _artifact(repo_root, pilot_path),
            "full": _artifact(repo_root, full_path),
        },
        "unattributed_worker_seconds": format(unattributed, "f"),
        "corrected_cost_model": model,
        "corrected_projection": {
            "total_rows": total_rows,
            "eligible_rows": eligible_rows,
            "ineligible_rows": ineligible_rows,
            "worker_count": 1,
            "projected_full_seconds": format(projected_seconds, "f"),
            "projected_full_compute_usd": format(projected_full_cost, "f"),
            "projected_cumulative_usd": format(projected_cumulative, "f"),
            "gate": "PASS" if projected_cumulative <= cap else "STOP_COST_CAP",
        },
        "original_projection": {
            "projected_full_seconds": format(original_seconds, "f"),
            "projected_cumulative_usd": format(original_cumulative, "f"),
            "gate": str(pilot.get("gate")),
        },
        "actual_cost": {
            "pilot_run_compute_usd": format(pilot_actual, "f"),
            "full_run_compute_usd": format(full_actual, "f"),
            "cumulative_compute_usd": format(cumulative_actual, "f"),
        },
        "caps": {
            "effective_run_cap_usd": format(cap, "f"),
            "original_total_exposure_cap_usd": format(
                _decimal(original_cost_cap_usd, "original total exposure cap"), "f"
            ),
        },
    }


def generate_retrospective_manifests(
    *,
    repo_root: Path,
    run_root: Path,
    dataset_root: Path,
    config_path: Path,
    git_revision: str,
    expected_active_rows: int = 11926,
    expected_pilot_rows: int = 256,
    expected_pilot_sha256: str | None = None,
    expected_full_sha256: str | None = None,
    original_cost_cap_usd: Decimal = Decimal("15.00"),
) -> tuple[dict[str, object], dict[str, object]]:
    """Revalidate current local bytes and build two deterministic manifests."""

    repo_root = Path(repo_root)
    run_root = Path(run_root)
    dataset_root = Path(dataset_root)
    config_path = Path(config_path)
    manifests = run_root / "manifests"
    caches = run_root / "caches"
    inputs = run_root / "inputs"
    pilot_path = manifests / "modal-pilot.json"
    full_path = manifests / "modal-full.json"
    pilot_sha = _sha256(pilot_path)
    full_sha = _sha256(full_path)
    if expected_pilot_sha256 is not None and pilot_sha != expected_pilot_sha256:
        raise ValueError("pilot manifest SHA-256 mismatch")
    if expected_full_sha256 is not None and full_sha != expected_full_sha256:
        raise ValueError("full manifest SHA-256 mismatch")
    pilot = _json_object(pilot_path, "Modal pilot manifest")
    full = _json_object(full_path, "Modal full manifest")
    if pilot.get("gate") != "PASS" or full.get("gate") != "PASS":
        raise ValueError("source Modal manifests must record gate=PASS")
    if int(pilot.get("n_total", -1)) != expected_pilot_rows:
        raise ValueError("pilot manifest row count mismatch")
    if int(full.get("n_total", -1)) != int(full.get("n_eligible", -2)) + int(
        full.get("n_ineligible", -3)
    ):
        raise ValueError("full Modal row strata mismatch")

    enabled_layers, disabled_layers = _enabled_and_disabled_layers(config_path)
    input_manifest_path = manifests / "input-corpora.json"
    input_by_stem, input_union = _input_inventory_by_stem(input_manifest_path)
    union_dataset = inputs / "all_corpora_union.jsonl"
    union_cache = caches / "all_corpora_union.features.jsonl"
    union_validation = manifests / "all_corpora_union.cache.json"
    union_summary = _json_value(validate_tocsin_cache(
        union_dataset,
        union_cache,
        expected_signature=TOCSIN_COMBINED_SIGNATURE,
        expected_layer_ids=enabled_layers,
        require_all_layers_seen=True,
    ).to_dict())
    assert isinstance(union_summary, dict)
    _assert_validation_manifest(union_validation, union_summary, disabled_layers)
    if int(union_summary["active_rows"]) != expected_active_rows:
        raise ValueError(
            "union active row count mismatch: "
            f"{union_summary['active_rows']} != {expected_active_rows}"
        )
    if pilot.get("source_dataset_sha256") != union_summary["dataset_sha256"]:
        raise ValueError("pilot source dataset SHA-256 mismatch")
    if (
        full.get("source_dataset_sha256") != union_summary["dataset_sha256"]
        or full.get("scored_payload_sha256") != union_summary["dataset_sha256"]
    ):
        raise ValueError("full source/scored dataset SHA-256 mismatch")
    if int(full["n_total"]) != int(union_summary["expected_unique_hashes"]):
        raise ValueError("full Modal row count disagrees with union validation")
    if int(full["n_eligible"]) != int(union_summary["active_rows"]):
        raise ValueError("full Modal eligible rows disagree with union validation")
    _validate_inventory_record(
        input_union,
        repo_root=repo_root,
        path=union_dataset,
        summary=union_summary,
    )

    slices: dict[str, object] = {}
    for stem in SLICE_STEMS:
        dataset = dataset_root / f"{stem}.jsonl"
        cache = caches / f"{stem}.features.jsonl"
        validation = manifests / f"{stem}.cache.json"
        summary = _json_value(validate_tocsin_cache(
            dataset,
            cache,
            expected_signature=TOCSIN_COMBINED_SIGNATURE,
            expected_layer_ids=enabled_layers,
        ).to_dict())
        assert isinstance(summary, dict)
        _assert_validation_manifest(validation, summary, disabled_layers)
        _validate_inventory_record(
            input_by_stem[stem], repo_root=repo_root, path=dataset, summary=summary
        )
        slices[stem] = {
            "dataset": _artifact(repo_root, dataset),
            "cache": _artifact(repo_root, cache),
            "validation_manifest": _artifact(repo_root, validation),
            "validator_summary": summary,
        }

    corroboration = _base_identity_corroboration(union_cache)
    if corroboration["matched_rows"] != expected_active_rows:
        raise ValueError("base_v corroboration row count mismatch")
    validator_source = repo_root / "cairn" / "tocsin_cache.py"
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "status": LEGACY_PROVENANCE_STATUS,
        "diagnostic_allowed": True,
        "promotion_allowed": False,
        "git_revision": git_revision,
        "config": _artifact(repo_root, config_path),
        "validator": {
            "implementation": "cairn.tocsin_cache.validate_tocsin_cache",
            "source": _artifact(repo_root, validator_source),
            "enabled_layer_ids": list(enabled_layers),
            "disabled_layer_ids": list(disabled_layers),
            "expected_signature": TOCSIN_COMBINED_SIGNATURE,
        },
        "source_manifests": {
            "modal_pilot": _artifact(repo_root, pilot_path),
            "modal_full": _artifact(repo_root, full_path),
            "input_corpora": _artifact(repo_root, input_manifest_path),
        },
        "union": {
            "dataset": _artifact(repo_root, union_dataset),
            "cache": _artifact(repo_root, union_cache),
            "validation_manifest": _artifact(repo_root, union_validation),
            "validator_summary": union_summary,
        },
        "slices": slices,
        "observed_static_signature": TOCSIN_COMBINED_SIGNATURE,
        "calibration_identity": None,
        "unbound": {
            "modal_output_lineage": None,
            "modal_output_sha256": None,
            "actual_base_identity": None,
            "huggingface_model_revisions": None,
            "modal_image_digest": None,
        },
        "unbound_fields": list(LEGACY_UNBOUND_FIELDS),
        "corroboration": {
            "active_rows": expected_active_rows,
            "base_v_vs_fast_detectgpt": corroboration,
        },
    }
    cost = _cost_retrospective(
        repo_root=repo_root,
        pilot_path=pilot_path,
        full_path=full_path,
        pilot=pilot,
        full=full,
        original_cost_cap_usd=original_cost_cap_usd,
    )
    return provenance, cost


def _find_repo_root(manifest_path: Path, payload: Mapping[str, object]) -> Path:
    union = payload.get("union")
    dataset = union.get("dataset") if isinstance(union, Mapping) else None
    relative = dataset.get("path") if isinstance(dataset, Mapping) else None
    if not isinstance(relative, str):
        raise ValueError("retrospective manifest is missing union dataset path")
    for parent in (manifest_path.parent, *manifest_path.parents):
        if (parent / relative).is_file():
            return parent
    raise ValueError("cannot resolve retrospective manifest repository root")


def verify_retrospective_cache_manifest(path: Path) -> dict[str, object]:
    """Revalidate every referenced byte and the declared identity status."""

    path = Path(path)
    payload = _json_object(path, "retrospective cache provenance manifest")
    require_exact_legacy_provenance_contract(payload)
    repo_root = _find_repo_root(path, payload)
    union = payload["union"]
    slices = payload.get("slices")
    source = payload.get("source_manifests")
    config = payload.get("config")
    if not all(isinstance(value, Mapping) for value in (union, slices, config)):
        raise ValueError("retrospective cache provenance structure is malformed")
    if not isinstance(source, Mapping):
        raise ValueError("legacy retrospective source manifests are malformed")
    dataset_root = repo_root / str(next(iter(slices.values()))["dataset"]["path"])
    dataset_root = dataset_root.parent
    run_root = (repo_root / str(union["dataset"]["path"])).parent.parent
    expected_pilot = str(source["modal_pilot"]["sha256"])
    expected_full = str(source["modal_full"]["sha256"])
    rebuilt, _cost = generate_retrospective_manifests(
        repo_root=repo_root,
        run_root=run_root,
        dataset_root=dataset_root,
        config_path=repo_root / str(config["path"]),
        git_revision=str(payload.get("git_revision")),
        expected_active_rows=int(payload["corroboration"]["active_rows"]),
        expected_pilot_rows=int(
            _json_object(
                repo_root / str(source["modal_pilot"]["path"]),
                "Modal pilot manifest",
            )["n_total"]
        ),
        expected_pilot_sha256=expected_pilot,
        expected_full_sha256=expected_full,
    )
    if dict(payload) != rebuilt:
        raise ValueError("retrospective cache provenance manifest content mismatch")
    result = dict(payload)
    result["manifest_path"] = _relative_path(repo_root, path)
    result["manifest_sha256"] = _sha256(path)
    return result

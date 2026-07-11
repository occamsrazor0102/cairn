"""Build a measured, provenance-locked feature cache on Modal.

The local entrypoint owns the hard cost gate.  A deterministic pilot measures
eligible and ineligible rows separately, writes its manifest unconditionally,
and projects the pilot plus full-run cost with ``Decimal`` arithmetic.  Later
runs must inherit that measured model before any remote invocation.

Example pilot::

    modal run scripts/modal_cache_features.py \
        --dataset-path outputs/tocsin/inputs/union.jsonl \
        --out-path outputs/tocsin/pilot/pilot.features.jsonl \
        --run-id tocsin-v2-20260711T120000Z-deadbeef1234-pilot \
        --manifest-path outputs/tocsin/manifests/modal-pilot.json \
        --compute-usd-per-second 0.00141996 \
        --pilot-rows 256

The required rate is supplied by the caller and recorded in every manifest so
future runs can substitute then-current Modal prices.  On 2026-07-10 the
accounted H200 worker rate was $0.00141996/s: H200 $0.001261/s, four physical
CPU cores $0.0000524/s, and 48 GiB host memory $0.00010656/s.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import modal

PILOT_SEED = 20260710
TOCSIN_MIN_WORDS = 60
GPU_TYPE = "H200"

REQUIRED_WORKER_MANIFEST_KEYS = frozenset(
    {
        "run_id",
        "source_dataset_sha256",
        "scored_payload_sha256",
        "config_path",
        "resolved_config_sha256",
        "git_revision",
        "n_total",
        "n_eligible",
        "n_ineligible",
        "n_timed_eligible",
        "n_timed_ineligible",
        "n_workers",
        "n_warmups",
        "warmup_seconds",
        "eligible_scoring_seconds",
        "ineligible_scoring_seconds",
        "worker_elapsed_seconds",
        "eligible_rows_per_second",
        "total_rows_per_second",
        "gpu_type",
        "remote_path",
        "python_version",
        "torch_version",
        "transformers_version",
        "cuda_version",
        "pilot_seed",
        "detector_seed",
        "pyproject_sha256",
        "package_versions",
    }
)


def validate_run_id(run_id: str) -> str:
    """Require a unique-looking, path-safe checkpoint identifier."""
    if run_id in {"v8", "default"} or len(run_id) < 12:
        raise ValueError("run_id must be a unique descriptive identifier")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]+", run_id) is None:
        raise ValueError("run_id contains unsafe path characters")
    return run_id


def _jsonl_rows(payload: bytes) -> list[tuple[str, dict[str, object]]]:
    """Decode validated JSONL while retaining each original line."""
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("dataset payload must be UTF-8 JSONL") from exc

    rows: list[tuple[str, dict[str, object]]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"dataset line {line_number} is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"dataset line {line_number} must be a JSON object")
        if not isinstance(value.get("text"), str):
            raise ValueError(
                f"dataset line {line_number} is missing string field 'text'"
            )
        rows.append((line, value))
    if not rows:
        raise ValueError("dataset payload contains no JSONL rows")
    return rows


def _is_eligible_text(text: str) -> bool:
    return len(text.split()) >= TOCSIN_MIN_WORDS


def _payload_counts(payload: bytes) -> tuple[int, int]:
    rows = _jsonl_rows(payload)
    eligible = sum(_is_eligible_text(str(row["text"])) for _line, row in rows)
    return len(rows), eligible


def ensure_unique_normalized_texts(
    payload: bytes,
    *,
    normalize_text: Callable[[str], str],
) -> None:
    """Reject duplicate normalized texts before timing or remote spend."""
    seen: dict[str, int] = {}
    for line_number, (_line, row) in enumerate(_jsonl_rows(payload), start=1):
        normalized = normalize_text(str(row["text"]))
        first_line = seen.get(normalized)
        if first_line is not None:
            raise ValueError(
                "duplicate normalized text at dataset lines "
                f"{first_line} and {line_number}"
            )
        seen[normalized] = line_number


def ensure_new_checkpoint(path: str | Path) -> None:
    """Refuse reused run IDs because earlier billed time is unaccounted."""
    checkpoint = Path(path)
    if checkpoint.exists():
        raise ValueError(
            f"checkpoint {checkpoint} already exists; choose a new unique run_id"
        )


def select_pilot_rows(
    payload: bytes,
    *,
    n_rows: int = 256,
    seed: int = PILOT_SEED,
) -> bytes:
    """Select equal eligible/ineligible strata reproducibly.

    The selected rows are emitted in original source order so the same source
    bytes, row count, and seed always produce the same scored payload.
    """
    if n_rows <= 0 or n_rows % 2:
        raise ValueError("pilot n_rows must be a positive even number")
    rows = _jsonl_rows(payload)
    eligible = [
        index
        for index, (_line, row) in enumerate(rows)
        if _is_eligible_text(str(row["text"]))
    ]
    ineligible = [
        index
        for index, (_line, row) in enumerate(rows)
        if not _is_eligible_text(str(row["text"]))
    ]
    per_stratum = n_rows // 2
    if len(eligible) < per_stratum:
        raise ValueError(
            "pilot requires "
            f"{per_stratum} eligible rows but dataset has {len(eligible)}"
        )
    if len(ineligible) < per_stratum:
        raise ValueError(
            "pilot requires "
            f"{per_stratum} ineligible rows but dataset has {len(ineligible)}"
        )

    rng = random.Random(seed)
    selected = set(rng.sample(eligible, per_stratum))
    selected.update(rng.sample(ineligible, per_stratum))
    return "".join(
        rows[index][0] + "\n" for index in sorted(selected)
    ).encode("utf-8")


def split_payload_shards(payload: bytes, shards: int) -> list[bytes]:
    """Split JSONL contiguously into at most one nonempty shard per row."""
    if shards <= 0:
        raise ValueError("shards must be a positive integer")
    rows = _jsonl_rows(payload)
    worker_count = min(shards, len(rows))
    shard_size = (len(rows) + worker_count - 1) // worker_count
    result: list[bytes] = []
    for index in range(worker_count):
        selected = rows[index * shard_size : (index + 1) * shard_size]
        if not selected:
            continue
        result.append(
            "".join(line + "\n" for line, _row in selected).encode("utf-8")
        )
    return result


def _first_eligible_text(payload: bytes) -> str:
    for _line, row in _jsonl_rows(payload):
        text = str(row["text"])
        if _is_eligible_text(text):
            return text
    raise ValueError("scored payload has no eligible row for warmup")


def project_modal_cost(
    pilot_manifest: Mapping[str, object],
    *,
    total_rows: int,
    eligible_rows: int,
    compute_usd_per_second: Decimal,
    cap_usd: Decimal = Decimal("15.00"),
    full_workers: int = 1,
) -> dict[str, object]:
    """Project the measured pilot plus one complete full run exactly."""
    if full_workers <= 0:
        raise ValueError("full_workers must be positive")
    n_eligible = int(pilot_manifest["n_eligible"])
    n_ineligible = int(pilot_manifest["n_ineligible"])
    eligible_seconds_per_row = (
        Decimal(str(pilot_manifest["eligible_scoring_seconds"]))
        / Decimal(n_eligible)
    )
    ineligible_seconds_per_row = (
        Decimal(str(pilot_manifest["ineligible_scoring_seconds"]))
        / Decimal(n_ineligible)
    )
    pilot_workers = int(pilot_manifest.get("n_warmups", 1))
    warmup_seconds_per_worker = (
        Decimal(str(pilot_manifest["warmup_seconds"]))
        / Decimal(pilot_workers)
    )
    projected_full_seconds = (
        Decimal(full_workers) * warmup_seconds_per_worker
        + Decimal(eligible_rows) * eligible_seconds_per_row
        + Decimal(total_rows - eligible_rows) * ineligible_seconds_per_row
    )
    projected_cumulative_usd = (
        Decimal(str(pilot_manifest["worker_elapsed_seconds"]))
        + projected_full_seconds
    ) * compute_usd_per_second
    return {
        "projected_full_seconds": float(projected_full_seconds),
        "projected_cumulative_usd": format(projected_cumulative_usd, "f"),
        "gate": (
            "PASS"
            if projected_cumulative_usd <= cap_usd
            else "STOP_COST_CAP"
        ),
    }


def cost_model_from_manifest(
    manifest: Mapping[str, object],
) -> dict[str, str]:
    """Convert measured stratum totals into a reusable exact cost model."""
    n_eligible = int(manifest.get("n_timed_eligible", manifest["n_eligible"]))
    n_ineligible = int(
        manifest.get("n_timed_ineligible", manifest["n_ineligible"])
    )
    if n_eligible <= 0 or n_ineligible <= 0:
        raise ValueError(
            "cost model requires nonzero eligible and ineligible timing strata"
        )
    eligible_seconds_per_row = (
        Decimal(str(manifest["eligible_scoring_seconds"]))
        / Decimal(n_eligible)
    )
    ineligible_seconds_per_row = (
        Decimal(str(manifest["ineligible_scoring_seconds"]))
        / Decimal(n_ineligible)
    )
    n_warmups = int(manifest.get("n_warmups", 1))
    if n_warmups <= 0:
        raise ValueError("cost model requires at least one measured warmup")
    warmup_seconds_per_worker = (
        Decimal(str(manifest["warmup_seconds"])) / Decimal(n_warmups)
    )
    return {
        "warmup_seconds": format(warmup_seconds_per_worker, "f"),
        "eligible_seconds_per_row": format(eligible_seconds_per_row, "f"),
        "ineligible_seconds_per_row": format(ineligible_seconds_per_row, "f"),
    }


def project_run_cost(
    prior_manifest: Mapping[str, object],
    *,
    total_rows: int,
    eligible_rows: int,
    compute_usd_per_second: Decimal,
    cap_usd: Decimal = Decimal("15.00"),
    worker_count: int = 1,
) -> dict[str, object]:
    """Project one later run from an inherited model and prior spend."""
    if total_rows < 0 or eligible_rows < 0 or eligible_rows > total_rows:
        raise ValueError("eligible_rows must be between zero and total_rows")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    model = prior_manifest.get("cost_model")
    if not isinstance(model, Mapping):
        raise ValueError("prior manifest is missing cost_model")
    if "cumulative_compute_usd" not in prior_manifest:
        raise ValueError("prior manifest is missing cumulative_compute_usd")

    warmup_seconds = Decimal(str(model["warmup_seconds"]))
    eligible_seconds_per_row = Decimal(
        str(model["eligible_seconds_per_row"])
    )
    ineligible_seconds_per_row = Decimal(
        str(model["ineligible_seconds_per_row"])
    )
    prior_cumulative_usd = Decimal(
        str(prior_manifest["cumulative_compute_usd"])
    )
    projected_run_seconds = (
        Decimal(worker_count) * warmup_seconds
        + Decimal(eligible_rows) * eligible_seconds_per_row
        + Decimal(total_rows - eligible_rows) * ineligible_seconds_per_row
    )
    projected_run_compute_usd = (
        projected_run_seconds * compute_usd_per_second
    )
    projected_cumulative_usd = (
        prior_cumulative_usd + projected_run_compute_usd
    )
    return {
        "projected_run_seconds": float(projected_run_seconds),
        "projected_run_compute_usd": format(projected_run_compute_usd, "f"),
        "projected_cumulative_usd": format(projected_cumulative_usd, "f"),
        "gate": (
            "PASS"
            if projected_cumulative_usd <= cap_usd
            else "STOP_COST_CAP"
        ),
    }


def validate_gate_manifest(
    gate_manifest: Mapping[str, object],
    *,
    source_dataset_sha256: str,
    resolved_config_sha256: str,
    git_revision: str,
) -> Mapping[str, object]:
    """Reject a stale or stopped pilot gate before any remote invocation."""
    if gate_manifest.get("gate") != "PASS":
        raise ValueError("gate manifest must record gate='PASS'")
    expected = {
        "source_dataset_sha256": source_dataset_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "git_revision": git_revision,
    }
    for key, value in expected.items():
        if gate_manifest.get(key) != value:
            raise ValueError(
                f"gate manifest {key} does not match the requested full run"
            )
    return gate_manifest


def validate_prior_financial_state(
    gate_manifest: Mapping[str, object],
    prior_manifest: Mapping[str, object],
) -> None:
    """Ensure an explicit prior cannot discard or replace pilot spend state."""
    gate_model = gate_manifest.get("cost_model")
    prior_model = prior_manifest.get("cost_model")
    if not isinstance(gate_model, Mapping):
        raise ValueError("gate manifest is missing cost_model")
    if not isinstance(prior_model, Mapping):
        raise ValueError("prior manifest is missing cost_model")
    if dict(prior_model) != dict(gate_model):
        raise ValueError("prior manifest cost_model is incompatible with gate")
    if "cumulative_compute_usd" not in gate_manifest:
        raise ValueError("gate manifest is missing cumulative_compute_usd")
    if "cumulative_compute_usd" not in prior_manifest:
        raise ValueError("prior manifest is missing cumulative_compute_usd")
    gate_cumulative = Decimal(str(gate_manifest["cumulative_compute_usd"]))
    prior_cumulative = Decimal(str(prior_manifest["cumulative_compute_usd"]))
    if prior_cumulative < gate_cumulative:
        raise ValueError(
            "prior manifest cumulative_compute_usd omits gate spend"
        )


def measure_scoring_times(
    pipeline: Any,
    rows: Sequence[Mapping[str, object]],
    *,
    normalize_text: Callable[[str], str],
    record_result: Callable[[str, object, object], None],
    clock: Callable[[], float] = time.perf_counter,
    warmup_row: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Warm exactly once on eligible text, then time every scored row."""
    if not rows:
        raise ValueError("cannot time an empty scored payload")
    normalized_rows: list[tuple[str, bool]] = []
    for index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError(f"scored row {index} is missing string field 'text'")
        normalized = normalize_text(text)
        normalized_rows.append((normalized, _is_eligible_text(normalized)))

    if warmup_row is None:
        warmup_text = next(
            (text for text, eligible in normalized_rows if eligible), None
        )
    else:
        raw_warmup_text = warmup_row.get("text")
        if not isinstance(raw_warmup_text, str):
            raise ValueError("warmup row is missing string field 'text'")
        warmup_text = normalize_text(raw_warmup_text)
        if not _is_eligible_text(warmup_text):
            raise ValueError("warmup row must be TOCSIN-eligible")
    if warmup_text is None:
        raise ValueError("scored payload has no eligible row for warmup")

    started = clock()
    pipeline.analyze_detailed(warmup_text)
    warmup_seconds = clock() - started

    eligible_seconds = 0.0
    ineligible_seconds = 0.0
    n_eligible = 0
    n_ineligible = 0
    for normalized, eligible in normalized_rows:
        started = clock()
        _determination, results, router_decision = pipeline.analyze_detailed(
            normalized
        )
        elapsed = clock() - started
        if eligible:
            n_eligible += 1
            eligible_seconds += elapsed
        else:
            n_ineligible += 1
            ineligible_seconds += elapsed
        record_result(normalized, results, router_decision)

    return {
        "n_eligible": n_eligible,
        "n_ineligible": n_ineligible,
        "warmup_seconds": warmup_seconds,
        "eligible_scoring_seconds": eligible_seconds,
        "ineligible_scoring_seconds": ineligible_seconds,
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolved_config_sha256(config_path: str | Path) -> str:
    """Hash the fully merged config, not only the overlay file bytes."""
    from cairn.config import load_config

    resolved = load_config(Path(config_path))
    canonical = json.dumps(
        resolved,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _detector_seed(config_path: str | Path) -> int:
    from cairn.config import load_config

    resolved = load_config(Path(config_path))
    detector = (resolved.get("detectors") or {}).get("token_cohesiveness") or {}
    return int(detector.get("seed", 42))


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if not revision:
        raise RuntimeError("git rev-parse returned an empty revision")
    return revision


def _pyproject_sha256() -> str:
    path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not path.is_file():
        raise FileNotFoundError(f"pyproject.toml not found at {path}")
    return _sha256_bytes(path.read_bytes())


def _decimal_cli(value: str | Decimal, *, name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal number") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{name} must be a finite nonnegative decimal")
    return parsed


def _read_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read manifest {manifest_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest {manifest_path} must contain a JSON object")
    return value


def _write_manifest(path: str | Path, manifest: Mapping[str, object]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def _validate_worker_manifest(
    manifest: Mapping[str, object],
    *,
    run_id: str,
    source_dataset_sha256: str,
    scored_payload_sha256: str,
    resolved_config_sha256: str,
    git_revision: str,
    n_total: int,
    n_eligible: int,
) -> None:
    missing = sorted(REQUIRED_WORKER_MANIFEST_KEYS - manifest.keys())
    if missing:
        raise RuntimeError(f"Modal worker manifest is missing keys: {missing}")
    expected = {
        "run_id": run_id,
        "source_dataset_sha256": source_dataset_sha256,
        "scored_payload_sha256": scored_payload_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "git_revision": git_revision,
        "n_total": n_total,
        "n_eligible": n_eligible,
        "n_ineligible": n_total - n_eligible,
        "n_timed_eligible": n_eligible,
        "n_timed_ineligible": n_total - n_eligible,
        "n_workers": 1,
        "n_warmups": 1,
        "gpu_type": GPU_TYPE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Modal worker manifest {key}={manifest.get(key)!r}; "
                f"expected {value!r}"
            )


def aggregate_worker_manifests(
    worker_manifests: Sequence[Mapping[str, object]],
    *,
    run_id: str,
    scored_payload_sha256: str,
) -> dict[str, object]:
    """Combine all billed worker timing while retaining worker provenance."""
    if not worker_manifests:
        raise ValueError("at least one worker manifest is required")
    workers = [dict(manifest) for manifest in worker_manifests]
    first = workers[0]
    common_keys = (
        "source_dataset_sha256",
        "config_path",
        "resolved_config_sha256",
        "git_revision",
        "gpu_type",
        "python_version",
        "torch_version",
        "transformers_version",
        "cuda_version",
        "pilot_seed",
        "detector_seed",
        "pyproject_sha256",
        "package_versions",
    )
    for index, worker in enumerate(workers[1:], start=1):
        for key in common_keys:
            if worker.get(key) != first.get(key):
                raise RuntimeError(
                    f"worker {index} manifest {key} disagrees with worker 0"
                )

    def summed_decimal(key: str) -> Decimal:
        return sum(
            (Decimal(str(worker[key])) for worker in workers),
            start=Decimal("0"),
        )

    n_total = sum(int(worker["n_total"]) for worker in workers)
    n_eligible = sum(int(worker["n_eligible"]) for worker in workers)
    n_ineligible = sum(int(worker["n_ineligible"]) for worker in workers)
    n_timed_eligible = sum(
        int(worker.get("n_timed_eligible", worker["n_eligible"]))
        for worker in workers
    )
    n_timed_ineligible = sum(
        int(worker.get("n_timed_ineligible", worker["n_ineligible"]))
        for worker in workers
    )
    n_warmups = sum(int(worker.get("n_warmups", 1)) for worker in workers)
    warmup_seconds = summed_decimal("warmup_seconds")
    eligible_scoring_seconds = summed_decimal("eligible_scoring_seconds")
    ineligible_scoring_seconds = summed_decimal("ineligible_scoring_seconds")
    worker_elapsed_seconds = summed_decimal("worker_elapsed_seconds")
    timed_total = n_timed_eligible + n_timed_ineligible

    aggregate = dict(first)
    remote_paths = [str(worker["remote_path"]) for worker in workers]
    aggregate.update(
        {
            "run_id": run_id,
            "scored_payload_sha256": scored_payload_sha256,
            "n_total": n_total,
            "n_eligible": n_eligible,
            "n_ineligible": n_ineligible,
            "n_timed_eligible": n_timed_eligible,
            "n_timed_ineligible": n_timed_ineligible,
            "n_workers": len(workers),
            "n_warmups": n_warmups,
            "warmup_seconds": float(warmup_seconds),
            "eligible_scoring_seconds": float(eligible_scoring_seconds),
            "ineligible_scoring_seconds": float(ineligible_scoring_seconds),
            "worker_elapsed_seconds": float(worker_elapsed_seconds),
            "eligible_rows_per_second": (
                n_timed_eligible / float(eligible_scoring_seconds)
                if eligible_scoring_seconds > 0
                else 0.0
            ),
            "total_rows_per_second": (
                timed_total / float(worker_elapsed_seconds)
                if worker_elapsed_seconds > 0
                else 0.0
            ),
            "remote_path": remote_paths[0] if len(workers) == 1 else remote_paths,
            "worker_remote_paths": remote_paths,
            "worker_run_ids": [str(worker["run_id"]) for worker in workers],
            "worker_manifests": workers,
            "n_new_this_run": sum(
                int(worker.get("n_new_this_run", worker["n_total"]))
                for worker in workers
            ),
            "n_duplicates": sum(
                int(worker.get("n_duplicates", 0)) for worker in workers
            ),
        }
    )
    return aggregate


def require_worker_result_count(
    worker_results: Sequence[object], expected: int
) -> None:
    """Reject incomplete fan-out before aggregating cost or row counts."""
    actual = len(worker_results)
    if actual != expected:
        raise RuntimeError(
            f"Modal returned {actual} manifests; expected {expected}"
        )


def _completed_cost_fields(
    manifest: Mapping[str, object],
    *,
    compute_usd_per_second: Decimal,
    prior_manifest: Mapping[str, object] | None,
) -> dict[str, object]:
    run_compute_usd = (
        Decimal(str(manifest["worker_elapsed_seconds"]))
        * compute_usd_per_second
    )
    prior_cumulative_usd = Decimal("0")
    if prior_manifest is not None:
        if "cumulative_compute_usd" not in prior_manifest:
            raise ValueError("prior manifest is missing cumulative_compute_usd")
        prior_cumulative_usd = Decimal(
            str(prior_manifest["cumulative_compute_usd"])
        )
    if prior_manifest is not None:
        inherited = prior_manifest.get("cost_model")
        if not isinstance(inherited, Mapping):
            raise ValueError("prior manifest is missing cost_model")
        cost_model = dict(inherited)
        cost_model_source = "prior_manifest"
    else:
        cost_model = cost_model_from_manifest(manifest)
        cost_model_source = "current_run"
    return {
        "compute_usd_per_second": format(compute_usd_per_second, "f"),
        "run_compute_usd": format(run_compute_usd, "f"),
        "cumulative_compute_usd": format(
            prior_cumulative_usd + run_compute_usd, "f"
        ),
        "cost_model": cost_model,
        "cost_model_source": cost_model_source,
    }


app = modal.App("cairn-cache-features")

# Container image with all heavy ML dependencies preinstalled.  The complete
# configs directory is mounted because resolved-config hashing follows profile
# inheritance rather than hashing only the requested overlay file.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "sentence-transformers",
        "scikit-learn",
        "numpy",
        "pyyaml",
        "huggingface_hub",
    )
    .add_local_dir("cairn", remote_path="/root/cairn")
    .add_local_dir("configs", remote_path="/root/configs")
)

hf_cache = modal.Volume.from_name("cairn-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="H200",
    cpu=4.0,
    memory=49152,
    timeout=21600,
    secrets=[modal.Secret.from_dotenv()],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def cache_features(
    dataset_jsonl: bytes,
    config_path: str,
    run_id: str,
    source_dataset_sha256: str,
    resolved_config_sha256: str,
    git_revision: str,
    pilot_seed: int | None,
    detector_seed: int,
    pyproject_sha256: str,
    warmup_text: str,
) -> dict[str, object]:
    """Score one payload and return measured runtime/provenance metadata."""
    import importlib.metadata
    import os
    import platform
    import sys

    worker_started = time.perf_counter()
    validate_run_id(run_id)
    sys.path.insert(0, "/root")
    os.chdir("/root")

    from cairn.dotenv import load_dotenv

    load_dotenv()

    import torch
    import transformers

    from cairn.feature_cache import text_hash, write_record
    from cairn.pipeline import CairnPipeline, normalize_text

    actual_config_sha256 = _resolved_config_sha256(config_path)
    if actual_config_sha256 != resolved_config_sha256:
        raise ValueError(
            "resolved config changed between local preflight and Modal worker"
        )
    actual_detector_seed = _detector_seed(config_path)
    if actual_detector_seed != detector_seed:
        raise ValueError(
            "detector seed changed between local preflight and Modal worker"
        )

    parsed_rows = _jsonl_rows(dataset_jsonl)
    rows = [row for _line, row in parsed_rows]
    ensure_unique_normalized_texts(
        dataset_jsonl,
        normalize_text=normalize_text,
    )
    payload_n_eligible = sum(
        _is_eligible_text(normalize_text(str(row["text"]))) for row in rows
    )
    normalized_warmup = normalize_text(warmup_text)
    if not _is_eligible_text(normalized_warmup):
        raise ValueError("warmup text must be TOCSIN-eligible")
    warmup_row = {"text": warmup_text}

    checkpoint_path = (
        f"/root/.cache/huggingface/cairn_cache_output/{run_id}_features.jsonl"
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    ensure_new_checkpoint(checkpoint_path)
    try:
        with open(checkpoint_path, "x", encoding="utf-8"):
            pass
    except FileExistsError as exc:
        raise ValueError(
            "checkpoint was created concurrently; choose a new unique run_id"
        ) from exc
    # Persist the reservation before costly model loading so a failed or
    # concurrent attempt cannot reuse the run ID without accounting for it.
    hf_cache.commit()

    pipeline = CairnPipeline.from_config_file(config_path)
    n_done = 0
    with open(checkpoint_path, "a", encoding="utf-8") as checkpoint:

        def record_result(
            normalized: str, results: object, router_decision: object
        ) -> None:
            nonlocal n_done
            write_record(
                checkpoint,
                text_hash(normalized),
                results,
                router_decision,
            )
            checkpoint.flush()
            n_done += 1
            if n_done % 100 == 0:
                hf_cache.commit()
                print(f"  {n_done} rows scored", flush=True)

        timing = measure_scoring_times(
            pipeline,
            rows,
            normalize_text=normalize_text,
            record_result=record_result,
            warmup_row=warmup_row,
        )

    hf_cache.commit()
    worker_elapsed_seconds = time.perf_counter() - worker_started

    distribution_names = (
        "torch",
        "transformers",
        "sentence-transformers",
        "scikit-learn",
        "numpy",
        "PyYAML",
        "huggingface-hub",
    )
    package_versions: dict[str, str] = {}
    for distribution in distribution_names:
        try:
            package_versions[distribution] = importlib.metadata.version(
                distribution
            )
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = "not-installed"

    eligible_scoring_seconds = float(timing["eligible_scoring_seconds"])
    total_scoring_seconds = eligible_scoring_seconds + float(
        timing["ineligible_scoring_seconds"]
    )
    n_timed_eligible = int(timing["n_eligible"])
    n_timed_ineligible = int(timing["n_ineligible"])
    eligible_rows_per_second = (
        n_timed_eligible / eligible_scoring_seconds
        if eligible_scoring_seconds > 0
        else 0.0
    )
    n_timed_total = n_timed_eligible + n_timed_ineligible
    total_rows_per_second = (
        n_timed_total / worker_elapsed_seconds
        if worker_elapsed_seconds > 0
        else 0.0
    )
    print(
        f"done: {n_done} scored, {total_scoring_seconds:.3f}s timed scoring",
        flush=True,
    )

    return {
        "run_id": run_id,
        "source_dataset_sha256": source_dataset_sha256,
        "scored_payload_sha256": _sha256_bytes(dataset_jsonl),
        "config_path": config_path,
        "resolved_config_sha256": resolved_config_sha256,
        "git_revision": git_revision,
        "n_total": len(rows),
        "n_eligible": payload_n_eligible,
        "n_ineligible": len(rows) - payload_n_eligible,
        "n_timed_eligible": n_timed_eligible,
        "n_timed_ineligible": n_timed_ineligible,
        "n_workers": 1,
        "n_warmups": 1,
        "warmup_seconds": float(timing["warmup_seconds"]),
        "eligible_scoring_seconds": eligible_scoring_seconds,
        "ineligible_scoring_seconds": float(
            timing["ineligible_scoring_seconds"]
        ),
        "worker_elapsed_seconds": worker_elapsed_seconds,
        "eligible_rows_per_second": eligible_rows_per_second,
        "total_rows_per_second": total_rows_per_second,
        "gpu_type": GPU_TYPE,
        "remote_path": f"cairn_cache_output/{run_id}_features.jsonl",
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "cuda_version": str(torch.version.cuda or "unavailable"),
        "pilot_seed": pilot_seed,
        "detector_seed": detector_seed,
        "pyproject_sha256": pyproject_sha256,
        "package_versions": package_versions,
        "n_new_this_run": n_done,
        "n_duplicates": 0,
    }


@app.local_entrypoint()
def main(
    dataset_path: str,
    out_path: str,
    run_id: str,
    manifest_path: str,
    compute_usd_per_second: str,
    config_path: str = "configs/default.yaml",
    pilot_rows: int = 0,
    gate_manifest_path: str | None = None,
    prior_manifest_path: str | None = None,
    cost_cap_usd: str = "15",
    shards: int = 1,
) -> None:
    """Preflight cost/provenance, invoke the workers, and persist results."""
    validate_run_id(run_id)
    if shards <= 0:
        raise ValueError("shards must be a positive integer")
    if pilot_rows not in (0, 256):
        raise ValueError("--pilot-rows must be omitted/0 or exactly 256")
    if pilot_rows and gate_manifest_path is not None:
        raise ValueError("a pilot cannot consume --gate-manifest-path")
    if pilot_rows and prior_manifest_path is not None:
        raise ValueError("a pilot cannot consume --prior-manifest-path")

    compute_rate = _decimal_cli(
        compute_usd_per_second, name="compute_usd_per_second"
    )
    if compute_rate <= 0:
        raise ValueError("compute_usd_per_second must be greater than zero")
    cap = _decimal_cli(cost_cap_usd, name="cost_cap_usd")

    source_payload = Path(dataset_path).read_bytes()
    from cairn.normalizer import normalize_text as normalize_local_text

    ensure_unique_normalized_texts(
        source_payload,
        normalize_text=normalize_local_text,
    )
    source_dataset_sha256 = _sha256_bytes(source_payload)
    source_n_total, source_n_eligible = _payload_counts(source_payload)
    scored_payload = (
        select_pilot_rows(
            source_payload,
            n_rows=pilot_rows,
            seed=PILOT_SEED,
        )
        if pilot_rows
        else source_payload
    )
    scored_n_total, scored_n_eligible = _payload_counts(scored_payload)
    scored_payload_sha256 = _sha256_bytes(scored_payload)
    shard_payloads = split_payload_shards(scored_payload, shards)
    worker_count = len(shard_payloads)
    warmup_text = _first_eligible_text(scored_payload)
    resolved_config_sha256 = _resolved_config_sha256(config_path)
    git_revision = _git_revision()
    pyproject_sha256 = _pyproject_sha256()
    detector_seed = _detector_seed(config_path)

    gate_manifest: dict[str, object] | None = None
    if gate_manifest_path is not None:
        gate_manifest = _read_manifest(gate_manifest_path)
        validate_gate_manifest(
            gate_manifest,
            source_dataset_sha256=source_dataset_sha256,
            resolved_config_sha256=resolved_config_sha256,
            git_revision=git_revision,
        )

    prior_manifest: dict[str, object] | None = None
    prior_projection: dict[str, object] | None = None
    if prior_manifest_path is not None:
        prior_manifest = _read_manifest(prior_manifest_path)
        if gate_manifest is not None:
            validate_prior_financial_state(gate_manifest, prior_manifest)
    elif gate_manifest is not None:
        prior_manifest = gate_manifest

    if prior_manifest is not None:
        prior_projection = project_run_cost(
            prior_manifest,
            total_rows=source_n_total,
            eligible_rows=source_n_eligible,
            compute_usd_per_second=compute_rate,
            cap_usd=cap,
            worker_count=worker_count,
        )
        if prior_projection["gate"] != "PASS":
            raise SystemExit(
                "STOP_COST_CAP: projected cumulative spend is "
                f"${prior_projection['projected_cumulative_usd']}"
            )

    print(
        f"sending {len(scored_payload):,} bytes / {scored_n_total:,} rows "
        f"to {worker_count} Modal {GPU_TYPE} worker(s) ({run_id})...",
        flush=True,
    )
    common_worker_args = (
        config_path,
        source_dataset_sha256,
        resolved_config_sha256,
        git_revision,
        PILOT_SEED if pilot_rows else None,
        detector_seed,
        pyproject_sha256,
        warmup_text,
    )
    worker_run_ids = [
        run_id if worker_count == 1 else f"{run_id}_shard{index}"
        for index in range(worker_count)
    ]
    if worker_count == 1:
        worker_results = [
            cache_features.remote(
                dataset_jsonl=shard_payloads[0],
                config_path=config_path,
                run_id=worker_run_ids[0],
                source_dataset_sha256=source_dataset_sha256,
                resolved_config_sha256=resolved_config_sha256,
                git_revision=git_revision,
                pilot_seed=PILOT_SEED if pilot_rows else None,
                detector_seed=detector_seed,
                pyproject_sha256=pyproject_sha256,
                warmup_text=warmup_text,
            )
        ]
    else:
        starmap_args = [
            (
                shard_payload,
                common_worker_args[0],
                worker_run_ids[index],
                *common_worker_args[1:],
            )
            for index, shard_payload in enumerate(shard_payloads)
        ]
        worker_results = list(cache_features.starmap(starmap_args))
    require_worker_result_count(worker_results, worker_count)

    worker_manifests: list[Mapping[str, object]] = []
    for index, worker_manifest in enumerate(worker_results):
        if not isinstance(worker_manifest, Mapping):
            raise RuntimeError(
                f"Modal worker {index} returned a non-mapping manifest"
            )
        shard_n_total, shard_n_eligible = _payload_counts(shard_payloads[index])
        _validate_worker_manifest(
            worker_manifest,
            run_id=worker_run_ids[index],
            source_dataset_sha256=source_dataset_sha256,
            scored_payload_sha256=_sha256_bytes(shard_payloads[index]),
            resolved_config_sha256=resolved_config_sha256,
            git_revision=git_revision,
            n_total=shard_n_total,
            n_eligible=shard_n_eligible,
        )
        worker_manifests.append(worker_manifest)

    manifest = aggregate_worker_manifests(
        worker_manifests,
        run_id=run_id,
        scored_payload_sha256=scored_payload_sha256,
    )
    manifest.update(
        _completed_cost_fields(
            manifest,
            compute_usd_per_second=compute_rate,
            prior_manifest=prior_manifest,
        )
    )
    manifest["cost_cap_usd"] = format(cap, "f")
    if pilot_rows:
        manifest.update(
            project_modal_cost(
                manifest,
                total_rows=source_n_total,
                eligible_rows=source_n_eligible,
                compute_usd_per_second=compute_rate,
                cap_usd=cap,
                full_workers=worker_count,
            )
        )
    elif prior_projection is not None:
        manifest["prelaunch_cost_projection"] = prior_projection
        manifest["gate"] = "PASS"
    elif gate_manifest is not None:
        manifest["gate"] = "PASS"
    else:
        manifest["gate"] = "PASS"

    if Decimal(str(manifest["cumulative_compute_usd"])) > cap:
        manifest["gate"] = "STOP_COST_CAP"

    # Persist before any deliberate nonzero exit and before volume download.
    _write_manifest(manifest_path, manifest)
    print(f"wrote Modal manifest to {manifest_path}", flush=True)
    if manifest["gate"] != "PASS":
        projected = manifest.get(
            "projected_cumulative_usd", manifest["cumulative_compute_usd"]
        )
        raise SystemExit(
            f"STOP_COST_CAP: projected cumulative spend is ${projected}"
        )

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    n_lines = 0
    n_merged_duplicates = 0
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for worker_manifest in worker_manifests:
            remote_path = str(worker_manifest["remote_path"])
            cache_bytes = bytearray()
            for chunk in hf_cache.read_file(remote_path):
                cache_bytes.extend(chunk)
            for raw in bytes(cache_bytes).decode("utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                key = record.get("hash")
                if not isinstance(key, str):
                    continue
                if key in seen_hashes:
                    n_merged_duplicates += 1
                    continue
                seen_hashes.add(key)
                destination.write(raw + "\n")
                n_lines += 1
    print(
        f"wrote {n_lines:,} cached features to {output} "
        f"({n_merged_duplicates} cross-shard duplicates merged)",
        flush=True,
    )

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
future runs can substitute then-current Modal prices.  On 2026-07-11 the
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


def project_modal_cost(
    pilot_manifest: Mapping[str, object],
    *,
    total_rows: int,
    eligible_rows: int,
    compute_usd_per_second: Decimal,
    cap_usd: Decimal = Decimal("15.00"),
) -> dict[str, object]:
    """Project the measured pilot plus one complete full run exactly."""
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
    projected_full_seconds = (
        Decimal(str(pilot_manifest["warmup_seconds"]))
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
    n_eligible = int(manifest["n_eligible"])
    n_ineligible = int(manifest["n_ineligible"])
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
    return {
        "warmup_seconds": format(
            Decimal(str(manifest["warmup_seconds"])), "f"
        ),
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
) -> dict[str, object]:
    """Project one later run from an inherited model and prior spend."""
    if total_rows < 0 or eligible_rows < 0 or eligible_rows > total_rows:
        raise ValueError("eligible_rows must be between zero and total_rows")
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
        warmup_seconds
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
        "gpu_type": GPU_TYPE,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"Modal worker manifest {key}={manifest.get(key)!r}; "
                f"expected {value!r}"
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
    try:
        cost_model = cost_model_from_manifest(manifest)
        cost_model_source = "current_run"
    except ValueError:
        inherited = None if prior_manifest is None else prior_manifest.get("cost_model")
        if not isinstance(inherited, Mapping):
            raise
        cost_model = {str(key): str(value) for key, value in inherited.items()}
        cost_model_source = "prior_manifest"
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
    payload_n_eligible = sum(
        _is_eligible_text(normalize_text(str(row["text"]))) for row in rows
    )
    warmup_row = next(
        (
            row
            for row in rows
            if _is_eligible_text(normalize_text(str(row["text"])))
        ),
        None,
    )
    if warmup_row is None:
        raise ValueError("scored payload has no eligible row for warmup")

    checkpoint_path = (
        f"/root/.cache/huggingface/cairn_cache_output/{run_id}_features.jsonl"
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    seen: set[str] = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as checkpoint:
            for line in checkpoint:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get("hash")
                if isinstance(key, str):
                    seen.add(key)
        print(
            f"resuming {run_id}: {len(seen)} checkpoint entries found",
            flush=True,
        )

    pending_rows: list[dict[str, object]] = []
    n_duplicates = 0
    for row in rows:
        normalized = normalize_text(str(row["text"]))
        key = text_hash(normalized)
        if key in seen:
            n_duplicates += 1
            continue
        seen.add(key)
        pending_rows.append(row)
    if not pending_rows:
        raise ValueError(
            "run_id has no uncached rows; choose a new unique run_id"
        )

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
            pending_rows,
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
    eligible_rows_per_second = (
        payload_n_eligible / eligible_scoring_seconds
        if eligible_scoring_seconds > 0
        else 0.0
    )
    total_rows_per_second = (
        len(rows) / worker_elapsed_seconds if worker_elapsed_seconds > 0 else 0.0
    )
    print(
        f"done: {n_done} scored, {n_duplicates} checkpoint duplicates, "
        f"{total_scoring_seconds:.3f}s timed scoring",
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
        "n_duplicates": n_duplicates,
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
    """Preflight cost/provenance, invoke one worker, and persist results."""
    validate_run_id(run_id)
    if shards != 1:
        raise ValueError(
            "measured timing requires shards=1 so the pipeline warms exactly once"
        )
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
        gate_projection = gate_manifest.get("projected_cumulative_usd")
        if gate_projection is not None and Decimal(str(gate_projection)) > cap:
            raise ValueError(
                "gate manifest projected_cumulative_usd exceeds current cap"
            )

    prior_manifest: dict[str, object] | None = None
    prior_projection: dict[str, object] | None = None
    if prior_manifest_path is not None:
        prior_manifest = _read_manifest(prior_manifest_path)
        prior_projection = project_run_cost(
            prior_manifest,
            total_rows=source_n_total,
            eligible_rows=source_n_eligible,
            compute_usd_per_second=compute_rate,
            cap_usd=cap,
        )
        if prior_projection["gate"] != "PASS":
            raise SystemExit(
                "STOP_COST_CAP: projected cumulative spend is "
                f"${prior_projection['projected_cumulative_usd']}"
            )

    print(
        f"sending {len(scored_payload):,} bytes / {scored_n_total:,} rows "
        f"to Modal {GPU_TYPE} ({run_id})...",
        flush=True,
    )
    worker_manifest = cache_features.remote(
        dataset_jsonl=scored_payload,
        config_path=config_path,
        run_id=run_id,
        source_dataset_sha256=source_dataset_sha256,
        resolved_config_sha256=resolved_config_sha256,
        git_revision=git_revision,
        pilot_seed=PILOT_SEED if pilot_rows else None,
        detector_seed=detector_seed,
        pyproject_sha256=pyproject_sha256,
    )
    if not isinstance(worker_manifest, Mapping):
        raise RuntimeError("Modal worker returned a non-mapping manifest")
    _validate_worker_manifest(
        worker_manifest,
        run_id=run_id,
        source_dataset_sha256=source_dataset_sha256,
        scored_payload_sha256=scored_payload_sha256,
        resolved_config_sha256=resolved_config_sha256,
        git_revision=git_revision,
        n_total=scored_n_total,
        n_eligible=scored_n_eligible,
    )

    manifest: dict[str, object] = dict(worker_manifest)
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

    remote_path = str(manifest["remote_path"])
    cache_bytes = bytearray()
    for chunk in hf_cache.read_file(remote_path):
        cache_bytes.extend(chunk)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(cache_bytes))
    n_lines = bytes(cache_bytes).count(b"\n")
    print(
        f"wrote {n_lines:,} cached features to {output}",
        flush=True,
    )

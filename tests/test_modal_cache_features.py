"""Offline tests for the measured Modal feature-cache runner.

The real script decorates its worker at import time.  These tests install a
small in-memory ``modal`` stub before loading the module so no test imports the
Modal SDK, authenticates, or opens a network connection.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import sys
import types
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cairn.feature_cache import text_hash
from cairn.normalizer import normalize_text

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "modal_cache_features.py"
PILOT_SEED = 20260710
COMBINED_SIGNATURE = "tocsin-bart-fastdetectgpt-combined-w-v1"
COHESIVENESS_SIGNATURE = "tocsin-bart-cohesiveness-u-v1"

REQUIRED_WORKER_MANIFEST_KEYS = {
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


class _ImageBuilder:
    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: self


class _MemoryVolume:
    def __init__(self, payload: bytes = b"") -> None:
        self.payload = payload

    def read_file(self, _path: str):
        if self.payload:
            yield self.payload

    def commit(self) -> None:
        return None


class _MappedVolume:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def read_file(self, path: str):
        yield self.payloads[path]


class _ChunkedVolume:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def read_file(self, _path: str):
        yield from self.chunks


class _StubApp:
    def __init__(self, _name: str) -> None:
        self.function_options: list[dict[str, object]] = []

    def function(self, **options):
        self.function_options.append(options)

        def decorate(function):
            function._modal_options = options
            function.remote = MagicMock(name=f"{function.__name__}.remote")
            function.starmap = MagicMock(name=f"{function.__name__}.starmap")
            return function

        return decorate

    def local_entrypoint(self):
        return lambda function: function


def _modal_stub() -> types.ModuleType:
    stub = types.ModuleType("modal")
    stub.App = _StubApp
    stub.Image = types.SimpleNamespace(
        debian_slim=lambda **_kwargs: _ImageBuilder()
    )
    stub.Volume = types.SimpleNamespace(
        from_name=lambda *_args, **_kwargs: _MemoryVolume()
    )
    stub.Secret = types.SimpleNamespace(
        from_dotenv=lambda: object()
    )
    return stub


@pytest.fixture(scope="module")
def modal_cache_module():
    name = "scripts_modal_cache_features_test"
    sentinel = object()
    previous_modal = sys.modules.get("modal", sentinel)
    previous_script = sys.modules.get(name, sentinel)
    sys.modules["modal"] = _modal_stub()
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        if previous_modal is sentinel:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = previous_modal
        if previous_script is sentinel:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_script


def _rows_payload(n_each: int) -> bytes:
    rows: list[dict[str, str]] = []
    for index in range(n_each):
        rows.append({"text": " ".join([f"long{index}"] * 80)})
        rows.append({"text": f"short {index}"})
    return "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")


def _cache_record(
    text: str,
    *,
    signature: str = COMBINED_SIGNATURE,
    calibration_identity: dict[str, str] | None = None,
) -> dict[str, object]:
    normalized = normalize_text(text)
    n_words = len(normalized.split())
    if n_words >= 60:
        if signature == COHESIVENESS_SIGNATURE or signature.startswith(
            COHESIVENESS_SIGNATURE + "-"
        ):
            raw_channel = "cohesiveness_u"
            raw_score = -1.25
            base_v: float | None = None
        else:
            raw_channel = "combined_w"
            raw_score = -0.25
            base_v = 0.32
        signals: dict[str, object] = {
            "cohesiveness_u": -1.25,
            "base_v": base_v,
            "combined_w": -0.25,
            "calibration_signature": signature,
            "calibration_raw_channel": raw_channel,
            "deletion_ratio": 0.015,
            "n_copies": 10,
        }
        if calibration_identity is not None:
            signals["calibration_identity"] = dict(calibration_identity)
        result = {
            "layer_id": "token_cohesiveness",
            "domain": "prose",
            "raw_score": raw_score,
            "p_llm": 0.6,
            "confidence": 0.7,
            "signals": signals,
            "determination": "AMBER",
            "attacker_tiers": ["A0", "A1", "A2", "A3"],
            "compute_cost": "moderate",
            "min_text_length": 60,
            "spans": [],
            "calibrator_applied": False,
        }
    else:
        result = {
            "layer_id": "token_cohesiveness",
            "domain": "prose",
            "raw_score": 0.0,
            "p_llm": 0.5,
            "confidence": 0.0,
            "signals": {"skipped": "too_short", "n_words": n_words},
            "determination": "SKIP",
            "attacker_tiers": ["A0", "A1", "A2", "A3"],
            "compute_cost": "moderate",
            "min_text_length": 60,
            "spans": [],
            "calibrator_applied": False,
        }
    return {
        "hash": text_hash(normalized),
        "results": [result],
        "router": {
            "domain": "prose",
            "confidence": 0.95,
            "prompt_score": 0.05,
            "prose_score": 0.95,
            "word_count": n_words,
            "recommended_detectors": [],
            "skip_detectors": [],
        },
    }


def _cache_records_for_source(source: bytes) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in source.decode("utf-8").splitlines()]
    return [_cache_record(str(row["text"])) for row in rows]


def _cache_records_for_source_with_signature(
    source: bytes,
    signature: str,
    calibration_identity: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in source.decode("utf-8").splitlines()]
    return [
        _cache_record(
            str(row["text"]),
            signature=signature,
            calibration_identity=calibration_identity,
        )
        for row in rows
    ]


def _cache_payload_for_source(
    source: bytes,
    *,
    final_newline: bool = True,
) -> bytes:
    return _cache_payload(
        _cache_records_for_source(source),
        final_newline=final_newline,
    )


def _cache_payload(
    records: list[dict[str, object]],
    *,
    final_newline: bool = True,
) -> bytes:
    text = "\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        for record in records
    )
    if final_newline:
        text += "\n"
    return text.encode("utf-8")


def _worker_manifest(
    call: dict[str, object],
    *,
    worker_elapsed_seconds: float = 4.0,
) -> dict[str, object]:
    payload = call["dataset_jsonl"]
    assert isinstance(payload, bytes)
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    eligible = sum(len(row["text"].split()) >= 60 for row in rows)
    n_total = len(rows)
    return {
        "run_id": call["run_id"],
        "source_dataset_sha256": call["source_dataset_sha256"],
        "scored_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "config_path": call["config_path"],
        "resolved_config_sha256": call["resolved_config_sha256"],
        "git_revision": call["git_revision"],
        "n_total": n_total,
        "n_eligible": eligible,
        "n_ineligible": n_total - eligible,
        "n_timed_eligible": eligible,
        "n_timed_ineligible": n_total - eligible,
        "n_workers": 1,
        "n_warmups": 1,
        "warmup_seconds": 0.3,
        "eligible_scoring_seconds": eligible * 0.2,
        "ineligible_scoring_seconds": (n_total - eligible) * 0.1,
        "worker_elapsed_seconds": worker_elapsed_seconds,
        "eligible_rows_per_second": eligible / (eligible * 0.2),
        "total_rows_per_second": n_total / max(worker_elapsed_seconds, 0.001),
        "gpu_type": "H200",
        "remote_path": f"cairn_cache_output/{call['run_id']}_features.jsonl",
        "python_version": "3.11-test",
        "torch_version": "2-test",
        "transformers_version": "4-test",
        "cuda_version": "12-test",
        "pilot_seed": call["pilot_seed"],
        "detector_seed": call["detector_seed"],
        "pyproject_sha256": call["pyproject_sha256"],
        "package_versions": {"torch": "2-test", "transformers": "4-test"},
    }


def _install_local_provenance(monkeypatch, module) -> None:
    monkeypatch.setattr(module, "_resolved_config_sha256", lambda _path: "c" * 64)
    monkeypatch.setattr(module, "_git_revision", lambda: "deadbeef1234567890")
    monkeypatch.setattr(module, "_pyproject_sha256", lambda: "p" * 64)
    monkeypatch.setattr(
        module,
        "_expected_cache_layer_ids",
        lambda _path: ("token_cohesiveness",),
    )
    monkeypatch.setattr(
        module,
        "_resolve_expected_tocsin_signature",
        lambda _path, explicit=None: explicit or COMBINED_SIGNATURE,
    )


@pytest.mark.parametrize(
    "run_id",
    ["v8", "default", "too-short", "bad/run-20260711", " bad-run-20260711"],
)
def test_validate_run_id_rejects_defaults_short_and_unsafe_values(
    modal_cache_module, run_id
):
    with pytest.raises(ValueError, match="run_id"):
        modal_cache_module.validate_run_id(run_id)


def test_validate_run_id_accepts_unique_safe_identifier(modal_cache_module):
    run_id = "tocsin-v2-20260711T153000Z-deadbeef1234-pilot"
    assert modal_cache_module.validate_run_id(run_id) == run_id


def test_modal_resources_and_required_cli_provenance_are_explicit(
    modal_cache_module,
):
    options = modal_cache_module.cache_features._modal_options
    assert options["gpu"] == "H200"
    assert options["cpu"] == 4.0
    assert options["memory"] == 49152
    assert options["timeout"] == 21600

    remote_parameters = inspect.signature(modal_cache_module.cache_features).parameters
    local_parameters = inspect.signature(modal_cache_module.main).parameters
    assert remote_parameters["run_id"].default is inspect.Parameter.empty
    assert local_parameters["run_id"].default is inspect.Parameter.empty
    assert local_parameters["manifest_path"].default is inspect.Parameter.empty
    assert (
        local_parameters["compute_usd_per_second"].default
        is inspect.Parameter.empty
    )


def test_script_parses_with_the_modal_container_python_311_grammar():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    # Backslashes inside f-string expressions are rejected by Python 3.11,
    # even though the local developer runtime may use the relaxed 3.12+ grammar.
    assert "count(b'\\n')" not in source
    ast.parse(
        source,
        filename=str(SCRIPT_PATH),
        feature_version=(3, 11),
    )


def test_pricing_provenance_uses_plan_accounting_date(modal_cache_module):
    assert "On 2026-07-10" in modal_cache_module.__doc__
    assert "On 2026-07-11" not in modal_cache_module.__doc__
    assert "$0.00141996/s" in modal_cache_module.__doc__


def test_auto_signature_resolution_uses_public_identity_builder(
    modal_cache_module,
    monkeypatch,
    tmp_path,
):
    from cairn.detectors.token_cohesiveness import (
        build_tocsin_calibration_signature,
    )

    monkeypatch.setattr("cairn.dotenv.load_dotenv", lambda: None)
    monkeypatch.setenv("CAIRN_FAST_DETECTGPT_MODEL", "org/custom-fast")
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "_extends: default\n"
        "detectors:\n"
        "  token_cohesiveness:\n"
        "    enabled: true\n"
        "    model: org/custom-bart\n",
        encoding="utf-8",
    )
    expected_signature, _identity = build_tocsin_calibration_signature(
        "combined_w",
        bart_model="org/custom-bart",
        base_model="org/custom-fast",
    )

    assert (
        modal_cache_module._resolve_expected_tocsin_signature(profile)
        == expected_signature
    )


def test_pilot_is_deterministic_and_stratified(modal_cache_module):
    payload = _rows_payload(300)

    first = modal_cache_module.select_pilot_rows(
        payload, n_rows=256, seed=PILOT_SEED
    )
    second = modal_cache_module.select_pilot_rows(
        payload, n_rows=256, seed=PILOT_SEED
    )
    selected = [json.loads(line) for line in first.decode().splitlines()]

    assert first == second
    assert len(selected) == 256
    assert sum(len(row["text"].split()) >= 60 for row in selected) == 128


@pytest.mark.parametrize("n_rows", [0, 255])
def test_pilot_requires_a_positive_even_row_count(modal_cache_module, n_rows):
    with pytest.raises(ValueError, match="positive even"):
        modal_cache_module.select_pilot_rows(
            _rows_payload(300), n_rows=n_rows, seed=PILOT_SEED
        )


def test_pilot_rejects_an_undersized_stratum(modal_cache_module):
    payload = "".join(
        json.dumps({"text": f"short {index}"}) + "\n" for index in range(300)
    ).encode("utf-8")
    with pytest.raises(ValueError, match="eligible"):
        modal_cache_module.select_pilot_rows(
            payload, n_rows=256, seed=PILOT_SEED
        )


def test_projection_includes_pilot_and_full_and_passes_at_exact_cap(
    modal_cache_module,
):
    pilot_manifest = {
        "n_eligible": 1,
        "n_ineligible": 1,
        "n_timed_eligible": 1,
        "n_timed_ineligible": 1,
        "n_warmups": 1,
        "eligible_scoring_seconds": 2,
        "ineligible_scoring_seconds": 1,
        "warmup_seconds": 1,
        "worker_elapsed_seconds": 5,
    }

    result = modal_cache_module.project_modal_cost(
        pilot_manifest,
        total_rows=4,
        eligible_rows=2,
        compute_usd_per_second=Decimal("1"),
        cap_usd=Decimal("15.00"),
        full_workers=2,
    )

    assert result == {
        "projected_full_seconds": 10.0,
        "projected_cumulative_usd": "15.0",
        "gate": "PASS",
    }


def test_projection_stops_above_cap(modal_cache_module):
    pilot_manifest = {
        "n_eligible": 1,
        "n_ineligible": 1,
        "n_timed_eligible": 1,
        "n_timed_ineligible": 1,
        "n_warmups": 1,
        "eligible_scoring_seconds": 2,
        "ineligible_scoring_seconds": 1,
        "warmup_seconds": 1,
        "worker_elapsed_seconds": 5,
    }
    result = modal_cache_module.project_modal_cost(
        pilot_manifest,
        total_rows=4,
        eligible_rows=2,
        compute_usd_per_second=Decimal("1.0000000001"),
        cap_usd=Decimal("15.00"),
        full_workers=2,
    )
    assert result["gate"] == "STOP_COST_CAP"
    assert Decimal(result["projected_cumulative_usd"]) > Decimal("15.00")


def test_cost_model_and_prior_projection_use_decimal_arithmetic(
    modal_cache_module,
):
    measured = {
        "n_eligible": 2,
        "n_ineligible": 4,
        "n_timed_eligible": 2,
        "n_timed_ineligible": 4,
        "n_warmups": 1,
        "eligible_scoring_seconds": "0.4",
        "ineligible_scoring_seconds": "0.4",
        "warmup_seconds": "0.3",
        "worker_elapsed_seconds": "1.7",
    }
    model = modal_cache_module.cost_model_from_manifest(measured)
    assert model == {
        "warmup_seconds": "0.3",
        "eligible_seconds_per_row": "0.2",
        "ineligible_seconds_per_row": "0.1",
        "unattributed_seconds_per_row": "0.1",
    }

    prior = {"cost_model": model, "cumulative_compute_usd": "2.00"}
    projected = modal_cache_module.project_run_cost(
        prior,
        total_rows=10,
        eligible_rows=4,
        compute_usd_per_second=Decimal("0.5"),
        cap_usd=Decimal("2.84"),
    )
    assert projected["projected_run_seconds"] == 2.7
    assert projected["projected_run_compute_usd"] == "1.35"
    assert projected["projected_cumulative_usd"] == "3.35"
    assert projected["gate"] == "STOP_COST_CAP"


def test_completed_cost_fields_preserve_inherited_model_exactly(
    modal_cache_module,
):
    inherited_model = {
        "warmup_seconds": 0.5,
        "eligible_seconds_per_row": "0.2",
        "ineligible_seconds_per_row": "0.1",
        "unattributed_seconds_per_row": "0.05",
        "model_version": 2,
    }
    fields = modal_cache_module._completed_cost_fields(
        {"worker_elapsed_seconds": "2"},
        compute_usd_per_second=Decimal("0.5"),
        prior_manifest={
            "cost_model": inherited_model,
            "cumulative_compute_usd": "3",
        },
    )

    assert fields["cost_model"] == inherited_model
    assert fields["cost_model_source"] == "prior_manifest"


def test_partial_retry_cost_model_uses_timed_counts_and_stops_understatement(
    modal_cache_module,
):
    partial_retry = {
        "n_eligible": 128,
        "n_ineligible": 128,
        "n_timed_eligible": 2,
        "n_timed_ineligible": 2,
        "n_warmups": 1,
        "eligible_scoring_seconds": "2",
        "ineligible_scoring_seconds": "1",
        "warmup_seconds": "0.5",
        "worker_elapsed_seconds": "4.5",
    }

    model = modal_cache_module.cost_model_from_manifest(partial_retry)
    assert model == {
        "warmup_seconds": "0.5",
        "eligible_seconds_per_row": "1",
        "ineligible_seconds_per_row": "0.5",
        "unattributed_seconds_per_row": "0.25",
    }
    projected = modal_cache_module.project_run_cost(
        {"cost_model": model, "cumulative_compute_usd": "0"},
        total_rows=10,
        eligible_rows=5,
        compute_usd_per_second=Decimal("1"),
        cap_usd=Decimal("1"),
    )
    assert projected["projected_cumulative_usd"] == "10.50"
    assert projected["gate"] == "STOP_COST_CAP"


def test_legacy_cost_model_derives_unattributed_time_when_timings_match(
    modal_cache_module,
):
    legacy = {
        "cost_model": {
            "warmup_seconds": "1",
            "eligible_seconds_per_row": "2",
            "ineligible_seconds_per_row": "1",
        },
        "n_timed_eligible": 1,
        "n_timed_ineligible": 1,
        "n_warmups": 1,
        "warmup_seconds": "1",
        "eligible_scoring_seconds": "2",
        "ineligible_scoring_seconds": "1",
        "worker_elapsed_seconds": "5",
        "cumulative_compute_usd": "5",
    }

    projected = modal_cache_module.project_run_cost(
        legacy,
        total_rows=4,
        eligible_rows=2,
        worker_count=2,
        compute_usd_per_second=Decimal("1"),
        cap_usd=Decimal("15"),
    )

    assert projected == {
        "projected_run_seconds": 10.0,
        "projected_run_compute_usd": "10.0",
        "projected_cumulative_usd": "15.0",
        "gate": "PASS",
    }


@pytest.mark.parametrize(
    "legacy",
    [
        {
            "cost_model": {
                "warmup_seconds": "1",
                "eligible_seconds_per_row": "2",
                "ineligible_seconds_per_row": "1",
            },
            "cumulative_compute_usd": "0",
        },
        {
            "cost_model": {
                "warmup_seconds": "9",
                "eligible_seconds_per_row": "2",
                "ineligible_seconds_per_row": "1",
            },
            "n_timed_eligible": 1,
            "n_timed_ineligible": 1,
            "n_warmups": 1,
            "warmup_seconds": "1",
            "eligible_scoring_seconds": "2",
            "ineligible_scoring_seconds": "1",
            "worker_elapsed_seconds": "5",
            "cumulative_compute_usd": "0",
        },
    ],
    ids=["missing-timings", "timings-do-not-describe-model"],
)
def test_legacy_cost_model_fails_closed_when_overhead_is_not_derivable(
    modal_cache_module,
    legacy,
):
    with pytest.raises(ValueError, match="unattributed"):
        modal_cache_module.project_run_cost(
            legacy,
            total_rows=1,
            eligible_rows=1,
            compute_usd_per_second=Decimal("1"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_elapsed_seconds", "NaN"),
        ("warmup_seconds", "-1"),
        ("eligible_scoring_seconds", "Infinity"),
    ],
)
def test_measured_cost_model_rejects_nonfinite_or_negative_timings(
    modal_cache_module,
    field,
    value,
):
    measured = {
        "n_timed_eligible": 1,
        "n_timed_ineligible": 1,
        "n_warmups": 1,
        "warmup_seconds": "1",
        "eligible_scoring_seconds": "2",
        "ineligible_scoring_seconds": "1",
        "worker_elapsed_seconds": "5",
    }
    measured[field] = value

    with pytest.raises(ValueError, match="finite nonnegative"):
        modal_cache_module.cost_model_from_manifest(measured)


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity"])
def test_prior_cumulative_spend_must_be_finite_nonnegative(
    modal_cache_module,
    value,
):
    prior = {
        "cost_model": {
            "warmup_seconds": "1",
            "eligible_seconds_per_row": "1",
            "ineligible_seconds_per_row": "1",
            "unattributed_seconds_per_row": "1",
        },
        "cumulative_compute_usd": value,
    }

    with pytest.raises(ValueError, match="finite nonnegative"):
        modal_cache_module.project_run_cost(
            prior,
            total_rows=1,
            eligible_rows=1,
            compute_usd_per_second=Decimal("1"),
        )


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity"])
def test_completed_cost_rejects_invalid_worker_elapsed_with_inherited_model(
    modal_cache_module,
    value,
):
    prior = {
        "cost_model": {
            "warmup_seconds": "1",
            "eligible_seconds_per_row": "1",
            "ineligible_seconds_per_row": "1",
            "unattributed_seconds_per_row": "1",
        },
        "cumulative_compute_usd": "1",
    }

    with pytest.raises(ValueError, match="finite nonnegative"):
        modal_cache_module._completed_cost_fields(
            {"worker_elapsed_seconds": value},
            compute_usd_per_second=Decimal("1"),
            prior_manifest=prior,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warmup_seconds", -1.0),
        ("eligible_scoring_seconds", float("nan")),
        ("ineligible_scoring_seconds", float("inf")),
        ("worker_elapsed_seconds", -1.0),
        ("eligible_rows_per_second", float("nan")),
        ("total_rows_per_second", -1.0),
    ],
)
def test_worker_manifest_rejects_invalid_timing_fields(
    modal_cache_module,
    field,
    value,
):
    payload = _rows_payload(1)
    call = {
        "dataset_jsonl": payload,
        "config_path": "configs/default.yaml",
        "run_id": "tocsin-v2-worker-timing",
        "source_dataset_sha256": "s" * 64,
        "resolved_config_sha256": "c" * 64,
        "git_revision": "deadbeef1234567890",
        "pilot_seed": None,
        "detector_seed": 42,
        "pyproject_sha256": "p" * 64,
    }
    manifest = _worker_manifest(call)
    manifest[field] = value

    with pytest.raises(ValueError, match="finite nonnegative"):
        modal_cache_module._validate_worker_manifest(
            manifest,
            run_id=str(call["run_id"]),
            source_dataset_sha256=str(call["source_dataset_sha256"]),
            scored_payload_sha256=hashlib.sha256(payload).hexdigest(),
            resolved_config_sha256=str(call["resolved_config_sha256"]),
            git_revision=str(call["git_revision"]),
            n_total=2,
            n_eligible=1,
        )


def test_worker_manifest_rejects_elapsed_below_accounted_timing(
    modal_cache_module,
):
    payload = _rows_payload(1)
    call = {
        "dataset_jsonl": payload,
        "config_path": "configs/default.yaml",
        "run_id": "tocsin-v2-worker-timing-total",
        "source_dataset_sha256": "s" * 64,
        "resolved_config_sha256": "c" * 64,
        "git_revision": "deadbeef1234567890",
        "pilot_seed": None,
        "detector_seed": 42,
        "pyproject_sha256": "p" * 64,
    }
    manifest = _worker_manifest(call)
    manifest["worker_elapsed_seconds"] = 0.1

    with pytest.raises(ValueError, match="below measured warmup plus scoring"):
        modal_cache_module._validate_worker_manifest(
            manifest,
            run_id=str(call["run_id"]),
            source_dataset_sha256=str(call["source_dataset_sha256"]),
            scored_payload_sha256=hashlib.sha256(payload).hexdigest(),
            resolved_config_sha256=str(call["resolved_config_sha256"]),
            git_revision=str(call["git_revision"]),
            n_total=2,
            n_eligible=1,
        )


def test_router_contract_allows_repeated_recommendations_used_by_current_cache(
    modal_cache_module,
    tmp_path,
):
    record = _cache_record(" ".join(["long"] * 80))
    router = record["router"]
    assert isinstance(router, dict)
    router["recommended_detectors"] = ["preamble", "preamble"]
    cache = tmp_path / "features.jsonl"
    cache.write_bytes(_cache_payload([record]))

    modal_cache_module._validate_serialized_cache_contract(cache)


def test_gate_manifest_requires_pass_and_matching_provenance(
    modal_cache_module,
):
    expected = {
        "source_dataset_sha256": "s" * 64,
        "resolved_config_sha256": "c" * 64,
        "git_revision": "deadbeef1234",
        "expected_tocsin_signature": COMBINED_SIGNATURE,
    }
    gate = {"gate": "PASS", **expected}
    assert (
        modal_cache_module.validate_gate_manifest(
            gate,
            source_dataset_sha256=expected["source_dataset_sha256"],
            resolved_config_sha256=expected["resolved_config_sha256"],
            git_revision=expected["git_revision"],
            expected_tocsin_signature=expected["expected_tocsin_signature"],
        )
        is gate
    )

    for key in (
        "source_dataset_sha256",
        "resolved_config_sha256",
        "git_revision",
        "expected_tocsin_signature",
    ):
        incompatible = dict(gate)
        incompatible[key] = "different"
        with pytest.raises(ValueError, match=key):
            modal_cache_module.validate_gate_manifest(
                incompatible,
                source_dataset_sha256=expected["source_dataset_sha256"],
                resolved_config_sha256=expected["resolved_config_sha256"],
                git_revision=expected["git_revision"],
                expected_tocsin_signature=expected[
                    "expected_tocsin_signature"
                ],
            )

    stopped = dict(gate, gate="STOP_COST_CAP")
    with pytest.raises(ValueError, match="PASS"):
        modal_cache_module.validate_gate_manifest(stopped, **expected)


def test_worker_timing_warms_once_then_times_each_stratum(modal_cache_module):
    rows = [
        {"text": "short row"},
        {"text": " ".join(["eligible"] * 60)},
    ]

    class Pipeline:
        def __init__(self):
            self.calls: list[str] = []

        def analyze_detailed(self, text):
            self.calls.append(text)
            return object(), [text], {"route": text}

    pipeline = Pipeline()
    clock_values = iter([0.0, 0.4, 1.0, 1.1, 2.0, 2.2])
    recorded: list[str] = []

    timing = modal_cache_module.measure_scoring_times(
        pipeline,
        rows,
        normalize_text=lambda text: text.strip(),
        record_result=lambda text, _results, _route: recorded.append(text),
        clock=lambda: next(clock_values),
    )

    eligible = rows[1]["text"]
    assert pipeline.calls == [eligible, "short row", eligible]
    assert recorded == ["short row", eligible]
    assert timing["warmup_seconds"] == pytest.approx(0.4)
    assert timing["eligible_scoring_seconds"] == pytest.approx(0.2)
    assert timing["ineligible_scoring_seconds"] == pytest.approx(0.1)
    assert timing["n_eligible"] == 1
    assert timing["n_ineligible"] == 1


def test_pilot_manifest_is_written_before_cost_stop(
    modal_cache_module, monkeypatch, tmp_path
):
    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    source = _rows_payload(300)
    dataset = tmp_path / "union.jsonl"
    dataset.write_bytes(source)
    manifest_path = tmp_path / "pilot.json"
    remote = MagicMock(
        side_effect=lambda **call: _worker_manifest(
            call, worker_elapsed_seconds=40.0
        )
    )
    monkeypatch.setattr(module.cache_features, "remote", remote)

    with pytest.raises(SystemExit, match="STOP_COST_CAP"):
        module.main(
            dataset_path=str(dataset),
            out_path=str(tmp_path / "pilot.features.jsonl"),
            run_id="tocsin-v2-20260711-pilot",
            manifest_path=str(manifest_path),
            compute_usd_per_second="0.01",
            config_path="configs/default.yaml",
            pilot_rows=256,
            cost_cap_usd="0.01",
        )

    assert remote.call_count == 1
    call = remote.call_args.kwargs
    assert call["source_dataset_sha256"] == hashlib.sha256(source).hexdigest()
    assert call["source_dataset_sha256"] != hashlib.sha256(
        call["dataset_jsonl"]
    ).hexdigest()
    scored_rows = [
        json.loads(line)
        for line in call["dataset_jsonl"].decode("utf-8").splitlines()
    ]
    assert len(scored_rows) == 256
    assert sum(len(row["text"].split()) >= 60 for row in scored_rows) == 128

    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert REQUIRED_WORKER_MANIFEST_KEYS <= written.keys()
    assert written["gate"] == "STOP_COST_CAP"
    assert Decimal(written["projected_cumulative_usd"]) > Decimal("0.01")


def test_incompatible_gate_blocks_before_remote_invocation(
    modal_cache_module, monkeypatch, tmp_path
):
    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    dataset = tmp_path / "union.jsonl"
    dataset.write_bytes(_rows_payload(1))
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "gate": "PASS",
                "source_dataset_sha256": "wrong",
                "resolved_config_sha256": "c" * 64,
                "git_revision": "deadbeef1234567890",
            }
        ),
        encoding="utf-8",
    )
    remote = MagicMock()
    monkeypatch.setattr(module.cache_features, "remote", remote)

    with pytest.raises(ValueError, match="source_dataset_sha256"):
        module.main(
            dataset_path=str(dataset),
            out_path=str(tmp_path / "full.features.jsonl"),
            run_id="tocsin-v2-20260711-full",
            manifest_path=str(tmp_path / "full.json"),
            compute_usd_per_second="0.00141996",
            config_path="configs/default.yaml",
            gate_manifest_path=str(gate_path),
        )

    remote.assert_not_called()


def test_prior_cost_projection_blocks_before_remote_invocation(
    modal_cache_module, monkeypatch, tmp_path
):
    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    dataset = tmp_path / "challenge.jsonl"
    dataset.write_bytes(_rows_payload(1))
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "expected_tocsin_signature": COMBINED_SIGNATURE,
                "cost_model": {
                    "warmup_seconds": "1",
                    "eligible_seconds_per_row": "1",
                    "ineligible_seconds_per_row": "1",
                    "unattributed_seconds_per_row": "0",
                },
                "cumulative_compute_usd": "14",
            }
        ),
        encoding="utf-8",
    )
    remote = MagicMock()
    monkeypatch.setattr(module.cache_features, "remote", remote)

    with pytest.raises(SystemExit, match="STOP_COST_CAP"):
        module.main(
            dataset_path=str(dataset),
            out_path=str(tmp_path / "challenge.features.jsonl"),
            run_id="tocsin-v2-20260711-challenge",
            manifest_path=str(tmp_path / "challenge.json"),
            compute_usd_per_second="1",
            config_path="configs/default.yaml",
            prior_manifest_path=str(prior_path),
            cost_cap_usd="15",
        )

    remote.assert_not_called()


def test_prior_manifest_signature_mismatch_blocks_before_remote_invocation(
    modal_cache_module,
    monkeypatch,
    tmp_path,
):
    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    dataset = tmp_path / "challenge.jsonl"
    dataset.write_bytes(_rows_payload(1))
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(
        json.dumps(
            {
                "expected_tocsin_signature": COHESIVENESS_SIGNATURE,
                "cost_model": {
                    "warmup_seconds": "0.1",
                    "eligible_seconds_per_row": "0.1",
                    "ineligible_seconds_per_row": "0.1",
                    "unattributed_seconds_per_row": "0.1",
                },
                "cumulative_compute_usd": "1",
            }
        ),
        encoding="utf-8",
    )
    remote = MagicMock()
    monkeypatch.setattr(module.cache_features, "remote", remote)

    with pytest.raises(ValueError, match="expected_tocsin_signature"):
        module.main(
            dataset_path=str(dataset),
            out_path=str(tmp_path / "challenge.features.jsonl"),
            run_id="tocsin-v2-20260711-prior-signature",
            manifest_path=str(tmp_path / "challenge.json"),
            compute_usd_per_second="1",
            config_path="configs/default.yaml",
            prior_manifest_path=str(prior_path),
            cost_cap_usd="15",
        )

    remote.assert_not_called()


def test_duplicate_input_is_rejected_before_remote_invocation(
    modal_cache_module, monkeypatch, tmp_path
):
    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    eligible = " ".join(["duplicate"] * 60)
    rows = [
        {"text": eligible},
        {"text": "short unique row"},
        {"text": eligible},
    ]
    dataset = tmp_path / "duplicates.jsonl"
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    remote = MagicMock()
    monkeypatch.setattr(module.cache_features, "remote", remote)

    with pytest.raises(ValueError, match="duplicate normalized text"):
        module.main(
            dataset_path=str(dataset),
            out_path=str(tmp_path / "features.jsonl"),
            run_id="tocsin-v2-20260711-duplicate-input",
            manifest_path=str(tmp_path / "manifest.json"),
            compute_usd_per_second="0.00141996",
            config_path="configs/default.yaml",
        )

    remote.assert_not_called()


def test_existing_checkpoint_requires_a_new_unique_run_id(
    modal_cache_module, tmp_path
):
    checkpoint = tmp_path / "existing.features.jsonl"
    checkpoint.write_text('{"hash":"already-billed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="unique run_id"):
        modal_cache_module.ensure_new_checkpoint(checkpoint)


def test_gate_only_full_run_reprices_at_current_rate_before_remote(
    modal_cache_module, monkeypatch, tmp_path
):
    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    source = _rows_payload(1)
    source_sha = hashlib.sha256(source).hexdigest()
    dataset = tmp_path / "union.jsonl"
    dataset.write_bytes(source)
    gate_path = tmp_path / "pilot.json"
    gate_path.write_text(
        json.dumps(
            {
                "gate": "PASS",
                "source_dataset_sha256": source_sha,
                "resolved_config_sha256": "c" * 64,
                "git_revision": "deadbeef1234567890",
                "expected_tocsin_signature": COMBINED_SIGNATURE,
                "cost_model": {
                    "warmup_seconds": "1",
                    "eligible_seconds_per_row": "1",
                    "ineligible_seconds_per_row": "1",
                    "unattributed_seconds_per_row": "0",
                },
                "cumulative_compute_usd": "1",
                "compute_usd_per_second": "0.01",
                "projected_cumulative_usd": "1.03",
            }
        ),
        encoding="utf-8",
    )
    remote = MagicMock()
    monkeypatch.setattr(module.cache_features, "remote", remote)

    with pytest.raises(SystemExit, match="STOP_COST_CAP"):
        module.main(
            dataset_path=str(dataset),
            out_path=str(tmp_path / "full.features.jsonl"),
            run_id="tocsin-v2-20260711-stale-rate",
            manifest_path=str(tmp_path / "full.json"),
            compute_usd_per_second="10",
            config_path="configs/default.yaml",
            gate_manifest_path=str(gate_path),
            cost_cap_usd="15",
        )

    remote.assert_not_called()


def test_gate_only_full_run_rejects_omitted_pilot_spend_before_remote(
    modal_cache_module, monkeypatch, tmp_path
):
    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    source = _rows_payload(1)
    source_sha = hashlib.sha256(source).hexdigest()
    dataset = tmp_path / "union.jsonl"
    dataset.write_bytes(source)
    gate_path = tmp_path / "pilot.json"
    gate_path.write_text(
        json.dumps(
            {
                "gate": "PASS",
                "source_dataset_sha256": source_sha,
                "resolved_config_sha256": "c" * 64,
                "git_revision": "deadbeef1234567890",
                "expected_tocsin_signature": COMBINED_SIGNATURE,
                "cost_model": {
                    "warmup_seconds": "0.1",
                    "eligible_seconds_per_row": "0.1",
                    "ineligible_seconds_per_row": "0.1",
                    "unattributed_seconds_per_row": "0",
                },
            }
        ),
        encoding="utf-8",
    )
    remote = MagicMock()
    monkeypatch.setattr(module.cache_features, "remote", remote)

    with pytest.raises(ValueError, match="cumulative_compute_usd"):
        module.main(
            dataset_path=str(dataset),
            out_path=str(tmp_path / "full.features.jsonl"),
            run_id="tocsin-v2-20260711-missing-spend",
            manifest_path=str(tmp_path / "full.json"),
            compute_usd_per_second="0.5",
            config_path="configs/default.yaml",
            gate_manifest_path=str(gate_path),
            cost_cap_usd="15",
        )

    remote.assert_not_called()


def test_completed_run_records_actual_and_cumulative_cost(
    modal_cache_module, monkeypatch, tmp_path
):
    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    source = _rows_payload(1)
    source_sha = hashlib.sha256(source).hexdigest()
    dataset = tmp_path / "union.jsonl"
    dataset.write_bytes(source)
    prior_path = tmp_path / "pilot.json"
    prior_path.write_text(
        json.dumps(
            {
                "gate": "PASS",
                "source_dataset_sha256": source_sha,
                "resolved_config_sha256": "c" * 64,
                "git_revision": "deadbeef1234567890",
                "expected_tocsin_signature": COMBINED_SIGNATURE,
                "cost_model": {
                    "warmup_seconds": "0.1",
                    "eligible_seconds_per_row": "0.1",
                    "ineligible_seconds_per_row": "0.1",
                    "unattributed_seconds_per_row": "0",
                },
                "cumulative_compute_usd": "2.00",
            }
        ),
        encoding="utf-8",
    )
    remote = MagicMock(
        side_effect=lambda **call: _worker_manifest(
            call, worker_elapsed_seconds=4.0
        )
    )
    monkeypatch.setattr(module.cache_features, "remote", remote)
    cache_payload = _cache_payload_for_source(source)
    monkeypatch.setattr(module, "hf_cache", _MemoryVolume(cache_payload))
    manifest_path = tmp_path / "full.json"
    out_path = tmp_path / "full.features.jsonl"

    module.main(
        dataset_path=str(dataset),
        out_path=str(out_path),
        run_id="tocsin-v2-20260711-full",
        manifest_path=str(manifest_path),
        compute_usd_per_second="0.5",
        config_path="configs/default.yaml",
        gate_manifest_path=str(prior_path),
        prior_manifest_path=str(prior_path),
        cost_cap_usd="15",
    )

    remote.assert_called_once()
    call = remote.call_args.kwargs
    assert call["source_dataset_sha256"] == source_sha
    assert hashlib.sha256(call["dataset_jsonl"]).hexdigest() == source_sha
    assert out_path.read_bytes() == cache_payload

    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert REQUIRED_WORKER_MANIFEST_KEYS <= written.keys()
    assert Decimal(written["run_compute_usd"]) == Decimal("2.00")
    assert Decimal(written["cumulative_compute_usd"]) == Decimal("4.00")
    assert written["compute_usd_per_second"] == "0.5"
    assert written["cost_model"] == {
        "warmup_seconds": "0.1",
        "eligible_seconds_per_row": "0.1",
        "ineligible_seconds_per_row": "0.1",
        "unattributed_seconds_per_row": "0",
    }
    assert written["cost_model_source"] == "prior_manifest"
    assert written["cache_output_status"] == "PASS"
    assert written["cache_output_validation"]["expected_unique_hashes"] == 2


def test_sharded_run_aggregates_worker_cost_and_validates_lossless_output(
    modal_cache_module, monkeypatch, tmp_path
):
    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    source = _rows_payload(2)
    source_sha = hashlib.sha256(source).hexdigest()
    dataset = tmp_path / "union.jsonl"
    dataset.write_bytes(source)
    gate_path = tmp_path / "pilot.json"
    inherited_model = {
        "warmup_seconds": "0.5",
        "eligible_seconds_per_row": "0.1",
        "ineligible_seconds_per_row": "0.1",
        "unattributed_seconds_per_row": "0.05",
    }
    gate_path.write_text(
        json.dumps(
            {
                "gate": "PASS",
                "source_dataset_sha256": source_sha,
                "resolved_config_sha256": "c" * 64,
                "git_revision": "deadbeef1234567890",
                "expected_tocsin_signature": COMBINED_SIGNATURE,
                "cost_model": inherited_model,
                "cumulative_compute_usd": "1.0",
            }
        ),
        encoding="utf-8",
    )

    worker_calls: list[dict[str, object]] = []

    def fake_starmap(args):
        manifests = []
        signature = inspect.signature(module.cache_features)
        for index, positional in enumerate(args):
            call = dict(signature.bind(*positional).arguments)
            worker_calls.append(call)
            manifests.append(
                _worker_manifest(
                    call,
                    worker_elapsed_seconds=float(index + 2),
                )
            )
        return iter(manifests)

    starmap = MagicMock(side_effect=fake_starmap)
    remote = MagicMock()
    monkeypatch.setattr(module.cache_features, "starmap", starmap)
    monkeypatch.setattr(module.cache_features, "remote", remote)
    run_id = "tocsin-v2-20260711-sharded"
    path0 = f"cairn_cache_output/{run_id}_shard0_features.jsonl"
    path1 = f"cairn_cache_output/{run_id}_shard1_features.jsonl"
    shard_payloads = module.split_payload_shards(source, 2)
    cache0 = _cache_payload_for_source(shard_payloads[0])
    cache1 = _cache_payload_for_source(shard_payloads[1], final_newline=False)
    monkeypatch.setattr(
        module,
        "hf_cache",
        _MappedVolume(
            {
                path0: cache0,
                path1: cache1,
            }
        ),
    )
    manifest_path = tmp_path / "full.json"
    out_path = tmp_path / "full.features.jsonl"

    module.main(
        dataset_path=str(dataset),
        out_path=str(out_path),
        run_id=run_id,
        manifest_path=str(manifest_path),
        compute_usd_per_second="1",
        config_path="configs/default.yaml",
        gate_manifest_path=str(gate_path),
        cost_cap_usd="15",
        shards=2,
    )

    remote.assert_not_called()
    starmap.assert_called_once()
    assert len(worker_calls) == 2
    assert [call["run_id"] for call in worker_calls] == [
        f"{run_id}_shard0",
        f"{run_id}_shard1",
    ]
    assert out_path.read_bytes() == cache0 + cache1

    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["source_dataset_sha256"] == source_sha
    assert written["scored_payload_sha256"] == source_sha
    assert written["n_workers"] == 2
    assert written["n_warmups"] == 2
    assert written["n_total"] == 4
    assert written["n_eligible"] == 2
    assert written["n_ineligible"] == 2
    assert written["n_timed_eligible"] == 2
    assert written["n_timed_ineligible"] == 2
    assert written["warmup_seconds"] == pytest.approx(0.6)
    assert written["eligible_scoring_seconds"] == pytest.approx(0.4)
    assert written["ineligible_scoring_seconds"] == pytest.approx(0.2)
    assert written["worker_elapsed_seconds"] == pytest.approx(5.0)
    assert Decimal(written["run_compute_usd"]) == Decimal("5")
    assert Decimal(written["cumulative_compute_usd"]) == Decimal("6.0")
    assert written["cost_model"] == inherited_model
    assert written["prelaunch_cost_projection"]["projected_run_seconds"] == 1.6
    assert len(written["worker_manifests"]) == 2
    assert written["cache_output_status"] == "PASS"


def test_worker_output_is_downloaded_losslessly_then_validated_before_pass(
    modal_cache_module, monkeypatch, tmp_path
):
    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    source = (
        json.dumps(
            {"text": " ".join(["café"] * 80)},
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps({"text": "short π"}, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    dataset = tmp_path / "union.jsonl"
    dataset.write_bytes(source)
    records = _cache_records_for_source(source)
    records[0]["download_marker"] = "café"
    payload = _cache_payload(records, final_newline=False)
    split_at = payload.index("é".encode("utf-8")) + 1
    monkeypatch.setattr(
        module,
        "hf_cache",
        _ChunkedVolume([payload[:split_at], payload[split_at:]]),
    )
    monkeypatch.setattr(
        module.cache_features,
        "remote",
        MagicMock(side_effect=lambda **call: _worker_manifest(call)),
    )
    writes: list[tuple[object, object]] = []
    real_write_manifest = module._write_manifest

    def track_manifest(path, manifest):
        writes.append(
            (manifest.get("gate"), manifest.get("cache_output_status"))
        )
        real_write_manifest(path, manifest)

    monkeypatch.setattr(module, "_write_manifest", track_manifest)
    out_path = tmp_path / "features.jsonl"
    manifest_path = tmp_path / "manifest.json"

    module.main(
        dataset_path=str(dataset),
        out_path=str(out_path),
        run_id="tocsin-v2-20260711-lossless",
        manifest_path=str(manifest_path),
        compute_usd_per_second="0.1",
        config_path="configs/default.yaml",
    )

    assert out_path.read_bytes() == payload
    assert ("PENDING_CACHE_VALIDATION", "PENDING") in writes
    assert writes[-1] == ("PASS", "PASS")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["cache_output_validation"]["cached_unique_hashes"] == 2
    assert manifest["cache_output_validation"]["identical_duplicate_hashes"] == 0


@pytest.mark.parametrize(
    "identity_case",
    ["default-cohesiveness", "custom-combined"],
    ids=["explicit-cohesiveness", "explicit-custom-identity"],
)
def test_explicit_expected_signature_is_provenanced_and_validated_exactly(
    modal_cache_module,
    monkeypatch,
    tmp_path,
    identity_case,
):
    from cairn.detectors.token_cohesiveness import (
        build_tocsin_calibration_signature,
    )

    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    if identity_case == "default-cohesiveness":
        expected_signature, identity = build_tocsin_calibration_signature(
            "cohesiveness_u",
            bart_model="facebook/bart-base",
        )
    else:
        expected_signature, identity = build_tocsin_calibration_signature(
            "combined_w",
            bart_model="org/custom-bart",
            base_model="org/custom-fast",
        )
    source = _rows_payload(1)
    dataset = tmp_path / "union.jsonl"
    dataset.write_bytes(source)
    records = _cache_records_for_source_with_signature(
        source,
        expected_signature,
        identity,
    )
    cache_payload = _cache_payload(records)
    monkeypatch.setattr(module, "hf_cache", _MemoryVolume(cache_payload))
    monkeypatch.setattr(
        module.cache_features,
        "remote",
        MagicMock(side_effect=lambda **call: _worker_manifest(call)),
    )
    manifest_path = tmp_path / "manifest.json"

    module.main(
        dataset_path=str(dataset),
        out_path=str(tmp_path / "features.jsonl"),
        run_id="tocsin-v2-20260711-explicit-signature",
        manifest_path=str(manifest_path),
        compute_usd_per_second="0.1",
        config_path="configs/default.yaml",
        expected_tocsin_signature=expected_signature,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["expected_tocsin_signature"] == expected_signature
    assert (
        manifest["cache_output_validation"]["calibration_signature"]
        == expected_signature
    )


@pytest.mark.parametrize(
    "violation",
    [
        "missing-coverage",
        "invalid-schema",
        "wrong-supported-signature",
        "invalid-signature",
        "custom-missing-identity",
        "custom-mismatched-identity",
        "missing-layer-id",
        "non-string-layer-id",
        "duplicate-layer-id",
        "malformed-router-entry",
        "duplicate",
    ],
)
def test_invalid_worker_output_preserves_prior_cache_and_records_failure(
    modal_cache_module,
    monkeypatch,
    tmp_path,
    violation,
):
    module = modal_cache_module
    _install_local_provenance(monkeypatch, module)
    expected_signature_for_run: str | None = None
    source = _rows_payload(1)
    dataset = tmp_path / "union.jsonl"
    dataset.write_bytes(source)
    records = _cache_records_for_source(source)
    if violation == "missing-coverage":
        records.pop()
    elif violation == "invalid-schema":
        records[0].pop("router")
    elif violation == "wrong-supported-signature":
        records = _cache_records_for_source_with_signature(
            source,
            COHESIVENESS_SIGNATURE,
        )
    elif violation == "invalid-signature":
        results = records[0]["results"]
        assert isinstance(results, list)
        signals = results[0]["signals"]
        assert isinstance(signals, dict)
        signals["calibration_signature"] = "unsupported-tocsin-signature"
    elif violation in {"custom-missing-identity", "custom-mismatched-identity"}:
        from cairn.detectors.token_cohesiveness import (
            build_tocsin_calibration_signature,
        )

        custom_signature, identity = build_tocsin_calibration_signature(
            "combined_w",
            bart_model="org/custom-bart",
            base_model="org/custom-fast",
        )
        expected_signature_for_run = custom_signature
        records = _cache_records_for_source_with_signature(
            source,
            custom_signature,
            None if violation == "custom-missing-identity" else identity,
        )
        if violation == "custom-mismatched-identity":
            results = records[0]["results"]
            assert isinstance(results, list)
            signals = results[0]["signals"]
            assert isinstance(signals, dict)
            attached_identity = signals["calibration_identity"]
            assert isinstance(attached_identity, dict)
            attached_identity["base_model"] = "org/spoofed-fast"
    elif violation == "missing-layer-id":
        results = records[0]["results"]
        assert isinstance(results, list)
        results.append({})
    elif violation == "non-string-layer-id":
        results = records[0]["results"]
        assert isinstance(results, list)
        results[0]["layer_id"] = 123
    elif violation == "duplicate-layer-id":
        results = records[0]["results"]
        assert isinstance(results, list)
        first = json.loads(json.dumps(results[0]))
        second = json.loads(json.dumps(results[0]))
        first["layer_id"] = "preamble"
        second["layer_id"] = "preamble"
        results.extend([first, second])
    elif violation == "malformed-router-entry":
        router = records[0]["router"]
        assert isinstance(router, dict)
        router["word_count"] = "not-an-integer"
    elif violation == "duplicate":
        records.append(json.loads(json.dumps(records[0])))
    invalid_payload = _cache_payload(records)
    monkeypatch.setattr(module, "hf_cache", _MemoryVolume(invalid_payload))
    monkeypatch.setattr(
        module.cache_features,
        "remote",
        MagicMock(side_effect=lambda **call: _worker_manifest(call)),
    )
    out_path = tmp_path / "features.jsonl"
    prior_bytes = b"prior-cache-must-survive\n"
    out_path.write_bytes(prior_bytes)
    manifest_path = tmp_path / "manifest.json"

    with pytest.raises(RuntimeError, match="INVALID_CACHE_OUTPUT"):
        module.main(
            dataset_path=str(dataset),
            out_path=str(out_path),
            run_id=f"tocsin-v2-20260711-{violation}",
            manifest_path=str(manifest_path),
                compute_usd_per_second="0.1",
                config_path="configs/default.yaml",
                expected_tocsin_signature=expected_signature_for_run,
        )

    assert out_path.read_bytes() == prior_bytes
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["gate"] == "INVALID_CACHE_OUTPUT"
    assert manifest["cache_output_status"] == "INVALID_CACHE_OUTPUT"
    expected_error_token = {
        "missing-coverage": "coverage",
        "invalid-schema": "router",
        "wrong-supported-signature": "expected tocsin signature",
        "invalid-signature": "signature",
        "custom-missing-identity": "calibration_identity",
        "custom-mismatched-identity": "calibration_identity",
        "missing-layer-id": "layer_id",
        "non-string-layer-id": "layer_id must be a nonempty string",
        "duplicate-layer-id": "duplicate layer_id",
        "malformed-router-entry": "word_count",
        "duplicate": "duplicate",
    }[violation]
    assert expected_error_token in manifest["cache_output_error"].lower()
    assert not list(tmp_path.glob("*.download.tmp"))


def test_sharded_run_rejects_missing_worker_results(modal_cache_module):
    with pytest.raises(RuntimeError, match="returned 1 manifests; expected 2"):
        modal_cache_module.require_worker_result_count([{"run_id": "one"}], 2)

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

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "modal_cache_features.py"
PILOT_SEED = 20260710

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
        "eligible_scoring_seconds": 5,
        "ineligible_scoring_seconds": 5,
        "warmup_seconds": 0,
        "worker_elapsed_seconds": 5,
    }

    result = modal_cache_module.project_modal_cost(
        pilot_manifest,
        total_rows=2,
        eligible_rows=1,
        compute_usd_per_second=Decimal("1"),
        cap_usd=Decimal("15.00"),
    )

    assert result == {
        "projected_full_seconds": 10.0,
        "projected_cumulative_usd": "15",
        "gate": "PASS",
    }


def test_projection_stops_above_cap(modal_cache_module):
    pilot_manifest = {
        "n_eligible": 1,
        "n_ineligible": 1,
        "eligible_scoring_seconds": 5,
        "ineligible_scoring_seconds": 5,
        "warmup_seconds": 0,
        "worker_elapsed_seconds": 5,
    }
    result = modal_cache_module.project_modal_cost(
        pilot_manifest,
        total_rows=2,
        eligible_rows=1,
        compute_usd_per_second=Decimal("1.0000000001"),
        cap_usd=Decimal("15.00"),
    )
    assert result["gate"] == "STOP_COST_CAP"
    assert Decimal(result["projected_cumulative_usd"]) > Decimal("15.00")


def test_cost_model_and_prior_projection_use_decimal_arithmetic(
    modal_cache_module,
):
    measured = {
        "n_eligible": 2,
        "n_ineligible": 4,
        "eligible_scoring_seconds": "0.4",
        "ineligible_scoring_seconds": "0.4",
        "warmup_seconds": "0.3",
    }
    model = modal_cache_module.cost_model_from_manifest(measured)
    assert model == {
        "warmup_seconds": "0.3",
        "eligible_seconds_per_row": "0.2",
        "ineligible_seconds_per_row": "0.1",
    }

    prior = {"cost_model": model, "cumulative_compute_usd": "2.00"}
    projected = modal_cache_module.project_run_cost(
        prior,
        total_rows=10,
        eligible_rows=4,
        compute_usd_per_second=Decimal("0.5"),
        cap_usd=Decimal("2.84"),
    )
    assert projected["projected_run_seconds"] == 1.7
    assert projected["projected_run_compute_usd"] == "0.85"
    assert projected["projected_cumulative_usd"] == "2.85"
    assert projected["gate"] == "STOP_COST_CAP"


def test_completed_cost_fields_preserve_inherited_model_exactly(
    modal_cache_module,
):
    inherited_model = {
        "warmup_seconds": 0.5,
        "eligible_seconds_per_row": "0.2",
        "ineligible_seconds_per_row": "0.1",
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
    }

    model = modal_cache_module.cost_model_from_manifest(partial_retry)
    assert model == {
        "warmup_seconds": "0.5",
        "eligible_seconds_per_row": "1",
        "ineligible_seconds_per_row": "0.5",
    }
    projected = modal_cache_module.project_run_cost(
        {"cost_model": model, "cumulative_compute_usd": "0"},
        total_rows=10,
        eligible_rows=5,
        compute_usd_per_second=Decimal("1"),
        cap_usd=Decimal("1"),
    )
    assert projected["projected_cumulative_usd"] == "8.0"
    assert projected["gate"] == "STOP_COST_CAP"


def test_gate_manifest_requires_pass_and_matching_provenance(
    modal_cache_module,
):
    expected = {
        "source_dataset_sha256": "s" * 64,
        "resolved_config_sha256": "c" * 64,
        "git_revision": "deadbeef1234",
    }
    gate = {"gate": "PASS", **expected}
    assert (
        modal_cache_module.validate_gate_manifest(
            gate,
            source_dataset_sha256=expected["source_dataset_sha256"],
            resolved_config_sha256=expected["resolved_config_sha256"],
            git_revision=expected["git_revision"],
        )
        is gate
    )

    for key in ("source_dataset_sha256", "resolved_config_sha256", "git_revision"):
        incompatible = dict(gate)
        incompatible[key] = "different"
        with pytest.raises(ValueError, match=key):
            modal_cache_module.validate_gate_manifest(
                incompatible,
                source_dataset_sha256=expected["source_dataset_sha256"],
                resolved_config_sha256=expected["resolved_config_sha256"],
                git_revision=expected["git_revision"],
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
            call, worker_elapsed_seconds=3.0
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
                "cost_model": {
                    "warmup_seconds": "1",
                    "eligible_seconds_per_row": "1",
                    "ineligible_seconds_per_row": "1",
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
                "cost_model": {
                    "warmup_seconds": "1",
                    "eligible_seconds_per_row": "1",
                    "ineligible_seconds_per_row": "1",
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
                "cost_model": {
                    "warmup_seconds": "0.1",
                    "eligible_seconds_per_row": "0.1",
                    "ineligible_seconds_per_row": "0.1",
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
                "cost_model": {
                    "warmup_seconds": "0.1",
                    "eligible_seconds_per_row": "0.1",
                    "ineligible_seconds_per_row": "0.1",
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
    monkeypatch.setattr(module, "hf_cache", _MemoryVolume(b'{"hash":"x"}\n'))
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
    assert out_path.read_bytes() == b'{"hash":"x"}\n'

    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert REQUIRED_WORKER_MANIFEST_KEYS <= written.keys()
    assert Decimal(written["run_compute_usd"]) == Decimal("2.00")
    assert Decimal(written["cumulative_compute_usd"]) == Decimal("4.00")
    assert written["compute_usd_per_second"] == "0.5"
    assert written["cost_model"] == {
        "warmup_seconds": "0.1",
        "eligible_seconds_per_row": "0.1",
        "ineligible_seconds_per_row": "0.1",
    }
    assert written["cost_model_source"] == "prior_manifest"


def test_sharded_run_aggregates_workers_cost_and_merged_output(
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
    }
    gate_path.write_text(
        json.dumps(
            {
                "gate": "PASS",
                "source_dataset_sha256": source_sha,
                "resolved_config_sha256": "c" * 64,
                "git_revision": "deadbeef1234567890",
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
    monkeypatch.setattr(
        module,
        "hf_cache",
        _MappedVolume(
            {
                path0: b'{"hash":"a"}\n{"hash":"shared"}\n',
                path1: b'{"hash":"shared"}\n{"hash":"b"}\n',
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
    assert out_path.read_text(encoding="utf-8").splitlines() == [
        '{"hash":"a"}',
        '{"hash":"shared"}',
        '{"hash":"b"}',
    ]

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
    assert written["prelaunch_cost_projection"]["projected_run_seconds"] == 1.4
    assert len(written["worker_manifests"]) == 2


def test_sharded_run_rejects_missing_worker_results(modal_cache_module):
    with pytest.raises(RuntimeError, match="returned 1 manifests; expected 2"):
        modal_cache_module.require_worker_result_count([{"run_id": "one"}], 2)

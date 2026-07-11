import hashlib
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from cairn.detectors.token_cohesiveness import TOCSIN_COMBINED_SIGNATURE
from cairn.evaluation.tocsin_legacy_provenance import (
    LEGACY_PROVENANCE_STATUS,
    SLICE_STEMS,
    generate_retrospective_manifests,
    verify_retrospective_cache_manifest,
)
import cairn.evaluation.tocsin_legacy_provenance as legacy_provenance
from cairn.feature_cache import text_hash
from cairn.normalizer import normalize_text
from cairn.tocsin_cache import validate_tocsin_cache
import scripts.generate_tocsin_legacy_provenance as legacy_cli


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _active_record(text: str, value: float) -> dict:
    rounded = round(value, 5)
    return {
        "hash": text_hash(normalize_text(text)),
        "results": [
            {
                "layer_id": "fast_detectgpt",
                "raw_score": value,
                "p_llm": 0.5,
                "confidence": 0.5,
                "signals": {"curvature": rounded, "scoring_model": "gpt2"},
                "determination": "AMBER",
                "domain": "prose",
                "attacker_tiers": [],
                "compute_cost": "moderate",
                "min_text_length": 60,
                "spans": [],
                "calibrator_applied": False,
            },
            {
                "layer_id": "token_cohesiveness",
                "raw_score": rounded,
                "p_llm": 0.5,
                "confidence": 0.5,
                "signals": {
                    "cohesiveness_u": -1.0,
                    "base_v": rounded,
                    "combined_w": rounded,
                    "calibration_signature": TOCSIN_COMBINED_SIGNATURE,
                    "calibration_raw_channel": "combined_w",
                    "deletion_ratio": 0.015,
                    "n_copies": 10,
                },
                "determination": "AMBER",
                "domain": "prose",
                "attacker_tiers": [],
                "compute_cost": "moderate",
                "min_text_length": 60,
                "spans": [],
                "calibrator_applied": False,
            },
        ],
        "router": {"skip_detectors": []},
    }


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    run_root = repo / "outputs" / "tocsin"
    dataset_root = repo / "data"
    manifests = run_root / "manifests"
    caches = run_root / "caches"
    inputs = run_root / "inputs"
    for path in (dataset_root, manifests, caches, inputs):
        path.mkdir(parents=True, exist_ok=True)
    validator_source = repo / "cairn" / "tocsin_cache.py"
    validator_source.parent.mkdir(parents=True, exist_ok=True)
    validator_source.write_text("# synthetic validator source\n", encoding="utf-8")

    config = repo / "config.yaml"
    config.write_text(
        "detectors:\n"
        "  fast_detectgpt:\n"
        "    enabled: true\n"
        "  token_cohesiveness:\n"
        "    enabled: true\n"
        "cascade: {}\n"
        "decision: {}\n",
        encoding="utf-8",
    )

    union_rows: list[dict] = []
    union_records: list[dict] = []
    input_datasets: list[dict] = []
    for index, stem in enumerate(SLICE_STEMS, start=1):
        text = " ".join([f"token{index}"] * 60)
        row = {"id": stem, "text": text, "label": index % 2}
        value = -0.010004 if index == 1 else 0.01 * index + 0.000004
        record = _active_record(text, value)
        dataset = dataset_root / f"{stem}.jsonl"
        cache = caches / f"{stem}.features.jsonl"
        dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
        cache.write_text(json.dumps(record) + "\n", encoding="utf-8")
        summary = validate_tocsin_cache(
            dataset,
            cache,
            expected_signature=TOCSIN_COMBINED_SIGNATURE,
            expected_layer_ids=("fast_detectgpt", "token_cohesiveness"),
            require_all_layers_seen=True,
        ).to_dict()
        summary["disabled_layer_ids"] = []
        _write_json(manifests / f"{stem}.cache.json", summary)
        input_datasets.append(
            {
                "path": dataset.relative_to(repo).as_posix(),
                "sha256": _sha256(dataset),
                "total_rows": 1,
                "unique_rows": 1,
                "duplicate_rows": 0,
                "eligible_rows": 1,
            }
        )
        union_rows.append(row)
        union_records.append(record)

    union_dataset = inputs / "all_corpora_union.jsonl"
    union_cache = caches / "all_corpora_union.features.jsonl"
    union_dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in union_rows),
        encoding="utf-8",
    )
    union_cache.write_text(
        "".join(json.dumps(row) + "\n" for row in union_records),
        encoding="utf-8",
    )
    union_summary = validate_tocsin_cache(
        union_dataset,
        union_cache,
        expected_signature=TOCSIN_COMBINED_SIGNATURE,
        expected_layer_ids=("fast_detectgpt", "token_cohesiveness"),
        require_all_layers_seen=True,
    ).to_dict()
    union_summary["disabled_layer_ids"] = []
    _write_json(manifests / "all_corpora_union.cache.json", union_summary)
    _write_json(
        manifests / "input-corpora.json",
        {
            "datasets": input_datasets,
            "union": {
                "path": union_dataset.relative_to(repo).as_posix(),
                "sha256": _sha256(union_dataset),
                "total_rows": len(union_rows),
                "unique_rows": len(union_rows),
                "duplicate_rows": 0,
                "eligible_rows": len(union_rows),
            },
        },
    )

    pilot = {
        "source_dataset_sha256": _sha256(union_dataset),
        "scored_payload_sha256": "a" * 64,
        "n_total": 2,
        "n_eligible": 1,
        "n_ineligible": 1,
        "n_timed_eligible": 1,
        "n_timed_ineligible": 1,
        "n_warmups": 1,
        "warmup_seconds": "1",
        "eligible_scoring_seconds": "2",
        "ineligible_scoring_seconds": "1",
        "worker_elapsed_seconds": "5",
        "compute_usd_per_second": "0.5",
        "run_compute_usd": "2.5",
        "cumulative_compute_usd": "2.5",
        "cost_cap_usd": "14.70",
        "projected_full_seconds": 20.0,
        "projected_cumulative_usd": "12.5",
        "gate": "PASS",
        "run_id": "pilot",
    }
    full = {
        "source_dataset_sha256": _sha256(union_dataset),
        "scored_payload_sha256": _sha256(union_dataset),
        "n_total": len(union_rows),
        "n_eligible": len(union_rows),
        "n_ineligible": 0,
        "worker_elapsed_seconds": "30",
        "run_compute_usd": "15",
        "cumulative_compute_usd": "17.5",
        "cost_cap_usd": "14.70",
        "gate": "PASS",
        "run_id": "full",
    }
    _write_json(manifests / "modal-pilot.json", pilot)
    _write_json(manifests / "modal-full.json", full)
    return {
        "repo": repo,
        "run_root": run_root,
        "dataset_root": dataset_root,
        "config": config,
        "pilot": manifests / "modal-pilot.json",
        "full": manifests / "modal-full.json",
    }


def _generate(paths: dict[str, Path]) -> tuple[dict, dict]:
    return generate_retrospective_manifests(
        repo_root=paths["repo"],
        run_root=paths["run_root"],
        dataset_root=paths["dataset_root"],
        config_path=paths["config"],
        git_revision="f" * 40,
        expected_active_rows=len(SLICE_STEMS),
        expected_pilot_rows=2,
        expected_pilot_sha256=_sha256(paths["pilot"]),
        expected_full_sha256=_sha256(paths["full"]),
        original_cost_cap_usd=Decimal("15.00"),
    )


def test_generator_is_deterministic_and_labels_legacy_identity_unbound(tmp_path):
    paths = _fixture(tmp_path)

    first, first_cost = _generate(paths)
    second, second_cost = _generate(paths)

    assert first == second
    assert first_cost == second_cost
    assert first["status"] == LEGACY_PROVENANCE_STATUS
    assert first["diagnostic_allowed"] is True
    assert first["promotion_allowed"] is False
    assert first["calibration_identity"] is None
    assert first["corroboration"]["active_rows"] == len(SLICE_STEMS)
    match = first["corroboration"]["base_v_vs_fast_detectgpt"]
    assert match["matched_rows"] == len(SLICE_STEMS)
    assert match["standalone_scoring_models"] == ["gpt2"]
    assert set(first["slices"]) == set(SLICE_STEMS)
    assert first["observed_static_signature"] == TOCSIN_COMBINED_SIGNATURE


def test_generator_fails_closed_when_anchored_source_manifest_changes(tmp_path):
    paths = _fixture(tmp_path)
    expected_pilot_sha = _sha256(paths["pilot"])
    payload = json.loads(paths["pilot"].read_text(encoding="utf-8"))
    payload["run_id"] = "tampered"
    _write_json(paths["pilot"], payload)

    with pytest.raises(ValueError, match="pilot manifest SHA-256 mismatch"):
        generate_retrospective_manifests(
            repo_root=paths["repo"],
            run_root=paths["run_root"],
            dataset_root=paths["dataset_root"],
            config_path=paths["config"],
            git_revision="f" * 40,
            expected_active_rows=len(SLICE_STEMS),
            expected_pilot_sha256=expected_pilot_sha,
            expected_full_sha256=_sha256(paths["full"]),
        )


def test_cost_correction_records_exact_unattributed_and_projection(tmp_path):
    paths = _fixture(tmp_path)

    _provenance, cost = _generate(paths)

    assert cost["unattributed_worker_seconds"] == "1"
    assert cost["corrected_cost_model"] == {
        "warmup_seconds": "1",
        "eligible_seconds_per_row": "2",
        "ineligible_seconds_per_row": "1",
        "unattributed_seconds_per_row": "0.5",
    }
    assert cost["corrected_projection"] == {
        "total_rows": len(SLICE_STEMS),
        "eligible_rows": len(SLICE_STEMS),
        "ineligible_rows": 0,
        "worker_count": 1,
        "projected_full_seconds": "21.0",
        "projected_full_compute_usd": "10.50",
        "projected_cumulative_usd": "13.00",
        "gate": "PASS",
    }
    assert cost["original_projection"] == {
        "projected_full_seconds": "20.0",
        "projected_cumulative_usd": "12.5",
        "gate": "PASS",
    }
    assert cost["actual_cost"] == {
        "pilot_run_compute_usd": "2.5",
        "full_run_compute_usd": "15",
        "cumulative_compute_usd": "17.5",
    }
    assert cost["caps"] == {
        "effective_run_cap_usd": "14.70",
        "original_total_exposure_cap_usd": "15.00",
    }


def test_retrospective_verifier_revalidates_every_source_byte(tmp_path):
    paths = _fixture(tmp_path)
    provenance, _cost = _generate(paths)
    manifest_path = paths["run_root"] / "manifests" / "retrospective.json"
    _write_json(manifest_path, provenance)

    verified = verify_retrospective_cache_manifest(manifest_path)
    assert verified["status"] == LEGACY_PROVENANCE_STATUS
    assert verified["manifest_sha256"] == _sha256(manifest_path)

    slice_path = paths["dataset_root"] / f"{SLICE_STEMS[0]}.jsonl"
    slice_path.write_text(
        slice_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="mismatch"):
        verify_retrospective_cache_manifest(manifest_path)


def test_v1_schema_rejects_relabelled_verified_manifest(tmp_path):
    paths = _fixture(tmp_path)
    provenance, _cost = _generate(paths)
    provenance.update(
        {
            "status": "VERIFIED",
            "diagnostic_allowed": True,
            "promotion_allowed": True,
            "calibration_identity": {"raw_channel": "combined_w"},
            "unbound_fields": [],
        }
    )
    manifest_path = paths["run_root"] / "manifests" / "relabelled.json"
    _write_json(manifest_path, provenance)

    with pytest.raises(
        ValueError, match="legacy provenance contract mismatch.*status"
    ):
        verify_retrospective_cache_manifest(manifest_path)


def test_cli_writes_both_manifests_without_rewriting_sources(tmp_path):
    paths = _fixture(tmp_path)
    pilot_before = paths["pilot"].read_bytes()
    full_before = paths["full"].read_bytes()
    provenance_out = paths["run_root"] / "manifests" / "retrospective.json"
    cost_out = paths["run_root"] / "manifests" / "retrospective-cost.json"

    exit_code = legacy_cli.main(
        [
            "--repo-root",
            str(paths["repo"]),
            "--run-root",
            str(paths["run_root"]),
            "--dataset-root",
            str(paths["dataset_root"]),
            "--config",
            str(paths["config"]),
            "--git-revision",
            "f" * 40,
            "--expected-active-rows",
            str(len(SLICE_STEMS)),
            "--expected-pilot-rows",
            "2",
            "--expected-pilot-sha256",
            _sha256(paths["pilot"]),
            "--expected-full-sha256",
            _sha256(paths["full"]),
            "--provenance-out",
            str(provenance_out),
            "--cost-out",
            str(cost_out),
        ]
    )

    assert exit_code == 0
    assert json.loads(provenance_out.read_text()) == _generate(paths)[0]
    assert json.loads(cost_out.read_text()) == _generate(paths)[1]
    assert paths["pilot"].read_bytes() == pilot_before
    assert paths["full"].read_bytes() == full_before


@pytest.mark.parametrize("historical_key", ["pilot", "full"])
def test_output_paths_reject_historical_manifest_aliases(
    tmp_path, historical_key
):
    paths = _fixture(tmp_path)
    safe = paths["run_root"] / "manifests" / "safe.json"

    with pytest.raises(ValueError, match="historical Modal manifest"):
        legacy_cli.validate_output_paths(
            paths["run_root"], paths[historical_key], safe
        )


def test_output_paths_reject_same_destination_after_normalization(tmp_path):
    paths = _fixture(tmp_path)
    destination = paths["run_root"] / "manifests" / "result.json"
    alias = destination.parent / "." / destination.name

    with pytest.raises(ValueError, match="must be distinct"):
        legacy_cli.validate_output_paths(paths["run_root"], destination, alias)


def test_output_paths_reject_hardlink_alias_of_historical_manifest(tmp_path):
    paths = _fixture(tmp_path)
    hardlink = paths["run_root"] / "manifests" / "pilot-hardlink.json"
    os.link(paths["pilot"], hardlink)

    with pytest.raises(ValueError, match="historical Modal manifest"):
        legacy_cli.validate_output_paths(
            paths["run_root"], hardlink, hardlink.parent / "safe.json"
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows case-folding contract")
def test_output_paths_reject_case_only_historical_alias(tmp_path):
    paths = _fixture(tmp_path)
    case_alias = Path(str(paths["pilot"]).swapcase())

    with pytest.raises(ValueError, match="historical Modal manifest"):
        legacy_cli.validate_output_paths(
            paths["run_root"], case_alias, paths["run_root"] / "safe.json"
        )


def test_path_identity_resolves_existing_worktree_data_junction():
    repo = Path(__file__).resolve().parents[1]
    junction_path = repo / "data" / "prompts_v2_train.jsonl"
    if not junction_path.is_file() or junction_path.resolve() == junction_path:
        pytest.skip("worktree data junction is not available")

    assert legacy_cli._same_resolved_path(junction_path, junction_path.resolve())


def test_cli_rejects_colliding_outputs_before_generation_or_write(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path)
    called = {"generate": False, "write": False}
    monkeypatch.setattr(
        legacy_cli,
        "generate_retrospective_manifests",
        lambda **kwargs: called.__setitem__("generate", True),
    )
    monkeypatch.setattr(
        legacy_cli,
        "_write_pair",
        lambda rendered: called.__setitem__("write", True),
    )

    with pytest.raises(ValueError, match="historical Modal manifest"):
        legacy_cli.main(
            [
                "--repo-root",
                str(paths["repo"]),
                "--run-root",
                str(paths["run_root"]),
                "--dataset-root",
                str(paths["dataset_root"]),
                "--config",
                str(paths["config"]),
                "--provenance-out",
                str(paths["pilot"]),
                "--cost-out",
                str(paths["run_root"] / "manifests" / "safe.json"),
            ]
        )

    assert called == {"generate": False, "write": False}


def test_module_and_direct_script_help_import_from_clean_process():
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    commands = (
        [sys.executable, "-m", "scripts.generate_tocsin_legacy_provenance", "--help"],
        [sys.executable, "scripts/generate_tocsin_legacy_provenance.py", "--help"],
    )

    for command in commands:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr


def test_worktree_data_junction_is_recorded_lexically_when_present():
    repo = Path(__file__).resolve().parents[1]
    dataset = repo / "data" / "prompts_v2_train.jsonl"
    if not dataset.is_file():
        pytest.skip("repository dataset is not available")

    assert legacy_provenance._relative_path(repo, dataset) == (
        "data/prompts_v2_train.jsonl"
    )

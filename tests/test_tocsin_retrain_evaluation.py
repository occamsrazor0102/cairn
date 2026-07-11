import ast
import csv
import hashlib
import inspect
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from cairn.calibration import CalibrationMetadata, DetectorCalibrator
from cairn.detectors.token_cohesiveness import TOCSIN_COMBINED_SIGNATURE
from cairn.evaluation.dataset import EvalSample
from cairn.evaluation.tocsin_retrain import (
    EvaluatedRow,
    ProfileEvaluation,
    binomial_target_interval,
    build_humanized_hard_slice,
    decide_promotion,
    detector_comparison,
    evaluate_profile,
    legacy_unsafe_transform_for_diagnostic,
    load_evaluation_slice,
    probability_metrics,
    profile_artifact_failures,
    profile_name,
    profile_metrics,
    preflight_profiles,
    reliability_curve,
)
from cairn.feature_cache import CachedPipeline, text_hash
from cairn.fusion.ebm import EBMFusion
from cairn.normalizer import normalize_text
from cairn.pipeline import CairnPipeline
import cairn.evaluation.tocsin_retrain as tocsin_retrain_module
import scripts.run_tocsin_retrain_benchmark as benchmark
from scripts.run_tocsin_retrain_benchmark import build_parser, main as benchmark_main


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_provenance_fixture(tmp_path: Path) -> dict[str, Path]:
    model = tmp_path / "ebm.skops"
    hmac_path = tmp_path / "ebm.skops.hmac"
    meta = tmp_path / "ebm.skops.meta.json"
    conformal = tmp_path / "conformal.json"
    calibration = tmp_path / "detector_calibration.json"
    for index, path in enumerate(
        (model, hmac_path, meta, conformal, calibration), start=1
    ):
        path.write_bytes(f"artifact-{index}".encode())
    config = tmp_path / "candidate.yaml"
    config.write_text(
        "\n".join(
            [
                "_profile: profile_prompts_only_tocsin_v2",
                "detectors: {}",
                "cascade: {}",
                "decision: {}",
                "fusion:",
                f"  model_path: {json.dumps(model.as_posix())}",
                f"  conformal_path: {json.dumps(conformal.as_posix())}",
                "  detector_calibration_path: "
                + json.dumps(calibration.as_posix()),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = (config, model, hmac_path, meta, conformal, calibration)
    candidate_manifest = tmp_path / "candidate-integrity.json"
    candidate_manifest.write_text(
        json.dumps(
            {
                str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in artifacts
            }
        ),
        encoding="utf-8",
    )

    dataset = tmp_path / "prompt.jsonl"
    rows = [
        {"id": "human", "text": "Human authored example", "label": 0},
        {"id": "llm", "text": "Machine generated example", "label": 1},
    ]
    _write_jsonl(dataset, rows)
    cache = tmp_path / "prompt.features.jsonl"
    _write_jsonl(
        cache,
        [
            {
                "hash": text_hash(normalize_text(row["text"])),
                "results": [],
                "router": {},
            }
            for row in rows
        ],
    )
    cache_manifest = tmp_path / "prompt.cache.json"
    cache_manifest.write_text(
        json.dumps(
            {
                "dataset_sha256": _sha256(dataset),
                "cache_sha256": _sha256(cache),
                "expected_unique_hashes": 2,
                "cached_unique_hashes": 2,
                "identical_duplicate_hashes": 0,
            }
        ),
        encoding="utf-8",
    )
    artifact = lambda path: {
        "path": path.relative_to(tmp_path).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }
    retrospective = tmp_path / "retrospective.json"
    retrospective.write_text(
        json.dumps(
            {
                "schema": "cairn.tocsin-cache-retrospective.v1",
                "status": "VERIFIED_BYTES_LEGACY_IDENTITY_UNBOUND",
                "diagnostic_allowed": True,
                "promotion_allowed": False,
                "calibration_identity": None,
                "unbound_fields": [
                    "modal_output_lineage",
                    "modal_output_sha256",
                    "actual_base_identity",
                    "huggingface_model_revisions",
                    "modal_image_digest",
                ],
                "config": artifact(config),
                "union": {
                    "dataset": artifact(dataset),
                    "cache": artifact(cache),
                    "validation_manifest": artifact(cache_manifest),
                },
                "slices": {
                    "prompts_v5_test": {
                        "dataset": artifact(dataset),
                        "cache": artifact(cache),
                        "validation_manifest": artifact(cache_manifest),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "config": config,
        "candidate_manifest": candidate_manifest,
        "dataset": dataset,
        "cache": cache,
        "cache_manifest": cache_manifest,
        "retrospective": retrospective,
    }


def test_benchmark_provenance_verifies_candidate_bytes_and_exact_slice_coverage(
    tmp_path, monkeypatch,
):
    paths = _write_provenance_fixture(tmp_path)
    monkeypatch.setattr(
        tocsin_retrain_module,
        "verify_retrospective_cache_manifest",
        lambda _path: {
            "schema": "cairn.tocsin-cache-retrospective.v1",
            "status": "VERIFIED_BYTES_LEGACY_IDENTITY_UNBOUND",
            "diagnostic_allowed": True,
            "promotion_allowed": False,
            "calibration_identity": None,
            "unbound_fields": ["actual_base_identity"],
            "manifest_path": str(paths["retrospective"]),
            "manifest_sha256": _sha256(paths["retrospective"]),
            "slices": {
                "prompts_v5_test": {
                    "dataset": {"sha256": _sha256(paths["dataset"])},
                    "cache": {"sha256": _sha256(paths["cache"])},
                    "validation_manifest": {
                        "sha256": _sha256(paths["cache_manifest"])
                    },
                }
            },
        },
    )

    provenance = tocsin_retrain_module.verify_benchmark_provenance(
        paths["config"],
        paths["candidate_manifest"],
        {"prompt_test": (paths["dataset"], paths["cache"])},
        [("prompt_test", paths["cache_manifest"])],
        paths["retrospective"],
    )

    assert provenance["status"] == "VERIFIED_BYTES_LEGACY_IDENTITY_UNBOUND"
    assert provenance["promotion_allowed"] is False
    assert provenance["candidate_integrity"]["manifest_sha256"] == _sha256(
        paths["candidate_manifest"]
    )
    assert provenance["slices"]["prompt_test"] == {
        "manifest_path": str(paths["cache_manifest"]),
        "manifest_sha256": _sha256(paths["cache_manifest"]),
        "dataset_path": str(paths["dataset"]),
        "dataset_sha256": _sha256(paths["dataset"]),
        "cache_path": str(paths["cache"]),
        "cache_sha256": _sha256(paths["cache"]),
        "unique_hashes": 2,
    }


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        ([], "missing cache manifest mapping: prompt_test"),
        (
            [
                ("prompt_test", "manifest"),
                ("prompt_test", "manifest"),
            ],
            "duplicate cache manifest mapping: prompt_test",
        ),
    ],
)
def test_benchmark_provenance_rejects_missing_or_duplicate_slice_mapping(
    tmp_path, mapping, expected
):
    paths = _write_provenance_fixture(tmp_path)
    resolved_mapping = [
        (name, paths["cache_manifest"] if value == "manifest" else Path(value))
        for name, value in mapping
    ]

    with pytest.raises(ValueError, match=expected):
        tocsin_retrain_module.verify_benchmark_provenance(
            paths["config"],
            paths["candidate_manifest"],
            {"prompt_test": (paths["dataset"], paths["cache"])},
            resolved_mapping,
            paths["retrospective"],
        )


def test_benchmark_provenance_rejects_changed_dataset_before_metrics(tmp_path):
    paths = _write_provenance_fixture(tmp_path)
    paths["dataset"].write_text(
        paths["dataset"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prompt_test dataset SHA-256 mismatch"):
        tocsin_retrain_module.verify_benchmark_provenance(
            paths["config"],
            paths["candidate_manifest"],
            {"prompt_test": (paths["dataset"], paths["cache"])},
            [("prompt_test", paths["cache_manifest"])],
            paths["retrospective"],
        )


def test_benchmark_provenance_rejects_duplicate_cache_hash_coverage(tmp_path):
    paths = _write_provenance_fixture(tmp_path)
    first = paths["cache"].read_text(encoding="utf-8").splitlines()[0]
    paths["cache"].write_text(first + "\n" + first + "\n", encoding="utf-8")
    manifest = json.loads(paths["cache_manifest"].read_text(encoding="utf-8"))
    manifest["cache_sha256"] = _sha256(paths["cache"])
    paths["cache_manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt_test cache duplicate hash"):
        tocsin_retrain_module.verify_benchmark_provenance(
            paths["config"],
            paths["candidate_manifest"],
            {"prompt_test": (paths["dataset"], paths["cache"])},
            [("prompt_test", paths["cache_manifest"])],
            paths["retrospective"],
        )


def test_preflight_explicit_provenance_request_fails_closed_when_manifest_missing(
    tmp_path,
):
    paths = _write_provenance_fixture(tmp_path)

    result = preflight_profiles(
        paths["config"],
        [paths["config"]],
        paths["dataset"],
        paths["cache"],
        candidate_integrity_manifest=None,
        slice_inputs={"prompt_test": (paths["dataset"], paths["cache"])},
        cache_manifest_mappings=[],
        retrospective_provenance_manifest=None,
    )

    assert result["provenance"]["status"] == "INVALID"
    assert any(
        failure.startswith("INVALID_PROVENANCE: missing candidate integrity manifest")
        for failure in result["failures"]
    )


def test_load_evaluation_slice_uses_stable_collision_safe_text_hash_ids(tmp_path):
    path = tmp_path / "slice.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "first",
                "source_id": "shared",
                "text": "Alpha text",
                "label": 0,
            },
            {
                "id": "second",
                "source_id": "shared",
                "text": "Beta text",
                "label": 1,
            },
        ],
    )

    first = load_evaluation_slice(path)
    second = load_evaluation_slice(path)

    assert [sample.id for sample in first] == [sample.id for sample in second]
    assert first[0].id == f"slice:{text_hash(normalize_text('Alpha text'))}"
    assert first[1].id == f"slice:{text_hash(normalize_text('Beta text'))}"
    assert first[0].id != first[1].id
    assert first[0].source_id == "shared|id=first"
    assert first[1].source_id == "shared|id=second"


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([0, 0], "duplicate normalized text"),
        ([0, 1], "cross-label duplicate"),
    ],
)
def test_load_evaluation_slice_rejects_duplicate_normalized_text(
    tmp_path, labels, message
):
    path = tmp_path / "duplicates.jsonl"
    _write_jsonl(
        path,
        [
            {"text": "same   text", "label": labels[0]},
            {"text": "same text", "label": labels[1]},
        ],
    )

    with pytest.raises(ValueError, match=message):
        load_evaluation_slice(path)


def test_load_evaluation_slice_reports_missing_fields_with_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    _write_jsonl(path, [{"text": "ok", "label": 0}, {"text": "missing"}])

    with pytest.raises(ValueError, match=r"bad\.jsonl:2: text and label are required"):
        load_evaluation_slice(path)


def test_reliability_curve_has_deterministic_half_open_bins():
    rows = reliability_curve([0, 1, 1], [0.0, 0.1, 1.0], n_bins=10)

    assert len(rows) == 10
    assert rows[0]["count"] == 1
    assert rows[1]["count"] == 1
    assert rows[-1]["count"] == 1
    assert rows[-1]["mean_probability"] == 1.0
    assert rows[2]["mean_probability"] is None


def test_probability_metrics_are_complete():
    metrics = probability_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert metrics["n"] == 4
    assert metrics["n_unique"] == 4
    assert metrics["finite"] is True
    assert metrics["in_unit_interval"] is True
    assert metrics["human_mean"] == pytest.approx(0.15)
    assert metrics["positive_mean"] == pytest.approx(0.85)
    assert metrics["endpoint_fraction"] == 0.0
    assert len(metrics["reliability_curve"]) == 10


def test_invalid_probabilities_make_probability_metrics_explicitly_missing():
    metrics = probability_metrics([0, 1], [math.nan, 2.0])

    assert metrics["finite"] is False
    assert metrics["in_unit_interval"] is False
    for key in (
        "brier",
        "log_loss",
        "auroc",
        "ece",
        "human_mean",
        "positive_mean",
        "endpoint_fraction",
        "reliability_curve",
    ):
        assert metrics[key] is None


def test_degenerate_class_metrics_use_none_instead_of_nan():
    metrics = profile_metrics(
        [0, 0],
        [0.1, 0.2],
        [["human"], ["human"]],
        [0.0, 0.0],
        fusion_modes=["ebm", "ebm"],
    )

    assert metrics["auroc"] is None
    assert metrics["balanced_accuracy"] is None
    assert metrics["tpr_at_fpr_05"] is None
    assert metrics["tpr_at_fpr_10"] is None


def test_detector_comparison_uses_one_support_mask_for_all_probability_metrics():
    result = detector_comparison(
        [0, 0, 1, 1],
        [-1.0, 0.0, 0.5, 2.0],
        [0.9, 0.8, 0.7, 0.6],
        [0.1, 0.2, 0.8, 0.9],
        [False, True, True, False],
        support=(0.0, 0.5),
        tolerance=0.0,
        detector_skip_rows=7,
    )

    assert result["eligible_rows"] == 4
    assert result["detector_skip_rows"] == 7
    assert result["in_support_rows"] == 2
    assert result["candidate"]["n"] == 2
    assert result["legacy_unsafe_clip_diagnostic"]["n"] == 2
    assert result["constant_prevalence"]["n"] == 2


def test_detector_comparison_rejects_mask_disagreement():
    with pytest.raises(
        ValueError, match="candidate apply mask disagrees with support guard"
    ):
        detector_comparison(
            [0, 1],
            [0.0, 0.5],
            [0.2, 0.8],
            [0.1, 0.9],
            [True, False],
            support=(0.0, 0.5),
            tolerance=0.0,
            detector_skip_rows=0,
        )


def test_profile_metrics_include_threshold_conformal_and_reconstruction_fields():
    metrics = profile_metrics(
        [0, 0, 1, 1],
        [0.1, 0.6, 0.4, 0.9],
        [["human"], ["human", "LLM"], ["human"], ["LLM"]],
        [0.0, 1e-12, 2e-12, 3e-12],
        fusion_modes=["ebm"] * 4,
    )

    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["human_fpr"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["conformal_truth_coverage_count"] == 3
    assert metrics["conformal_truth_coverage"] == 0.75
    assert metrics["singleton_rate"] == 0.75
    assert metrics["selective_accuracy"] == pytest.approx(2 / 3)
    assert metrics["max_ebm_reconstruction_error"] == 3e-12
    assert metrics["all_ebm"] is True


def test_binomial_target_interval_is_two_sided_count_interval():
    assert binomial_target_interval(100, 0.95) == (90, 99)


def _profile_block(
    *,
    brier=0.20,
    log_loss=0.40,
    balanced_accuracy=0.79,
    auroc=0.79,
    human_fpr=0.10,
    tpr_at_fpr_05=0.78,
    tpr_at_fpr_10=0.78,
    coverage_count=96,
    n=100,
):
    return {
        "n": n,
        "brier": brier,
        "log_loss": log_loss,
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "human_fpr": human_fpr,
        "specificity": 1.0 - human_fpr,
        "tpr_at_fpr_05": tpr_at_fpr_05,
        "tpr_at_fpr_10": tpr_at_fpr_10,
        "conformal_truth_coverage_count": coverage_count,
        "conformal_truth_coverage_n": n,
        "finite": True,
        "in_unit_interval": True,
        "all_ebm": True,
        "max_ebm_reconstruction_error": 1e-9,
    }


def _legacy_provenance() -> dict:
    return {
        "schema": "cairn.tocsin-cache-retrospective.v1",
        "status": "VERIFIED_BYTES_LEGACY_IDENTITY_UNBOUND",
        "diagnostic_allowed": True,
        "promotion_allowed": False,
        "calibration_identity": None,
        "unbound_fields": [
            "modal_output_lineage",
            "modal_output_sha256",
            "actual_base_identity",
            "huggingface_model_revisions",
            "modal_image_digest",
        ],
    }


def _assert_only_legacy_provenance_lock(report: dict) -> None:
    decision = decide_promotion(report)
    assert decision["status"] == "DO_NOT_PROMOTE"
    assert len(decision["failures"]) == 1
    assert decision["failures"][0].startswith("UNPROMOTABLE_PROVENANCE:")


def _passing_report():
    candidate_prompt = _profile_block()
    candidate_humanized = _profile_block()
    candidate_ood = {
        name: _profile_block(auroc=0.78)
        for name in ("ood_m4_test", "ood_raid_test", "ood_mage_test")
    }
    baseline_prompt = _profile_block(
        balanced_accuracy=0.80,
        auroc=0.80,
        human_fpr=0.10,
    )
    baseline_humanized = _profile_block(
        auroc=0.80,
        tpr_at_fpr_05=0.80,
        tpr_at_fpr_10=0.80,
    )
    baseline_ood = {
        name: _profile_block(auroc=0.80)
        for name in ("ood_m4_test", "ood_raid_test", "ood_mage_test")
    }
    return {
        "preflight": {
            "failures": [],
            "provenance": _legacy_provenance(),
        },
        "detector_calibration": {
            "eligible_rows": 1822,
            "support_coverage": 1.0,
            "candidate": {
                "finite": True,
                "in_unit_interval": True,
                "n_unique": 4,
                "brier": 0.10,
                "log_loss": 0.30,
                "endpoint_fraction": 0.949999,
            },
            "legacy_unsafe_clip_diagnostic": {
                "brier": 0.11,
                "log_loss": 0.31,
            },
            "constant_prevalence": {"brier": 0.12},
        },
        "candidate": {
            "prompt_test": candidate_prompt,
            "humanized_hard": candidate_humanized,
            "ood": candidate_ood,
        },
        "comparators": {
            "prompts_only": {
                "prompt_test": baseline_prompt,
                "humanized_hard": baseline_humanized,
                "ood": baseline_ood,
            },
            "prompts_specdetect": {
                "prompt_test": deepcopy(baseline_prompt),
                "humanized_hard": deepcopy(baseline_humanized),
                "ood": deepcopy(baseline_ood),
            },
        },
        "challenge": {"brier": 999.0, "balanced_accuracy": 0.0},
    }


def _mixed_gate_audit_report():
    report = _passing_report()
    report["candidate_name"] = "prompts_only_tocsin_v2"
    report["challenge"] = None
    invalid_comparator_failure = (
        "INVALID_COMPARATOR prompts_specdetect:\n"
        "missing models/calib2000/profiles/prompts_specdetect/ebm.skops\n"
        "missing models/calib2000/profiles/prompts_specdetect/ebm.skops.hmac"
    )
    report["preflight"] = {
        "candidate": {
            "name": "prompts_only_tocsin_v2",
            "valid": True,
            "failures": [],
        },
        "comparators": {
            "prompts_only": {
                "name": "prompts_only",
                "valid": True,
                "failures": [],
            },
            "prompts_specdetect": {
                "name": "prompts_specdetect",
                "valid": False,
                "failures": [
                    "missing models/calib2000/profiles/prompts_specdetect/ebm.skops",
                    "missing models/calib2000/profiles/prompts_specdetect/ebm.skops.hmac",
                ],
            },
        },
        "comparator_identity_valid": True,
        "provenance": _legacy_provenance(),
        "failures": [invalid_comparator_failure],
    }
    del report["comparators"]["prompts_specdetect"]
    report["candidate"]["prompt_test"]["human_fpr"] = math.nextafter(
        0.10, math.inf
    )
    return report, invalid_comparator_failure


def _expected_locked_gate_ids():
    slices = (
        "prompt_test",
        "humanized_hard",
        "ood_m4_test",
        "ood_raid_test",
        "ood_mage_test",
    )
    integrity = (
        "all_ebm",
        "finite",
        "in_unit_interval",
        "max_ebm_reconstruction_error",
    )
    ids = [
        "preflight.comparator_identity",
        "preflight.provenance",
        "preflight.candidate_artifacts",
        "preflight.comparator.prompts_only.artifacts",
        "preflight.comparator.prompts_specdetect.artifacts",
        "detector.eligible_rows",
        "detector.support_coverage",
        "detector.candidate.finite",
        "detector.candidate.in_unit_interval",
        "detector.candidate.n_unique",
        "detector.candidate.brier_vs_legacy",
        "detector.candidate.log_loss_vs_legacy",
        "detector.candidate.brier_vs_constant",
        "detector.candidate.endpoint_fraction",
    ]
    ids.extend(
        f"candidate.{slice_name}.{metric}"
        for slice_name in slices
        for metric in integrity
    )
    ids.append("candidate.prompt_test.conformal_truth_coverage_count")
    for profile in ("prompts_only", "prompts_specdetect"):
        ids.extend(
            f"comparator.{profile}.{slice_name}.{metric}"
            for slice_name in slices
            for metric in integrity
        )
        ids.extend(
            [
                f"comparator.{profile}.prompt_test.brier",
                f"comparator.{profile}.prompt_test.log_loss",
                f"comparator.{profile}.prompt_test.balanced_accuracy",
                f"comparator.{profile}.prompt_test.auroc",
                f"comparator.{profile}.prompt_test.human_fpr",
                f"comparator.{profile}.humanized_hard.auroc",
                f"comparator.{profile}.humanized_hard.tpr_at_fpr_05",
                f"comparator.{profile}.humanized_hard.tpr_at_fpr_10",
                f"comparator.{profile}.ood_m4_test.auroc",
                f"comparator.{profile}.ood_raid_test.auroc",
                f"comparator.{profile}.ood_mage_test.auroc",
            ]
        )
    return ids


def test_promotion_gate_audit_is_complete_ordered_and_preserves_decision():
    report, invalid_comparator_failure = _mixed_gate_audit_report()

    audit = tocsin_retrain_module.promotion_gate_audit(report)
    gate_ids = [row["gate_id"] for row in audit]

    assert gate_ids == _expected_locked_gate_ids()
    assert len(gate_ids) == len(set(gate_ids))
    by_id = {row["gate_id"]: row for row in audit}
    assert by_id["preflight.comparator.prompts_only.artifacts"]["status"] == "PASS"
    assert (
        by_id["preflight.comparator.prompts_specdetect.artifacts"]["status"]
        == "FAIL"
    )
    assert by_id["detector.eligible_rows"] == {
        "gate_id": "detector.eligible_rows",
        "label": "Detector eligible row count",
        "scope": "detector",
        "profile": "token_cohesiveness",
        "slice": "prompt_test",
        "status": "PASS",
        "observed": 1822,
        "operator": "==",
        "reference": 1822,
        "failure_detail": None,
        "not_evaluated_reason": None,
    }
    assert by_id["candidate.prompt_test.conformal_truth_coverage_count"][
        "observed"
    ] == 96
    assert by_id["candidate.prompt_test.conformal_truth_coverage_count"][
        "reference"
    ] == [90, 99]
    failed_fpr = by_id["comparator.prompts_only.prompt_test.human_fpr"]
    assert failed_fpr["status"] == "FAIL"
    assert failed_fpr["observed"] == math.nextafter(0.10, math.inf)
    assert failed_fpr["operator"] == "<="
    assert failed_fpr["reference"] == 0.10
    assert failed_fpr["failure_detail"] == (
        "prompts_only prompt_test Human FPR: candidate regressed"
    )
    specdetect_evaluation = [
        row
        for row in audit
        if row["gate_id"].startswith("comparator.prompts_specdetect.")
        and not row["gate_id"].endswith(".artifacts")
    ]
    assert len(specdetect_evaluation) == 31
    assert {row["status"] for row in specdetect_evaluation} == {"NOT_EVALUATED"}
    failures = decide_promotion(report)["failures"]
    assert failures[0].startswith("UNPROMOTABLE_PROVENANCE:")
    assert failures[1:] == [
        invalid_comparator_failure,
        "prompts_only prompt_test Human FPR: candidate regressed",
    ]


def test_markdown_renders_every_gate_status_from_shared_audit():
    report, _ = _mixed_gate_audit_report()
    decision = decide_promotion(report)

    rendered = benchmark._markdown_report(decision, report)

    assert "| Status | Gate ID | Scope | Profile | Slice | Observed | Operator | Reference | Detail |" in rendered
    assert "| PASS | `detector.eligible_rows` |" in rendered
    assert "| FAIL | `comparator.prompts_only.prompt_test.human_fpr` |" in rendered
    assert (
        "| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.brier` |"
        in rendered
    )
    assert rendered.count("`detector.eligible_rows`") == 1
    assert rendered.count("`comparator.prompts_only.prompt_test.human_fpr`") == 1


def test_legacy_provenance_allows_diagnostics_but_is_an_automatic_promotion_fail():
    report = _passing_report()
    report["preflight"].update(_valid_preflight())
    report["preflight"]["provenance"] = _legacy_provenance()
    report["preflight"]["comparators"]["prompts_specdetect"]["valid"] = False
    report["preflight"]["comparators"]["prompts_specdetect"]["failures"] = [
        "missing comparator"
    ]
    report["preflight"]["failures"] = [
        "INVALID_COMPARATOR prompts_specdetect: missing comparator"
    ]

    audit = tocsin_retrain_module.promotion_gate_audit(report)
    row = next(row for row in audit if row["gate_id"] == "preflight.provenance")

    assert row["status"] == "FAIL"
    assert row["observed"] == "VERIFIED_BYTES_LEGACY_IDENTITY_UNBOUND"
    assert "UNPROMOTABLE_PROVENANCE" in row["failure_detail"]
    assert decide_promotion(report)["status"] == "DO_NOT_PROMOTE"
    assert benchmark._diagnostic_block_reason(report["preflight"]) is None


@pytest.mark.parametrize("status", ["VERIFIED", "FORGED"])
def test_diagnostic_rejects_nonlegacy_status_even_when_diagnostic_flag_is_true(
    status,
):
    preflight = _valid_preflight()
    preflight["comparators"]["prompts_specdetect"].update(
        {"valid": False, "failures": ["missing comparator"]}
    )
    preflight["failures"] = [
        "INVALID_COMPARATOR prompts_specdetect: missing comparator"
    ]
    preflight["provenance"] = {
        **_legacy_provenance(),
        "status": status,
        "diagnostic_allowed": True,
    }

    reason = benchmark._diagnostic_block_reason(preflight)

    assert reason is not None
    assert "exact legacy provenance" in reason


def test_relabelled_verified_v1_provenance_stays_an_automatic_fail():
    report = _passing_report()
    report["preflight"]["provenance"] = {
        "schema": "cairn.tocsin-cache-retrospective.v1",
        "status": "VERIFIED",
        "diagnostic_allowed": True,
        "promotion_allowed": True,
        "calibration_identity": {"raw_channel": "combined_w"},
        "unbound_fields": [],
    }

    decision = decide_promotion(report)
    row = next(
        row
        for row in tocsin_retrain_module.promotion_gate_audit(report)
        if row["gate_id"] == "preflight.provenance"
    )

    assert row["status"] == "FAIL"
    assert decision["status"] == "DO_NOT_PROMOTE"
    assert any("UNPROMOTABLE_PROVENANCE" in item for item in decision["failures"])


def test_invalid_provenance_never_allows_diagnostic_full_run():
    preflight = _valid_preflight()
    preflight["provenance"] = {
        "status": "INVALID",
        "diagnostic_allowed": False,
        "promotion_allowed": False,
    }

    assert "exact legacy provenance" in benchmark._diagnostic_block_reason(
        preflight
    )


def test_not_evaluated_gate_observed_values_are_always_null():
    report, _ = _mixed_gate_audit_report()
    audit = tocsin_retrain_module.promotion_gate_audit(report)

    not_evaluated = [row for row in audit if row["status"] == "NOT_EVALUATED"]
    assert not_evaluated
    assert {row["observed"] for row in not_evaluated} == {None}
    rendered = benchmark._markdown_report(decide_promotion(report), report)
    assert all(
        " | NA | " in line
        for line in rendered.splitlines()
        if "| NOT_EVALUATED |" in line
    )


def test_every_metric_gate_passes_but_legacy_provenance_locks_promotion():
    _assert_only_legacy_provenance_lock(_passing_report())


def test_decide_promotion_rejects_clean_but_incomplete_preflight():
    decision = decide_promotion({"preflight": _valid_preflight()})

    assert decision["status"] == "DO_NOT_PROMOTE"
    assert len(decision["failures"]) == 1
    assert decision["failures"][0].startswith("UNPROMOTABLE_PROVENANCE:")


@pytest.mark.parametrize(
    ("mutate", "failure_fragment"),
    [
        (
            lambda r: r["detector_calibration"].__setitem__("eligible_rows", 1821),
            "eligible rows",
        ),
        (
            lambda r: r["detector_calibration"].__setitem__("support_coverage", 0.999),
            "support coverage",
        ),
        (
            lambda r: r["detector_calibration"]["candidate"].__setitem__(
                "n_unique", 3
            ),
            "unique probabilities",
        ),
        (
            lambda r: r["detector_calibration"]["candidate"].__setitem__(
                "finite", False
            ),
            "non-finite",
        ),
        (
            lambda r: r["detector_calibration"]["candidate"].__setitem__(
                "in_unit_interval", False
            ),
            "outside [0, 1]",
        ),
        (
            lambda r: r["detector_calibration"]["candidate"].__setitem__(
                "brier", 0.11
            ),
            "stale Brier",
        ),
        (
            lambda r: r["detector_calibration"]["candidate"].__setitem__(
                "log_loss", 0.31
            ),
            "stale log loss",
        ),
        (
            lambda r: r["detector_calibration"]["constant_prevalence"].__setitem__(
                "brier", 0.10
            ),
            "constant prevalence",
        ),
        (
            lambda r: r["detector_calibration"]["candidate"].__setitem__(
                "endpoint_fraction", 0.95
            ),
            "endpoint fraction",
        ),
        (
            lambda r: r["candidate"]["prompt_test"].__setitem__(
                "all_ebm", False
            ),
            "not every row",
        ),
        (
            lambda r: r["candidate"]["prompt_test"].__setitem__(
                "finite", False
            ),
            "non-finite p_llm",
        ),
        (
            lambda r: r["candidate"]["prompt_test"].__setitem__(
                "in_unit_interval", False
            ),
            "p_llm outside [0, 1]",
        ),
        (
            lambda r: r["comparators"]["prompts_only"]["prompt_test"].__setitem__(
                "all_ebm", False
            ),
            "comparator prompts_only prompt_test: not every row",
        ),
        (
            lambda r: r["candidate"]["prompt_test"].__setitem__(
                "conformal_truth_coverage_count", 89
            ),
            "conformal coverage count",
        ),
        (
            lambda r: r["candidate"]["prompt_test"].__setitem__(
                "conformal_truth_coverage_count", 100
            ),
            "conformal coverage count",
        ),
    ],
)
def test_decide_promotion_rejects_each_gate_immediately_beyond_boundary(
    mutate, failure_fragment
):
    report = _passing_report()
    mutate(report)

    decision = decide_promotion(report)

    assert decision["status"] == "DO_NOT_PROMOTE"
    assert any(failure_fragment in failure for failure in decision["failures"])


def _nested_get(mapping, path):
    for key in path:
        mapping = mapping[key]
    return mapping


def _nested_set(mapping, path, value):
    for key in path[:-1]:
        mapping = mapping[key]
    mapping[path[-1]] = value


@pytest.mark.parametrize(
    ("candidate_path", "baseline_path", "allowance", "direction"),
    [
        (("candidate", "prompt_test", "brier"), ("comparators", "prompts_only", "prompt_test", "brier"), 0.0, "max"),
        (("candidate", "prompt_test", "log_loss"), ("comparators", "prompts_only", "prompt_test", "log_loss"), 0.0, "max"),
        (("candidate", "prompt_test", "balanced_accuracy"), ("comparators", "prompts_only", "prompt_test", "balanced_accuracy"), 0.01, "min"),
        (("candidate", "prompt_test", "auroc"), ("comparators", "prompts_only", "prompt_test", "auroc"), 0.01, "min"),
        (("candidate", "prompt_test", "human_fpr"), ("comparators", "prompts_only", "prompt_test", "human_fpr"), 0.0, "max"),
        (("candidate", "humanized_hard", "auroc"), ("comparators", "prompts_only", "humanized_hard", "auroc"), 0.02, "min"),
        (("candidate", "humanized_hard", "tpr_at_fpr_05"), ("comparators", "prompts_only", "humanized_hard", "tpr_at_fpr_05"), 0.02, "min"),
        (("candidate", "humanized_hard", "tpr_at_fpr_10"), ("comparators", "prompts_only", "humanized_hard", "tpr_at_fpr_10"), 0.02, "min"),
        (("candidate", "ood", "ood_m4_test", "auroc"), ("comparators", "prompts_only", "ood", "ood_m4_test", "auroc"), 0.02, "min"),
        (("candidate", "ood", "ood_raid_test", "auroc"), ("comparators", "prompts_only", "ood", "ood_raid_test", "auroc"), 0.02, "min"),
        (("candidate", "ood", "ood_mage_test", "auroc"), ("comparators", "prompts_only", "ood", "ood_mage_test", "auroc"), 0.02, "min"),
    ],
)
def test_comparator_gates_use_literal_nextafter_boundaries(
    candidate_path, baseline_path, allowance, direction
):
    baseline = _nested_get(_passing_report(), baseline_path)
    boundary = baseline - allowance if direction == "min" else baseline
    inside_direction = math.inf if direction == "min" else -math.inf
    outside_direction = -math.inf if direction == "min" else math.inf

    for passing_value in (boundary, math.nextafter(boundary, inside_direction)):
        report = _passing_report()
        _nested_set(report, candidate_path, passing_value)
        _assert_only_legacy_provenance_lock(report)

    report = _passing_report()
    _nested_set(report, candidate_path, math.nextafter(boundary, outside_direction))
    assert decide_promotion(report)["status"] == "DO_NOT_PROMOTE"


@pytest.mark.parametrize(
    "metric_path",
    [
        ("candidate", "prompt_test", "max_ebm_reconstruction_error"),
        (
            "comparators",
            "prompts_only",
            "prompt_test",
            "max_ebm_reconstruction_error",
        ),
    ],
)
def test_reconstruction_gate_uses_literal_nextafter_boundary(metric_path):
    for passing_value in (1e-9, math.nextafter(1e-9, -math.inf)):
        report = _passing_report()
        _nested_set(report, metric_path, passing_value)
        _assert_only_legacy_provenance_lock(report)

    report = _passing_report()
    _nested_set(report, metric_path, math.nextafter(1e-9, math.inf))
    assert decide_promotion(report)["status"] == "DO_NOT_PROMOTE"


@pytest.mark.parametrize("coverage_count", [90, 99])
def test_decide_promotion_accepts_both_conformal_interval_boundaries(
    coverage_count,
):
    report = _passing_report()
    report["candidate"]["prompt_test"][
        "conformal_truth_coverage_count"
    ] = coverage_count
    _assert_only_legacy_provenance_lock(report)


def test_decide_promotion_never_reads_adaptive_challenge_block():
    report = _passing_report()
    passing = decide_promotion(report)
    report["challenge"] = None
    assert decide_promotion(report) == passing


def test_decide_promotion_preserves_preflight_failures():
    report = _passing_report()
    report["preflight"]["failures"] = [
        "INVALID_COMPARATOR prompts_specdetect: missing ebm.skops"
    ]
    decision = decide_promotion(report)
    assert decision["status"] == "DO_NOT_PROMOTE"
    assert any(item.startswith("INVALID_COMPARATOR") for item in decision["failures"])


def _samples(prefix: str, label: int, count: int) -> list[EvalSample]:
    return [
        EvalSample(id=f"{prefix}:{i}", text=f"{prefix} unique text {i}", label=label)
        for i in range(count)
    ]


def test_build_humanized_hard_slice_filters_to_102_positive_and_296_human():
    humanized = _samples("positive", 1, 102) + _samples("control", 0, 400)
    hard_human = _samples("hard", 0, 296)

    combined = build_humanized_hard_slice(humanized, hard_human)

    assert len(combined) == 398
    assert sum(sample.label == 1 for sample in combined) == 102
    assert sum(sample.label == 0 for sample in combined) == 296


def test_build_humanized_hard_slice_rejects_cross_label_text_conflict():
    humanized = _samples("positive", 1, 102)
    hard_human = _samples("hard", 0, 296)
    hard_human[0] = EvalSample(
        id=hard_human[0].id,
        text=humanized[0].text,
        label=0,
    )

    with pytest.raises(ValueError, match="cross-label hashes"):
        build_humanized_hard_slice(humanized, hard_human)


def test_legacy_unsafe_transform_is_an_isolated_diagnostic_only():
    calibrator = DetectorCalibrator()
    calibrator.fit("token_cohesiveness", [0.0, 1.0], [0, 1])

    assert legacy_unsafe_transform_for_diagnostic(
        calibrator, "token_cohesiveness", 2.0
    ) == 1.0
    for production_callable in (
        CairnPipeline.analyze_detailed,
        CachedPipeline.analyze_detailed,
        EBMFusion.fuse,
        evaluate_profile,
    ):
        assert "legacy_unsafe_transform_for_diagnostic" not in inspect.getsource(
            production_callable
        )


def test_legacy_unsafe_helper_has_one_audited_production_call_site():
    source = inspect.getsource(tocsin_retrain_module)
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "legacy_unsafe_transform_for_diagnostic"
    ]

    assert len(calls) == 1
    parent = parents[calls[0]]
    while not isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parent = parents[parent]
    assert parent.name == "evaluate_detector_calibration"
    assert source.count("AUDITED_UNSAFE_STALE_DIAGNOSTIC") == 1


def test_locked_direct_transform_audit_requires_marker_on_call_line():
    repo_root = Path(tocsin_retrain_module.__file__).resolve().parents[2]
    direct_transform_tokens = (
        "calibrator.transform",
        "cal.transform",
        "_detector_calibrator.transform",
    )
    direct_transforms = [
        f"{path.relative_to(repo_root)}:{line_number}:{line}"
        for source_root in (repo_root / "cairn", repo_root / "scripts")
        for path in source_root.rglob("*.py")
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if any(token in line for token in direct_transform_tokens)
    ]
    unexpected_transforms = [
        entry
        for entry in direct_transforms
        if not (
            entry.startswith("cairn\\evaluation\\tocsin_retrain.py:")
            or entry.startswith("cairn/evaluation/tocsin_retrain.py:")
        )
        or "AUDITED_UNSAFE_STALE_DIAGNOSTIC" not in entry
    ]

    assert len(direct_transforms) == 1
    assert unexpected_transforms == []


def test_current_specdetect_comparator_hard_fails_for_exact_missing_artifacts():
    failures = profile_artifact_failures(
        Path("configs/.profile_prompts_specdetect.yaml")
    )
    assert failures == [
        "missing models/calib2000/profiles/prompts_specdetect/ebm.skops",
        "missing models/calib2000/profiles/prompts_specdetect/ebm.skops.hmac",
    ]


def test_profile_names_use_comparator_contract_names_without_config_prefix():
    assert profile_name(Path("configs/.profile_prompts_only.yaml")) == "prompts_only"
    assert (
        profile_name(Path("configs/.profile_prompts_specdetect.yaml"))
        == "prompts_specdetect"
    )


def test_benchmark_parser_appends_baselines_and_uses_two_path_dataset_pairs(tmp_path):
    args = build_parser().parse_args(
        [
            "--preflight-only",
            "--candidate-config",
            "candidate.yaml",
            "--baseline-config",
            "baseline-a.yaml",
            "--baseline-config",
            "baseline-b.yaml",
            "--prompt-test",
            "prompt.jsonl",
            "prompt.features.jsonl",
            "--candidate-integrity-manifest",
            "candidate-integrity.json",
            "--retrospective-provenance-manifest",
            "retrospective.json",
            "--cache-manifest",
            "prompt_test",
            "prompt.cache.json",
            "--out-dir",
            str(tmp_path),
        ]
    )

    assert args.baseline_config == [Path("baseline-a.yaml"), Path("baseline-b.yaml")]
    assert args.prompt_test == [Path("prompt.jsonl"), Path("prompt.features.jsonl")]
    assert args.candidate_integrity_manifest == Path("candidate-integrity.json")
    assert args.retrospective_provenance_manifest == Path("retrospective.json")
    assert args.cache_manifest == [["prompt_test", "prompt.cache.json"]]


def test_preflight_only_writes_decision_and_markdown_without_reading_dataset(tmp_path):
    config = tmp_path / "invalid.yaml"
    config.write_text(
        """
_profile: invalid
detectors: {}
cascade: {}
decision: {}
fusion:
  model_path: missing/ebm.skops
  conformal_path: missing/conformal.json
  detector_calibration_path: missing/detector_calibration.json
""".strip(),
        encoding="utf-8",
    )
    out_dir = tmp_path / "reports"

    exit_code = benchmark_main(
        [
            "--preflight-only",
            "--candidate-config",
            str(config),
            "--baseline-config",
            str(config),
            "--prompt-test",
            str(tmp_path / "does-not-exist.jsonl"),
            str(tmp_path / "does-not-exist.features.jsonl"),
            "--retrospective-provenance-manifest",
            str(tmp_path / "missing-retrospective.json"),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 1
    decision = json.loads((out_dir / "promotion_decision.json").read_text())
    assert decision["status"] == "DO_NOT_PROMOTE"
    assert (out_dir / "PROMOTION_REPORT.md").exists()


def test_preflight_rejects_an_existing_but_unloadable_detector_calibrator(
    tmp_path, monkeypatch
):
    model = tmp_path / "ebm.skops"
    hmac = tmp_path / "ebm.skops.hmac"
    conformal = tmp_path / "conformal.json"
    calibration = tmp_path / "detector_calibration.json"
    for artifact in (model, hmac, conformal, calibration):
        artifact.write_text("present", encoding="utf-8")
    config = tmp_path / "profile.yaml"
    config.write_text(
        json.dumps(
            {
                "_profile": "profile_fake",
                "detectors": {},
                "cascade": {},
                "decision": {},
                "fusion": {
                    "model_path": model.as_posix(),
                    "conformal_path": conformal.as_posix(),
                    "detector_calibration_path": calibration.as_posix(),
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = tmp_path / "smoke.jsonl"
    _write_jsonl(dataset, [{"text": "smoke text", "label": 0}])

    class FakeCachedPipeline:
        def __init__(self, cache_path, config_path):
            self._inner = SimpleNamespace(
                _calibration_status="calibrated",
                _detector_calibrator=None,
                _fusion=SimpleNamespace(
                    _model=SimpleNamespace(intercept_=[0.0])
                ),
            )

        def analyze(self, text):
            return SimpleNamespace(
                fusion_mode="ebm",
                p_llm=0.5,
                feature_contributions={"feature": 0.0},
            )

    monkeypatch.setattr(
        "cairn.evaluation.tocsin_retrain.CachedPipeline", FakeCachedPipeline
    )

    result = preflight_profiles(config, [config], dataset, tmp_path / "cache")

    assert result["candidate"]["valid"] is False
    assert result["candidate"]["failures"] == [
        "smoke detector calibrator did not load"
    ]


def test_preflight_rejects_legacy_tocsin_comparator_but_accepts_matching_v2(
    tmp_path,
    monkeypatch,
):
    def write_calibration(path: Path, schema: str) -> None:
        calibrator = DetectorCalibrator()
        calibrator.fit("preamble", [0.0, 1.0], [0, 1])
        if schema == "v1":
            calibrator.fit("token_cohesiveness", [-0.4, 0.1], [0, 1])
        elif schema == "v2":
            calibrator.fit(
                "token_cohesiveness",
                [-0.4, -0.2, 0.0, 0.1],
                [0, 0, 1, 1],
                metadata=CalibrationMetadata(
                    detector_signature=TOCSIN_COMBINED_SIGNATURE,
                    raw_channel="combined_w",
                    training_dataset_hash="sha256:test",
                    weighting_policy={"policy": "uniform"},
                ),
            )
        calibrator.save(path)
        if schema == "v1":
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["token_cohesiveness"].update(
                {
                    "detector_signature": TOCSIN_COMBINED_SIGNATURE,
                    "raw_channel": "combined_w",
                }
            )
            path.write_text(json.dumps(payload), encoding="utf-8")

    def write_profile(name: str, schema: str) -> Path:
        calibration = tmp_path / f"{name}.calibration.json"
        write_calibration(calibration, schema)
        config = tmp_path / f"{name}.yaml"
        config.write_text(
            json.dumps(
                {
                    "_profile": f"profile_{name}",
                    "detectors": {"token_cohesiveness": {"enabled": True}},
                    "cascade": {},
                    "decision": {},
                    "fusion": {
                        "detector_calibration_path": calibration.as_posix(),
                    },
                }
            ),
            encoding="utf-8",
        )
        return config

    candidate = write_profile("prompts_only_tocsin_v2", "v2")
    legacy = write_profile("prompts_only", "v1")
    spec = write_profile("prompts_specdetect", "absent")
    dataset = tmp_path / "smoke.jsonl"
    _write_jsonl(dataset, [{"text": "smoke text", "label": 0}])

    class LoadedConformal:
        calibrated = True

    class LoadedFusion:
        _model = SimpleNamespace(intercept_=[0.0])
        _conformal = LoadedConformal()

    class FaithfulCachedPipeline:
        def __init__(self, cache_path, config_path):
            self._inner = CairnPipeline.from_config_file(config_path)

        def analyze(self, text):
            return SimpleNamespace(
                fusion_mode="ebm",
                p_llm=0.5,
                feature_contributions={"feature": 0.0},
            )

    monkeypatch.setattr(
        CairnPipeline,
        "_build_fusion",
        staticmethod(lambda _config: LoadedFusion()),
    )
    monkeypatch.setattr(
        tocsin_retrain_module,
        "profile_artifact_failures",
        lambda _path: [],
    )
    monkeypatch.setattr(
        tocsin_retrain_module,
        "CachedPipeline",
        FaithfulCachedPipeline,
    )
    monkeypatch.setattr(
        tocsin_retrain_module,
        "_reconstruction_error",
        lambda _pipeline, _determination: 0.0,
    )

    result = preflight_profiles(
        candidate,
        [legacy, spec],
        dataset,
        tmp_path / "unused-cache.jsonl",
    )

    assert result["candidate"]["valid"] is True
    assert result["comparators"]["prompts_specdetect"]["valid"] is True
    assert result["comparators"]["prompts_only"]["valid"] is False
    assert result["comparators"]["prompts_only"]["failures"] == [
        "smoke calibration_status=degraded, expected calibrated*"
    ]
    assert any(
        failure.startswith("INVALID_COMPARATOR prompts_only:")
        for failure in result["failures"]
    )


def _write_missing_artifact_profile(tmp_path: Path, name: str, suffix: str) -> Path:
    path = tmp_path / f"{suffix}.yaml"
    path.write_text(
        json.dumps(
            {
                "_profile": f"profile_{name}",
                "detectors": {},
                "cascade": {},
                "decision": {},
                "fusion": {
                    "model_path": (tmp_path / suffix / "ebm.skops").as_posix(),
                    "conformal_path": (
                        tmp_path / suffix / "conformal.json"
                    ).as_posix(),
                    "detector_calibration_path": (
                        tmp_path / suffix / "detector_calibration.json"
                    ).as_posix(),
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("comparator_names", "identity_fragments"),
    [
        (["prompts_only"], ["missing prompts_specdetect"]),
        (
            ["prompts_only", "prompts_only"],
            ["duplicate prompts_only", "missing prompts_specdetect"],
        ),
        (
            ["prompts_only", "unexpected"],
            ["unexpected unexpected", "missing prompts_specdetect"],
        ),
    ],
)
def test_cli_preflight_locks_missing_duplicate_and_unexpected_comparator_identities(
    tmp_path, comparator_names, identity_fragments
):
    candidate = _write_missing_artifact_profile(tmp_path, "candidate", "candidate")
    baselines = [
        _write_missing_artifact_profile(tmp_path, name, f"baseline-{index}")
        for index, name in enumerate(comparator_names)
    ]
    out_dir = tmp_path / "reports"
    argv = [
        "--preflight-only",
        "--candidate-config",
        str(candidate),
        "--prompt-test",
        str(tmp_path / "absent.jsonl"),
        str(tmp_path / "absent.features.jsonl"),
        "--retrospective-provenance-manifest",
        str(tmp_path / "missing-retrospective.json"),
        "--out-dir",
        str(out_dir),
    ]
    for baseline in baselines:
        argv.extend(["--baseline-config", str(baseline)])

    preflight = preflight_profiles(
        candidate,
        baselines,
        tmp_path / "absent.jsonl",
        tmp_path / "absent.features.jsonl",
    )
    exit_code = benchmark_main(argv)
    decision = json.loads((out_dir / "promotion_decision.json").read_text())
    combined_failures = "\n".join(decision["failures"])

    assert exit_code == 1
    assert preflight["comparator_identity_valid"] is False
    assert len(preflight["comparators"]) == len(comparator_names)
    assert all(fragment in combined_failures for fragment in identity_fragments)


def _diagnostic_args(tmp_path: Path) -> list[str]:
    pair_flags = (
        "--prompt-test",
        "--humanized",
        "--hard-human",
        "--ood-m4",
        "--ood-raid",
        "--ood-mage",
    )
    argv = [
        "--diagnostic-on-invalid-comparator",
        "--candidate-config",
        "candidate.yaml",
        "--baseline-config",
        "prompts-only.yaml",
        "--baseline-config",
        "prompts-specdetect.yaml",
        "--stale-calibration",
        "stale.json",
        "--retrospective-provenance-manifest",
        "retrospective.json",
        "--out-dir",
        str(tmp_path / "reports"),
    ]
    for flag in pair_flags:
        argv.extend([flag, "dataset.jsonl", "cache.jsonl"])
    return argv


def _normal_full_args(tmp_path: Path) -> list[str]:
    argv = _diagnostic_args(tmp_path)
    argv.remove("--diagnostic-on-invalid-comparator")
    return argv


def _valid_preflight() -> dict:
    return {
        "candidate": {
            "name": "prompts_only_tocsin_v2",
            "valid": True,
            "failures": [],
        },
        "comparators": {
            "prompts_only": {
                "name": "prompts_only",
                "valid": True,
                "failures": [],
            },
            "prompts_specdetect": {
                "name": "prompts_specdetect",
                "valid": True,
                "failures": [],
            },
        },
        "comparator_identity_valid": True,
        "provenance": _legacy_provenance(),
        "failures": [],
    }


def test_preflight_reports_serialize_verified_provenance_hashes(tmp_path):
    provenance = {
        **_legacy_provenance(),
        "retrospective_manifest": {
            "path": "manifests/retrospective.json",
            "sha256": "f" * 64,
        },
        "candidate_integrity": {
            "manifest_path": "manifests/candidate-integrity.json",
            "manifest_sha256": "a" * 64,
            "artifacts": {
                "candidate.yaml": {"sha256": "b" * 64, "bytes": 123}
            },
        },
        "slices": {
            "prompt_test": {
                "manifest_path": "manifests/prompt.cache.json",
                "manifest_sha256": "c" * 64,
                "dataset_path": "data/prompt.jsonl",
                "dataset_sha256": "d" * 64,
                "cache_path": "caches/prompt.features.jsonl",
                "cache_sha256": "e" * 64,
                "unique_hashes": 2,
            }
        },
    }
    preflight = _valid_preflight()
    preflight["provenance"] = provenance
    report = {"preflight": preflight}
    decision = decide_promotion(report)

    benchmark._write_decision_reports(tmp_path, decision, report)

    stored = json.loads((tmp_path / "promotion_decision.json").read_text())
    markdown = (tmp_path / "PROMOTION_REPORT.md").read_text()
    assert stored["provenance"] == provenance
    assert "## Provenance status" in markdown
    assert "`actual_base_identity`" in markdown
    assert "`" + "f" * 64 + "`" in markdown
    assert "`" + "a" * 64 + "`" in markdown
    assert "`" + "d" * 64 + "`" in markdown
    assert "`" + "e" * 64 + "`" in markdown


def test_valid_preflight_cli_exits_zero_with_non_promotable_incomplete_decision(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        benchmark, "preflight_profiles", lambda *args, **kwargs: _valid_preflight()
    )

    exit_code = benchmark.main(
        [
            "--preflight-only",
            "--candidate-config",
            "candidate.yaml",
            "--baseline-config",
            "prompts-only.yaml",
            "--baseline-config",
            "prompts-specdetect.yaml",
            "--prompt-test",
            "prompt.jsonl",
            "cache.jsonl",
            "--retrospective-provenance-manifest",
            "retrospective.json",
            "--out-dir",
            str(tmp_path),
        ]
    )

    decision = json.loads((tmp_path / "promotion_decision.json").read_text())
    assert exit_code == 0
    assert decision["status"] == "DO_NOT_PROMOTE"
    assert len(decision["failures"]) == 1
    assert decision["failures"][0].startswith("UNPROMOTABLE_PROVENANCE:")
    assert decision["provenance"] == _valid_preflight()["provenance"]


@pytest.mark.parametrize(
    "preflight",
    [
        {
            "candidate": {
                "name": "prompts_only_tocsin_v2",
                "valid": False,
                "failures": ["missing candidate model"],
            },
            "comparators": {
                "prompts_only": {"name": "prompts_only", "valid": True, "failures": []},
                "prompts_specdetect": {
                    "name": "prompts_specdetect",
                    "valid": False,
                    "failures": ["missing comparator model"],
                },
            },
            "comparator_identity_valid": True,
            "failures": [
                "INVALID_CANDIDATE prompts_only_tocsin_v2: missing candidate model",
                "INVALID_COMPARATOR prompts_specdetect: missing comparator model",
            ],
        },
        {
            "candidate": {
                "name": "prompts_only_tocsin_v2",
                "valid": False,
                "failures": ["smoke cache error"],
            },
            "comparators": {
                "prompts_only": {"name": "prompts_only", "valid": True, "failures": []},
                "prompts_specdetect": {
                    "name": "prompts_specdetect",
                    "valid": False,
                    "failures": ["missing comparator model"],
                },
            },
            "comparator_identity_valid": True,
            "failures": [
                "INVALID_CANDIDATE prompts_only_tocsin_v2: smoke cache error",
                "INVALID_COMPARATOR prompts_specdetect: missing comparator model",
            ],
        },
        {
            "candidate": {
                "name": "prompts_only_tocsin_v2",
                "valid": True,
                "failures": [],
            },
            "comparators": {
                "prompts_only": {"name": "prompts_only", "valid": True, "failures": []},
                "prompts_specdetect": {
                    "name": "prompts_specdetect",
                    "valid": True,
                    "failures": [],
                },
            },
            "comparator_identity_valid": True,
            "failures": [],
        },
    ],
)
def test_diagnostic_lock_never_enters_slices_for_invalid_candidate_or_no_invalid_comparator(
    tmp_path, monkeypatch, preflight
):
    calls = {"run_full": 0, "slice": 0}

    monkeypatch.setattr(benchmark, "preflight_profiles", lambda *args: deepcopy(preflight))

    def forbidden_run_full(*args):
        calls["run_full"] += 1
        raise AssertionError("_run_full must not be entered")

    def forbidden_slice(*args):
        calls["slice"] += 1
        raise AssertionError("evaluation slices must not be read")

    monkeypatch.setattr(benchmark, "_run_full", forbidden_run_full)
    monkeypatch.setattr(benchmark, "load_evaluation_slice", forbidden_slice)

    exit_code = benchmark.main(_diagnostic_args(tmp_path))

    assert exit_code == 1
    assert calls == {"run_full": 0, "slice": 0}
    assert (tmp_path / "reports" / "promotion_decision.json").exists()
    assert (tmp_path / "reports" / "PROMOTION_REPORT.md").exists()


def _full_report_with_preflight() -> dict:
    report = _passing_report()
    report["preflight"] = _valid_preflight()
    report["candidate_name"] = "prompts_only_tocsin_v2"
    report["challenge"] = None
    return report


@pytest.mark.parametrize(
    ("mutate", "failure_fragment"),
    [
        (
            lambda report: report["candidate"]["prompt_test"].__setitem__(
                "max_ebm_reconstruction_error", math.inf
            ),
            "reconstruction",
        ),
        (
            lambda report: report["candidate"]["prompt_test"].__setitem__(
                "auroc", math.nan
            ),
            "auroc",
        ),
        (
            lambda report: report["candidate"]["prompt_test"].update(
                {
                    "finite": False,
                    "in_unit_interval": False,
                    "brier": None,
                    "log_loss": None,
                    "auroc": None,
                    "balanced_accuracy": None,
                    "human_fpr": None,
                }
            ),
            "non-finite p_llm",
        ),
    ],
)
def test_invalid_gate_metrics_still_publish_complete_json_safe_reports(
    tmp_path, monkeypatch, mutate, failure_fragment
):
    report = _full_report_with_preflight()
    mutate(report)
    monkeypatch.setattr(benchmark, "preflight_profiles", lambda *args: _valid_preflight())
    monkeypatch.setattr(
        benchmark,
        "_run_full",
        lambda *args: (report, {"challenge_rows": []}),
    )

    exit_code = benchmark.main(_normal_full_args(tmp_path))

    assert exit_code == 2
    required = {
        "detector_calibration.json",
        "benchmarks.json",
        "benchmarks_long.csv",
        "july10_per_task.csv",
        "promotion_decision.json",
        "PROMOTION_REPORT.md",
    }
    assert {path.name for path in (tmp_path / "reports").iterdir()} == required
    for name in (
        "detector_calibration.json",
        "benchmarks.json",
        "promotion_decision.json",
    ):
        raw = (tmp_path / "reports" / name).read_text(encoding="utf-8")
        assert "NaN" not in raw and "Infinity" not in raw
        json.loads(raw)
    decision = json.loads(
        (tmp_path / "reports" / "promotion_decision.json").read_text()
    )
    assert decision["status"] == "DO_NOT_PROMOTE"
    assert any(failure_fragment in failure for failure in decision["failures"])
    assert (
        tmp_path / "reports" / "benchmarks_long.csv"
    ).read_text(encoding="utf-8").splitlines() == [
        "profile,slice,id,label,p_llm,predicted_label,conformal_set,fusion_mode,reconstruction_error"
    ]


def test_full_benchmark_stores_complete_gate_audit_in_benchmarks_json(
    tmp_path, monkeypatch
):
    report = _full_report_with_preflight()
    monkeypatch.setattr(benchmark, "preflight_profiles", lambda *args: _valid_preflight())
    monkeypatch.setattr(
        benchmark,
        "_run_full",
        lambda *args: (report, {"challenge_rows": []}),
    )

    exit_code = benchmark.main(_normal_full_args(tmp_path))

    assert exit_code == 2
    stored = json.loads((tmp_path / "reports" / "benchmarks.json").read_text())
    assert [row["gate_id"] for row in stored["gate_audit"]] == (
        _expected_locked_gate_ids()
    )
    assert all(
        set(row)
        == {
            "gate_id",
            "label",
            "scope",
            "profile",
            "slice",
            "status",
            "observed",
            "operator",
            "reference",
            "failure_detail",
            "not_evaluated_reason",
        }
        for row in stored["gate_audit"]
    )


def test_report_render_failure_leaves_no_partial_report_set(tmp_path, monkeypatch):
    report = _full_report_with_preflight()
    monkeypatch.setattr(benchmark, "preflight_profiles", lambda *args: _valid_preflight())
    monkeypatch.setattr(
        benchmark,
        "_run_full",
        lambda *args: (report, {"challenge_rows": []}),
    )
    monkeypatch.setattr(
        benchmark,
        "_markdown_report",
        lambda *args: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    with pytest.raises(RuntimeError, match="render failed"):
        benchmark.main(_normal_full_args(tmp_path))

    assert list((tmp_path / "reports").iterdir()) == []


def _evaluated_row(
    digest_char: str,
    *,
    label: int,
    p_llm: float,
    conformal_set=("LLM",),
    fusion_mode="ebm",
    reconstruction_error=1e-12,
) -> EvaluatedRow:
    return EvaluatedRow(
        id=f"locked:{digest_char * 64}",
        source_id=None,
        label=label,
        p_llm=p_llm,
        conformal_set=conformal_set,
        fusion_mode=fusion_mode,
        reconstruction_error=reconstruction_error,
    )


def _diagnostic_preflight() -> dict:
    preflight = _valid_preflight()
    preflight["comparators"]["prompts_specdetect"] = {
        "name": "prompts_specdetect",
        "valid": False,
        "failures": ["missing comparator model"],
    }
    preflight["failures"] = [
        "INVALID_COMPARATOR prompts_specdetect: missing comparator model"
    ]
    return preflight


def test_diagnostic_benchmarks_long_is_exact_per_prediction_schema(tmp_path, monkeypatch):
    preflight = _diagnostic_preflight()
    report = _passing_report()
    report["preflight"] = preflight
    report["candidate_name"] = "prompts_only_tocsin_v2"
    report["comparators"].pop("prompts_specdetect")
    report["challenge"] = None
    candidate = ProfileEvaluation(
        name="prompts_only_tocsin_v2",
        metrics={},
        rows={
            "ood_m4_test": (
                _evaluated_row("b", label=0, p_llm=0.25, conformal_set=("human",)),
            ),
            "prompt_test": (
                _evaluated_row(
                    "a",
                    label=1,
                    p_llm=0.75,
                    conformal_set=("LLM", "human"),
                ),
            ),
        },
    )
    baseline = ProfileEvaluation(
        name="prompts_only",
        metrics={},
        rows={
            "ood_m4_test": (
                _evaluated_row("b", label=0, p_llm=0.2, conformal_set=("human",)),
            ),
            "prompt_test": (
                _evaluated_row("a", label=1, p_llm=0.6),
            ),
        },
    )
    outputs = {
        "candidate_eval": candidate,
        "comparator_evals": {"prompts_only": baseline},
        "challenge_rows": [],
    }
    monkeypatch.setattr(benchmark, "preflight_profiles", lambda *args: preflight)
    monkeypatch.setattr(benchmark, "_run_full", lambda *args: (report, outputs))

    exit_code = benchmark.main(_diagnostic_args(tmp_path))

    assert exit_code == 1
    csv_path = tmp_path / "reports" / "benchmarks_long.csv"
    raw = csv_path.read_text(encoding="utf-8")
    assert raw.splitlines()[0] == (
        "profile,slice,id,label,p_llm,predicted_label,conformal_set,"
        "fusion_mode,reconstruction_error"
    )
    rows = list(csv.DictReader(raw.splitlines()))
    assert [(row["profile"], row["slice"], row["id"]) for row in rows] == [
        ("prompts_only_tocsin_v2", "prompt_test", f"locked:{'a' * 64}"),
        ("prompts_only_tocsin_v2", "ood_m4_test", f"locked:{'b' * 64}"),
        ("prompts_only", "prompt_test", f"locked:{'a' * 64}"),
        ("prompts_only", "ood_m4_test", f"locked:{'b' * 64}"),
    ]
    assert rows[0] == {
        "profile": "prompts_only_tocsin_v2",
        "slice": "prompt_test",
        "id": f"locked:{'a' * 64}",
        "label": "1",
        "p_llm": "0.75",
        "predicted_label": "1",
        "conformal_set": '["human","LLM"]',
        "fusion_mode": "ebm",
        "reconstruction_error": "1e-12",
    }
    assert "role" not in rows[0]
    assert "text" not in raw and "source_rows" not in raw


def test_prediction_csv_blanks_invalid_row_values_and_gate_stays_failed(
    tmp_path, monkeypatch
):
    report = _full_report_with_preflight()
    report["candidate"]["prompt_test"].update(
        {
            "finite": False,
            "in_unit_interval": False,
            "brier": None,
            "log_loss": None,
            "auroc": None,
            "balanced_accuracy": None,
            "human_fpr": None,
            "max_ebm_reconstruction_error": math.inf,
        }
    )
    invalid_candidate = ProfileEvaluation(
        name="prompts_only_tocsin_v2",
        metrics={},
        rows={
            "prompt_test": (
                _evaluated_row(
                    "c",
                    label=1,
                    p_llm=math.nan,
                    conformal_set=None,
                    reconstruction_error=math.inf,
                ),
            )
        },
    )
    valid_comparators = {
        name: ProfileEvaluation(
            name=name,
            metrics={},
            rows={
                "prompt_test": (
                    _evaluated_row("c", label=1, p_llm=0.6),
                )
            },
        )
        for name in ("prompts_only", "prompts_specdetect")
    }
    outputs = {
        "candidate_eval": invalid_candidate,
        "comparator_evals": valid_comparators,
        "challenge_rows": [],
    }
    monkeypatch.setattr(benchmark, "preflight_profiles", lambda *args: _valid_preflight())
    monkeypatch.setattr(benchmark, "_run_full", lambda *args: (report, outputs))

    exit_code = benchmark.main(_normal_full_args(tmp_path))

    assert exit_code == 2
    rows = list(
        csv.DictReader(
            (tmp_path / "reports" / "benchmarks_long.csv")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    )
    assert len(rows) == 3
    assert rows[0]["profile"] == "prompts_only_tocsin_v2"
    assert rows[0]["p_llm"] == ""
    assert rows[0]["predicted_label"] == ""
    assert rows[0]["conformal_set"] == ""
    assert rows[0]["reconstruction_error"] == ""
    decision = json.loads(
        (tmp_path / "reports" / "promotion_decision.json").read_text()
    )
    assert decision["status"] == "DO_NOT_PROMOTE"
    assert any("non-finite p_llm" in failure for failure in decision["failures"])

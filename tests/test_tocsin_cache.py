from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cairn.detectors.token_cohesiveness import (
    TOCSIN_COHESIVENESS_SIGNATURE,
    TOCSIN_COMBINED_SIGNATURE,
)
from cairn.feature_cache import text_hash
from cairn.normalizer import normalize_text
from cairn.tocsin_cache import (
    TocsinCacheSummary,
    build_union_dataset,
    extract_cache_for_dataset,
    hash_artifacts,
    validate_tocsin_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TOCSIN_MIN_WORDS = 60


def long_text(token: str = "long") -> str:
    return " ".join([token] * 80)


def short_text(token: str = "short") -> str:
    return " ".join([token] * 12)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(path: Path, texts: list[str]) -> Path:
    rows = [
        {"id": f"row-{index}", "text": text, "label": index % 2}
        for index, text in enumerate(texts)
    ]
    return write_dataset_rows(path, rows)


def write_dataset_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def layer_result(
    layer_id: str,
    *,
    determination: str = "AMBER",
    raw_score: float = 0.2,
) -> dict[str, object]:
    return {
        "layer_id": layer_id,
        "domain": "prose",
        "raw_score": raw_score,
        "p_llm": 0.6,
        "confidence": 0.7,
        "signals": {"evidence": True},
        "determination": determination,
        "attacker_tiers": ["A0", "A1", "A2", "A3"],
        "compute_cost": "cheap",
        "min_text_length": 0,
        "spans": [],
        "calibrator_applied": False,
    }


def tocsin_result(
    signature: str = TOCSIN_COMBINED_SIGNATURE,
    *,
    raw_score: float = -0.25,
) -> dict[str, object]:
    channel = {
        TOCSIN_COMBINED_SIGNATURE: "combined_w",
        TOCSIN_COHESIVENESS_SIGNATURE: "cohesiveness_u",
    }.get(signature, "combined_w")
    signals: dict[str, object] = {
        "cohesiveness_u": -1.25,
        "base_v": (
            0.32 if signature == TOCSIN_COMBINED_SIGNATURE else None
        ),
        "combined_w": raw_score,
        "calibration_signature": signature,
        "calibration_raw_channel": channel,
        "deletion_ratio": 0.015,
        "n_copies": 10,
    }
    signals[channel] = raw_score
    result = layer_result("token_cohesiveness", raw_score=raw_score)
    result.update(
        {
            "compute_cost": "moderate",
            "min_text_length": TOCSIN_MIN_WORDS,
            "signals": signals,
        }
    )
    return result


def tocsin_skip(*, reason: str = "too_short") -> dict[str, object]:
    result = layer_result(
        "token_cohesiveness", determination="SKIP", raw_score=0.0
    )
    result.update(
        {
            "p_llm": 0.5,
            "confidence": 0.0,
            "compute_cost": "moderate",
            "min_text_length": TOCSIN_MIN_WORDS,
            "signals": {"skipped": reason, "n_words": 12},
        }
    )
    return result


def cache_record(
    text: str,
    signature: str = TOCSIN_COMBINED_SIGNATURE,
    *,
    extra_results: list[dict[str, object]] | None = None,
    skip_detectors: list[str] | None = None,
) -> dict[str, object]:
    tocsin = (
        tocsin_result(signature)
        if len(normalize_text(text).split()) >= TOCSIN_MIN_WORDS
        else tocsin_skip()
    )
    return {
        "hash": text_hash(normalize_text(text)),
        "results": [tocsin, *(extra_results or [])],
        "router": {
            "domain": "prose",
            "confidence": 0.95,
            "prompt_score": 0.05,
            "prose_score": 0.95,
            "word_count": len(normalize_text(text).split()),
            "recommended_detectors": [],
            "skip_detectors": list(skip_detectors or []),
        },
    }


def write_cache(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_valid_combined_signature_cache_passes(tmp_path: Path) -> None:
    texts = [long_text("active"), short_text("skip")]
    dataset = write_dataset(tmp_path / "data.jsonl", texts)
    cache = write_cache(
        tmp_path / "features.jsonl", [cache_record(text) for text in texts]
    )

    summary = validate_tocsin_cache(
        dataset,
        cache,
        expected_signature=TOCSIN_COMBINED_SIGNATURE,
        expected_layer_ids=("token_cohesiveness",),
        require_all_layers_seen=True,
    )

    assert summary == TocsinCacheSummary(
        dataset_sha256=sha256_file(dataset),
        cache_sha256=sha256_file(cache),
        expected_unique_hashes=2,
        cached_unique_hashes=2,
        active_rows=1,
        skip_rows=1,
        calibration_signature=TOCSIN_COMBINED_SIGNATURE,
        calibration_raw_channel="combined_w",
        raw_min=-0.25,
        raw_max=-0.25,
        identical_duplicate_hashes=0,
        api_layer_ids=(),
        layer_counts={
            "token_cohesiveness": {
                "present": 2,
                "active": 1,
                "skip": 1,
                "router_skipped": 0,
                "unexpected_absent": 0,
            }
        },
        unexpected_layer_ids=(),
    )
    assert summary.to_dict()["layer_counts"] == summary.layer_counts
    with pytest.raises(FrozenInstanceError):
        summary.active_rows = 99  # type: ignore[misc]


def test_rejects_normalized_text_hash_coverage_mismatch(tmp_path: Path) -> None:
    texts = ["  alpha   beta  " + " word" * 70, long_text("missing")]
    dataset = write_dataset(tmp_path / "data.jsonl", texts)
    cache = write_cache(tmp_path / "features.jsonl", [cache_record(texts[0])])
    missing_hash = text_hash(normalize_text(texts[1]))

    with pytest.raises(ValueError, match="hash coverage mismatch") as exc:
        validate_tocsin_cache(dataset, cache)

    assert str(cache) in str(exc.value)
    assert missing_hash in str(exc.value)
    assert "layer=<record>" in str(exc.value)


def test_rejects_non_full_cache_hash(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    record["hash"] = str(record["hash"])[:12]
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match="full lowercase SHA-256") as exc:
        validate_tocsin_cache(dataset, cache)

    assert "layer=<record>" in str(exc.value)


def test_rejects_more_than_one_tocsin_result_per_record(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text, extra_results=[tocsin_result()])
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match="more than one TOCSIN result"):
        validate_tocsin_cache(dataset, cache)


def test_rejects_eligible_row_without_active_tocsin(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    record["results"] = [tocsin_skip(reason="tier2_unavailable")]
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match="eligible row.*active TOCSIN"):
        validate_tocsin_cache(dataset, cache)


@pytest.mark.parametrize(
    "replacement",
    [tocsin_result(), tocsin_skip(reason="tier2_unavailable")],
    ids=["active", "wrong_skip_reason"],
)
def test_rejects_short_row_not_represented_as_too_short_skip(
    tmp_path: Path, replacement: dict[str, object]
) -> None:
    text = short_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    record["results"] = [replacement]
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match="short row.*SKIP.*too_short"):
        validate_tocsin_cache(dataset, cache)


def test_rejects_retired_mean_deletion_impact_evidence(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    record["results"][0]["signals"]["mean_deletion_impact"] = 0.1  # type: ignore[index]
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match="mean_deletion_impact"):
        validate_tocsin_cache(dataset, cache)


def test_rejects_missing_current_signature(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    del record["results"][0]["signals"]["calibration_signature"]  # type: ignore[index]
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match="missing TOCSIN calibration_signature"):
        validate_tocsin_cache(dataset, cache)


def test_rejects_mixed_current_signatures(tmp_path: Path) -> None:
    texts = [long_text("combined"), long_text("fallback")]
    dataset = write_dataset(tmp_path / "data.jsonl", texts)
    cache = write_cache(
        tmp_path / "features.jsonl",
        [
            cache_record(texts[0], TOCSIN_COMBINED_SIGNATURE),
            cache_record(texts[1], TOCSIN_COHESIVENESS_SIGNATURE),
        ],
    )

    with pytest.raises(ValueError, match="mixed TOCSIN signatures"):
        validate_tocsin_cache(dataset, cache)


def test_rejects_expected_signature_mismatch(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    cache = write_cache(tmp_path / "features.jsonl", [cache_record(text)])

    with pytest.raises(ValueError, match="expected TOCSIN signature"):
        validate_tocsin_cache(
            dataset,
            cache,
            expected_signature=TOCSIN_COHESIVENESS_SIGNATURE,
        )


def test_rejects_channel_signature_disagreement(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    record["results"][0]["signals"][  # type: ignore[index]
        "calibration_raw_channel"
    ] = "cohesiveness_u"
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match="raw channel.*signature"):
        validate_tocsin_cache(dataset, cache)


@pytest.mark.parametrize(
    ("signature", "base_v", "message"),
    [
        (TOCSIN_COMBINED_SIGNATURE, None, "combined.*base_v"),
        (TOCSIN_COHESIVENESS_SIGNATURE, 0.32, "cohesiveness-only.*base_v"),
    ],
)
def test_rejects_signature_base_channel_presence_disagreement(
    tmp_path: Path, signature: str, base_v: float | None, message: str
) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text, signature)
    record["results"][0]["signals"]["base_v"] = base_v  # type: ignore[index]
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match=message):
        validate_tocsin_cache(dataset, cache)


@pytest.mark.parametrize(
    "field", ["raw_score", "combined_w", "cohesiveness_u", "base_v"]
)
def test_rejects_nonfinite_tocsin_score(
    tmp_path: Path, field: str
) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    if field == "raw_score":
        record["results"][0][field] = float("nan")  # type: ignore[index]
    else:
        record["results"][0]["signals"][field] = float("inf")  # type: ignore[index]
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match="nonfinite TOCSIN"):
        validate_tocsin_cache(dataset, cache)


def test_rejects_raw_score_named_channel_disagreement(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    record["results"][0]["signals"]["combined_w"] = -0.24998  # type: ignore[index]
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match="raw_score.*combined_w"):
        validate_tocsin_cache(dataset, cache)


@pytest.mark.parametrize("field,value", [("min_text_length", 0)])
def test_rejects_wrong_tocsin_support_field(
    tmp_path: Path, field: str, value: object
) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    record["results"][0][field] = value  # type: ignore[index]
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match="TOCSIN min_text_length"):
        validate_tocsin_cache(dataset, cache)


@pytest.mark.parametrize(
    ("field", "value"),
    [("deletion_ratio", 0.02), ("n_copies", 9)],
)
def test_rejects_wrong_tocsin_signature_support_parameter(
    tmp_path: Path, field: str, value: object
) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    record["results"][0]["signals"][field] = value  # type: ignore[index]
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match=f"TOCSIN {field}"):
        validate_tocsin_cache(dataset, cache)


def test_rejects_short_skip_with_active_calibration_identity(tmp_path: Path) -> None:
    text = short_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    record["results"][0]["signals"].update(  # type: ignore[index]
        {
            "calibration_signature": TOCSIN_COMBINED_SIGNATURE,
            "calibration_raw_channel": "combined_w",
        }
    )
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match="SKIP.*calibration identity"):
        validate_tocsin_cache(dataset, cache)


def test_rejects_conflicting_duplicate_hashes_before_indexing(
    tmp_path: Path,
) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    first = cache_record(text)
    second = json.loads(json.dumps(first))
    second["results"][0]["raw_score"] = -0.1
    cache = write_cache(tmp_path / "features.jsonl", [first, second])

    with pytest.raises(ValueError, match="conflicting duplicate hash") as exc:
        validate_tocsin_cache(dataset, cache)

    assert text_hash(normalize_text(text)) in str(exc.value)


def test_counts_identical_duplicate_hash_once(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text)
    cache = write_cache(tmp_path / "features.jsonl", [record, record])

    summary = validate_tocsin_cache(dataset, cache)

    assert summary.cached_unique_hashes == 1
    assert summary.active_rows == 1
    assert summary.identical_duplicate_hashes == 1


@pytest.mark.parametrize("layer_id", ["contrastive_gen", "dna_gpt"])
def test_rejects_any_api_backed_layer_result(
    tmp_path: Path, layer_id: str
) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    record = cache_record(text, extra_results=[layer_result(layer_id)])
    cache = write_cache(tmp_path / "features.jsonl", [record])

    with pytest.raises(ValueError, match=f"API-backed layer.*{layer_id}"):
        validate_tocsin_cache(dataset, cache)


def test_rejects_router_eligible_record_missing_enabled_layer(
    tmp_path: Path,
) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    cache = write_cache(tmp_path / "features.jsonl", [cache_record(text)])

    with pytest.raises(ValueError, match="preamble.*unexpected_absent") as exc:
        validate_tocsin_cache(
            dataset,
            cache,
            expected_layer_ids=("preamble", "token_cohesiveness"),
        )

    assert str(cache) in str(exc.value)
    assert text_hash(normalize_text(text)) in str(exc.value)


def test_router_skipped_layer_may_be_never_present_in_slice(
    tmp_path: Path,
) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    cache = write_cache(
        tmp_path / "features.jsonl",
        [cache_record(text, skip_detectors=["preamble"])],
    )

    summary = validate_tocsin_cache(
        dataset,
        cache,
        expected_layer_ids=("preamble", "token_cohesiveness"),
        require_all_layers_seen=False,
    )

    assert summary.layer_counts["preamble"] == {
        "present": 0,
        "active": 0,
        "skip": 0,
        "router_skipped": 1,
        "unexpected_absent": 0,
    }


def test_strict_union_rejects_enabled_layer_never_seen(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    cache = write_cache(
        tmp_path / "features.jsonl",
        [cache_record(text, skip_detectors=["preamble"])],
    )

    with pytest.raises(ValueError, match="preamble.*never present"):
        validate_tocsin_cache(
            dataset,
            cache,
            expected_layer_ids=("preamble", "token_cohesiveness"),
            require_all_layers_seen=True,
        )


def test_reports_active_skip_router_and_unexpected_layers(tmp_path: Path) -> None:
    texts = [long_text("one"), short_text("two")]
    dataset = write_dataset(tmp_path / "data.jsonl", texts)
    cache = write_cache(
        tmp_path / "features.jsonl",
        [
            cache_record(
                texts[0],
                extra_results=[layer_result("preamble")],
                skip_detectors=["voice_spec"],
            ),
            cache_record(
                texts[1],
                extra_results=[
                    layer_result("preamble", determination="SKIP"),
                    layer_result("lapd"),
                ],
                skip_detectors=["voice_spec"],
            ),
        ],
    )

    summary = validate_tocsin_cache(
        dataset,
        cache,
        expected_layer_ids=(
            "preamble",
            "voice_spec",
            "token_cohesiveness",
        ),
    )

    assert summary.layer_counts["preamble"] == {
        "present": 2,
        "active": 1,
        "skip": 1,
        "router_skipped": 0,
        "unexpected_absent": 0,
    }
    assert summary.layer_counts["voice_spec"]["router_skipped"] == 2
    assert summary.layer_counts["lapd"]["present"] == 1
    assert summary.unexpected_layer_ids == ("lapd",)


def test_build_union_preserves_first_complete_object_and_manifest(
    tmp_path: Path,
) -> None:
    alpha = long_text("alpha")
    alpha_variant = "  " + alpha.replace(" ", "   ") + "  "
    beta = short_text("beta")
    gamma = long_text("gamma")
    first_rows = [
        {
            "id": "alpha-first",
            "text": alpha,
            "label": 1,
            "nested": {"source": "first", "values": [1, 2]},
        },
        {"id": "alpha-local-duplicate", "text": alpha_variant, "label": 0},
        {"id": "beta", "text": beta, "label": 0, "extra": "kept"},
    ]
    second_rows = [
        {"id": "alpha-later", "text": alpha, "label": 0, "replace": False},
        {"id": "gamma", "text": gamma, "label": 1, "extra": ["all", "kept"]},
    ]
    first = write_dataset_rows(tmp_path / "first.jsonl", first_rows)
    second = write_dataset_rows(tmp_path / "second.jsonl", second_rows)
    output = tmp_path / "union.jsonl"

    manifest = build_union_dataset([first, second], output)

    union_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert union_rows == [first_rows[0], first_rows[2], second_rows[1]]
    assert manifest["datasets"] == [
        {
            "path": first.as_posix(),
            "sha256": sha256_file(first),
            "total_rows": 3,
            "unique_rows": 2,
            "duplicate_rows": 1,
            "eligible_rows": 1,
        },
        {
            "path": second.as_posix(),
            "sha256": sha256_file(second),
            "total_rows": 2,
            "unique_rows": 2,
            "duplicate_rows": 0,
            "eligible_rows": 2,
        },
    ]
    assert manifest["union"] == {
        "path": output.as_posix(),
        "sha256": sha256_file(output),
        "total_rows": 5,
        "unique_rows": 3,
        "duplicate_rows": 2,
        "eligible_rows": 2,
    }


def test_extract_cache_writes_once_in_dataset_hash_order(tmp_path: Path) -> None:
    texts = [long_text("one"), long_text("two"), long_text("one")]
    dataset = write_dataset(tmp_path / "data.jsonl", texts)
    records = [cache_record(texts[1]), cache_record(texts[0])]
    source = write_cache(tmp_path / "source.jsonl", records)
    output = tmp_path / "slice.jsonl"

    count = extract_cache_for_dataset(source, dataset, output)

    sliced = [json.loads(line) for line in output.read_text().splitlines()]
    assert count == 2
    assert [row["hash"] for row in sliced] == [
        text_hash(normalize_text(texts[0])),
        text_hash(normalize_text(texts[1])),
    ]
    assert sliced == [records[1], records[0]]


def test_extract_cache_fails_before_partial_output_on_first_absent_hash(
    tmp_path: Path,
) -> None:
    texts = [long_text("present"), long_text("first-missing"), long_text("later")]
    dataset = write_dataset(tmp_path / "data.jsonl", texts)
    source = write_cache(tmp_path / "source.jsonl", [cache_record(texts[0])])
    output = tmp_path / "slice.jsonl"
    output.write_text("sentinel\n", encoding="utf-8")
    first_absent = text_hash(normalize_text(texts[1]))

    with pytest.raises(ValueError, match=first_absent):
        extract_cache_for_dataset(source, dataset, output)

    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_hash_artifacts_is_sorted_and_changes_when_one_byte_changes(
    tmp_path: Path,
) -> None:
    z_path = tmp_path / "z.bin"
    a_path = tmp_path / "a.bin"
    z_path.write_bytes(b"abc")
    a_path.write_bytes(b"xyz")

    before = hash_artifacts([z_path, a_path])
    z_path.write_bytes(b"abd")
    after = hash_artifacts([z_path, a_path])

    assert list(before) == sorted([z_path.as_posix(), a_path.as_posix()])
    assert before[a_path.as_posix()] == after[a_path.as_posix()]
    assert before[z_path.as_posix()]["bytes"] == 3
    assert before[z_path.as_posix()]["sha256"] != after[z_path.as_posix()]["sha256"]
    assert len(str(after[z_path.as_posix()]["sha256"])) == 64


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_hash_artifacts_rejects_non_file_paths(tmp_path: Path, kind: str) -> None:
    path = tmp_path / kind
    if kind == "directory":
        path.mkdir()

    with pytest.raises(ValueError, match=str(path).replace("\\", r"\\")):
        hash_artifacts([path])


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(REPO_ROOT), environment.get("PYTHONPATH", ""))
        if part
    )
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_build_union_cli_writes_union_and_manifest(tmp_path: Path) -> None:
    first = write_dataset(tmp_path / "first.jsonl", [long_text("first")])
    second = write_dataset(tmp_path / "second.jsonl", [long_text("second")])
    union = tmp_path / "union.jsonl"
    manifest_path = tmp_path / "manifest.json"

    completed = run_script(
        "scripts/build_tocsin_cache_inputs.py",
        "--union-out",
        str(union),
        "--manifest-out",
        str(manifest_path),
        "--datasets",
        str(first),
        str(second),
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(manifest_path.read_text())["union"]["unique_rows"] == 2


def test_slice_cache_cli_uses_documented_flags(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    source = write_cache(tmp_path / "source.jsonl", [cache_record(text)])
    output = tmp_path / "slice.jsonl"

    completed = run_script(
        "scripts/slice_tocsin_feature_cache.py",
        "--source-cache",
        str(source),
        "--dataset",
        str(dataset),
        "--out",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    assert len(output.read_text().splitlines()) == 1


def test_hash_artifacts_cli_writes_indented_json(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"payload")
    output = tmp_path / "hashes.json"

    completed = run_script(
        "scripts/hash_artifacts.py", "--out", str(output), str(artifact)
    )

    assert completed.returncode == 0, completed.stderr
    raw = output.read_text(encoding="utf-8")
    assert "\n  \"" in raw
    assert json.loads(raw)[artifact.as_posix()]["sha256"] == sha256_file(artifact)


def test_validate_cli_resolves_only_enabled_phase_1_to_3_layers(
    tmp_path: Path,
) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    cache = write_cache(tmp_path / "features.jsonl", [cache_record(text)])
    config = tmp_path / "profile.yaml"
    config.write_text(
        """
detectors:
  token_cohesiveness:
    enabled: true
  preamble:
    enabled: false
  cross_similarity:
    enabled: true
  contributor_graph:
    enabled: true
cascade: {}
decision: {}
""".lstrip(),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"

    completed = run_script(
        "scripts/validate_tocsin_cache.py",
        "--dataset",
        str(dataset),
        "--cache",
        str(cache),
        "--config",
        str(config),
        "--expected-signature",
        TOCSIN_COMBINED_SIGNATURE,
        "--require-all-layers-seen",
        "--out-manifest",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(output.read_text())
    assert set(manifest["layer_counts"]) == {"token_cohesiveness"}


def test_validate_cli_leaves_manifest_unchanged_on_failure(tmp_path: Path) -> None:
    text = long_text()
    dataset = write_dataset(tmp_path / "data.jsonl", [text])
    cache = write_cache(tmp_path / "features.jsonl", [])
    config = tmp_path / "profile.yaml"
    config.write_text(
        "detectors: {}\ncascade: {}\ndecision: {}\n", encoding="utf-8"
    )
    output = tmp_path / "summary.json"
    output.write_text("sentinel\n", encoding="utf-8")

    completed = run_script(
        "scripts/validate_tocsin_cache.py",
        "--dataset",
        str(dataset),
        "--cache",
        str(cache),
        "--config",
        str(config),
        "--out-manifest",
        str(output),
    )

    assert completed.returncode != 0
    assert output.read_text(encoding="utf-8") == "sentinel\n"

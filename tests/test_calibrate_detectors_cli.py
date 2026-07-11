"""CLI and merge behavior for isolated detector calibration training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from cairn.contracts import LayerResult, RouterDecision
from cairn.evaluation.dataset import EvalSample, save_dataset
from cairn.feature_cache import text_hash, write_record
from cairn.pipeline import normalize_text
from scripts import calibrate_detectors as calibrate_cli


SIGNATURE = "tocsin-bart-fastdetectgpt-combined-w-v1"
RAW_CHANNEL = "combined_w"
TARGET = "token_cohesiveness"
def _text(word_count: int, token: str) -> str:
    return " ".join([token] * word_count)


def _sample(
    sample_id: str,
    *,
    word_count: int,
    label: int,
    source: str = "source",
) -> EvalSample:
    return EvalSample(
        id=sample_id,
        text=_text(word_count, sample_id),
        label=label,
        source=source,
    )


def _layer(
    raw_score: float | None,
    *,
    layer_id: str = TARGET,
    signature: str | None = SIGNATURE,
    raw_channel: str | None = RAW_CHANNEL,
    determination: str = "AMBER",
) -> LayerResult:
    signals: dict[str, object] = {"native": "untouched"}
    if signature is not None:
        signals["calibration_signature"] = signature
    if raw_channel is not None:
        signals["calibration_raw_channel"] = raw_channel
    return LayerResult(
        layer_id=layer_id,
        domain="universal",
        raw_score=raw_score,
        p_llm=0.31,
        confidence=0.8,
        signals=signals,
        determination=determination,
        attacker_tiers=["A0"],
        compute_cost="moderate",
        min_text_length=60,
    )


def _router(word_count: int) -> RouterDecision:
    return RouterDecision(
        domain="prose",
        confidence=0.9,
        prompt_score=0.2,
        prose_score=0.8,
        word_count=word_count,
        recommended_detectors=[TARGET],
        skip_detectors=[],
    )


def _merge_payload(*, include_target: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "legacy_other": {
            "schema": "isotonic-thresholds-v1",
            "X": [0.0, 1.0],
            "y": [0.0, 1.0],
            "custom": {"preserve": [3, 2, 1]},
        },
        "versioned_other": {
            "schema": "isotonic-thresholds-v2",
            "X": [-1.0, 1.0],
            "y": [0.0, 1.0],
            "detector_signature": "other-v1",
            "raw_channel": "native",
            "training_dataset_hash": "a" * 64,
            "weighting_policy": {"policy": "other", "extra": True},
            "support_min": -1.0,
            "support_max": 1.0,
            "training_rows": 2,
            "class_counts": {"0": 1, "1": 1},
            "unrecognized_but_safe": {"keep": "exactly"},
        },
    }
    if include_target:
        payload[TARGET] = {
            "schema": "isotonic-thresholds-v1",
            "X": [-9.0, 9.0],
            "y": [0.0, 1.0],
            "old": "replace-me",
        }
    return payload


def _valid_observations() -> list[tuple[LayerResult, EvalSample]]:
    observations: list[tuple[LayerResult, EvalSample]] = []
    raw = -0.8
    for bin_index, word_count in enumerate((60, 100, 150, 250)):
        for label in (0, 1):
            for source_index in range(2):
                sample = _sample(
                    f"valid-{bin_index}-{label}-{source_index}",
                    word_count=word_count,
                    label=label,
                    source=f"source-{source_index}",
                )
                observations.append((_layer(raw), sample))
                raw += 0.1
    return observations


def _isolated_args(
    *,
    dataset: Path,
    config: Path,
    out: Path,
    merge_from: Path,
    manifest: Path,
    cache: Path | None = None,
) -> list[str]:
    args = [
        "--dataset",
        str(dataset),
        "--config",
        str(config),
        "--out",
        str(out),
        "--only-detector",
        TARGET,
        "--merge-from",
        str(merge_from),
        "--detector-signature",
        SIGNATURE,
        "--raw-channel",
        RAW_CHANNEL,
        "--source-length-weighting",
        "--manifest-out",
        str(manifest),
    ]
    if cache is not None:
        args.extend(["--features-cache", str(cache)])
    return args


def _write_cached_fixture(
    dataset_path: Path,
    cache_path: Path,
    samples: list[EvalSample],
    *,
    router_word_count_delta: int = 0,
    two_legacy_detectors: bool = False,
) -> None:
    save_dataset(samples, dataset_path)
    with cache_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, sample in enumerate(samples):
            word_count = len(normalize_text(sample.text).split())
            if two_legacy_detectors:
                results = [
                    _layer(
                        float(sample.label),
                        layer_id="legacy_a",
                        signature=None,
                        raw_channel=None,
                    ),
                    _layer(
                        0.2 + 0.6 * sample.label,
                        layer_id="legacy_b",
                        signature=None,
                        raw_channel=None,
                    ),
                ]
            else:
                results = [_layer(float(sample.label) + index / 10_000.0)]
            write_record(
                handle,
                text_hash(normalize_text(sample.text)),
                results,
                _router(word_count + router_word_count_delta),
            )


@pytest.mark.parametrize("include_target", [False, True])
def test_isolated_fit_adds_or_replaces_target_and_preserves_every_other_entry(
    tmp_path,
    include_target,
):
    merge_from = tmp_path / "base.json"
    base_payload = _merge_payload(include_target=include_target)
    merge_from.write_text(json.dumps(base_payload, indent=4) + "\n", encoding="utf-8")
    source_bytes = merge_from.read_bytes()
    observations = _valid_observations()
    rejected = [
        (_layer(0.1, layer_id="wrong"), _sample("wrong-det", word_count=60, label=0)),
        (_layer(0.1, determination="SKIP"), _sample("skip", word_count=60, label=0)),
        (_layer(0.1, signature="wrong"), _sample("signature", word_count=60, label=0)),
        (_layer(0.1, raw_channel="wrong"), _sample("channel", word_count=60, label=0)),
        (_layer(None), _sample("missing", word_count=60, label=0)),
        (_layer(float("nan")), _sample("nonfinite", word_count=60, label=0)),
    ]

    calibrator, manifest = calibrate_cli.fit_isolated_detector_calibration(
        observations + rejected,
        detector_id=TARGET,
        merge_from=merge_from,
        detector_signature=SIGNATURE,
        raw_channel=RAW_CHANNEL,
        training_dataset_hash="d" * 64,
        min_class_rows=2,
    )

    assert merge_from.read_bytes() == source_bytes
    assert manifest["observed_rows"] == 22
    assert manifest["accepted_rows"] == 16
    assert manifest["rejected_rows"] == 6
    assert manifest["accepted_reasons"] == {"matched": 16}
    assert manifest["rejected_reasons"] == {
        "detector_id_mismatch": 1,
        "inactive": 1,
        "missing_raw_score": 1,
        "nonfinite_raw_score": 1,
        "raw_channel_mismatch": 1,
        "signature_mismatch": 1,
    }
    assert manifest["class_counts"] == {"0": 8, "1": 8}
    accepted_scores = [float(result.raw_score) for result, _ in observations]
    assert manifest["support"] == {
        "min": pytest.approx(min(accepted_scores)),
        "max": pytest.approx(max(accepted_scores)),
    }
    assert manifest["weighting_policy"]["policy"] == "source-length-balanced-v1"
    assert len(manifest["weighting_policy"]["strata"]) == 8

    out = tmp_path / "merged.json"
    calibrator.save(out)
    output_payload = json.loads(out.read_text(encoding="utf-8"))
    assert {
        key: value for key, value in output_payload.items() if key != TARGET
    } == {
        key: value for key, value in base_payload.items() if key != TARGET
    }
    assert output_payload[TARGET]["schema"] == "isotonic-thresholds-v2"
    assert output_payload[TARGET]["detector_signature"] == SIGNATURE
    assert output_payload[TARGET]["raw_channel"] == RAW_CHANNEL
    assert output_payload[TARGET]["training_dataset_hash"] == "d" * 64
    assert output_payload[TARGET]["weighting_policy"] == manifest["weighting_policy"]


def test_isolated_fit_enforces_500_rows_per_accepted_class_by_default(tmp_path):
    merge_from = tmp_path / "base.json"
    merge_from.write_text(json.dumps(_merge_payload(include_target=False)), encoding="utf-8")

    with pytest.raises(ValueError, match=r"accepted class counts .* below minimum 500"):
        calibrate_cli.fit_isolated_detector_calibration(
            _valid_observations(),
            detector_id=TARGET,
            merge_from=merge_from,
            detector_signature=SIGNATURE,
            raw_channel=RAW_CHANNEL,
            training_dataset_hash="d" * 64,
        )


def test_isolated_fit_rejects_non_object_merge_json(tmp_path):
    merge_from = tmp_path / "base.json"
    merge_from.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="merge source must be a JSON object"):
        calibrate_cli.fit_isolated_detector_calibration(
            _valid_observations(),
            detector_id=TARGET,
            merge_from=merge_from,
            detector_signature=SIGNATURE,
            raw_channel=RAW_CHANNEL,
            training_dataset_hash="d" * 64,
            min_class_rows=2,
        )


def test_live_isolated_collection_disables_configured_calibration_before_analysis(
    tmp_path,
    monkeypatch,
):
    sample = _sample("native-live", word_count=60, label=0)

    class PoisonCalibrator:
        def apply(self, result):
            result.determination = "SKIP"

    class LivePipeline:
        def __init__(self):
            self._detector_calibrator = PoisonCalibrator()

        def analyze_detailed(self, text):
            result = _layer(0.42)
            if self._detector_calibrator is not None:
                self._detector_calibrator.apply(result)
            return object(), [result], _router(60)

    pipeline = LivePipeline()
    monkeypatch.setattr(
        calibrate_cli.CairnPipeline,
        "from_config_file",
        lambda _path: pipeline,
    )

    observations = calibrate_cli._collect_isolated_observations(
        [sample],
        detector_id=TARGET,
        config_path=tmp_path / "config.yaml",
        features_cache=None,
    )

    assert pipeline._detector_calibrator is None
    assert len(observations) == 1
    result, observed_sample = observations[0]
    assert observed_sample is sample
    assert result.determination == "AMBER"
    assert result.raw_score == pytest.approx(0.42)
    assert result.p_llm == pytest.approx(0.31)
    assert result.signals["native"] == "untouched"


@pytest.mark.parametrize(
    ("cached_state", "error_fragment"),
    [
        ("applied", "calibrator_applied=True"),
        ("guard_rejected", "calibrator_signature_mismatch"),
    ],
)
def test_cached_isolated_collection_rejects_calibrator_mutated_target_rows(
    tmp_path,
    cached_state,
    error_fragment,
):
    sample = _sample(f"cached-{cached_state}", word_count=60, label=0)
    result = _layer(0.42)
    if cached_state == "applied":
        result.calibrator_applied = True
        result.p_llm = 0.99
    else:
        result.determination = "SKIP"
        result.signals["calibrator_signature_mismatch"] = {
            "result_signature": SIGNATURE,
            "artifact_signature": "stale-signature",
            "result_raw_channel": RAW_CHANNEL,
            "artifact_raw_channel": RAW_CHANNEL,
        }
    cache = tmp_path / "cache.jsonl"
    with cache.open("w", encoding="utf-8", newline="\n") as handle:
        write_record(
            handle,
            text_hash(normalize_text(sample.text)),
            [result],
            _router(60),
        )

    with pytest.raises(ValueError) as exc:
        calibrate_cli._collect_isolated_observations(
            [sample],
            detector_id=TARGET,
            config_path=tmp_path / "unused-config.yaml",
            features_cache=cache,
        )

    message = str(exc.value)
    assert f"sample {sample.id!r}" in message
    assert f"detector {TARGET!r}" in message
    assert error_fragment in message


def test_cached_isolated_collection_accepts_native_target_identity_fields(tmp_path):
    sample = _sample("cached-native", word_count=60, label=0)
    native = _layer(0.42)
    cache = tmp_path / "cache.jsonl"
    with cache.open("w", encoding="utf-8", newline="\n") as handle:
        write_record(
            handle,
            text_hash(normalize_text(sample.text)),
            [native],
            _router(60),
        )

    observations = calibrate_cli._collect_isolated_observations(
        [sample],
        detector_id=TARGET,
        config_path=tmp_path / "unused-config.yaml",
        features_cache=cache,
    )

    assert observations == [(native, sample)]
    assert native.calibrator_applied is False
    assert native.signals["calibration_signature"] == SIGNATURE
    assert native.signals["calibration_raw_channel"] == RAW_CHANNEL


def test_isolated_mode_rejects_cached_router_word_count_mismatch_before_fit(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    cache = tmp_path / "cache.jsonl"
    sample = _sample("mismatch", word_count=60, label=0)
    _write_cached_fixture(
        dataset,
        cache,
        [sample],
        router_word_count_delta=1,
    )
    config = tmp_path / "config.yaml"
    config.write_text("_extends: default\n", encoding="utf-8")
    merge_from = tmp_path / "base.json"
    merge_from.write_text(json.dumps(_merge_payload(include_target=False)), encoding="utf-8")
    manifest = tmp_path / "manifest.json"

    with pytest.raises(
        ValueError,
        match=r"cached router word count 61 does not match normalized text count 60",
    ):
        calibrate_cli.main(
            _isolated_args(
                dataset=dataset,
                config=config,
                out=tmp_path / "out.json",
                merge_from=merge_from,
                manifest=manifest,
                cache=cache,
            )
        )

    assert not manifest.exists()


def test_isolated_cached_cli_uses_native_rows_and_writes_complete_hashed_manifest(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    cache = tmp_path / "cache.jsonl"
    samples: list[EvalSample] = []
    for word_count in (60, 100, 150, 250):
        for label in (0, 1):
            for row in range(125):
                samples.append(
                    _sample(
                        f"row-{word_count}-{label}-{row}",
                        word_count=word_count,
                        label=label,
                        source=f"source-{label}-{row % 5}",
                    )
                )
    _write_cached_fixture(dataset, cache, samples)

    poison = tmp_path / "poison-calibration.json"
    poison.write_text(
        json.dumps(
            {
                TARGET: {
                    "schema": "isotonic-thresholds-v1",
                    "X": [0.0, 1.0],
                    "y": [0.0, 1.0],
                }
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "fusion:\n  detector_calibration_path: " + json.dumps(str(poison)) + "\n",
        encoding="utf-8",
    )
    merge_from = tmp_path / "base.json"
    base_payload = _merge_payload(include_target=True)
    merge_from.write_text(json.dumps(base_payload, indent=2) + "\n", encoding="utf-8")
    source_bytes = merge_from.read_bytes()
    out = tmp_path / "out.json"
    manifest_path = tmp_path / "manifest.json"

    calibrate_cli.main(
        _isolated_args(
            dataset=dataset,
            config=config,
            out=out,
            merge_from=merge_from,
            manifest=manifest_path,
            cache=cache,
        )
    )

    assert merge_from.read_bytes() == source_bytes
    output = json.loads(out.read_text(encoding="utf-8"))
    assert {key: value for key, value in output.items() if key != TARGET} == {
        key: value for key, value in base_payload.items() if key != TARGET
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    cache_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    assert len(dataset_hash) == len(cache_hash) == 64
    assert manifest["dataset_path"] == str(dataset)
    assert manifest["dataset_sha256"] == dataset_hash
    assert manifest["features_cache_path"] == str(cache)
    assert manifest["features_cache_sha256"] == cache_hash
    assert manifest["detector_id"] == TARGET
    assert manifest["detector_signature"] == SIGNATURE
    assert manifest["raw_channel"] == RAW_CHANNEL
    assert manifest["observed_rows"] == 1_000
    assert manifest["accepted_rows"] == 1_000
    assert manifest["rejected_rows"] == 0
    assert manifest["accepted_reasons"] == {"matched": 1_000}
    assert manifest["rejected_reasons"] == {}
    assert manifest["class_counts"] == {"0": 500, "1": 500}
    assert manifest["support"] == {
        "min": pytest.approx(0.0),
        "max": pytest.approx(1.0999),
    }
    assert manifest["weighting_policy"]["policy"] == "source-length-balanced-v1"
    assert len(manifest["weighting_policy"]["strata"]) == 8
    assert output[TARGET]["training_dataset_hash"] == dataset_hash
    assert output[TARGET]["weighting_policy"] == manifest["weighting_policy"]


def test_manifest_is_not_written_until_non_target_preservation_is_proven(tmp_path):
    merge_from = tmp_path / "base.json"
    base_payload = _merge_payload(include_target=False)
    merge_from.write_text(json.dumps(base_payload, indent=2), encoding="utf-8")
    source_bytes = merge_from.read_bytes()
    out = tmp_path / "corrupt.json"
    manifest_path = tmp_path / "manifest.json"

    class CorruptingCalibrator:
        def save(self, path):
            path.write_text(
                json.dumps(
                    {
                        "legacy_other": {"changed": True},
                        TARGET: {
                            "schema": "isotonic-thresholds-v2",
                            "X": [0.0, 1.0],
                            "y": [0.0, 1.0],
                        },
                    }
                ),
                encoding="utf-8",
            )

    with pytest.raises(RuntimeError, match="changed non-target entries"):
        calibrate_cli._save_isolated_outputs(
            CorruptingCalibrator(),
            merge_from=merge_from,
            out=out,
            detector_id=TARGET,
            manifest_out=manifest_path,
            manifest={"must": "not be written"},
        )

    assert merge_from.read_bytes() == source_bytes
    assert not manifest_path.exists()


@pytest.mark.parametrize("alias_kind", ["out", "manifest"])
def test_save_preflight_rejects_hardlink_aliases_to_merge_source(
    tmp_path,
    alias_kind,
):
    merge_from = tmp_path / "base.json"
    base_payload = _merge_payload(include_target=False)
    merge_from.write_text(json.dumps(base_payload, indent=2), encoding="utf-8")
    source_bytes = merge_from.read_bytes()
    out = tmp_path / "out.json"
    manifest_path = tmp_path / "manifest.json"
    alias = out if alias_kind == "out" else manifest_path
    try:
        os.link(merge_from, alias)
    except OSError as exc:  # pragma: no cover - platform capability guard
        pytest.skip(f"hard links unavailable: {exc}")

    class PreservingCalibrator:
        save_called = False

        def save(self, path):
            self.save_called = True
            path.write_text(
                json.dumps(
                    {
                        **base_payload,
                        TARGET: {
                            "schema": "isotonic-thresholds-v2",
                            "X": [0.0, 1.0],
                            "y": [0.0, 1.0],
                        },
                    }
                ),
                encoding="utf-8",
            )

    calibrator = PreservingCalibrator()
    with pytest.raises(ValueError, match="aliases merge source"):
        calibrate_cli._save_isolated_outputs(
            calibrator,
            merge_from=merge_from,
            out=out,
            detector_id=TARGET,
            manifest_out=manifest_path,
            manifest={"safe": True},
        )

    assert calibrator.save_called is False
    assert merge_from.read_bytes() == source_bytes


def test_legacy_no_flag_mode_still_fits_all_detector_ids_unweighted(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    cache = tmp_path / "cache.jsonl"
    samples = [
        _sample("human", word_count=60, label=0),
        _sample("positive", word_count=60, label=1),
    ]
    _write_cached_fixture(
        dataset,
        cache,
        samples,
        two_legacy_detectors=True,
    )
    config = tmp_path / "config.yaml"
    config.write_text("_extends: default\n", encoding="utf-8")
    out = tmp_path / "legacy.json"

    calibrate_cli.main(
        [
            "--dataset",
            str(dataset),
            "--config",
            str(config),
            "--features-cache",
            str(cache),
            "--out",
            str(out),
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) == {"legacy_a", "legacy_b"}
    assert payload["legacy_a"]["schema"] == "isotonic-thresholds-v1"
    assert payload["legacy_b"]["schema"] == "isotonic-thresholds-v1"
    assert "weighting_policy" not in payload["legacy_a"]
    assert "weighting_policy" not in payload["legacy_b"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--merge-from", "base.json"],
        ["--detector-signature", SIGNATURE],
        ["--raw-channel", RAW_CHANNEL],
        ["--source-length-weighting"],
        ["--manifest-out", "manifest.json"],
    ],
)
def test_legacy_mode_rejects_every_isolated_only_flag(extra, capsys):
    with pytest.raises(SystemExit) as exc:
        calibrate_cli.main(
            [
                "--dataset",
                "dataset.jsonl",
                "--config",
                "config.yaml",
                "--out",
                "out.json",
                *extra,
            ]
        )

    assert exc.value.code == 2
    assert "require --only-detector" in capsys.readouterr().err


@pytest.mark.parametrize(
    "missing_flag",
    [
        "--merge-from",
        "--detector-signature",
        "--raw-channel",
        "--source-length-weighting",
        "--manifest-out",
    ],
)
def test_isolated_mode_requires_every_companion_flag(missing_flag, capsys):
    full = [
        "--dataset",
        "dataset.jsonl",
        "--config",
        "config.yaml",
        "--out",
        "out.json",
        "--only-detector",
        TARGET,
        "--merge-from",
        "base.json",
        "--detector-signature",
        SIGNATURE,
        "--raw-channel",
        RAW_CHANNEL,
        "--source-length-weighting",
        "--manifest-out",
        "manifest.json",
    ]
    index = full.index(missing_flag)
    del full[index : index + (1 if missing_flag == "--source-length-weighting" else 2)]

    with pytest.raises(SystemExit) as exc:
        calibrate_cli.main(full)

    assert exc.value.code == 2
    assert f"requires {missing_flag}" in capsys.readouterr().err


def test_isolated_mode_refuses_to_overwrite_merge_source(capsys):
    with pytest.raises(SystemExit) as exc:
        calibrate_cli.main(
            [
                "--dataset",
                "dataset.jsonl",
                "--config",
                "config.yaml",
                "--out",
                "same.json",
                "--only-detector",
                TARGET,
                "--merge-from",
                "same.json",
                "--detector-signature",
                SIGNATURE,
                "--raw-channel",
                RAW_CHANNEL,
                "--source-length-weighting",
                "--manifest-out",
                "manifest.json",
            ]
        )

    assert exc.value.code == 2
    assert "--out must differ from --merge-from" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("out", "merge_from", "manifest_out"),
    [
        ("out.json", "base.json", "base.json"),
        ("out.json", "base.json", "out.json"),
    ],
)
def test_isolated_mode_refuses_manifest_path_aliases(
    out,
    merge_from,
    manifest_out,
    capsys,
):
    with pytest.raises(SystemExit) as exc:
        calibrate_cli.main(
            [
                "--dataset",
                "dataset.jsonl",
                "--config",
                "config.yaml",
                "--out",
                out,
                "--only-detector",
                TARGET,
                "--merge-from",
                merge_from,
                "--detector-signature",
                SIGNATURE,
                "--raw-channel",
                RAW_CHANNEL,
                "--source-length-weighting",
                "--manifest-out",
                manifest_out,
            ]
        )

    assert exc.value.code == 2
    assert "--manifest-out must differ from --merge-from and --out" in capsys.readouterr().err

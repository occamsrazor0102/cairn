"""Deterministic source/length weighting for detector calibration."""

from __future__ import annotations

from pathlib import Path

import pytest

from cairn.calibration_training import (
    LENGTH_BINS,
    _length_bin,
    compute_source_length_weights,
)
from cairn.evaluation.dataset import EvalSample, load_dataset
from cairn.pipeline import normalize_text


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATASET = REPO_ROOT / "data" / "prompts_v2_train.jsonl"


def _text(word_count: int, token: str) -> str:
    return " ".join([token] * word_count)


def _sample(
    sample_id: str,
    *,
    word_count: int,
    label: int,
    source: str | None,
    source_id: str | None = None,
    nonbreaking_spaces: bool = False,
) -> EvalSample:
    separator = "\u00a0" if nonbreaking_spaces else " "
    return EvalSample(
        id=sample_id,
        text=separator.join([sample_id] * word_count),
        label=label,
        source=source,
        source_id=source_id,
    )


def _synthetic_inventory() -> list[EvalSample]:
    samples: list[EvalSample] = []
    for bin_index, word_count in enumerate((60, 100, 150, 250)):
        for row in range(20):
            samples.append(
                _sample(
                    f"h-{bin_index}-common-{row}",
                    word_count=word_count,
                    label=0,
                    source="human-common",
                    nonbreaking_spaces=row == 0,
                )
            )
        samples.append(
            _sample(
                f"h-{bin_index}-rare",
                word_count=word_count,
                label=0,
                source="",
                source_id="human-rare",
            )
        )
        for row in range(2):
            samples.append(
                _sample(
                    f"p-{bin_index}-common-{row}",
                    word_count=word_count,
                    label=1,
                    source="positive-common",
                )
            )
        samples.append(
            _sample(
                f"p-{bin_index}-missing",
                word_count=word_count,
                label=1,
                source=None,
                source_id=None,
            )
        )
    return samples


def test_length_bins_have_exact_boundaries_and_reject_short_rows():
    assert LENGTH_BINS == ((60, 100), (100, 150), (150, 250), (250, None))
    assert [_length_bin(n) for n in (60, 99, 100, 149, 150, 249, 250, 999)] == [
        "60-100",
        "60-100",
        "100-150",
        "100-150",
        "150-250",
        "150-250",
        "250-inf",
        "250-inf",
    ]
    with pytest.raises(ValueError, match="ineligible row: 59 words"):
        _length_bin(59)


def test_weighting_is_inverse_source_clipped_and_renormalized_deterministically():
    samples = _synthetic_inventory()

    first = compute_source_length_weights(samples)
    second = compute_source_length_weights(samples)

    assert first == second
    assert first.manifest["policy"] == "source-length-balanced-v1"
    assert first.manifest["bins"] == [
        "60-100",
        "100-150",
        "150-250",
        "250-inf",
    ]
    assert first.manifest["clip_multiple"] == 10.0
    assert len(first.weights) == len(samples)

    strata = first.manifest["strata"]
    assert [(row["length_bin"], row["label"]) for row in strata] == [
        (bin_name, label)
        for bin_name in ("60-100", "100-150", "150-250", "250-inf")
        for label in (0, 1)
    ]
    for bin_index in range(4):
        human = strata[bin_index * 2]
        positive = strata[bin_index * 2 + 1]
        assert human == {
            "length_bin": human["length_bin"],
            "label": 0,
            "rows": 21,
            "target_weight": 3.0,
            "actual_weight": pytest.approx(3.0),
            "median_inverse_source_weight": pytest.approx(0.05),
            "clip_cap": pytest.approx(0.5),
            "effective_sample_size": pytest.approx(7.5),
            "source_rows": {"human-common": 20, "human-rare": 1},
            "source_total_weights": {
                "human-common": pytest.approx(2.0),
                "human-rare": pytest.approx(1.0),
            },
        }
        assert positive == {
            "length_bin": positive["length_bin"],
            "label": 1,
            "rows": 3,
            "target_weight": 3.0,
            "actual_weight": pytest.approx(3.0),
            "median_inverse_source_weight": pytest.approx(0.5),
            "clip_cap": pytest.approx(5.0),
            "effective_sample_size": pytest.approx(8.0 / 3.0),
            "source_rows": {"<missing>": 1, "positive-common": 2},
            "source_total_weights": {
                "<missing>": pytest.approx(1.5),
                "positive-common": pytest.approx(1.5),
            },
        }

        offset = bin_index * 24
        assert first.weights[offset : offset + 20] == pytest.approx([0.1] * 20)
        assert first.weights[offset + 20] == pytest.approx(1.0)
        assert first.weights[offset + 21 : offset + 24] == pytest.approx(
            [0.75, 0.75, 1.5]
        )


def test_weighting_uses_canonical_normalized_word_count():
    samples = _synthetic_inventory()
    samples[0] = EvalSample(
        id="normalized-60",
        text=("word\u00a0" * 59) + "word",
        label=0,
        source="human-common",
    )

    weighted = compute_source_length_weights(samples)

    assert normalize_text(samples[0].text).split() == ["word"] * 60
    assert weighted.manifest["strata"][0]["rows"] == 21


@pytest.mark.skipif(
    not TRAIN_DATASET.is_file(),
    reason="data/prompts_v2_train.jsonl is git-ignored and absent from this checkout",
)
def test_real_train_inventory_matches_locked_source_length_targets():
    samples = load_dataset(TRAIN_DATASET)
    eligible = [
        sample
        for sample in samples
        if len(normalize_text(sample.text).split()) >= 60
    ]

    weighted = compute_source_length_weights(eligible)

    assert len(samples) == 20_732
    assert len(eligible) == 8_484
    expected = {
        ("60-100", 0): (1_968, 442.0),
        ("60-100", 1): (442, 442.0),
        ("100-150", 0): (1_583, 320.0),
        ("100-150", 1): (320, 320.0),
        ("150-250", 0): (912, 912.0),
        ("150-250", 1): (1_642, 912.0),
        ("250-inf", 0): (53, 53.0),
        ("250-inf", 1): (1_564, 53.0),
    }
    actual = {
        (str(row["length_bin"]), int(row["label"])): (
            int(row["rows"]),
            float(row["target_weight"]),
        )
        for row in weighted.manifest["strata"]
    }
    assert actual == expected
    assert sum(row["rows"] for row in weighted.manifest["strata"] if row["label"] == 0) == 4_516
    assert sum(row["rows"] for row in weighted.manifest["strata"] if row["label"] == 1) == 3_968
    assert sum(row["target_weight"] for row in weighted.manifest["strata"] if row["label"] == 0) == pytest.approx(1_727.0)
    assert sum(row["target_weight"] for row in weighted.manifest["strata"] if row["label"] == 1) == pytest.approx(1_727.0)
    for row in weighted.manifest["strata"]:
        assert row["actual_weight"] == pytest.approx(row["target_weight"])
        assert 0.0 < row["effective_sample_size"] <= row["rows"]

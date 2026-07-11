"""Deterministic weighting policies for detector calibration training."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Sequence

from cairn.evaluation.dataset import EvalSample
from cairn.pipeline import normalize_text


LENGTH_BINS = ((60, 100), (100, 150), (150, 250), (250, None))


@dataclass(frozen=True)
class SourceLengthWeights:
    weights: tuple[float, ...]
    manifest: dict[str, object]


def _source(sample: EvalSample) -> str:
    return sample.source or sample.source_id or "<missing>"


def _length_bin(word_count: int) -> str:
    for lower, upper in LENGTH_BINS:
        if word_count >= lower and (upper is None or word_count < upper):
            return f"{lower}-{upper if upper is not None else 'inf'}"
    raise ValueError(
        f"TOCSIN weighting received ineligible row: {word_count} words"
    )


def compute_source_length_weights(
    samples: Sequence[EvalSample],
) -> SourceLengthWeights:
    rows = [
        {
            "index": index,
            "label": int(sample.label),
            "source": _source(sample),
            "bin": _length_bin(len(normalize_text(sample.text).split())),
        }
        for index, sample in enumerate(samples)
    ]
    by_stratum: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_stratum[(str(row["bin"]), int(row["label"]))].append(row)

    weights = [0.0] * len(samples)
    strata: list[dict[str, object]] = []
    for bin_name in ("60-100", "100-150", "150-250", "250-inf"):
        n_human = len(by_stratum[(bin_name, 0)])
        n_positive = len(by_stratum[(bin_name, 1)])
        if not n_human or not n_positive:
            raise ValueError(f"length bin {bin_name} lacks both classes")
        target = float(min(n_human, n_positive))

        for label in (0, 1):
            members = by_stratum[(bin_name, label)]
            source_counts = Counter(str(row["source"]) for row in members)
            initial = [
                1.0 / source_counts[str(row["source"])]
                for row in members
            ]
            initial_median = median(initial)
            cap = 10.0 * initial_median
            clipped = [min(value, cap) for value in initial]
            scale = target / sum(clipped)
            for row, value in zip(members, clipped):
                weights[int(row["index"])] = value * scale

            stratum_weights = [
                weights[int(row["index"])] for row in members
            ]
            source_totals = {
                source: sum(
                    weights[int(row["index"])]
                    for row in members
                    if str(row["source"]) == source
                )
                for source in sorted(source_counts)
            }
            effective_sample_size = (
                sum(stratum_weights) ** 2
                / sum(value * value for value in stratum_weights)
            )
            strata.append(
                {
                    "length_bin": bin_name,
                    "label": label,
                    "rows": len(members),
                    "target_weight": target,
                    "actual_weight": sum(stratum_weights),
                    "median_inverse_source_weight": initial_median,
                    "clip_cap": cap,
                    "effective_sample_size": effective_sample_size,
                    "source_rows": dict(sorted(source_counts.items())),
                    "source_total_weights": source_totals,
                }
            )

    return SourceLengthWeights(
        weights=tuple(weights),
        manifest={
            "policy": "source-length-balanced-v1",
            "bins": ["60-100", "100-150", "150-250", "250-inf"],
            "clip_multiple": 10.0,
            "strata": strata,
        },
    )

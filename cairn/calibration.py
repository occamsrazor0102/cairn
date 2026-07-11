# cairn/calibration.py
"""Detector-level probability calibration via isotonic regression.

Maps each detector's raw_score -> calibrated p_llm using a dataset of
(raw_score, label) pairs. Stored as JSON for portability.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression

if TYPE_CHECKING:
    from cairn.contracts import LayerResult


CALIBRATION_SUPPORT_TOLERANCE_FRACTION = 0.25
CALIBRATION_SUPPORT_TOLERANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class CalibrationMetadata:
    detector_signature: str
    raw_channel: str
    training_dataset_hash: str
    weighting_policy: dict[str, object]


class DetectorCalibrator:
    """Fits and stores isotonic regression per detector."""

    def __init__(self) -> None:
        self._models: dict[str, IsotonicRegression] = {}
        self._metadata: dict[str, CalibrationMetadata | None] = {}
        self._fit_stats: dict[str, dict[str, object]] = {}
        self._serialized_entries: dict[str, dict[str, object]] = {}

    def fit(
        self,
        detector_id: str,
        raw_scores: Iterable[float],
        labels: Iterable[int],
        *,
        sample_weight: Iterable[float] | None = None,
        metadata: CalibrationMetadata | None = None,
    ) -> None:
        x = np.asarray(list(raw_scores), dtype=float)
        y = np.asarray(list(labels), dtype=float)
        weights = (
            None
            if sample_weight is None
            else np.asarray(list(sample_weight), dtype=float)
        )
        if x.shape != y.shape:
            raise ValueError("raw_scores and labels must have the same shape")
        if weights is not None and weights.shape != x.shape:
            raise ValueError("sample_weight must match raw_scores")
        if not np.isfinite(x).all():
            raise ValueError("raw_scores must all be finite")
        if weights is not None and (
            not np.isfinite(weights).all() or (weights <= 0).any()
        ):
            raise ValueError("sample_weight must be finite and positive")
        if len(x) < 2 or len(set(y.tolist())) < 2:
            # Not enough variation to fit; skip silently.
            self._models.pop(detector_id, None)
            self._metadata.pop(detector_id, None)
            self._fit_stats.pop(detector_id, None)
            self._serialized_entries.pop(detector_id, None)
            return
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(x, y, sample_weight=weights)
        self._models[detector_id] = ir
        self._metadata[detector_id] = metadata
        self._fit_stats[detector_id] = {
            "support_min": float(x.min()),
            "support_max": float(x.max()),
            "training_rows": int(len(x)),
            "class_counts": {
                "0": int((y == 0).sum()),
                "1": int((y == 1).sum()),
            },
        }
        self._serialized_entries.pop(detector_id, None)

    def transform(self, detector_id: str, raw_score: float) -> float:
        model = self._models.get(detector_id)
        if model is None:
            return float(raw_score)
        return float(model.predict([raw_score])[0])

    def has(self, detector_id: str) -> bool:
        return detector_id in self._models

    def support(self, detector_id: str) -> tuple[float, float]:
        metadata = self._metadata.get(detector_id)
        if metadata is not None:
            stats = self._fit_stats[detector_id]
            return float(stats["support_min"]), float(stats["support_max"])

        model = self._models[detector_id]
        return (
            float(model.X_thresholds_[0]),
            float(model.X_thresholds_[-1]),
        )

    def _entry_metadata(self, detector_id: str) -> dict[str, object]:
        metadata = self._metadata.get(detector_id)
        if metadata is not None:
            return asdict(metadata)
        return self._serialized_entries.get(detector_id, {})

    def apply(self, result: LayerResult) -> bool:
        if result.determination == "SKIP":
            return False
        result.calibrator_applied = False
        model = self._models.get(result.layer_id)
        if model is None:
            return False
        raw = float(result.raw_score)
        if not np.isfinite(raw):
            result.signals["calibrator_nonfinite_raw"] = {"raw_score": repr(raw)}
            result.determination = "SKIP"
            return False

        entry = self._entry_metadata(result.layer_id)
        artifact_signature = entry.get("detector_signature")
        artifact_channel = entry.get("raw_channel")
        result_signature = result.signals.get("calibration_signature")
        result_channel = result.signals.get("calibration_raw_channel")
        artifact_is_versioned = self._metadata.get(result.layer_id) is not None
        artifact_identity_valid = all(
            isinstance(value, str) and bool(value.strip())
            for value in (artifact_signature, artifact_channel)
        )
        result_identity_valid = all(
            isinstance(value, str) and bool(value.strip())
            for value in (result_signature, result_channel)
        )
        identity_mismatch = (
            not artifact_identity_valid
            or not result_identity_valid
            or artifact_signature != result_signature
            or artifact_channel != result_channel
        )
        legacy_identity_declared = any(
            value is not None for value in (result_signature, result_channel)
        )
        if (
            artifact_is_versioned and identity_mismatch
        ) or (
            not artifact_is_versioned and legacy_identity_declared
        ):
            result.signals["calibrator_signature_mismatch"] = {
                "result_signature": result_signature,
                "artifact_signature": artifact_signature,
                "result_raw_channel": result_channel,
                "artifact_raw_channel": artifact_channel,
            }
            result.determination = "SKIP"
            return False

        support_min, support_max = self.support(result.layer_id)
        tolerance = max(
            CALIBRATION_SUPPORT_TOLERANCE_FRACTION
            * (support_max - support_min),
            CALIBRATION_SUPPORT_TOLERANCE_FLOOR,
        )
        if raw < support_min - tolerance or raw > support_max + tolerance:
            result.signals["calibrator_out_of_domain"] = {
                "raw_score": raw,
                "support_min": support_min,
                "support_max": support_max,
                "tolerance": tolerance,
            }
            result.determination = "SKIP"
            return False

        result.p_llm = float(model.predict([raw])[0])
        result.calibrator_applied = True
        return True

    def save(self, path: Path) -> None:
        data: dict[str, dict[str, object]] = {}
        for det_id, ir in self._models.items():
            serialized = self._serialized_entries.get(det_id)
            if serialized is not None:
                data[det_id] = serialized
                continue

            # sklearn's IsotonicRegression stores the fitted knots on the
            # estimator. These are public attributes on recent sklearn
            # versions; guard so a change in internal naming produces a
            # clear error rather than a silent skip or crash later.
            if not (hasattr(ir, "X_thresholds_") and hasattr(ir, "y_thresholds_")):
                raise RuntimeError(
                    f"IsotonicRegression for '{det_id}' lacks X_thresholds_/"
                    "y_thresholds_ attributes; your sklearn version is too new "
                    "or too old for the save format expected here"
                )
            entry: dict[str, object] = {
                "X": ir.X_thresholds_.tolist(),
                "y": ir.y_thresholds_.tolist(),
            }
            metadata = self._metadata.get(det_id)
            if metadata is None:
                entry["schema"] = "isotonic-thresholds-v1"
            else:
                entry["schema"] = "isotonic-thresholds-v2"
                entry.update(asdict(metadata))
                entry.update(self._fit_stats[det_id])
            data[det_id] = entry
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for det_id, params in data.items():
            schema = params.get("schema")
            if schema not in (
                None,
                "isotonic-thresholds-v1",
                "isotonic-thresholds-v2",
            ):
                raise RuntimeError(
                    f"calibration artifact for '{det_id}' has unknown schema "
                    f"{schema!r}; regenerate with the current code"
                )
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            x = np.asarray(params["X"], dtype=float)
            y = np.asarray(params["y"], dtype=float)
            ir.fit(x, y)
            self._models[det_id] = ir
            if schema == "isotonic-thresholds-v2":
                self._metadata[det_id] = CalibrationMetadata(
                    detector_signature=params.get("detector_signature"),
                    raw_channel=params.get("raw_channel"),
                    training_dataset_hash=params["training_dataset_hash"],
                    weighting_policy=params["weighting_policy"],
                )
                self._fit_stats[det_id] = {
                    "support_min": float(params["support_min"]),
                    "support_max": float(params["support_max"]),
                    "training_rows": int(params["training_rows"]),
                    "class_counts": dict(params["class_counts"]),
                }
                self._serialized_entries.pop(det_id, None)
            else:
                self._metadata[det_id] = None
                self._fit_stats.pop(det_id, None)
                self._serialized_entries[det_id] = dict(params)

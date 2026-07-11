# cairn/calibration.py
"""Detector-level probability calibration via isotonic regression.

Maps each detector's raw_score -> calibrated p_llm using a dataset of
(raw_score, label) pairs. Stored as JSON for portability.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
from sklearn.isotonic import IsotonicRegression

if TYPE_CHECKING:
    from cairn.contracts import LayerResult


CALIBRATION_SUPPORT_TOLERANCE_FRACTION = 0.25
CALIBRATION_SUPPORT_TOLERANCE_FLOOR = 1e-12


def _finite_json_metadata(value: object) -> bool:
    """Return whether nested JSON-like metadata contains no non-finite number."""
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        try:
            return math.isfinite(float(value))
        except (OverflowError, TypeError, ValueError):
            return False
    if isinstance(value, list):
        return all(_finite_json_metadata(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _finite_json_metadata(item)
            for key, item in value.items()
        )
    return False


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
        if not np.isfinite(y).all() or not set(y.tolist()).issubset({0.0, 1.0}):
            raise ValueError("labels must be finite binary values")
        if weights is not None and (
            not np.isfinite(weights).all() or (weights <= 0).any()
        ):
            raise ValueError("sample_weight must be finite and positive")
        if metadata is not None:
            valid_identity = all(
                isinstance(value, str) and bool(value.strip())
                for value in (
                    metadata.detector_signature,
                    metadata.raw_channel,
                    metadata.training_dataset_hash,
                )
            )
            if (
                not valid_identity
                or not isinstance(metadata.weighting_policy, dict)
                or not _finite_json_metadata(metadata.weighting_policy)
            ):
                raise ValueError("invalid v2 calibration metadata")
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

    def identity(
        self,
        detector_id: str,
    ) -> tuple[str | None, str | None] | None:
        """Return the persisted detector identity, or ``None`` if absent.

        Legacy v1 entries deliberately return ``(None, None)``.  They remain
        valid for unsigned detectors, while identity-aware callers can detect
        that the entry cannot safely calibrate their signed raw channel.
        """
        if detector_id not in self._models:
            return None
        metadata = self._metadata.get(detector_id)
        if metadata is None:
            return (None, None)
        return (metadata.detector_signature, metadata.raw_channel)

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

    @staticmethod
    def _validated_v2_entry(
        detector_id: str,
        entry: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray, CalibrationMetadata, dict[str, object]]:
        """Validate the complete persisted v2 contract before sklearn sees it."""
        prefix = f"invalid v2 calibration artifact for '{detector_id}': "

        def invalid(reason: str) -> RuntimeError:
            return RuntimeError(prefix + reason)

        try:
            x = np.asarray(entry["X"], dtype=float)
            y = np.asarray(entry["y"], dtype=float)
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            raise invalid(f"invalid knots: {exc}") from exc
        if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or x.shape != y.shape:
            raise invalid("X and y must be equal-length one-dimensional arrays")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise invalid("X and y must contain only finite values")
        if not np.all(np.diff(x) > 0):
            raise invalid("X knots must be strictly increasing")
        if not np.all(np.diff(y) >= 0):
            raise invalid("y predictions must be monotone nondecreasing")
        if np.any(y < 0.0) or np.any(y > 1.0):
            raise invalid("y predictions must lie in [0, 1]")

        try:
            support_min = float(entry["support_min"])
            support_max = float(entry["support_max"])
        except (KeyError, OverflowError, TypeError, ValueError) as exc:
            raise invalid(f"invalid support bounds: {exc}") from exc
        if not math.isfinite(support_min) or not math.isfinite(support_max):
            raise invalid("support bounds must be finite")
        if support_min >= support_max:
            raise invalid("support_min must be less than support_max")
        if support_min != float(x[0]) or support_max != float(x[-1]):
            raise invalid("support bounds must equal the X knot endpoints")

        training_rows = entry.get("training_rows")
        if (
            isinstance(training_rows, bool)
            or not isinstance(training_rows, int)
            or training_rows < 2
        ):
            raise invalid("training_rows must be an integer of at least 2")
        class_counts = entry.get("class_counts")
        if not isinstance(class_counts, dict) or set(class_counts) != {"0", "1"}:
            raise invalid("class_counts must contain exactly string keys '0' and '1'")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in class_counts.values()
        ):
            raise invalid("class_counts values must be positive integers")
        if sum(class_counts.values()) != training_rows:
            raise invalid("class_counts must sum to training_rows")

        identity: dict[str, object] = {}
        for field in ("detector_signature", "raw_channel", "training_dataset_hash"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise invalid(f"{field} must be a non-empty string")
            identity[field] = value
        weighting_policy = entry.get("weighting_policy")
        if not isinstance(weighting_policy, dict) or not _finite_json_metadata(
            weighting_policy
        ):
            raise invalid("weighting_policy must be finite JSON object metadata")

        metadata = CalibrationMetadata(
            detector_signature=str(identity["detector_signature"]),
            raw_channel=str(identity["raw_channel"]),
            training_dataset_hash=str(identity["training_dataset_hash"]),
            weighting_policy=dict(weighting_policy),
        )
        stats: dict[str, object] = {
            "support_min": support_min,
            "support_max": support_max,
            "training_rows": training_rows,
            "class_counts": dict(class_counts),
        }
        return x, y, metadata, stats

    def _validate_versioned_runtime_state(
        self,
        detector_id: str,
        model: IsotonicRegression,
    ) -> None:
        metadata = self._metadata.get(detector_id)
        if metadata is None:
            return
        if not (
            hasattr(model, "X_thresholds_")
            and hasattr(model, "y_thresholds_")
        ):
            raise RuntimeError(
                f"invalid v2 calibration artifact for '{detector_id}': "
                "fitted knot attributes are missing"
            )
        x_knots = model.X_thresholds_
        y_knots = model.y_thresholds_
        if not isinstance(x_knots, np.ndarray) or not isinstance(
            y_knots, np.ndarray
        ):
            raise RuntimeError(
                f"invalid v2 calibration artifact for '{detector_id}': "
                "fitted knots must be numpy arrays"
            )
        try:
            entry: dict[str, object] = {
                "X": x_knots.tolist(),
                "y": y_knots.tolist(),
                **asdict(metadata),
                **self._fit_stats.get(detector_id, {}),
            }
            self._validated_v2_entry(detector_id, entry)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"invalid v2 calibration artifact for '{detector_id}': "
                f"runtime state could not be inspected ({type(exc).__name__})"
            ) from exc

    def apply(self, result: LayerResult) -> bool:
        if result.determination == "SKIP":
            return False
        result.calibrator_applied = False
        model = self._models.get(result.layer_id)
        if model is None:
            return False
        if result.raw_score is None:
            result.signals["calibrator_missing_raw"] = {"raw_score": None}
            result.determination = "SKIP"
            return False
        raw = float(result.raw_score)
        if not np.isfinite(raw):
            result.signals["calibrator_nonfinite_raw"] = {"raw_score": repr(raw)}
            result.determination = "SKIP"
            return False

        try:
            self._validate_versioned_runtime_state(result.layer_id, model)
        except RuntimeError as exc:
            result.signals["calibrator_invalid_artifact"] = {"reason": str(exc)}
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

        try:
            prediction = float(model.predict([raw])[0])
        except Exception as exc:
            result.signals["calibrator_invalid_artifact"] = {
                "reason": f"calibrator prediction failed: {exc}"
            }
            result.determination = "SKIP"
            return False
        if not math.isfinite(prediction):
            result.signals["calibrator_invalid_artifact"] = {
                "reason": "calibrator prediction was nonfinite"
            }
            result.determination = "SKIP"
            return False
        result.p_llm = prediction
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
        if not isinstance(data, dict):
            raise RuntimeError("calibration artifact must be a JSON object")
        loaded: list[
            tuple[
                str,
                IsotonicRegression,
                CalibrationMetadata | None,
                dict[str, object] | None,
                dict[str, object] | None,
            ]
        ] = []
        for det_id, params in data.items():
            if not isinstance(det_id, str) or not det_id or not isinstance(params, dict):
                raise RuntimeError("calibration artifact entries must be named objects")
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
            if schema == "isotonic-thresholds-v2":
                x, y, metadata, stats = self._validated_v2_entry(det_id, params)
                serialized = None
            else:
                ir_x = np.asarray(params["X"], dtype=float)
                ir_y = np.asarray(params["y"], dtype=float)
                x, y = ir_x, ir_y
                metadata = None
                stats = None
                serialized = dict(params)
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(x, y)
            loaded.append((det_id, ir, metadata, stats, serialized))

        for det_id, ir, metadata, stats, serialized in loaded:
            self._models[det_id] = ir
            self._metadata[det_id] = metadata
            if metadata is not None:
                assert stats is not None
                self._fit_stats[det_id] = stats
                self._serialized_entries.pop(det_id, None)
            else:
                self._fit_stats.pop(det_id, None)
                assert serialized is not None
                self._serialized_entries[det_id] = serialized

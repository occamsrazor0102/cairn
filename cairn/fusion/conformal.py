"""Split-conformal prediction wrapper for binary (LLM vs human) classification.

Produces valid marginal-coverage prediction sets: after calibration on a
held-out set of scores + labels, `predict_set(p)` returns a subset of
{"LLM", "human"} that contains the true label with probability at least
`1 - alpha` over the calibration distribution, as long as the test
point is exchangeable with the calibration set.

This replaces a prior implementation that mapped conformal scores to
four UI severity bands (RED/AMBER/YELLOW/GREEN) via a midpoint
heuristic — that mapping did **not** preserve marginal coverage and
made the documented guarantee false. The severity-band mapping has
moved to `beet/fusion/ebm.py::_p_llm_to_labels` where it's honestly
labelled as a cosmetic display mapping.

Math:
  - Nonconformity score s_i = 1 - p(true_label_i) for calibration point i.
  - Quantile q_hat = ceil((n + 1) * (1 - alpha)) / n quantile of {s_i}.
  - For a test point with score p_llm, include class c in the set iff
    1 - p(c | p_llm) <= q_hat, i.e. the nonconformity score of the
    hypothesis "the label is c" doesn't exceed the calibrated quantile.
  - Uncalibrated: return both classes (maximally conservative, honest).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


_LABELS = ["human", "LLM"]  # index 0 = label 0 = human; index 1 = LLM


class ConformalWrapper:
    """Honest split-conformal for the binary LLM/human decision (Mondrian-capable)."""

    _MIN_PER_CLASS = 10

    def __init__(self, alpha: float = 0.05):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._alpha = alpha
        self._threshold: float | None = None
        self._threshold_per_class: dict[int, float] | None = None
        self._n_calibration: int = 0
        self._n_per_class: dict[int, int] | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def calibrated(self) -> bool:
        return self._threshold is not None

    @property
    def threshold_per_class(self) -> dict[int, float] | None:
        return self._threshold_per_class

    def threshold_for(self, cls: int) -> float | None:
        if self._threshold_per_class is None:
            return self._threshold
        return self._threshold_per_class.get(int(cls))

    def _quantile_for(self, nonconformity: np.ndarray) -> float:
        n = len(nonconformity)
        q_level = math.ceil((n + 1) * (1.0 - self._alpha)) / n
        q_level = min(q_level, 1.0)
        return float(np.quantile(nonconformity, q_level, method="higher"))

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit the conformal threshold on a held-out calibration set.

        scores: array of fused p(LLM) estimates, shape (n,), values in [0, 1].
        labels: array of ground-truth labels, shape (n,), 0 = human, 1 = LLM.

        Atomicity: all four state fields (`_threshold`, `_n_calibration`,
        `_threshold_per_class`, `_n_per_class`) are written together only
        after every per-class check has passed. If any check raises, the
        wrapper retains its prior state (typically all-None, i.e.
        ``calibrated == False``).
        """
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if scores.shape != labels.shape:
            raise ValueError("scores and labels must have the same shape")
        if scores.ndim != 1:
            raise ValueError("scores and labels must be 1-D arrays")
        n = len(scores)
        if n < self._MIN_PER_CLASS:
            raise ValueError(f"too few calibration points ({n})")

        nonconformity = np.where(labels == 1, 1.0 - scores, scores)
        global_threshold = self._quantile_for(nonconformity)

        per_class: dict[int, float] = {}
        n_per_class: dict[int, int] = {}
        for cls in (0, 1):
            mask = labels == cls
            n_cls = int(mask.sum())
            if n_cls < self._MIN_PER_CLASS:
                raise ValueError(f"too few class-{cls} calibration points ({n_cls})")
            nc_cls = (1.0 - scores[mask]) if cls == 1 else scores[mask]
            per_class[cls] = self._quantile_for(nc_cls)
            n_per_class[cls] = n_cls

        # All checks passed — commit the calibration atomically.
        self._threshold = global_threshold
        self._n_calibration = int(n)
        self._threshold_per_class = per_class
        self._n_per_class = n_per_class

    def predict_set(self, p_llm: float) -> list[str]:
        """Return the conformal prediction set for one fused score.

        Returns a list drawn from ["human", "LLM"]. Possible values:
          - ["LLM"]            — confident LLM
          - ["human"]           — confident human
          - ["human", "LLM"]   — abstain; both hypotheses are conformal
        When uncalibrated, always returns ["human", "LLM"] (honest
        maximum-uncertainty set, not a coverage guarantee).
        """
        p_llm = max(0.0, min(1.0, float(p_llm)))
        if self._threshold is None:
            return list(_LABELS)
        if self._threshold_per_class is not None:
            out: list[str] = []
            nc = {0: p_llm, 1: 1.0 - p_llm}
            idx_to_label = {0: "human", 1: "LLM"}
            for cls in (0, 1):
                if nc[cls] <= self._threshold_per_class[cls]:
                    out.append(idx_to_label[cls])
            return out
        nc_named = {"human": p_llm, "LLM": 1.0 - p_llm}
        return [c for c in _LABELS if nc_named[c] <= self._threshold]

    _SCHEMA_V1 = "binary-split-conformal-v1"
    _SCHEMA_V2 = "binary-mondrian-conformal-v2"

    def save(self, path: Path) -> None:
        if self._threshold is None:
            raise RuntimeError("ConformalWrapper.save: not calibrated")
        payload: dict[str, object] = {
            "alpha": self._alpha,
            "threshold": self._threshold,
            "n_calibration": self._n_calibration,
            "schema": self._SCHEMA_V2,
        }
        if self._threshold_per_class is not None:
            payload["threshold_per_class"] = {
                str(k): v for k, v in self._threshold_per_class.items()
            }
        if self._n_per_class is not None:
            payload["n_per_class"] = {str(k): v for k, v in self._n_per_class.items()}
        Path(path).write_text(json.dumps(payload))

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text())
        schema = data.get("schema")
        if schema not in (None, self._SCHEMA_V1, self._SCHEMA_V2):
            raise ValueError(
                f"ConformalWrapper.load: unknown schema {schema!r}"
            )
        self._alpha = float(data["alpha"])
        self._threshold = float(data["threshold"])
        self._n_calibration = int(data.get("n_calibration", 0))
        tpc = data.get("threshold_per_class")
        if tpc is not None:
            self._threshold_per_class = {int(k): float(v) for k, v in tpc.items()}
        elif schema == self._SCHEMA_V1 or schema is None:
            self._threshold_per_class = {0: self._threshold, 1: self._threshold}
        else:
            # schema == _SCHEMA_V2 but threshold_per_class is missing — malformed.
            raise ValueError(
                f"ConformalWrapper.load: schema {self._SCHEMA_V2!r} requires threshold_per_class"
            )
        npc = data.get("n_per_class")
        self._n_per_class = (
            {int(k): int(v) for k, v in npc.items()} if npc is not None else None
        )

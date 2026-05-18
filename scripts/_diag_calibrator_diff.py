"""Diff v1's detector_calibration vs v2's for non-specdetect detectors.

Compares fit anchor lists. Used to determine whether the v1-vs-new
retrain regression is partly explained by calibrator drift, NOT just
EBM input schema differences.

One-off scratch invocation (underscore-prefix).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V1 = Path("models/calib2000/profiles/master_minus_arxiv_robust_v1/detector_calibration.json")
V2 = Path("models/calib2000/profiles/master_minus_arxiv_robust_v2/detector_calibration.json")


def main() -> int:
    v1 = json.loads(V1.read_text())
    v2 = json.loads(V2.read_text())

    detectors_v1 = set(v1.keys())
    detectors_v2 = set(v2.keys())
    print(f"v1 detectors: {len(detectors_v1)} -- {sorted(detectors_v1)}")
    print(f"v2 detectors: {len(detectors_v2)} -- {sorted(detectors_v2)}")
    print(f"v1 only: {sorted(detectors_v1 - detectors_v2)}")
    print(f"v2 only: {sorted(detectors_v2 - detectors_v1)}")

    common = detectors_v1 & detectors_v2
    print(f"\nPer-detector anchor diff (common: {len(common)}):")
    for det in sorted(common):
        a1 = v1[det]
        a2 = v2[det]
        x1 = a1.get("x_") or a1.get("x") or []
        y1 = a1.get("y_") or a1.get("y") or []
        x2 = a2.get("x_") or a2.get("x") or []
        y2 = a2.get("y_") or a2.get("y") or []
        if len(x1) != len(x2) or x1 != x2 or y1 != y2:
            max_dy = 0.0
            if len(x1) == len(x2):
                for v1_y, v2_y in zip(y1, y2):
                    max_dy = max(max_dy, abs(float(v1_y) - float(v2_y)))
            print(f"  {det}: v1 n={len(x1)} v2 n={len(x2)} max|Δy|={max_dy:.4f}")
        else:
            print(f"  {det}: IDENTICAL")

    return 0


if __name__ == "__main__":
    sys.exit(main())

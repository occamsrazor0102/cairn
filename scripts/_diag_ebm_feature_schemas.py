"""Compare _cairn_feature_names across robust_v1 + robust_v2 + ablations.

Shows which features each EBM was trained on, and which schemas differ.
One-off scratch invocation (underscore-prefix).
"""
from __future__ import annotations

import sys
from pathlib import Path

from cairn.fusion.serialization import load_model

VARIANTS = [
    ("v1", "models/calib2000/profiles/master_minus_arxiv_robust_v1/ebm.pkl"),
    ("v2", "models/calib2000/profiles/master_minus_arxiv_robust_v2/ebm.pkl"),
    ("v2e_no_specdetect", "models/calib2000/profiles/master_minus_arxiv_robust_v2e_no_specdetect/ebm.pkl"),
    ("v2f_pllm_only", "models/calib2000/profiles/master_minus_arxiv_robust_v2f_pllm_only/ebm.pkl"),
]


def main() -> int:
    schemas: dict[str, set[str]] = {}
    for name, path in VARIANTS:
        if not Path(path).exists():
            print(f"{name}: MISSING ({path})")
            continue
        m, _ = load_model(Path(path), verify_hmac=False)
        names = getattr(m, "_cairn_feature_names", None) or []
        schemas[name] = set(names)
        print(f"{name}: {len(names)} features")

    print()
    if "v1" in schemas:
        base = schemas["v1"]
        for name, s in schemas.items():
            if name == "v1":
                continue
            extra = sorted(s - base)
            missing = sorted(base - s)
            print(f"{name} vs v1:")
            print(f"  extra in {name}: {extra}")
            print(f"  missing in {name}: {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

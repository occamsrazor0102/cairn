"""Post-process a feature cache to simulate per-router specdetect gating.

For each record in the input cache, if the router domain is not "prose",
strip the specdetect entry from the results list. The result is a cache
where specdetect features are absent (and thus NaN-imputed downstream
by FeatureAssembler) for any sample that would not have hit specdetect
under the production cascade.

Used by robust_v2 follow-up experiment 4: train an EBM on the gated
cache to test whether eliminating the train/inference distribution
mismatch for specdetect closes the v2 regression. One-off scratch
invocation (underscore-prefix).

Usage:
    python scripts/_build_specdetect_gated_cache.py \\
        --in data/all_features_v8.jsonl \\
        --out data/all_features_v8_specdetect_gated.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    args = ap.parse_args()

    n_in = 0
    n_specdetect_gated = 0
    n_specdetect_kept = 0
    n_specdetect_absent = 0
    with args.in_path.open("r", encoding="utf-8") as fin, \
         args.out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            router_domain = rec.get("router", {}).get("domain", "prose")
            results = rec.get("results", [])
            had_specdetect = any(r.get("layer_id") == "specdetect" for r in results)
            if router_domain != "prose":
                rec["results"] = [r for r in results if r.get("layer_id") != "specdetect"]
                if had_specdetect:
                    n_specdetect_gated += 1
                else:
                    n_specdetect_absent += 1
            else:
                if had_specdetect:
                    n_specdetect_kept += 1
                else:
                    n_specdetect_absent += 1
            fout.write(json.dumps(rec) + "\n")

    print(
        f"processed {n_in} records: "
        f"{n_specdetect_kept} prose-with-specdetect kept, "
        f"{n_specdetect_gated} non-prose-with-specdetect gated, "
        f"{n_specdetect_absent} specdetect already absent",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

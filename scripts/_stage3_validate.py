"""Validate a retrained prompt EBM on the humanized slice (CPU, cache-based).

Fuses the 502 validation prompts (102 humanized label=1 + 400 human label=0)
through a given EBM using their cached features, and reports AUC + TPR at
5/10/20% FPR. Run for the with-SpecDetect and without-SpecDetect models to
isolate SpecDetect's contribution.

Usage: python scripts/_stage3_validate.py <ebm.skops> [<label>]
"""
import json
import math
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("CAIRN_ALLOW_LEGACY_PICKLE", "1")

from sklearn.metrics import roc_auc_score
from cairn.fusion.serialization import load_model
from cairn.fusion.ebm import EBMFusion
from cairn.feature_cache import (
    load_cache, text_hash, normalize_text, _result_from_dict, _router_from_dict,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "_specdetect_spike_features.jsonl"
VAL = ROOT / "data" / "_humanized_eval.jsonl"


def main() -> int:
    ebm_path = sys.argv[1] if len(sys.argv) > 1 else "models/_spike/prompts_specdetect_ebm.skops"
    label = sys.argv[2] if len(sys.argv) > 2 else Path(ebm_path).stem
    model, _ = load_model(Path(ebm_path))
    fusion = EBMFusion(model=model)
    cache = load_cache(CACHE)

    rows = []  # (p, label, words)
    miss = 0
    for line in open(VAL, encoding="utf-8"):
        r = json.loads(line)
        c = cache.get(text_hash(normalize_text(r["text"])))
        if c is None:
            miss += 1
            continue
        results = [_result_from_dict(x) for x in c["results"]]
        rd = _router_from_dict(c["router"])
        p = fusion.fuse(results, word_count=rd.word_count, domain=rd.domain).p_llm
        rows.append((p, r["label"], len(r["text"].split())))
    pos = [p for p, l, _ in rows if l == 1]
    neg = [p for p, l, _ in rows if l == 0]

    def tpr_at(fpr):
        nn = sorted(neg)
        thr = nn[max(0, min(len(nn) - 1, int(math.ceil((1 - fpr) * len(nn))) - 1))]
        return sum(1 for s in pos if s > thr) / len(pos), thr

    auc = roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg)
    # length confound checks
    ps = [p for p, _, _ in rows]; ws = [w for _, _, w in rows]; n = len(rows)
    mp, mw = sum(ps) / n, sum(ws) / n
    cov = sum((p - mp) * (w - mw) for p, w in zip(ps, ws))
    cw = corr = cov / ((sum((p - mp) ** 2 for p in ps) ** .5) * (sum((w - mw) ** 2 for w in ws) ** .5))
    print(f"\n=== {label} ===  (miss={miss})")
    print(f"  AUC = {auc:.3f}   corr(words, p_llm) = {cw:+.3f}")
    for fpr in (0.05, 0.10, 0.20):
        tpr, thr = tpr_at(fpr)
        print(f"  FPR={fpr:.0%}  thr={thr:.3f}  TPR={tpr:.2f}{'  <- >=0.70' if tpr >= 0.70 else ''}")
    print("  within-band AUC (length held ~constant -> real, not length):")
    for lo, hi in [(50, 150), (150, 260), (260, 9999)]:
        sub = [(p, l) for p, l, w in rows if lo <= w < hi]
        nh = sum(1 for _, l in sub if l == 0); nl = sum(1 for _, l in sub if l == 1)
        if nh >= 3 and nl >= 3:
            a = roc_auc_score([l for _, l in sub], [p for p, _ in sub])
            print(f"    {lo}-{hi}w n={len(sub)} (h={nh} l={nl}) AUC={a:.3f}")
        else:
            print(f"    {lo}-{hi}w n={len(sub)} (h={nh} l={nl}) -- too few one class")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

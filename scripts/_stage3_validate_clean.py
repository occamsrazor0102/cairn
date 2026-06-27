"""LEAK-FREE re-validation of prompts_specdetect: humanized positives vs
hard-human negatives that are DISJOINT from the EBM train+calib sets.

Positives: 102 humanized (spike cache, 0% train overlap).
Negatives: 296 clean hard humans (>=150w, NOT in prompts_only_train/_calib).
Reports AUC + TPR@FPR + within-band AUC + conformal-set distribution.
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
from cairn.feature_cache import CachedPipeline

ROOT = Path(__file__).resolve().parent.parent
CFG = "configs/.profile_prompts_specdetect.yaml"
MERGED = ROOT / "data" / "_val_clean_merged_features.jsonl"


def main() -> int:
    with open(MERGED, "w", encoding="utf-8") as out:
        for f in ("_specdetect_spike_features.jsonl", "_hard_human_clean_features.jsonl"):
            for line in open(ROOT / "data" / f, encoding="utf-8"):
                if line.strip():
                    out.write(line)
    pipe = CachedPipeline(MERGED, CFG)
    print("profile status:", pipe._inner._calibration_status)

    rows = []  # (p, label, conformal_set, words)
    miss = 0
    for line in open(ROOT / "data" / "_humanized_eval.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["label"] != 1:
            continue
        det = pipe.analyze(r["text"])
        rows.append((det.p_llm, 1, tuple(det.conformal_set or []), len(r["text"].split())))
    for line in open(ROOT / "data" / "_hard_human_clean.jsonl", encoding="utf-8"):
        r = json.loads(line)
        try:
            det = pipe.analyze(r["text"])
        except KeyError:
            miss += 1
            continue
        rows.append((det.p_llm, 0, tuple(det.conformal_set or []), len(r["text"].split())))

    pos = [p for p, l, _, _ in rows if l == 1]
    neg = [p for p, l, _, _ in rows if l == 0]

    def tpr_at(fpr):
        nn = sorted(neg)
        thr = nn[max(0, min(len(nn) - 1, int(math.ceil((1 - fpr) * len(nn))) - 1))]
        return sum(1 for s in pos if s > thr) / len(pos), thr

    auc = roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg)
    print(f"\n=== prompts_specdetect (LEAK-FREE: humanized vs clean hard-human) ===")
    print(f"  humanized n={len(pos)}  clean-human n={len(neg)}  (miss={miss})")
    print(f"  AUC = {auc:.3f}")
    for fpr in (0.05, 0.10, 0.20):
        tpr, thr = tpr_at(fpr)
        print(f"  FPR={fpr:.0%}  thr={thr:.3f}  TPR={tpr:.2f}{'  <- >=0.70' if tpr >= 0.70 else ''}")
    for lo, hi in [(150, 260), (200, 320)]:
        sub = [(p, l) for p, l, _, w in rows if lo <= w < hi]
        nh = sum(1 for _, l in sub if l == 0); nl = sum(1 for _, l in sub if l == 1)
        if nh >= 5 and nl >= 5:
            print(f"  within-band {lo}-{hi}w n={len(sub)} (h={nh} l={nl}) AUC={roc_auc_score([l for _,l in sub],[p for p,_ in sub]):.3f}")
    from collections import Counter
    for name, lab in (("humanized(+)", 1), ("clean-human(-)", 0)):
        c = Counter(cs for _, l, cs, _ in rows if l == lab)
        fmt = {("LLM",): "conf[LLM]", ("human",): "conf[human]", ("human", "LLM"): "abstain", ("LLM", "human"): "abstain", (): "empty"}
        print(f"  conformal {name}: { {fmt.get(k, str(k)): v for k, v in c.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

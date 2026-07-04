"""End-to-end validation of the full prompts_specdetect PROFILE (EBM + detector
calibration + conformal), via CachedPipeline on the merged validation caches.

Reports TPR@FPR + AUC on humanized vs hard-human, plus the honest conformal_set
distribution (binary ['LLM']/['human']/['human','LLM']).
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
MERGED = ROOT / "data" / "_val_merged_features.jsonl"


def main() -> int:
    # merge spike (has humanized) + hard caches into one file CachedPipeline reads
    with open(MERGED, "w", encoding="utf-8") as out:
        for f in ("_specdetect_spike_features.jsonl", "_hard_human_features.jsonl"):
            for line in open(ROOT / "data" / f, encoding="utf-8"):
                if line.strip():
                    out.write(line)

    pipe = CachedPipeline(MERGED, CFG)
    print("profile calibration_status:", pipe._inner._calibration_status)

    rows = []  # (p_llm, label, conformal_set, words)
    for line in open(ROOT / "data" / "_humanized_eval.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["label"] != 1:
            continue
        det = pipe.analyze(r["text"])
        rows.append((det.p_llm, 1, tuple(det.conformal_set or []), len(r["text"].split())))
    for line in open(ROOT / "data" / "_hard_human.jsonl", encoding="utf-8"):
        r = json.loads(line)
        det = pipe.analyze(r["text"])
        rows.append((det.p_llm, 0, tuple(det.conformal_set or []), len(r["text"].split())))

    pos = [p for p, l, _, _ in rows if l == 1]
    neg = [p for p, l, _, _ in rows if l == 0]

    def tpr_at(fpr):
        nn = sorted(neg)
        thr = nn[max(0, min(len(nn) - 1, int(math.ceil((1 - fpr) * len(nn))) - 1))]
        return sum(1 for s in pos if s > thr) / len(pos), thr

    auc = roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg)
    print("\n=== FULL PROFILE: prompts_specdetect (humanized vs hard-human) ===")
    print(f"  humanized n={len(pos)}  hard-human n={len(neg)}")
    print(f"  AUC = {auc:.3f}")
    for fpr in (0.05, 0.10, 0.20):
        tpr, thr = tpr_at(fpr)
        print(f"  FPR={fpr:.0%}  thr={thr:.3f}  TPR={tpr:.2f}{'  <- >=0.70' if tpr >= 0.70 else ''}")
    # conformal set distribution (the honest binary set) at the profile's alpha
    from collections import Counter
    for name, lab in (("humanized(+)", 1), ("hard-human(-)", 0)):
        c = Counter(cs for _, l, cs, _ in rows if l == lab)
        tot = sum(c.values())
        fmt = {("LLM",): "conf[LLM]", ("human",): "conf[human]",
               ("human", "LLM"): "abstain", ("LLM", "human"): "abstain", (): "empty"}
        summary = {fmt.get(k, str(k)): v for k, v in c.items()}
        print(f"  conformal {name}: {summary}  (n={tot})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

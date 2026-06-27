"""Hard re-validation: prompts_specdetect on humanized LLM vs LONG-DETAILED
human prompts (medqa / mmlu-pro law/history/health), mean 235w ~ the humanized
267w. Length-matched by construction, so a high TPR@low-FPR here is real
AI-discrimination, not length or generic-vs-dense.

Positives: 102 humanized (spike cache). Negatives: 380 hard humans (hard cache).
Usage: python scripts/_stage3_validate_hard.py <ebm.skops> [label]
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
SPIKE = ROOT / "data" / "_specdetect_spike_features.jsonl"
HARD = ROOT / "data" / "_hard_human_features.jsonl"
VAL = ROOT / "data" / "_humanized_eval.jsonl"
HARDSET = ROOT / "data" / "_hard_human.jsonl"


def main() -> int:
    ebm_path = sys.argv[1] if len(sys.argv) > 1 else "models/_spike/prompts_specdetect_ebm.skops"
    label = sys.argv[2] if len(sys.argv) > 2 else Path(ebm_path).stem
    model, _ = load_model(Path(ebm_path))
    fusion = EBMFusion(model=model)
    cache = load_cache(SPIKE)
    cache.update(load_cache(HARD))

    def score(text):
        c = cache.get(text_hash(normalize_text(text)))
        if c is None:
            return None, None
        results = [_result_from_dict(x) for x in c["results"]]
        rd = _router_from_dict(c["router"])
        return fusion.fuse(results, word_count=rd.word_count, domain=rd.domain).p_llm, rd.word_count

    rows = []  # (p, label, words)
    for line in open(VAL, encoding="utf-8"):
        r = json.loads(line)
        if r["label"] != 1:
            continue  # humanized positives only
        p, w = score(r["text"])
        if p is not None:
            rows.append((p, 1, w))
    for line in open(HARDSET, encoding="utf-8"):
        r = json.loads(line)
        p, w = score(r["text"])
        if p is not None:
            rows.append((p, 0, w))

    pos = [p for p, l, _ in rows if l == 1]
    neg = [p for p, l, _ in rows if l == 0]

    def tpr_at(fpr):
        nn = sorted(neg)
        thr = nn[max(0, min(len(nn) - 1, int(math.ceil((1 - fpr) * len(nn))) - 1))]
        return sum(1 for s in pos if s > thr) / len(pos), thr

    auc = roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg)
    print(f"\n=== {label}  (HARD: humanized vs long-detailed human) ===")
    print(f"  humanized n={len(pos)}  hard-human n={len(neg)}")
    print(f"  AUC = {auc:.3f}")
    for fpr in (0.05, 0.10, 0.20):
        tpr, thr = tpr_at(fpr)
        print(f"  FPR={fpr:.0%}  thr={thr:.3f}  TPR={tpr:.2f}{'  <- >=0.70' if tpr >= 0.70 else ''}")
    # length-matched band (both classes dense)
    for lo, hi in [(200, 320), (260, 9999)]:
        sub = [(p, l) for p, l, w in rows if lo <= w < hi]
        nh = sum(1 for _, l in sub if l == 0); nl = sum(1 for _, l in sub if l == 1)
        if nh >= 5 and nl >= 5:
            print(f"  band {lo}-{hi}w n={len(sub)} (h={nh} l={nl}) AUC={roc_auc_score([l for _,l in sub],[p for p,_ in sub]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 1 (CPU-only): clamp-OFF TPR@FPR, using pre-clamp positive scores from
git history (no live detector run -> cannot be killed).

Positives (clamp OFF):
  - screen 40  from git 73b755d:scripts/_lighthouse_screen_results.json
  - bench  62  from git 282e47e:scripts/_lighthouse_benchmark_results.json
Negatives: human prompts fused through prompts_only with clamp OFF, via cache.
"""
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("CAIRN_ALLOW_LEGACY_PICKLE", "1")

from sklearn.metrics import roc_auc_score  # noqa: E402
from cairn.config import load_config  # noqa: E402
from cairn.pipeline import CairnPipeline  # noqa: E402
from cairn.feature_cache import (  # noqa: E402
    load_cache, text_hash, normalize_text, _result_from_dict, _router_from_dict,
)

ROOT = Path(__file__).resolve().parent.parent


def git_json(ref):
    out = subprocess.run(["git", "show", ref], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    return json.loads(out.stdout)


def main() -> int:
    pos = []
    for r in git_json("73b755d:scripts/_lighthouse_screen_results.json"):
        if isinstance(r.get("p_llm"), (int, float)):
            pos.append((r["p_llm"], len(r["text"].split())))
    for r in git_json("282e47e:scripts/_lighthouse_benchmark_results.json"):
        if isinstance(r.get("p_llm"), (int, float)):
            pos.append((r["p_llm"], len(r["text"].split())))
    print(f"positives (clamp OFF): {len(pos)}")

    cfg = load_config("configs/.profile_prompts_only.yaml")
    cfg.setdefault("fusion", {})["word_count_clamp"] = None   # clamp OFF for negatives too
    pipe = CairnPipeline(cfg)
    cal = pipe._detector_calibrator
    cache = load_cache(ROOT / "data" / "all_features_v6.jsonl")
    humans = [json.loads(l)["text"] for l in open(ROOT / "data" / "prompts_v5_test.jsonl", encoding="utf-8")
              if json.loads(l).get("label") == 0]
    random.Random(0).shuffle(humans)
    neg = []
    for txt in humans[:3000]:
        c = cache.get(text_hash(normalize_text(txt)))
        if c is None:
            continue
        res = [_result_from_dict(r) for r in c["results"]]
        rd = _router_from_dict(c["router"])
        if cal is not None:
            for r in res:
                cal.apply(r)
        p = pipe._decision.decide(pipe._fusion.fuse(res, word_count=rd.word_count, domain=rd.domain), res).p_llm
        neg.append((p, rd.word_count))
    print(f"negatives: {len(neg)}")

    pos_p = [p for p, _ in pos]; pos_w = [w for _, w in pos]
    neg_p = [p for p, _ in neg]

    def tpr_at_fpr(fpr):
        nn = sorted(neg_p)
        thr = nn[max(0, min(len(nn) - 1, int(math.ceil((1 - fpr) * len(nn))) - 1))]
        return sum(1 for s in pos_p if s > thr) / len(pos_p), thr

    auc = roc_auc_score([1] * len(pos_p) + [0] * len(neg_p), pos_p + neg_p)
    print("\n" + "=" * 64)
    print("prompts_only (CLAMP OFF): humanized vs human")
    print("=" * 64)
    print(f"  AUC = {auc:.3f}")
    for fpr in (0.05, 0.10, 0.20, 0.30):
        tpr, thr = tpr_at_fpr(fpr)
        print(f"  FPR={fpr:.0%}  thr={thr:.3f}  TPR={tpr:.2f}{'  <- >=0.70' if tpr >= 0.70 else ''}")
    # length split: long positives (>=300w) vs short
    _, thr10 = tpr_at_fpr(0.10)
    longp = [p for p, w in zip(pos_p, pos_w) if w >= 300]
    shortp = [p for p, w in zip(pos_p, pos_w) if w < 300]
    print(f"\n  @10%FPR: TPR on LONG (>=300w, n={len(longp)}) = {sum(1 for s in longp if s>thr10)/len(longp):.2f}"
          f"   TPR on SHORT (<300w, n={len(shortp)}) = {sum(1 for s in shortp if s>thr10)/len(shortp):.2f}")
    print("  (the long-vs-short gap = how much TPR is length-driven, not content)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

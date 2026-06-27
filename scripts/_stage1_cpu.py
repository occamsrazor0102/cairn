"""Stage 1 (CPU-only, no heavy model loads): TPR@FPR for prompts_only on the
102 humanized prompts, using already-saved positive scores + cache negatives.

Positives = saved p_llm from the screen (40) + benchmark (62) runs (clamp ON,
current config). Negatives = human prompts fused via the cache on CPU. No
detector models are loaded, so this cannot be killed by GPU/RAM pressure.
"""
import json
import math
import os
import random
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
RNG = random.Random(0)


def main() -> int:
    # positives (clamp-on saved scores)
    pos = []
    for f in ("_lighthouse_screen_results.json", "_lighthouse_benchmark_results.json"):
        for r in json.loads((ROOT / "scripts" / f).read_text(encoding="utf-8")):
            if isinstance(r.get("p_llm"), (int, float)):
                pos.append((r["p_llm"], len(r["text"].split())))
    print(f"positives (clamp-on): {len(pos)}")

    cfg = load_config("configs/.profile_prompts_only.yaml")  # clamp on, matches positives
    pipe = CairnPipeline(cfg)
    cal = pipe._detector_calibrator
    cache = load_cache(ROOT / "data" / "all_features_v6.jsonl")
    humans = [json.loads(l)["text"] for l in open(ROOT / "data" / "prompts_v5_test.jsonl", encoding="utf-8")
              if json.loads(l).get("label") == 0]
    RNG.shuffle(humans)
    neg = []
    for txt in humans[:3000]:
        c = cache.get(text_hash(normalize_text(txt)))
        if c is None:
            continue
        res = [_result_from_dict(r) for r in c["results"]]
        rd = _router_from_dict(c["router"])
        if cal is not None:
            for r in res:
                rs = getattr(r, "raw_score", float("nan"))
                if cal.has(r.layer_id) and isinstance(rs, (int, float)) and math.isfinite(rs):
                    r.p_llm = cal.transform(r.layer_id, rs)
        p = pipe._decision.decide(pipe._fusion.fuse(res, word_count=rd.word_count, domain=rd.domain), res).p_llm
        neg.append((p, rd.word_count))
    print(f"negatives: {len(neg)}")

    pos_p = [p for p, _ in pos]
    neg_p = [p for p, _ in neg]

    def tpr_at_fpr(fpr):
        nn = sorted(neg_p)
        thr = nn[max(0, min(len(nn) - 1, int(math.ceil((1 - fpr) * len(nn))) - 1))]
        return sum(1 for s in pos_p if s > thr) / len(pos_p), thr

    auc = roc_auc_score([1] * len(pos_p) + [0] * len(neg_p), pos_p + neg_p)
    print("\n" + "=" * 64)
    print("prompts_only (CLAMP ON): humanized vs human")
    print("=" * 64)
    print(f"  AUC = {auc:.3f}")
    for fpr in (0.05, 0.10, 0.20, 0.30):
        tpr, thr = tpr_at_fpr(fpr)
        print(f"  FPR={fpr:.0%}  thr={thr:.3f}  TPR={tpr:.2f}{'  <- >=0.70' if tpr >= 0.70 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

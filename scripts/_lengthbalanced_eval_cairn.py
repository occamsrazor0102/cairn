"""Length-balanced human-vs-LLM eval for prompts_only — via the feature cache.

Scores the FULL prompts_v5_test set (6098 rows, 100% cached) through the
prompts_only EBM using CachedPipeline (fusion only, no GPT-J), then isolates
CAIRN's real discrimination from length leakage:

  - NATURAL AUC      : full test set, natural length skew (human longer, LLM
                       shorter) — length CAN help.
  - LENGTH-ONLY AUC  : word count alone as the predictor (free baseline) — how
                       much length alone separates the classes.
  - WITHIN-BIN AUC   : AUC inside narrow word-count bins (length ~constant) —
                       length removed; this is REAL content discrimination.
  - MATCHED AUC      : balanced subsample with identical class length
                       distributions — second length-neutral view.

Run:  python scripts/_lengthbalanced_eval_cairn.py
"""
import json
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sklearn.metrics import roc_auc_score  # noqa: E402

from cairn.feature_cache import CachedPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "prompts_v5_test.jsonl"
CACHE = ROOT / "data" / "all_features_v6.jsonl"
CFG = ROOT / "configs" / ".profile_prompts_only.yaml"
BINS = [(5, 15), (15, 30), (30, 50), (50, 80), (80, 130), (130, 260), (260, 4000)]
RNG = random.Random(0)


def _bin(w):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= w < hi:
            return i
    return -1


def auc(items):
    y = [it["label"] for it in items]
    p = [it["p_llm"] for it in items]
    return roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")


def auc_score(items, key):
    y = [it["label"] for it in items]
    s = [it[key] for it in items]
    return roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")


def corr_words_p(items):
    n = len(items)
    xs = [it["words"] for it in items]
    ys = [it["p_llm"] for it in items]
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs) ** 0.5
    vy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def main() -> int:
    pool = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("label") not in (0, 1):
                continue
            w = len(r.get("text", "").split())
            if _bin(w) < 0:
                continue
            pool.append({"text": r["text"], "words": w, "label": r["label"]})
    print(f"pool: {len(pool)}  (human={sum(1 for x in pool if x['label']==0)} "
          f"llm={sum(1 for x in pool if x['label']==1)})")

    print("loading feature cache + prompts_only EBM ...")
    pipe = CachedPipeline(CACHE, CFG)
    for k, it in enumerate(pool):
        it["p_llm"] = pipe.analyze(it["text"]).p_llm
        if (k + 1) % 1000 == 0:
            print(f"  scored {k+1}/{len(pool)}")

    def meanw(items, lab):
        ws = [it["words"] for it in items if it["label"] == lab]
        return sum(ws) / len(ws) if ws else float("nan")

    print("\n" + "=" * 90)
    print("prompts_only — human(0) vs LLM(1) discrimination on prompts_v5_test")
    print("=" * 90)
    print(f"NATURAL (full {len(pool)}):  mean words human={meanw(pool,0):.1f}  llm={meanw(pool,1):.1f}")
    print(f"   CAIRN p_llm AUC = {auc(pool):.3f}")
    print(f"   LENGTH-ONLY AUC = {auc_score(pool,'words'):.3f}   (word count alone, same set)")
    print(f"   corr(words, p_llm) = {corr_words_p(pool):+.3f}")

    # within-bin AUC (length ~constant)
    print("\nWITHIN-BIN AUC (length held ~constant -> real content discrimination):")
    wsum = 0.0
    wn = 0
    for i, (lo, hi) in enumerate(BINS):
        sub = [x for x in pool if _bin(x["words"]) == i]
        nh = sum(1 for x in sub if x["label"] == 0)
        nl = sum(1 for x in sub if x["label"] == 1)
        a = auc(sub)
        astr = f"{a:.3f}" if a == a else " n/a "
        if a == a:
            wsum += a * len(sub)
            wn += len(sub)
        print(f"   {lo:3d}-{hi:<4d} words  n={len(sub):4d} (h={nh:4d} l={nl:4d})   AUC={astr}")
    print(f"   -> size-weighted within-bin AUC = {wsum/wn:.3f}")

    # matched subsample: equal class counts per bin
    matched = []
    for i in range(len(BINS)):
        hs = [x for x in pool if _bin(x["words"]) == i and x["label"] == 0]
        ls = [x for x in pool if _bin(x["words"]) == i and x["label"] == 1]
        RNG.shuffle(hs)
        RNG.shuffle(ls)
        k = min(len(hs), len(ls))
        matched += hs[:k] + ls[:k]
    print(f"\nMATCHED subsample n={len(matched)}: mean words "
          f"human={meanw(matched,0):.1f}  llm={meanw(matched,1):.1f}")
    print(f"   CAIRN p_llm AUC = {auc(matched):.3f}   LENGTH-ONLY AUC = {auc_score(matched,'words'):.3f}")

    out = ROOT / "scripts" / "_lengthbalanced_results.json"
    out.write_text(json.dumps({
        "auc_natural": auc(pool), "auc_length_only_natural": auc_score(pool, "words"),
        "within_bin_weighted_auc": wsum / wn, "auc_matched": auc(matched),
        "n_pool": len(pool), "n_matched": len(matched),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

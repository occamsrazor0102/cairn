"""Re-rank the Form B profile grid by REAL discrimination (AUC + human control).

mean-p(LLM) on a known-LLM set is invalid: a profile that flags everything wins.
This adds the human control — AUC and class means on a balanced v5_test sample —
so degenerate (saturated) profiles are exposed. Reuses the feature cache once
across all profiles. Merges with _formb_profile_grid_results.json.

Run: CAIRN_ALLOW_LEGACY_PICKLE=1 python scripts/_formb_profile_grid_auc.py
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
PROF_DIR = ROOT / "models" / "calib2000" / "profiles"
RNG = random.Random(0)


def main() -> int:
    # balanced v5_test sample
    H, L = [], []
    with open(ROOT / "data" / "prompts_v5_test.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            (H if r.get("label") == 0 else L).append(r["text"])
    sample = [(t, 0) for t in RNG.sample(H, 200)] + [(t, 1) for t in RNG.sample(L, 200)]

    print("loading feature cache once...")
    cache = load_cache(ROOT / "data" / "all_features_v6.jsonl")
    # pre-resolve cache hits
    keyed = []
    for t, y in sample:
        c = cache.get(text_hash(normalize_text(t)))
        if c is not None:
            keyed.append((c, y))
    print(f"sample with cache coverage: {len(keyed)}/{len(sample)}")

    formb = {r["profile"]: r for r in json.loads(
        (ROOT / "scripts" / "_formb_profile_grid_results.json").read_text(encoding="utf-8"))["rows"]}

    rows = []
    for pd in sorted(d for d in PROF_DIR.iterdir() if d.is_dir()):
        name = pd.name
        cfg_file = ROOT / "configs" / f".profile_{name}.yaml"
        if cfg_file.exists():
            cfg = load_config(str(cfg_file))
        else:
            cfg = load_config("configs/.profile_master_minus_arxiv.yaml")
            ebm = pd / ("ebm.skops" if (pd / "ebm.skops").exists() else "ebm.pkl")
            cfg["fusion"] = {"model_path": str(ebm),
                             "conformal_path": str(pd / "conformal.json"),
                             "detector_calibration_path": str(pd / "detector_calibration.json")}
        mp = (cfg.get("fusion") or {}).get("model_path")
        if not mp or not Path(mp).exists():
            continue
        try:
            pipe = CairnPipeline(cfg)
        except Exception:
            continue
        cal = pipe._detector_calibrator
        ys, ps = [], []
        for c, y in keyed:
            results = [_result_from_dict(r) for r in c["results"]]
            rd = _router_from_dict(c["router"])
            if cal is not None:
                for r in results:
                    rs = getattr(r, "raw_score", float("nan"))
                    if cal.has(r.layer_id) and isinstance(rs, (int, float)) and math.isfinite(rs):
                        r.p_llm = cal.transform(r.layer_id, rs)
            try:
                p = pipe._decision.decide(
                    pipe._fusion.fuse(results, word_count=rd.word_count, domain=rd.domain), results).p_llm
            except Exception:
                continue
            ys.append(y); ps.append(p)
        if len(set(ys)) < 2:
            continue
        import statistics as st
        hu = [p for p, y in zip(ps, ys) if y == 0]
        ll = [p for p, y in zip(ps, ys) if y == 1]
        fb = formb.get(name, {})
        rows.append({"profile": name, "auc": roc_auc_score(ys, ps),
                     "human_mean": st.mean(hu), "llm_mean": st.mean(ll),
                     "formb_mean": fb.get("mean"), "formb_flag": fb.get("flag")})
        print(f"  {name:42s} AUC={rows[-1]['auc']:.3f}")

    print("\n" + "=" * 104)
    print("ALL PROFILES re-ranked by REAL discrimination (AUC on balanced v5_test) vs Form B mean")
    print("=" * 104)
    print(f"{'profile':42s} {'v5 AUC':>7s} {'hum':>5s} {'llm':>5s} | {'FormB mean':>10s} {'flag':>6s}  verdict")
    for r in sorted(rows, key=lambda x: x["auc"], reverse=True):
        degen = "DEGENERATE (flags ~all)" if r["auc"] < 0.6 and r["human_mean"] > 0.7 else ""
        fbm = f"{r['formb_mean']:.3f}" if r['formb_mean'] is not None else "  -  "
        print(f"{r['profile']:42s} {r['auc']:7.3f} {r['human_mean']:5.2f} {r['llm_mean']:5.2f} | "
              f"{fbm:>10s} {str(r['formb_flag'])+'/20':>6s}  {degen}")
    out = ROOT / "scripts" / "_formb_profile_grid_auc_results.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

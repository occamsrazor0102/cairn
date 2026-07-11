"""Score the 20 Form B prompts (all known LLM-generated) against EVERY trained
calibration profile.

Detector outputs are profile-independent, so we run the full cascade ONCE per
prompt (full-coverage + specdetect, so all features exist), then fast-fuse the
cached detector outputs through each profile's EBM + conformal. Each profile's
own config is used (so e.g. prompts_only's word_count_clamp applies).

Run:  CAIRN_ALLOW_LEGACY_PICKLE=1 python scripts/_formb_profile_grid.py
(legacy flag: most profile EBMs are pre-skops local pickles we trust.)
"""
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ.setdefault("CAIRN_ALLOW_LEGACY_PICKLE", "1")

from cairn.config import load_config  # noqa: E402
from cairn.pipeline import CairnPipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROF_DIR = ROOT / "models" / "calib2000" / "profiles"
FC_CFG = "configs/.profile_full_coverage_with_specdetect.yaml"


def form_b_prompts() -> dict:
    spec = importlib.util.spec_from_file_location("lh", str(ROOT / "scripts" / "_lighthouse_screen_cairn.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return {k: v for k, v in m.extract_items().items() if k.startswith("B")}


def main() -> int:
    items = form_b_prompts()
    labels = sorted(items, key=lambda x: int(x[1:]))
    print(f"Form B prompts: {len(labels)}")

    # 1. detector outputs once, via full coverage; no detector calibrator so the
    #    cached p_llm is the native detector value (each profile re-calibrates).
    print("running full-coverage detectors once (slow)...")
    fc = CairnPipeline(load_config(FC_CFG))
    fc._detector_calibrator = None
    cache = {}
    for lab in labels:
        _, results, rd = fc.analyze_detailed(items[lab])
        cache[lab] = ([copy.deepcopy(r) for r in results], rd)
    del fc

    # 2. enumerate profiles
    profiles = sorted(d for d in PROF_DIR.iterdir() if d.is_dir())
    rows = []
    for pd in profiles:
        name = pd.name
        cfg_file = ROOT / "configs" / f".profile_{name}.yaml"
        if cfg_file.exists():
            cfg = load_config(str(cfg_file))
        else:
            cfg = load_config("configs/.profile_master_minus_arxiv.yaml")
            ebm = pd / ("ebm.skops" if (pd / "ebm.skops").exists() else "ebm.pkl")
            cfg["fusion"] = {
                "model_path": str(ebm),
                "conformal_path": str(pd / "conformal.json"),
                "detector_calibration_path": str(pd / "detector_calibration.json"),
            }
        mp = (cfg.get("fusion") or {}).get("model_path")
        if not mp or not Path(mp).exists():
            rows.append({"profile": name, "note": "no EBM artifact"})
            continue
        try:
            pipe = CairnPipeline(cfg)
        except Exception as e:
            rows.append({"profile": name, "note": f"load fail: {str(e)[:70]}"})
            continue
        cal = pipe._detector_calibrator
        pll = []
        for lab in labels:
            res0, rd = cache[lab]
            res = copy.deepcopy(res0)
            if cal is not None:
                for r in res:
                    cal.apply(r)
            try:
                fr = pipe._fusion.fuse(res, word_count=rd.word_count, domain=rd.domain)
                det = pipe._decision.decide(fr, res)
                pll.append(det.p_llm)
            except Exception:
                pll.append(None)
        valid = [p for p in pll if isinstance(p, (int, float))]
        rows.append({
            "profile": name,
            "status": pipe._calibration_status,
            "mean": (sum(valid) / len(valid)) if valid else None,
            "flag": sum(1 for p in valid if p >= 0.5),
            "n": len(valid),
            "min": min(valid) if valid else None,
            "max": max(valid) if valid else None,
            "pll": pll,
        })
        m = rows[-1]
        if m.get("mean") is not None:
            print(f"  {name:42s} status={m['status']:10s} mean={m['mean']:.3f} flag={m['flag']:2d}/{m['n']}")
        else:
            print(f"  {name:42s} -> {m.get('note')}")

    # report
    print("\n" + "=" * 92)
    print("FORM B (20 known-LLM prompts) vs ALL profiles — sorted by mean p_llm")
    print("=" * 92)
    print(f"{'profile':42s} {'status':10s} {'mean':>6s} {'flag':>7s} {'min':>5s} {'max':>5s}")
    scored = [r for r in rows if r.get("mean") is not None]
    for r in sorted(scored, key=lambda x: x["mean"], reverse=True):
        print(f"{r['profile']:42s} {r['status']:10s} {r['mean']:6.3f} {r['flag']:3d}/{r['n']:<3d} "
              f"{r['min']:5.2f} {r['max']:5.2f}")
    for r in rows:
        if r.get("mean") is None:
            print(f"{r['profile']:42s} SKIPPED: {r.get('note')}")

    out = ROOT / "scripts" / "_formb_profile_grid_results.json"
    out.write_text(json.dumps({"labels": labels, "rows": rows}, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

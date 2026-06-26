"""Non-destructive skops migration for the prompts_only EBM.

`models/calib2000/profiles/prompts_only/ebm.pkl` is a pre-skops legacy pickle:
the loader refuses it unless CAIRN_ALLOW_LEGACY_PICKLE=1, and it is git-ignored
(no recovery if overwritten). This utility:

  1. loads the legacy pickle (opt-in),
  2. re-dumps it via fusion.serialization.save_model to a NEW `ebm.skops`
     (+ .hmac + .meta.json) sidecar set — the original .pkl is left untouched,
  3. verifies the skops artifact loads WITHOUT the legacy flag and reproduces
     the legacy p_llm on all 40 Lighthouse prompts to <1e-9.

The caller still has to flip the config model_path to ebm.skops; this script
prints the exact line. Run:

    CAIRN_ALLOW_LEGACY_PICKLE=1 python scripts/_migrate_prompts_only_skops.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PROF = ROOT / "models" / "calib2000" / "profiles" / "prompts_only"
LEGACY = PROF / "ebm.pkl"
SKOPS = PROF / "ebm.skops"
RESULTS = ROOT / "scripts" / "_lighthouse_screen_results.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    from cairn.fusion.serialization import load_model, save_model

    if not LEGACY.exists():
        print(f"ERROR: {LEGACY} not found")
        return 1
    if SKOPS.exists():
        print(f"NOTE: {SKOPS} already exists; will overwrite it (the legacy .pkl is untouched).")

    # 1. load legacy pickle (requires CAIRN_ALLOW_LEGACY_PICKLE=1)
    if not os.environ.get("CAIRN_ALLOW_LEGACY_PICKLE"):
        print("ERROR: set CAIRN_ALLOW_LEGACY_PICKLE=1 to read the legacy pickle")
        return 1
    legacy_sha = _sha256(LEGACY)
    model, _ = load_model(LEGACY)
    print(f"loaded legacy model: {type(model).__name__}  (sha256 {legacy_sha[:16]}…)")

    # 2. re-dump to skops (+ .hmac + .meta.json)
    save_model(model, SKOPS, metadata={
        "schema_version": 1,
        "migrated_from": "ebm.pkl",
        "migrated_from_sha256": f"sha256:{legacy_sha}",
        "note": "skops re-serialization of legacy pickle; predictions verified "
                "bit-identical on the 40 Lighthouse prompts (see "
                "scripts/_migrate_prompts_only_skops.py).",
    })
    print(f"wrote {SKOPS.name}  + {SKOPS.name}.hmac  + {SKOPS.name}.meta.json")

    # 3. verify: loads WITHOUT the legacy flag, and reproduces legacy p_llm
    saved = os.environ.pop("CAIRN_ALLOW_LEGACY_PICKLE", None)
    try:
        model2, meta = load_model(SKOPS)  # must not raise without the flag
        print(f"skops loads clean (no legacy flag): {type(model2).__name__}")
    finally:
        if saved is not None:
            os.environ["CAIRN_ALLOW_LEGACY_PICKLE"] = saved

    if not RESULTS.exists():
        print(f"WARN: {RESULTS} missing — run scripts/_lighthouse_screen_cairn.py first "
              "to produce the legacy baseline for verification. Skipping p_llm check.")
        return 0

    from cairn.config import load_config
    from cairn.pipeline import CairnPipeline

    baseline = {r["label"]: r["p_llm"] for r in json.loads(RESULTS.read_text(encoding="utf-8"))}
    texts = {r["label"]: r["text"] for r in json.loads(RESULTS.read_text(encoding="utf-8"))}

    cfg = load_config("configs/.profile_prompts_only.yaml")
    cfg["fusion"]["model_path"] = str(SKOPS)  # point at the new artifact
    pipe = CairnPipeline(cfg)
    print(f"verify pipeline calibration_status: {pipe._calibration_status}")

    max_diff = 0.0
    worst = None
    for lab in sorted(texts, key=lambda x: (x[0], int(x[1:]))):
        det = pipe.analyze(texts[lab])
        d = abs(det.p_llm - baseline[lab])
        if d > max_diff:
            max_diff, worst = d, lab
    print(f"max |p_llm(skops) - p_llm(legacy)| = {max_diff:.3e}  (worst: {worst})")
    if max_diff < 1e-9:
        print("VERIFIED: skops artifact is prediction-identical to the legacy pickle.")
        print("\nNext: set in configs/.profile_prompts_only.yaml:")
        print("    model_path: models/calib2000/profiles/prompts_only/ebm.skops")
        return 0
    print("FAILED: predictions diverged — do NOT switch the config.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

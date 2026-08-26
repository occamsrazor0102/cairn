# Retrain runbook — new detectors (fast_detectgpt, faithful TOCSIN, DivEye surprisal)

PR #51 wires three replacement detectors into the cascade and EBM
assembler but ships **code only**. The new detectors do not affect any
*calibrated* profile until its EBM + conformal artifacts are retrained:
`FeatureAssembler` now emits new columns, and existing artifacts simply
ignore them (logged as `extra_in_assembler`) until a fresh model is
trained. This runbook regenerates the artifacts.

Run this **locally** (or on your Modal-enabled box): the calib2000 corpus
and feature caches are gitignored and live only on your machine, and the
GPU caching step needs your Modal token. A cloud Claude session has
neither.

## What changed that forces a retrain

- `fast_detectgpt` — new Phase-2 detector (replaces classic `perturbation`).
- `token_cohesiveness` — faithful TOCSIN (BART-base cohesiveness + a
  Fast-DetectGPT base channel).
- `surprisal_dynamics` — heuristic suppressed (`UNCERTAIN`/confidence 0),
  now exports DivEye-style fluctuation features.

New EBM feature columns (additive; nothing removed, so old artifacts keep
loading): `fast_detectgpt_{p_llm,curvature,missing,confidence}`,
`tocsin_{cohesiveness_u,combined_w}`, and
`surprisal_{skew,kurtosis,d1_std,d1_absmean,d1_autocorr,d2_std,d2_absmean,
moment_mean,moment_std}`. Assembler went 77 → 90 columns.

## Pre-flight already done (cloud session)

The 4-step pipeline was exercised end-to-end on a synthetic corpus with
real models (gpt2 + bart-base) on CPU. Verified: all scripts run with the
new detectors; the new EBM artifact carries all 90 columns and loads with
no `extra_in_artifact` error; and the **current production artifact (77
cols) still loads under the new 90-col assembler** (`cols-it-needs =
[]`). So this runbook is known-good mechanically; what remains is running
it on the real corpus and confirming no metric regression.

## 0. Prereqs

```bash
pip install -e '.[tier2,fusion]'          # torch, transformers, sentence-transformers, interpret, sklearn, skops
pip install modal && modal token new      # for the GPU caching step
# Confirm these local inputs exist (names per docs/benchmarks/attack_coverage.md
# and CLAUDE.md — adjust to your actual filenames):
#   data/calib2000_v6_train.jsonl     training rows
#   data/calib2000_v6_calib.jsonl     conformal calibration split
#   data/calib2000_full_test.jsonl    in-domain test (per-attack baseline)
#   data/ood_raid_test.jsonl data/ood_mage_test.jsonl data/ood_m4_test.jsonl
```

> **CRITICAL — cache with full coverage.** Use
> `configs/.profile_full_coverage.yaml` for the feature cache so every
> detector fires on every sample. Under the default cascade the new
> Phase-2 detectors only fire on the ambiguous-middle slice (the
> pre-flight measured 24/48 rows), which understates their strength and
> starves the EBM of their features. The full-coverage profile is
> uncalibrated (no `model_path`/`detector_calibration_path`), so cached
> `p_llm` stays at the heuristic value — correct for ablation-friendly
> caching.

## 1. Rebuild the feature cache (GPU)

```bash
modal run scripts/modal_cache_features.py \
    --dataset-path data/calib2000_v6_train.jsonl \
    --out-path data/all_features_v8.jsonl \
    --config-path configs/.profile_full_coverage.yaml

# Repeat for any test/calib sets not already covered by all_features_v8,
# e.g. the OOD slices used in validation:
modal run scripts/modal_cache_features.py \
    --dataset-path data/calib2000_full_test.jsonl \
    --out-path data/calib2000_full_test_features_v8.jsonl \
    --config-path configs/.profile_full_coverage.yaml
```

CPU fallback (no Modal): same args to
`scripts/cache_pipeline_features.py --dataset … --config … --out …`.
Much slower (~1–4 s/sample); fine for the test sets, painful for the
full train corpus.

## 2. Retrain per production profile

Do this for **both** `master_minus_arxiv_robust_v1` and
`master_minus_arxiv` (swap the `--config` and output dir). Example for
robust_v1:

```bash
P=models/calib2000/profiles/master_minus_arxiv_robust_v1
CFG=configs/.profile_master_minus_arxiv_robust_v1.yaml

python scripts/calibrate_detectors.py \
    --dataset data/calib2000_v6_train.jsonl --config "$CFG" \
    --features-cache data/all_features_v8.jsonl \
    --out "$P/detector_calibration.json"

python scripts/train_fusion.py \
    --dataset data/calib2000_v6_train.jsonl --config "$CFG" \
    --features-cache data/all_features_v8.jsonl \
    --out "$P/ebm.pkl"            # regenerates ebm.pkl.hmac + ebm.pkl.meta.json

python scripts/calibrate_conformal.py \
    --dataset data/calib2000_v6_calib.jsonl --config "$CFG" \
    --features-cache data/all_features_v8.jsonl --alpha 0.05 \
    --out "$P/conformal.json"
```

Set `CAIRN_MODEL_HMAC_KEY` before `train_fusion` if you want the artifact
HMAC keyed to your secret rather than the default public key.

## 3. Validation gates (must pass before un-drafting)

```bash
# Per-detector AUROC: new TOCSIN + fast_detectgpt should clear the ~0.50
# floor the old detectors sat at; surprisal_dynamics emits no standalone
# p_llm now (EBM-only), so judge it by EBM feature importance instead.
python scripts/per_detector_auroc.py \
    --calibrator "$P/detector_calibration.json" \
    --datasets data/calib2000_full_test.jsonl data/ood_raid_test.jsonl \
    --features-caches data/calib2000_full_test_features_v8.jsonl data/ood_raid_test_features_v8.jsonl \
    --out-csv docs/benchmarks/calib2000/per_detector_auroc_v8.csv

# Per-attack + aggregate vs the current numbers in
# docs/benchmarks/attack_coverage.md — confirm NO regression on the
# in-domain / RAID / MAGE rows and the per-attack table.
python scripts/run_per_attack_baseline.py \
    --dataset data/calib2000_full_test.jsonl --config "$CFG" --limit 100 \
    --out-csv docs/benchmarks/calib2000/per_attack_auroc_v8.csv
```

Compare against `docs/benchmarks/attack_coverage.md`:
in-domain ≥ 0.94, RAID ≥ 0.73, MAGE ≥ 0.60, and no per-attack regression
> 0.02 vs the robust_v1 baseline. The `summarization_attack` cell (0.42 in
the current grid) is the one to watch — improvement there would be a bonus,
but at minimum it must not get worse.

## 4. Ship

If the gates hold: commit the regenerated `$P/{ebm.pkl,ebm.pkl.hmac,
ebm.pkl.meta.json,detector_calibration.json,conformal.json}` for both
profiles plus the new benchmark CSVs, push to
`claude/last-update-timestamp-qyls30`, and mark PR #51 ready for review.
If a profile regresses, the additive schema means you can flip the
offending new detector to `enabled: false` in that profile and retrain
without touching the others.

---

# Slate 2026-08-26 -- compression_profile / typographic_signature / reasoning_artifact

Same shape as the PR-#51 runbook above, and same constraint: the branch
ships **code only**, all three `enabled: false`. They change nothing
until you rebuild the cache and retrain locally. Design spec:
`docs/superpowers/specs/2026-08-26-detector-slate-design.md`.

All three are **stdlib-only** -- no torch, no transformers, no API key.
That makes the cache rebuild much cheaper than the PR-#51 pass: the new
detectors add microseconds per sample, so the cost is dominated by the
tier-2 detectors you are re-running anyway.

## What changed

27 new EBM columns (assembler 102 -> 129, strictly additive -- existing
artifacts keep loading and log the extras until retrained):

- `compression_profile_{p_llm,missing,confidence,zlib_ratio,lzma_ratio,
  mattr_100,distinct_4,max_ngram_repeat,redundancy}`
- `typographic_signature_{p_llm,missing,confidence,dash_style_ratio,
  curly_quote_ratio,fancy_bullet_ratio,oxford_comma_ratio,
  heading_title_case,score}`
- `reasoning_artifact_{p_llm,missing,confidence,deliberation_density,
  self_correction_density,intent_density,enum_scaffold_density,
  register_uniformity,score}`

## 1. Enable, then rebuild the cache

Flip the three `enabled: false` flags in your cache-build profile, then:

```bash
modal run scripts/modal_cache_features.py \
    --dataset-path data/calib2000_v6_train.jsonl \
    --out-path data/all_features_v11.jsonl \
    --config-path configs/.profile_full_coverage.yaml
```

Same **CRITICAL** caveat as above: build with
`configs/.profile_full_coverage.yaml`. `reasoning_artifact` is a Phase-2
detector, so under the default cascade it only fires on the
ambiguous-middle slice and starves the EBM of its features.

Repeat for the calib split and every OOD/test slice you validate on.

## 2. Retrain

Identical to §2 above, swapping `--features-cache data/all_features_v11.jsonl`.
Do it for both production profiles.

## 3. Gates

The slate-wide gates from §3 still apply (in-domain >= 0.94, RAID >= 0.73,
MAGE >= 0.60, no per-attack regression > 0.02). Plus one gate per
detector, since each targets a specific measured hole:

| detector | gate | if it fails |
|---|---|---|
| `compression_profile` | `summarization_attack` improves on 0.4167 and clears 0.55 | `enabled: false`; keep the columns as EBM-only features |
| `typographic_signature` | short-form (< 150 words) in-domain AUROC improves >= 0.02, or `verb_low` improves | `enabled: false` |
| `reasoning_artifact` | OpenAI per-provider AUROC >= 0.96 AND Gemini drops <= 0.02 | `enabled: false` -- a Gemini drop means it learned one provider's UX |

`summarization_attack` is the number to watch. It is the only cell in
the per-attack grid below the random floor (0.4167), and
`compression_profile` exists specifically for it. If that cell does not
move, the detector has not earned its slot regardless of what the
aggregate does.

**Also check cascade-phase distribution, not just AUROC.**
`compression_profile` and `typographic_signature` are Phase-1 detectors,
so they feed the confidence-weighted `aggregate_p_llm` that gates phases
2 and 3. Both cap confidence low (0.80 / 0.65) to limit this, but a
slate that lifts AUROC while short-circuiting more samples away from the
phase-2 detectors has not improved the system. Compare the
`cascade_phases` histogram before and after.

## 4. Ship

As §4: commit the regenerated artifacts for both profiles plus a
refreshed `per_detector_auroc_v11.csv`. Any detector that misses its
gate flips to `enabled: false` -- the additive schema means that is a
config change, not another retrain.

## Optional fast-follow (not in this PR)

- `semantic_kinematics` suppress branch uses `SKIP`, which the assembler
  drops — its `sk_*` features never reach the EBM. One-line fix
  (`SKIP` → `UNCERTAIN`, mirroring the new surprisal path); fold into the
  same retrain to finally activate those features.

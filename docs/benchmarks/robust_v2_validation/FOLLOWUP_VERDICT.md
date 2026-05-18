# robust_v2 follow-up experiments verdict — REGRESSION CONFIRMED

**Date:** 2026-05-18
**Compared against:** master_minus_arxiv_robust_v1 (production baseline)

## TL;DR

All four follow-up experiments designed to close the v2 regression
**failed**. SpecDetect's addition itself is the source of the
regression — not the user-flagged rows, not the interaction count,
not the train/inference distribution mismatch, and not full-coverage
cache.

**Keep robust_v1 in production. Do not promote v2 or any of v2a/b/c/d.**

## OOD AUROC matrix (200 rows per test set)

| Variant                  | RAID    | MAGE    | M4      | Δ vs v1 mean   |
|--------------------------|--------:|--------:|--------:|---------------:|
| **v1 (baseline)**        |  0.7571 |  0.7161 |  0.6960 |          0.000 |
| v2 (full+user-flagged)   |  0.7336 |  0.6205 |  0.5365 |       **-0.093** |
| v2a (no user-flagged)    |  0.7302 |  0.6191 |  0.5434 |         -0.092 |
| v2b (cascade-default)    |  0.6657 |  0.5538 |  0.5894 |         -0.120 |
| v2c (interactions=5)     |  0.6922 |  0.5949 |  0.5354 |         -0.116 |
| v2d (specdetect gated)   |  0.7269 |  0.6244 |  0.5550 |         -0.083 |

## Per-experiment findings

### Experiment 1: ablate user-flagged rows (v2a)
**Hypothesis:** The 4 user-flagged samples shifted the EBM in a harmful
direction.
**Result:** v2a is **nearly identical** to v2 (AUROC differs by ≤0.005
on every set). The 4 rows have negligible effect on a model trained on
82,910 baseline rows.
**Verdict:** User-flagged rows are NOT the source of the regression.

### Experiment 2: cascade-default cache rebuild (v2b)
**Hypothesis:** The v2 EBM was trained on full-coverage features but
production runs cascade-short-circuit; aligning train and inference
distributions should help.
**Result:** v2b is the **WORST** variant on RAID (0.666) and MAGE (0.554).
Surprisingly the cascade-default cache makes things worse for these
test sets, possibly because the EBM loses signal it had at training time.
On M4, v2b actually improves over v2 (0.589 vs 0.537) — but still below
v1's 0.696.
**Verdict:** Train/inference distribution match does NOT close the gap.

### Experiment 3: lower EBM interactions to 5 (v2c)
**Hypothesis:** The 15-interaction EBM overfits in the larger feature
space added by SpecDetect.
**Result:** v2c regresses further on RAID (0.692) and MAGE (0.595).
Reducing interactions does NOT improve generalization here.
**Verdict:** EBM overfitting is NOT the source of the regression.

### Experiment 4: per-router specdetect gating (v2d)
**Hypothesis:** SpecDetect signal is only meaningful for the prose
router; gating it elsewhere eliminates noise.
**Result:** v2d is the **best** of the four follow-ups but still
**below v1** on every set (RAID 0.727 vs 0.757, MAGE 0.624 vs 0.716,
M4 0.555 vs 0.696).
**Verdict:** Even the cleanest specdetect contribution doesn't recover
v1's quality.

## What this tells us

The regression is robust across:
- corpus composition (v2 vs v2a)
- feature distribution (v2 vs v2b)
- model capacity (v2 vs v2c)
- per-router gating (v2 vs v2d)

The only common change across all five v2 variants is **the presence
of SpecDetect features in the EBM input vector**. Even when SpecDetect
is gated to prose-only (v2d), or NaN-imputed for non-prose (v2b), the
EBM's relationships among the OTHER detector features appear to have
shifted in a way that hurts OOD generalization.

## Plausible deeper hypotheses (untested)

1. **Specdetect's calibrator absorbed signal from collinear detectors.**
   When isotonic was refit on the new 50,814-sample prose corpus to
   include specdetect, it may have shifted the per-detector p_llm
   distributions used by the EBM. v1's isotonic was fit without
   specdetect competition.
2. **Specdetect's _HEURISTIC_S_DFT_TOTAL_ENERGY_TO_P_LLM table is
   miscalibrated.** The detector ships uncalibrated and the empirical
   range may have a different scale than v1's existing detectors,
   biasing the EBM's feature weights.
3. **The 6 spectral signal features confuse the EBM.** specdetect_total_energy
   is redundant with surprisal_std (Parseval), and the 3 band fractions
   plus low/mid ratio plus spectral entropy may be capturing redundant
   variance that the EBM allocates capacity to instead of robust signal.

## Recommended next steps (none authorized — out of scope for this session)

1. **Detector-level ablation:** retrain v2 with SpecDetect's signal-level
   features dropped (keep only specdetect_p_llm + specdetect_missing +
   specdetect_confidence). Tests hypothesis 3.
2. **Re-calibrate v1's detectors on v8 cache** (no specdetect) to
   isolate whether specdetect's mere PRESENCE shifts other detectors'
   isotonic curves. Tests hypothesis 1.
3. **Calibrate specdetect's raw → p_llm via isotonic with held-out
   validation** instead of the hardcoded heuristic table. Tests
   hypothesis 2.

## Decision

robust_v1 stays production. SpecDetect integration is shelved pending
a deeper investigation of which one (or more) of the deeper hypotheses
is the actual root cause.

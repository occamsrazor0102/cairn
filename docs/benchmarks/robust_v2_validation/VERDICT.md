# robust_v2 validation — verdict: REGRESSION

**Date:** 2026-05-18
**Compared against:** master_minus_arxiv_robust_v1 (production baseline)

## TL;DR

robust_v2 underperforms robust_v1 on every OOD benchmark and, most
critically, makes the math-heavy LLM false-negative class **worse**
rather than better. Do not promote v2 to production.

## OOD AUROC (200 rows each)

| Test set | v1 AUROC | v2 AUROC | Δ          | v1 TPR@1%FPR | v2 TPR@1%FPR | Δ          |
|----------|---------:|---------:|-----------:|-------------:|-------------:|-----------:|
| RAID     |   0.7571 |   0.7336 | **-0.024** |       0.4800 |       0.2000 | **-0.28**  |
| MAGE     |   0.7161 |   0.6205 | **-0.096** |       0.1000 |       0.0900 |    -0.01   |
| M4       |   0.6960 |   0.5365 | **-0.160** |       0.3600 |       0.1300 | **-0.23**  |

With n=200, RAID is within 1 standard error (could be noise) but MAGE
(~3 std err) and M4 (~5 std err) are statistically real regressions.

## Targeted user-prompt comparison (NINJ1 + FCHSD2 batches)

| Prompt              | truth   | v1 label    | v1 p_llm | v2 label    | v2 p_llm | Net    |
|---------------------|---------|-------------|---------:|-------------|---------:|--------|
| A_judge_rebuttal    | LLM     | RED         |    0.997 | RED         |    0.989 | same   |
| **B_solution**      | **LLM** | **AMBER**   |    0.677 | **UNCERTAIN** |  **0.335** | **WORSE** |
| C_error_analysis    | LLM     | UNCERTAIN   |    0.957 | RED         |    0.979 | better |
| D_original_prompt   | human   | RED (wrong) |    0.959 | UNCERTAIN   |    0.711 | better |
| E_combined_DCBA     | mixed   | UNCERTAIN   |    0.434 | GREEN       |    0.168 | worse  |
| b_A_solution        | LLM     | AMBER       |    0.687 | RED         |    0.844 | better |
| b_B_error_analysis  | LLM     | RED         |    0.986 | RED         |    0.975 | same   |
| b_C_original_prompt | ?       | YELLOW      |    0.351 | RED         |    0.763 | n/a    |
| b_D_combined_CBA    | mixed   | RED (wrong) |    0.864 | RED (wrong) |    0.780 | same   |

**The single most important case — B_solution (math-heavy step-by-step
LLM solution that the user flagged as a known false-negative class) —
regressed from v1's leaning-LLM verdict to v2's leaning-human verdict.**

## What went wrong (working hypotheses)

1. **Specdetect spike standalone gave RAID 0.855 / MAGE 0.711, but the
   full ensemble with specdetect gives 0.734 / 0.620.** Adding a strong
   single detector to an ensemble doesn't always help -- other detectors
   may have learned signals that conflict with specdetect's.
2. **Train/inference distribution mismatch.** The v8 cache was built with
   `_extends: .profile_full_coverage` so every detector fires every
   sample. At production inference time the cascade short-circuits, so
   specdetect only runs for the prose router (~half of samples). The
   v2 EBM was trained expecting specdetect_p_llm to always be present;
   at inference it's often NaN-imputed. (Note: this hypothesis does NOT
   explain the OOD validation regression here, since the OOD features
   cache was also built with full-coverage and thus specdetect IS
   present at OOD test time.)
3. **SpecDetect may be unreliable on math-heavy text.** Math text has
   a different token distribution than prose; per-token logprob
   spectrums may not carry the same LLM signature there. This is a
   plausible explanation for the B_solution regression specifically.
4. **4 user-flagged samples in 82,914 is too few** to shift a model with
   15 interactions. The bulk of the v2 change is from adding specdetect
   as a feature, not from the 4 new training rows.

## Recommended next experiments

(in increasing order of work)

1. **Ablate the 4 user-flagged samples**: retrain v2 without them to
   isolate the SpecDetect effect. If results are unchanged, the new
   rows are a no-op and SpecDetect is the regression source.
2. **Cascade-default cache rebuild**: re-cache the training corpus with
   the production cascade (short-circuit on), so specdetect features
   are NaN-imputed for non-prose. Retrain EBM. Hypothesis: this aligns
   train and inference distributions.
3. **Lower EBM interactions**: try interactions=5 (v1 had 5, current v2
   has 15) and see if overfitting eases.
4. **Per-router specdetect gating**: only allow specdetect to contribute
   to the EBM when the router is "prose"; mask its contribution otherwise.

## Decision

robust_v2 is **not production-ready**. Keep
master_minus_arxiv_robust_v1 as the production profile until one of
the above experiments closes the gap.

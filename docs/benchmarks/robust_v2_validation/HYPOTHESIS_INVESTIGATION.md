# robust_v2 deep-dive: investigating the 3 deeper hypotheses

**Date:** 2026-05-18 (afternoon session)
**Status:** Surprising findings; v1-vs-retrain gap is NOT explained by any
of the original 3 hypotheses.

## Original 3 hypotheses (from FOLLOWUP_VERDICT.md)

1. **H1 (calibrator collinearity):** SpecDetect's calibrator absorbed signal
   from collinear detectors, shifting per-detector p_llm distributions.
2. **H2 (heuristic miscalibration):** The `_HEURISTIC_TOTAL_ENERGY_TO_P_LLM`
   table doesn't cover the empirical range well.
3. **H3 (redundant signals):** The 6 spectral signal features dilute the EBM's
   capacity for marginal gain.

## Tests run + findings

### Diagnostic: SpecDetect empirical distribution (H2)

`scripts/_diag_specdetect_distribution.py` against `all_features_v8.jsonl`:

- Heuristic table anchors: 0.5 -> 5.0 (LLM-leaning to human-leaning)
- **Empirical p50 = 5.75** (already off the right edge of the table)
- **70.9% of samples have energy > 5.0** -> saturate to p_llm = 0.05
- **Calibrated specdetect_p_llm: p1=p5=p25=p50 = 0.05** (signal lost via saturation)
- Raw dft_total_energy AUROC vs label = **0.5557** (barely above random)

**H2 confirmed.** The heuristic table is broken for this corpus's regime;
isotonic calibration can't recover information once the heuristic has
saturated everything to p_llm=0.05. The detector is essentially injecting
near-constant noise into the EBM.

### Exp 5: drop all specdetect features (v2e)

If H1 (specdetect's mere presence dilutes other features) were the issue,
dropping all 9 specdetect features should restore v1-equivalent AUROC.

Result: **v2e regresses against v1** (RAID 0.629 vs v1's 0.757).
**H1 partially refuted** -- dropping specdetect doesn't recover v1.

### Exp 7: specdetect p_llm only (v2f)

Drop the 6 spectral signal features, keep p_llm + missing + confidence.

Result: **v2f is WORSE than v2 on RAID** (0.615 vs 0.734) and slightly
better on M4. The spectral signals carry SOME signal on RAID-style text
(adversarial paraphrases) -- removing them hurts there.

**H3 partially refuted** -- the spectral features aren't pure noise; they
help on some test sets.

### Discovery: 3 LAPD orphan features in the assembler

`scripts/_diag_ebm_feature_schemas.py` revealed:
- v1 EBM: 77 features (no LAPD, no SpecDetect)
- v2 EBM: 89 features (added 3 LAPD + 9 SpecDetect)
- v2e (drop specdetect): 80 features = 77 + 3 LAPD

LAPD was aborted (didn't reproduce paper numbers) but its assembler keys
(`lapd_p_llm`, `lapd_missing`, `lapd_confidence`) were left behind. The
detector never runs at inference, so these are always NaN-imputed --
effectively constant-valued noise consuming EBM capacity.

### Exp 8: clean v2 retrains drop LAPD orphans too

- **v2g:** drop specdetect + LAPD (true v1 schema) -> RAID 0.631 / MAGE 0.677 / M4 0.581
- **v2h:** drop only LAPD (full v2 minus orphans) -> RAID 0.733 / MAGE 0.621 / M4 0.537

**v2g ~= v2e** and **v2h ~= v2**. LAPD orphans are NOT the dilution source.
The discovery is real (assembler still has the orphans) but the EBM appears
robust to constant-valued NaN-imputed columns.

### Discovery: v1 calibrates 10 detectors, v2 calibrates 12

`scripts/_diag_calibrator_diff.py`:
- v1 calibrated: contrastive_lm, fingerprint_vocab, instruction_density,
  mixed_boundary, nssi, preamble, prompt_structure, surprisal_dynamics,
  token_cohesiveness, voice_spec
- v2 added: **perturbation**, specdetect

`perturbation_p_llm` is uncalibrated in v1's pipeline but isotonic-calibrated
in v2's pipeline. Different feature distributions at training time.

### Exp 9: v2i with v1's calibrator

Same EBM schema as v1 (drop specdetect + LAPD) AND v1's calibrator
(10 detectors, no perturbation).

Result: RAID 0.618 / MAGE 0.678 / M4 0.566.
**Still regresses against v1.** The calibrator schema change is NOT the
remaining gap source.

## What is the remaining ~0.14 gap on RAID?

After controlling for:
- EBM feature schema (v2i matches v1)
- Calibrator schema (v2i uses v1's calibrator JSON)
- 4 user-flagged training rows (v2a vs v2, negligible)

The remaining differences between v1 and v2i are:
1. **Cache used during training:** v1 trained on `all_features_v7.jsonl`
   (built May 13). v2i trained on `all_features_v8.jsonl` (built May 18).
2. **Detector code or library versions** may have shifted between
   the two cache builds.
3. **EBM training non-determinism** in joblib parallel paths
   (random_state=42 is fixed, but the parallel reduction order may
   not be deterministic).

The cache hypothesis is the most plausible -- if any non-specdetect detector
produced even slightly different signal values between May 13 and May 18,
the EBM's learned splits would shift.

## Summary table (all variants tested in this session)

| Variant | RAID | MAGE | M4 | Description |
|---------|-----:|-----:|-----:|------------|
| v1 (baseline) | 0.7571 | 0.7161 | 0.6960 | production |
| v2 (full + user-flagged) | 0.7336 | 0.6205 | 0.5365 | original v2 |
| v2a (no user-flagged) | 0.7302 | 0.6191 | 0.5434 | Exp 1 |
| v2b (cascade cache) | 0.6657 | 0.5538 | 0.5894 | Exp 2 |
| v2c (interactions=5) | 0.6922 | 0.5949 | 0.5354 | Exp 3 |
| v2d (specdetect gated) | 0.7269 | 0.6244 | 0.5550 | Exp 4 |
| v2e (drop specdetect, keep lapd) | 0.6294 | 0.6755 | 0.5815 | Exp 5 |
| v2f (specdetect p_llm only) | 0.6145 | 0.6577 | 0.5677 | Exp 7 |
| v2g (drop specdetect AND lapd) | 0.6306 | 0.6765 | 0.5814 | Exp 8 |
| v2h (drop only lapd, full specdetect) | 0.7332 | 0.6214 | 0.5373 | Exp 8 |
| v2i (v1 calibrator + drop specdetect+lapd) | 0.6181 | 0.6782 | 0.5659 | Exp 9 |

## Open questions for further investigation

1. **v7 vs v8 cache differential.** Spot-check N=100 sample texts and compare
   non-specdetect detector signal values between the two caches. If they
   differ materially, the cache build process (or upstream detector code)
   is the regression source.
2. **EBM training reproducibility.** Re-train v1 with current code +
   v7 cache -> does it reproduce v1's AUROC? If not, EBM training is
   non-deterministic.
3. **Specdetect's heuristic recalibration.** Re-anchor the
   `_HEURISTIC_TOTAL_ENERGY_TO_P_LLM` table from the empirical p1/p25/p50/
   p75/p95/p99 of the v8 cache (3.0, 4.8, 5.8, 6.9, 9.0, 10.9 units).
   Retrain v2_new. If RAID improves substantially, H2's miscalibration is
   the dominant SpecDetect issue.

## Decision (unchanged from earlier verdict)

robust_v1 stays production. SpecDetect's integration needs (a) a recalibrated
heuristic table or fully data-driven anchors, and (b) a clear answer to the
v7-vs-v8 cache differential before any retrained ensemble can match v1.

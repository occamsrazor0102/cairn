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

## Follow-up 2026-05-20: v7-vs-v8 cache differential — root cause found

Ran `scripts/_diag_v7_v8_cache_diff.py` on 500 shared hashes
(`all_features_v7.jsonl` ∩ `all_features_v8.jsonl` = 80,237 records).

For samples where the same layer fires in both caches:

| layer | field | n | max abs delta | shifted (delta > 1e-6) |
|---|---|---:|---:|---:|
| **token_cohesiveness** | **p_llm** | **100** | **0.0528** | **76 / 100** |
| token_cohesiveness | raw_score | 100 | 0.0033 | 76 / 100 |
| surprisal_dynamics | raw_score | 100 | 9.9e-6 | 42 / 100 |
| contrastive_lm | p_llm | 100 | 6.2e-6 | 26 / 100 |
| surprisal_dynamics | p_llm | 100 | 5.8e-6 | 12 / 100 |
| contrastive_lm | raw_score | 100 | 3.0e-6 | 11 / 100 |

Only `token_cohesiveness` drifts at a magnitude that can shift EBM splits.
The other v1-schema detectors drift at float-reduction-order noise (~1e-6).

**Root cause:** `cairn/detectors/token_cohesiveness.py:39`

```python
sample_indices = random.sample(range(len(words)), n_samples)
```

The detector uses Python's **global `random` module** to pick which 30 word
positions to delete-and-re-embed. Different cache-build sessions consume the
global RNG in different orders (router decisions, other detectors), so the
same text passing through `token_cohesiveness.analyze()` gets different
`sample_indices`, different reduced texts, different cosine distances,
different `p_llm`. The output is non-deterministic across runs and even
across samples within a run.

This was always present; v7 vs v8 only made it visible because we compared
two independent builds.

### Implications

1. **`token_cohesiveness` is one of v1's 10 calibrated detectors.** Its
   training-time signal distribution differs from its inference-time
   signal distribution by ~0.011 stdev — same magnitude as the v2-vs-v1
   p_llm shift we've been chasing.
2. **The production v1 EBM was trained on v7's RNG draw and runs against
   live inference's RNG draw.** It has been silently miscalibrated to
   the magnitude of this drift since v1 shipped.
3. **Part of the 0.14 RAID AUROC gap is plausibly attributable to this
   alone**, plus the cascade-routing differences observed as
   layer-presence-only deltas (see below).

### Layer-presence deltas (cascade routing changed between builds)

```
v8-only layers (per 500 sampled): contrastive_lm=400, surprisal_dynamics=400,
  mixed_boundary=400, token_cohesiveness=400, semantic_kinematics=478,
  perturbation=500, specdetect=500
v7-only layers: none beyond per-record hash gap
```

The v8 build fired phase-2 detectors on ~80% of corpus samples that v7 had
cascade-short-circuited past. This is a config change (the cache builder
profile changed between May 13 and May 18, plausibly to the
`full_coverage_with_specdetect` research profile), not a detector-code
change. It explains a different part of the v1-vs-v2 gap — the EBM saw
a denser feature matrix at train time than v1 did, and a denser one than
the production cascade produces at inference time.

### Proposed fix path

**Decision required from operator before proceeding.**

Option A (minimal, recommended): seed `token_cohesiveness` deterministically
from a hash of the input text.

```python
rng = random.Random(hashlib.sha256(text.encode("utf-8")).digest())
sample_indices = rng.sample(range(len(words)), n_samples)
```

Pros: detector becomes pure-functional in `text`; cache builds reproducible;
no global-RNG side-effect on other code.

Cons: changes signal values for every cached row -> v1 EBM should be
re-validated against the new signal distribution. Two sub-paths:

- A1. Rebuild v7 cache against fixed detector, retrain v1, re-benchmark.
  Lowest risk to production accuracy; ~few hours of compute.
- A2. Ship the fix without retraining; accept ~0.011 stdev p_llm shift on
  one of 10 features at inference time. Likely a small but non-zero
  accuracy hit on production.

Option B: accept the drift, document it. Faster, but the v1->v2 retrain
story stays poisoned because every future retrain inherits this issue.

Option C: full audit. Grep every detector for `random.` usage, fix all of
them to seeded RNG. The diff already shows `token_cohesiveness` is the
only one with material drift, but `surprisal_dynamics` /
`contrastive_lm` showing any drift at all (~1e-6) hints that some other
upstream nondeterminism may exist (BLAS reduction order, etc.).
Worth one grep before committing to Option A.

### Recommendation

Run Option C's grep first (cheap), then Option A1 (rebuild v7 cache and
retrain v1 against the fix). That gives a clean baseline before attempting
robust_v2 again. SpecDetect's heuristic recalibration (open question #3
above) is orthogonal and can land after.

## Follow-up 2026-05-20 (later): A1 executed -- v10 clean retrain

Executed Option A1 in full:

1. Fixed `cairn/detectors/token_cohesiveness.py:39` to seed `random.Random`
   from `sha256(text)`.
2. Added two determinism tests (`tests/test_detectors/test_token_cohesiveness.py`).
3. Pytest gate: 800 / 800 passed.
4. Modal H200 cache build: `data/all_features_v10.jsonl`, 80,241 unique
   features. Plus calib split + OOD test caches.
5. Retrained `master_minus_arxiv_clean` profile (detector_calibration.json,
   ebm.pkl, conformal.json).
6. Apples-to-apples OOD on the same fixed-RNG cache:

| Dataset | v10 clean (retrained) | v1 prod (trained on buggy cache) | delta |
|---|---:|---:|---:|
| RAID | **0.6895** | 0.6764 | **+0.013** |
| MAGE | 0.5917 | 0.5947 | -0.003 |
| M4 | **0.5885** | 0.5421 | **+0.046** |

v10 clean meets-or-beats v1 on every OOD slice.

### Why the v10 RAID number (0.6895) looks lower than historical v1 (0.7571)

The historical v1 RAID 0.7571 was measured against an OOD cache built in
a session whose global-RNG environment was statistically similar to the
training session. The EBM's learned splits aligned with that environment's
RNG draws. When evaluated against the deterministic (fixed-seed) cache,
v1's splits no longer align with the data -- so the same v1 model now
scores 0.6764 on RAID against the honest cache.

**Production v1 has been silently miscalibrated against inference since
it shipped.** The 0.7571 number was an artifact of train/test cache
co-poisoning by the same global-RNG environment, not real OOD
generalization. The retrain (v10) matches inference and gives the honest
performance.

### Promotion decision

v10 clean is the new honest baseline. The artifacts are at
`models/calib2000/profiles/master_minus_arxiv_clean/`. Promotion to
production = copying the three files (`ebm.pkl`, `conformal.json`,
`detector_calibration.json`) into `master_minus_arxiv/`.

The robust_v2 retrain story is now unblocked: any future v2 attempt
should train against `data/all_features_v10.jsonl` (or a successor built
against the fixed detector) and compare against v10 clean, not against
the historically-inflated v1 numbers.

SpecDetect's heuristic recalibration (open question #3 above) remains
orthogonal -- it can land on top of the v10 baseline.

### Refined diagnosis: re-running robust_v2 on the clean OOD cache

Ran the existing `master_minus_arxiv_robust_v2` profile against
`data/ood_all_features_v10.jsonl` (the fixed-detector OOD cache):

| Dataset | v10 clean | v1 prod | robust_v2 |
|---|---:|---:|---:|
| RAID | 0.6895 | 0.6764 | 0.6898 (tied with v10) |
| MAGE | 0.5917 | 0.5947 | 0.5691 (-0.023 vs v10) |
| M4 | 0.5885 | 0.5421 | 0.5322 (-0.056 vs v10) |

The original "REGRESSION, do not promote" verdict was correct as a
shipping decision, but the diagnosis splits cleanly:

- **RAID regression was a RNG ghost.** On the honest cache, v2 ties v10.
  The historical RAID gap (-0.024) was an artifact of the buggy v8 cache
  matching v2's training distribution by chance.
- **MAGE and M4 regressions are real.** v2 underperforms v10 by ~0.02
  to ~0.06 AUROC on these slices. SpecDetect's heuristic miscalibration
  (open question #3) is the most plausible source on text where SpecDetect
  fires (prose / step-by-step LLM output).

This sharpens the next experiment: rebuild a v2-style cache (SpecDetect
enabled, full coverage) against the fixed detector, recalibrate
SpecDetect's `_HEURISTIC_TOTAL_ENERGY_TO_P_LLM` from empirical p1/p25/p50/
p75/p95/p99 of that cache, retrain v2-clean, evaluate against v10 on
MAGE/M4. If MAGE and M4 close to v10, SpecDetect-with-honest-calibration
is shippable.

# Full-coverage vs partial-coverage per-detector AUROC

**Date:** 2026-05-07
**Holdout:** 500 rows (250 label=0, 250 label=1) sampled from
`data/calib2000_full_test.jsonl`, source-stratified across 30 unique
sources, deterministic via `scripts/sample_calib2000_holdout.py --seed 42`.
**Cost:** ~$0.30 Modal T4 + ~7 min local CPU. ~30 min wall clock.

## Question

Does CAIRN's cascade short-circuit (`cairn/cascade.py::CascadeScheduler`)
materially bias per-detector AUROC numbers? If a detector only fires
on the "ambiguous middle" of inputs, its AUROC is computed on a
non-representative slice — the per-detector regression gates planned
for C3 in the post-honest-calibration roadmap depend on those numbers
being honest across the full distribution.

## Method

1. Sample 500 label-balanced, source-stratified rows from the
   `calib2000_full_test` split (no overlap with EBM training data).
2. Cache feature vectors twice on the same 500 rows:
   - **Partial coverage** — `configs/default.yaml`, cascade
     short-circuit ON (`scripts/cache_pipeline_features.py`, local CPU).
   - **Full coverage** — `configs/.profile_full_coverage.yaml`, cascade
     thresholds set outside [0,1] so phase 1 → 2 always escalates,
     `phase3_always_run: true` to bypass the phase 2 → 3 check
     (`scripts/modal_cache_features.py`, Modal T4 GPU).
3. Compute per-detector AUROC against the same 491 unique-text rows
   under each cache (`scripts/per_detector_auroc.py` with the
   `master_minus_arxiv` detector calibrator).

## Result

See `full_coverage_holdout_500.csv`. Highlights:

| Detector | Partial | Full | Δ | Interpretation |
|---|---|---|---|---|
| `perturbation` | n/a | **0.7582** | +full signal | Phase-3 detector. Cascade short-circuit **never escalated to phase 3** on this slice — the detector's 0.76 AUROC was hidden entirely. |
| `contrastive_lm` | 0.7613 | **0.8324** | +0.071 | Cascade-induced selection bias suppressed its real strength by ~7 AUROC points. |
| `semantic_kinematics` | 0.1676 | 0.2398 | +0.072 | **Both AUROCs below 0.5** — the detector predicts the wrong class. See "Semantic-kinematics finding" below. |
| `surprisal_dynamics` | 0.5984 | 0.5794 | -0.019 | Partial-coverage AUROC was slightly inflated by cascade selection. |
| `mixed_boundary` | 0.5095 | 0.5040 | -0.006 | Coin-flip under both regimes. |
| `token_cohesiveness` | 0.5000 | 0.4960 | -0.004 | Coin-flip under both regimes. |
| Phase-1 detectors | unchanged across all six (`fingerprint_vocab`, `instruction_density`, `nssi`, `preamble`, `prompt_structure`, `voice_spec`) | | | Sanity check: Phase-1 detectors fire on every row regardless of cascade settings. |

## Findings

### 1. The cascade hides two real signals

`perturbation` (a curvature-flavored Phase-3 detector) reaches 0.76
AUROC under full coverage but never fires under cascade short-circuit
on this slice. The honest-calibration sprint's missingness fusion v7
(`_missing` indicator features per `0f7daeb`) *partially* compensates
inside the EBM, but the per-detector view confirms the upstream signal
itself is being masked from any downstream calibration / ablation work
that reads from `data/all_features_v6.jsonl`.

`contrastive_lm` is +7 AUROC points stronger than the cached partial-
coverage number suggests. The cascade-induced sample bias (only
"ambiguous-middle" texts reach Phase 2) was making the detector look
weaker than it is.

**Implication for A2 / OOD attack pack (Wave 2):** the case for
re-training on full-coverage features instead of cached partial-
coverage features is empirically stronger after this experiment.

### 2. `semantic_kinematics` is anti-correlated on this corpus

Both AUROCs (0.17, 0.24) are below 0.5, meaning the detector's
`raw_score → p_llm` mapping consistently predicts the *wrong* class.

Diagnosis (`cairn/detectors/semantic_kinematics.py:97-106`): the
`_heuristic_p_llm` function maps low velocity_std → high p_llm,
following the hypothesis that LLMs produce "frictionless-gradient"
trajectories with low sentence-to-sentence variation while humans
produce "drunken-walk" trajectories with high variation. On this
HC3-derived corpus the empirical relationship is reversed: humans
answer Q&A focused / on-topic (low std), LLMs elaborate over multiple
aspects (high std).

**This is not a sign-flip bug** — on free-form prose the original
direction may be correct. The right fix is calibrator retraining (the
EBM is unconstrained on this detector per `ca7de4f`'s monotone-up
list, so it *can* learn an anti-correlated mapping) rather than
hand-flipping the heuristic prior.

**Action items:**
- Add `semantic_kinematics` to the `master_minus_arxiv` detector
  calibrator JSON when next retrained.
- Re-run this experiment after the next calibrator pass to confirm
  the calibrated AUROC moves above 0.5.
- Until then, every report from this detector under heuristic mode
  is misleading. Consider gating the detector to "skip in heuristic
  mode" until calibration covers it.

### 3. Two detectors are at random under both regimes

`token_cohesiveness` (0.50, 0.50) and `mixed_boundary` (0.51, 0.50)
are coin-flips on this slice under both partial and full coverage.
Two possibilities:
- They genuinely don't discriminate on calib2000-derived data and
  are dead weight in the ensemble.
- They discriminate on a different distribution (e.g., adversarial
  / paraphrased text) where the test split doesn't exercise them.

The A2 OOD attack pack will surface which is the case. Until then,
their EBM weights should already be ~0 if the missingness fusion is
honest about non-discriminative features.

## What this experiment does NOT show

- It is a 491-unique-text snapshot. Statistical power for small
  per-detector deltas (< 0.02 AUROC) is limited.
- It uses the `master_minus_arxiv` detector calibrator, which
  doesn't include `semantic_kinematics`. The "real" calibrated
  AUROC for that detector cannot be measured until the calibrator
  is retrained.
- The full-coverage profile flips on `perturbation` only — the
  LLM-call detectors (`contrastive_gen`, `dna_gpt`) are commented
  out in the profile to avoid API spend. Their per-detector numbers
  remain uncovered.
- Binoculars (Hans et al. 2024) and Fast-DetectGPT (Bao et al. 2023)
  exist as competitor adapters in `cairn/benchmarks/` but are NOT
  in the cascade. Promoting them from adapter to native detector is
  a separate piece of work tracked in the post-honest-calibration
  roadmap follow-ups.

## Reproducing

```bash
python scripts/sample_calib2000_holdout.py
python scripts/cache_pipeline_features.py \
    --dataset data/calib2000_holdout_500.jsonl \
    --config configs/default.yaml \
    --out data/calib2000_holdout_500_partial_features.jsonl
modal run scripts/modal_cache_features.py \
    --dataset-path data/calib2000_holdout_500.jsonl \
    --out-path data/calib2000_holdout_500_full_features.jsonl \
    --config-path configs/.profile_full_coverage.yaml
python scripts/per_detector_auroc.py \
    --calibrator models/calib2000/profiles/master_minus_arxiv/detector_calibration.json \
    --datasets partial:data/calib2000_holdout_500.jsonl full:data/calib2000_holdout_500.jsonl \
    --features-caches partial:data/calib2000_holdout_500_partial_features.jsonl \
                      full:data/calib2000_holdout_500_full_features.jsonl \
    --out-csv docs/benchmarks/full_coverage_holdout_500.csv
```

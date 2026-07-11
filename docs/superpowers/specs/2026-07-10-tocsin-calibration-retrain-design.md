# TOCSIN Calibration Retrain Design

**Date:** 2026-07-10
**Status:** Approved design; implementation pending written-spec review
**Target:** A versioned `prompts_only_tocsin_v2` candidate profile. The existing `prompts_only` profile remains unchanged unless the candidate later passes every promotion gate.

## Problem statement

The deployed `prompts_only` TOCSIN calibration is not compatible with the current detector implementation.

- The May 1 calibrator was fitted to the retired MiniLM deletion-impact score under the `token_cohesiveness` detector ID. Its isotonic support is approximately `0.00077275` to `0.01604344`.
- The current faithful detector emits BARTScore `cohesiveness_u`, Fast-DetectGPT `base_v`, and their `combined_w`; when the base channel is available, `combined_w` is the layer `raw_score`.
- Current scores on the July 10 challenge set span approximately `-0.351` to `0.092`. Because the old isotonic model clips out-of-range inputs, it collapses the current detector into three high probability values and materially contributes to Human false positives.
- The current `prompts_only` EBM is stale as well. It consumes `tocsin_p_llm` and the obsolete `tocsin_mean_deletion_impact`, but it does not consume `tocsin_cohesiveness_u` or `tocsin_combined_w`.
- Existing feature caches contain the retired TOCSIN schema. No cache under `data/` can train or validate the current TOCSIN implementation without a fresh scoring pass.

Replacing only `detector_calibration.json` would therefore feed a new probability distribution into an EBM trained on the old one. The safe unit of change is a matched detector calibrator, EBM, and conformal wrapper.

## Goals

1. Prevent stale detector calibrators from silently clipping scores produced by a different detector implementation or raw channel.
2. Produce fresh, current-schema TOCSIN features on provenance-controlled prompt corpora.
3. Fit a source- and length-weighted TOCSIN calibrator without using the July 10 challenge labels for training or calibration.
4. Retrain a matched prompt-domain EBM that consumes the current TOCSIN channels.
5. Recalibrate conformal prediction on a disjoint calibration corpus.
6. Evaluate a new candidate against the current profile on locked prompt, humanized, OOD, and challenge sets.
7. Keep Modal compute within the user-approved `$15` soft cap.

## Non-goals

- Do not overwrite `models/calib2000/profiles/prompts_only/`.
- Do not fit any artifact on the 14 labelled July 10 rows.
- Do not reinterpret `AI-Edited` as equivalent to untouched LLM output; the supplied rows remain a challenge set for the current binary target.
- Do not enable API-backed detectors during feature generation.
- Do not promote an in-sample threshold such as `0.836` from the July 10 analysis.
- Do not claim production readiness without a fresh, group-disjoint AI-edited-versus-Human benchmark.

## Approaches considered

### 1. Matched candidate-profile retrain — selected

Build fresh current-schema caches, fit a versioned TOCSIN calibration entry, retrain the EBM, recalibrate conformal prediction, and validate the coherent trio. This is the only approach that avoids feeding a changed TOCSIN probability distribution into a legacy EBM.

### 2. Replace only the TOCSIN JSON entry — rejected for deployment

This is useful only as a detector-level diagnostic. It is not deployable because the old EBM learned shape functions for the retired calibrated probability and obsolete raw feature.

### 3. Fit on the 14 supplied rows — rejected

The rows contain only five correlated scenario families, were already used for profile selection and diagnosis, and include a cross-label near-duplicate pair. Fitting on them would create adaptive test leakage and unstable isotonic knots.

## Data design

### Training corpus

Use `data/prompts_v2_train.jsonl`:

- `20,732` rows, balanced at `10,366` per binary class.
- `2,799` positive rows are `tier=humanized`, providing the best available local proxy for human-like or edited AI prompts.
- At the current TOCSIN minimum length of 60 words, `8,484` rows are eligible: `3,968` positive and `4,516` Human.
- The corpus has no reliable lineage-level `source_id`. The calibrator fit must therefore weight source and length strata so MedQA Humans or a small set of positive generators cannot dominate the mapping.

Use four deterministic word-count bins: `[60, 100)`, `[100, 150)`, `[150, 250)`, and `[250, infinity)`. For each bin, assign the Human and positive classes equal total weight proportional to the smaller class count in that bin. Within each `(label, length_bin)` stratum, start with inverse-source-frequency row weights, clip row weights to at most 10 times that stratum's median weight, and renormalize to the stratum target. Record the resulting effective class, bin, and source weights in the training manifest.

The EBM trains on the full corpus, including explicit TOCSIN missingness for rows below the 60-word floor. The TOCSIN isotonic fit uses only active, finite, current-signature TOCSIN rows.

### Conformal calibration corpus

Use `data/prompts_only_calib.jsonl`:

- `3,660` balanced rows, disjoint by exact normalized text from the training corpus.
- `1,026` rows meet the TOCSIN length floor: `272` positive and `754` Human.
- Preserve this corpus exclusively for conformal calibration and calibration-quality reporting; do not add it to EBM training.

### Locked evaluation corpora

1. `data/prompts_v5_test.jsonl`: `6,098` balanced prompt rows; the primary in-domain test.
2. `data/_humanized_eval.jsonl` plus `data/_hard_human_clean.jsonl`: humanized-positive and leak-free long/dense Human regression slices.
3. `data/ood_m4_test.jsonl`, `data/ood_raid_test.jsonl`, and `data/ood_mage_test.jsonl`: cross-domain regression tests only.
4. The July 10 labelled workbook rows: `14` labelled examples across five scenario families. These remain an adaptive challenge set and are reported separately from promotion metrics.

## Detector and calibration contract

### Current TOCSIN signatures

The detector must surface an explicit calibration signature in `LayerResult.signals`:

- Base scorer available: `tocsin-bart-fastdetectgpt-combined-w-v1`, raw channel `combined_w`.
- Base scorer unavailable: `tocsin-bart-cohesiveness-u-v1`, raw channel `cohesiveness_u`.

A calibrator fitted to one signature must never apply to the other.

### Versioned calibration entry

The new TOCSIN calibration entry records:

- artifact schema version;
- detector implementation signature;
- raw channel name;
- fitted support minimum and maximum;
- training-row count and class counts;
- training dataset hash;
- source/length weighting policy;
- fitted isotonic knots.

Legacy detector entries remain loadable. If a result supplies a signature and the selected calibration entry lacks that signature or names a different signature, calibration is rejected for that result.

### Centralized application guard

Live inference, `CachedPipeline`, and analysis/replay code must use one calibration-application function. It must:

1. ignore `SKIP` results;
2. require a finite raw score;
3. require a matching detector signature when one is present;
4. reject scores outside the artifact's supported domain plus the established tolerance;
5. leave `calibrator_applied=False` on rejection;
6. surface `calibrator_signature_mismatch` or `calibrator_out_of_domain` in signals and make the detector missing to fusion rather than clipping it.

This guard prevents recurrence even if a detector ID is reused after another algorithm change.

## GPU scoring design

The faithful TOCSIN implementation currently loads BART and its tensors on CPU. Before remote scoring:

1. select `cuda` when available and otherwise use CPU;
2. move the BART model and every encoded tensor to the selected device;
3. configure the shared Fast-DetectGPT adapter for the same device;
4. preserve CPU-only behavior and lazy optional-dependency loading;
5. add tests that use lightweight fakes rather than downloading model weights.

Feature generation uses `configs/.profile_full_coverage_with_specdetect.yaml`. API-backed layers remain disabled. Unique Modal run IDs prevent accidental resume from stale `v8` checkpoints.

### Cost gate

Run a 256-row throughput pilot first. Use its observed eligible-row and total-row throughput to project the complete train, calibration, and evaluation cost. Proceed only when projected cumulative Modal spend is at most `$15`. If the projection exceeds the cap, terminate before the full pass and report the measured estimate.

The full run may use one H200 or a cheaper GPU/shard layout, but the selected layout must satisfy the measured cost gate rather than a hard-coded speed assumption.

## Training flow

1. Generate fresh current-code feature caches for train, conformal calibration, and locked evaluation corpora.
2. Verify cache coverage by normalized-text hash, current TOCSIN signature, active/SKIP counts, raw ranges, and absence of API results.
3. Fit only the TOCSIN detector-calibration entry using active training rows with the specified source/length weights.
4. Combine that entry with the unchanged non-TOCSIN entries from `models/calib2000/profiles/prompts_specdetect/detector_calibration.json` so TOCSIN is the isolated detector-calibration change.
5. Train a new EBM on `prompts_v2_train.jsonl` with the matched candidate calibrator and current full-coverage feature cache.
6. Confirm the EBM feature schema contains `tocsin_p_llm`, `tocsin_cohesiveness_u`, `tocsin_combined_w`, and `tocsin_missing`, and does not require `tocsin_mean_deletion_impact`.
7. Fit a new conformal wrapper on `prompts_only_calib.jsonl` using the candidate EBM and matched calibration cache.
8. Write the candidate under `models/calib2000/profiles/prompts_only_tocsin_v2/` with a dedicated `configs/.profile_prompts_only_tocsin_v2.yaml` extending the full-coverage prompt recipe and retaining `word_count_clamp: 200`.

## Evaluation and promotion gates

### Detector-level gates

On eligible locked prompt-test rows, compare the current raw detector, stale calibration, and new calibration. The candidate must:

- have finite outputs in `[0, 1]`;
- produce more than three distinct calibrated levels;
- improve Brier score and log loss over the stale calibrator;
- improve Brier score over the constant-prevalence predictor;
- avoid a calibration curve collapsed at either endpoint;
- report AUROC, ECE, class means, support coverage, and active/SKIP counts even though monotonic calibration does not itself improve rank ordering.

### End-to-end gates

Relative to the existing `prompts_only` and `prompts_specdetect` comparators:

- in-domain `prompts_v5_test` Brier score and log loss must not regress;
- balanced accuracy and AUROC must not regress by more than `0.01` absolute;
- Human false-positive rate must not increase;
- humanized-versus-clean-hard-Human AUROC and TPR at 5% and 10% FPR must not regress by more than `0.02` absolute;
- OOD AUROC must not regress by more than `0.02` on any one locked corpus without a documented compensating improvement;
- conformal empirical truth coverage must be within its binomial uncertainty interval around the configured 95% target, and singleton/selective accuracy must be reported;
- every EBM result must use `fusion_mode=ebm`, have finite probability, and reconstruct from intercept plus term contributions within `1e-9`.

The July 10 challenge results are secondary because the set already informed this retrain. Report Brier, balanced accuracy, Human specificity, conformal coverage, and per-task changes, but do not use the set as the sole promotion gate.

### Promotion decision

- **Pass:** retain the candidate artifacts, publish the comparison report, and recommend a separate deployment review. Do not automatically replace `prompts_only`.
- **Fail:** retain the diagnostic report and training manifest, mark the candidate `DO_NOT_PROMOTE`, and leave all production profile paths unchanged.

## Test strategy

Follow red-green-refactor for each behavior change.

1. Calibration unit tests prove a current TOCSIN result rejects the old artifact by signature and by support instead of clipping.
2. Live-pipeline and `CachedPipeline` tests prove identical application, rejection, SKIP, and signal-surfacing behavior.
3. TOCSIN device tests prove the model and encoded tensors select CUDA when available and CPU otherwise without loading real weights.
4. Cache validation tests reject legacy TOCSIN fields and mixed-signature caches.
5. Training-script tests prove `--only-detector token_cohesiveness` preserves all non-TOCSIN calibration entries and excludes SKIP rows.
6. Candidate-artifact tests load the EBM, conformal wrapper, and detector calibrator through the normal profile loader.
7. Targeted tests run before the full suite; `pytest -q` is the final repository gate.

## Artifacts and reporting

Store run outputs under `outputs/tocsin_retrain_2026-07-10/`:

- immutable input and cache manifests with SHA-256 hashes;
- Modal pilot and full-run manifests, elapsed time, GPU type, and estimated cost;
- current-versus-candidate detector calibration metrics;
- end-to-end benchmark JSON and long-form CSV;
- July 10 per-task comparison;
- a concise Markdown promotion report stating `PASS` or `DO_NOT_PROMOTE`.

Large feature caches remain untracked under `outputs/`. Candidate model artifacts and profile configuration are versioned only after their integrity checks pass. Production artifacts are never modified in place.

## Failure handling and rollback

- Stop before the full Modal run if the pilot projects spend above `$15`.
- Abort a cache if it contains the retired `mean_deletion_impact` schema for current TOCSIN rows, mixed signatures, missing hashes, duplicate hashes with conflicting results, or unexpected API activity.
- Abort training if source/length balancing leaves either class with fewer than 500 active TOCSIN examples.
- Abort candidate loading if artifact hashes, HMAC verification, EBM schema, or conformal calibration status fail.
- Rollback is path-based: continue using `configs/.profile_prompts_only.yaml`. No production file needs restoration because the candidate uses separate paths.

## Privacy and reproducibility

- No submission text is sent to LLM APIs during calibration.
- Modal receives the approved calibration corpora for local model inference inside the project job; API-backed CAIRN detectors remain disabled.
- Seeds, package versions, GPU type, dataset hashes, source/length weights, and artifact hashes are recorded in the run manifest.
- The 14 user-labelled rows are retained only in the existing local output directory and are not incorporated into training data.

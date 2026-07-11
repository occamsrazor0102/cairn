# DO_NOT_PROMOTE

## Preflight

- Candidate `prompts_only_tocsin_v2`: VALID
- Comparator `prompts_only`: VALID
- Comparator `prompts_specdetect`: INVALID

## Automatic promotion gates

- FAIL: INVALID_COMPARATOR prompts_specdetect:; missing models/calib2000/profiles/prompts_specdetect/ebm.skops; missing models/calib2000/profiles/prompts_specdetect/ebm.skops.hmac
- FAIL: detector: candidate Brier must beat constant prevalence
- FAIL: prompts_only prompt_test Human FPR: candidate regressed
- FAIL: prompts_only ood_m4_test auroc: regression exceeds 0.02
- FAIL: prompts_only ood_raid_test auroc: regression exceeds 0.02
- FAIL: prompts_only ood_mage_test auroc: regression exceeds 0.02

## Detector calibration

- Eligible rows: 1822
- Detector SKIP rows: 4276
- Support coverage: 1.000000
- Raw combined_w range: -49.912093 to 0.393239
- Candidate Brier: 0.249355
- Legacy unsafe clipped diagnostic Brier: 0.618695

## Adaptive July 10 challenge (secondary)

Not run: `invalid_comparator_avoided_additional_compute`.

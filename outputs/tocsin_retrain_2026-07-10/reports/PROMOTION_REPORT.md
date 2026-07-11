# DO_NOT_PROMOTE

## Preflight

- Candidate `prompts_only_tocsin_v2`: VALID
- Comparator `prompts_only`: VALID
- Comparator `prompts_specdetect`: INVALID

## Automatic promotion gates

| Status | Gate ID | Scope | Profile | Slice | Observed | Operator | Reference | Detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | `preflight.comparator_identity` | preflight | NA | NA | true | is | true |  |
| PASS | `preflight.candidate_artifacts` | preflight | prompts_only_tocsin_v2 | NA | true | is | true |  |
| PASS | `preflight.comparator.prompts_only.artifacts` | preflight | prompts_only | NA | true | is | true |  |
| FAIL | `preflight.comparator.prompts_specdetect.artifacts` | preflight | prompts_specdetect | NA | false | is | true | INVALID_COMPARATOR prompts_specdetect:<br>missing models/calib2000/profiles/prompts_specdetect/ebm.skops<br>missing models/calib2000/profiles/prompts_specdetect/ebm.skops.hmac |
| PASS | `detector.eligible_rows` | detector | token_cohesiveness | prompt_test | 1822 | == | 1822 |  |
| PASS | `detector.support_coverage` | detector | token_cohesiveness | prompt_test | 1.0 | == | 1.0 |  |
| PASS | `detector.candidate.finite` | detector | token_cohesiveness | prompt_test | true | is | true |  |
| PASS | `detector.candidate.in_unit_interval` | detector | token_cohesiveness | prompt_test | true | is | true |  |
| PASS | `detector.candidate.n_unique` | detector | token_cohesiveness | prompt_test | 5 | > | 3 |  |
| PASS | `detector.candidate.brier_vs_legacy` | detector | token_cohesiveness | prompt_test | 0.2493550781715259 | < | 0.6186946433314154 |  |
| PASS | `detector.candidate.log_loss_vs_legacy` | detector | token_cohesiveness | prompt_test | 0.6925921704459008 | < | 17.266872555143678 |  |
| FAIL | `detector.candidate.brier_vs_constant` | detector | token_cohesiveness | prompt_test | 0.2493550781715259 | < | 0.2030000445825567 | detector: candidate Brier must beat constant prevalence |
| PASS | `detector.candidate.endpoint_fraction` | detector | token_cohesiveness | prompt_test | 0.0005488474204171241 | < | 0.95 |  |
| PASS | `candidate.prompt_test.all_ebm` | candidate | prompts_only_tocsin_v2 | prompt_test | true | is | true |  |
| PASS | `candidate.prompt_test.finite` | candidate | prompts_only_tocsin_v2 | prompt_test | true | is | true |  |
| PASS | `candidate.prompt_test.in_unit_interval` | candidate | prompts_only_tocsin_v2 | prompt_test | true | is | true |  |
| PASS | `candidate.prompt_test.max_ebm_reconstruction_error` | candidate | prompts_only_tocsin_v2 | prompt_test | 5.551115123125783e-16 | <= | 1e-09 |  |
| PASS | `candidate.humanized_hard.all_ebm` | candidate | prompts_only_tocsin_v2 | humanized_hard | true | is | true |  |
| PASS | `candidate.humanized_hard.finite` | candidate | prompts_only_tocsin_v2 | humanized_hard | true | is | true |  |
| PASS | `candidate.humanized_hard.in_unit_interval` | candidate | prompts_only_tocsin_v2 | humanized_hard | true | is | true |  |
| PASS | `candidate.humanized_hard.max_ebm_reconstruction_error` | candidate | prompts_only_tocsin_v2 | humanized_hard | 5.551115123125783e-16 | <= | 1e-09 |  |
| PASS | `candidate.ood_m4_test.all_ebm` | candidate | prompts_only_tocsin_v2 | ood_m4_test | true | is | true |  |
| PASS | `candidate.ood_m4_test.finite` | candidate | prompts_only_tocsin_v2 | ood_m4_test | true | is | true |  |
| PASS | `candidate.ood_m4_test.in_unit_interval` | candidate | prompts_only_tocsin_v2 | ood_m4_test | true | is | true |  |
| PASS | `candidate.ood_m4_test.max_ebm_reconstruction_error` | candidate | prompts_only_tocsin_v2 | ood_m4_test | 4.440892098500626e-16 | <= | 1e-09 |  |
| PASS | `candidate.ood_raid_test.all_ebm` | candidate | prompts_only_tocsin_v2 | ood_raid_test | true | is | true |  |
| PASS | `candidate.ood_raid_test.finite` | candidate | prompts_only_tocsin_v2 | ood_raid_test | true | is | true |  |
| PASS | `candidate.ood_raid_test.in_unit_interval` | candidate | prompts_only_tocsin_v2 | ood_raid_test | true | is | true |  |
| PASS | `candidate.ood_raid_test.max_ebm_reconstruction_error` | candidate | prompts_only_tocsin_v2 | ood_raid_test | 4.718447854656915e-16 | <= | 1e-09 |  |
| PASS | `candidate.ood_mage_test.all_ebm` | candidate | prompts_only_tocsin_v2 | ood_mage_test | true | is | true |  |
| PASS | `candidate.ood_mage_test.finite` | candidate | prompts_only_tocsin_v2 | ood_mage_test | true | is | true |  |
| PASS | `candidate.ood_mage_test.in_unit_interval` | candidate | prompts_only_tocsin_v2 | ood_mage_test | true | is | true |  |
| PASS | `candidate.ood_mage_test.max_ebm_reconstruction_error` | candidate | prompts_only_tocsin_v2 | ood_mage_test | 4.440892098500626e-16 | <= | 1e-09 |  |
| PASS | `candidate.prompt_test.conformal_truth_coverage_count` | candidate | prompts_only_tocsin_v2 | prompt_test | 5816 | between_inclusive | [5759,5826] |  |
| PASS | `comparator.prompts_only.prompt_test.all_ebm` | comparator | prompts_only | prompt_test | true | is | true |  |
| PASS | `comparator.prompts_only.prompt_test.finite` | comparator | prompts_only | prompt_test | true | is | true |  |
| PASS | `comparator.prompts_only.prompt_test.in_unit_interval` | comparator | prompts_only | prompt_test | true | is | true |  |
| PASS | `comparator.prompts_only.prompt_test.max_ebm_reconstruction_error` | comparator | prompts_only | prompt_test | 2.220446049250313e-16 | <= | 1e-09 |  |
| PASS | `comparator.prompts_only.humanized_hard.all_ebm` | comparator | prompts_only | humanized_hard | true | is | true |  |
| PASS | `comparator.prompts_only.humanized_hard.finite` | comparator | prompts_only | humanized_hard | true | is | true |  |
| PASS | `comparator.prompts_only.humanized_hard.in_unit_interval` | comparator | prompts_only | humanized_hard | true | is | true |  |
| PASS | `comparator.prompts_only.humanized_hard.max_ebm_reconstruction_error` | comparator | prompts_only | humanized_hard | 3.3306690738754696e-16 | <= | 1e-09 |  |
| PASS | `comparator.prompts_only.ood_m4_test.all_ebm` | comparator | prompts_only | ood_m4_test | true | is | true |  |
| PASS | `comparator.prompts_only.ood_m4_test.finite` | comparator | prompts_only | ood_m4_test | true | is | true |  |
| PASS | `comparator.prompts_only.ood_m4_test.in_unit_interval` | comparator | prompts_only | ood_m4_test | true | is | true |  |
| PASS | `comparator.prompts_only.ood_m4_test.max_ebm_reconstruction_error` | comparator | prompts_only | ood_m4_test | 2.220446049250313e-16 | <= | 1e-09 |  |
| PASS | `comparator.prompts_only.ood_raid_test.all_ebm` | comparator | prompts_only | ood_raid_test | true | is | true |  |
| PASS | `comparator.prompts_only.ood_raid_test.finite` | comparator | prompts_only | ood_raid_test | true | is | true |  |
| PASS | `comparator.prompts_only.ood_raid_test.in_unit_interval` | comparator | prompts_only | ood_raid_test | true | is | true |  |
| PASS | `comparator.prompts_only.ood_raid_test.max_ebm_reconstruction_error` | comparator | prompts_only | ood_raid_test | 2.220446049250313e-16 | <= | 1e-09 |  |
| PASS | `comparator.prompts_only.ood_mage_test.all_ebm` | comparator | prompts_only | ood_mage_test | true | is | true |  |
| PASS | `comparator.prompts_only.ood_mage_test.finite` | comparator | prompts_only | ood_mage_test | true | is | true |  |
| PASS | `comparator.prompts_only.ood_mage_test.in_unit_interval` | comparator | prompts_only | ood_mage_test | true | is | true |  |
| PASS | `comparator.prompts_only.ood_mage_test.max_ebm_reconstruction_error` | comparator | prompts_only | ood_mage_test | 1.6653345369377348e-16 | <= | 1e-09 |  |
| PASS | `comparator.prompts_only.prompt_test.brier` | comparator | prompts_only | prompt_test | 0.11969087765849079 | <= | 0.16873460543911678 |  |
| PASS | `comparator.prompts_only.prompt_test.log_loss` | comparator | prompts_only | prompt_test | 0.3771571715517195 | <= | 0.5242738183610238 |  |
| PASS | `comparator.prompts_only.prompt_test.balanced_accuracy` | comparator | prompts_only | prompt_test | 0.832896031485733 | >= | 0.7645162348310921 |  |
| PASS | `comparator.prompts_only.prompt_test.auroc` | comparator | prompts_only | prompt_test | 0.9103609558150515 | >= | 0.8123902454293872 |  |
| FAIL | `comparator.prompts_only.prompt_test.human_fpr` | comparator | prompts_only | prompt_test | 0.1774352246638242 | <= | 0.15874057067891112 | prompts_only prompt_test Human FPR: candidate regressed |
| PASS | `comparator.prompts_only.humanized_hard.auroc` | comparator | prompts_only | humanized_hard | 0.9179583995760466 | >= | 0.5341202967673556 |  |
| PASS | `comparator.prompts_only.humanized_hard.tpr_at_fpr_05` | comparator | prompts_only | humanized_hard | 0.6372549019607843 | >= | 0.0976470588235294 |  |
| PASS | `comparator.prompts_only.humanized_hard.tpr_at_fpr_10` | comparator | prompts_only | humanized_hard | 0.7058823529411765 | >= | 0.15647058823529414 |  |
| FAIL | `comparator.prompts_only.ood_m4_test.auroc` | comparator | prompts_only | ood_m4_test | 0.5334 | >= | 0.607 | prompts_only ood_m4_test auroc: regression exceeds 0.02 |
| FAIL | `comparator.prompts_only.ood_raid_test.auroc` | comparator | prompts_only | ood_raid_test | 0.5489 | >= | 0.6476000000000001 | prompts_only ood_raid_test auroc: regression exceeds 0.02 |
| FAIL | `comparator.prompts_only.ood_mage_test.auroc` | comparator | prompts_only | ood_mage_test | 0.43839999999999996 | >= | 0.4992 | prompts_only ood_mage_test auroc: regression exceeds 0.02 |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.all_ebm` | comparator | prompts_specdetect | prompt_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.finite` | comparator | prompts_specdetect | prompt_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.in_unit_interval` | comparator | prompts_specdetect | prompt_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.max_ebm_reconstruction_error` | comparator | prompts_specdetect | prompt_test | NA | <= | 1e-09 | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.humanized_hard.all_ebm` | comparator | prompts_specdetect | humanized_hard | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.humanized_hard.finite` | comparator | prompts_specdetect | humanized_hard | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.humanized_hard.in_unit_interval` | comparator | prompts_specdetect | humanized_hard | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.humanized_hard.max_ebm_reconstruction_error` | comparator | prompts_specdetect | humanized_hard | NA | <= | 1e-09 | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_m4_test.all_ebm` | comparator | prompts_specdetect | ood_m4_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_m4_test.finite` | comparator | prompts_specdetect | ood_m4_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_m4_test.in_unit_interval` | comparator | prompts_specdetect | ood_m4_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_m4_test.max_ebm_reconstruction_error` | comparator | prompts_specdetect | ood_m4_test | NA | <= | 1e-09 | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_raid_test.all_ebm` | comparator | prompts_specdetect | ood_raid_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_raid_test.finite` | comparator | prompts_specdetect | ood_raid_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_raid_test.in_unit_interval` | comparator | prompts_specdetect | ood_raid_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_raid_test.max_ebm_reconstruction_error` | comparator | prompts_specdetect | ood_raid_test | NA | <= | 1e-09 | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_mage_test.all_ebm` | comparator | prompts_specdetect | ood_mage_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_mage_test.finite` | comparator | prompts_specdetect | ood_mage_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_mage_test.in_unit_interval` | comparator | prompts_specdetect | ood_mage_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_mage_test.max_ebm_reconstruction_error` | comparator | prompts_specdetect | ood_mage_test | NA | <= | 1e-09 | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.brier` | comparator | prompts_specdetect | prompt_test | 0.11969087765849079 | <= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.log_loss` | comparator | prompts_specdetect | prompt_test | 0.3771571715517195 | <= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.balanced_accuracy` | comparator | prompts_specdetect | prompt_test | 0.832896031485733 | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.auroc` | comparator | prompts_specdetect | prompt_test | 0.9103609558150515 | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.human_fpr` | comparator | prompts_specdetect | prompt_test | 0.1774352246638242 | <= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.humanized_hard.auroc` | comparator | prompts_specdetect | humanized_hard | 0.9179583995760466 | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.humanized_hard.tpr_at_fpr_05` | comparator | prompts_specdetect | humanized_hard | 0.6372549019607843 | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.humanized_hard.tpr_at_fpr_10` | comparator | prompts_specdetect | humanized_hard | 0.7058823529411765 | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_m4_test.auroc` | comparator | prompts_specdetect | ood_m4_test | 0.5334 | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_raid_test.auroc` | comparator | prompts_specdetect | ood_raid_test | 0.5489 | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_mage_test.auroc` | comparator | prompts_specdetect | ood_mage_test | 0.43839999999999996 | >= | NA | comparator artifacts are invalid |

## Detector calibration

- Eligible rows: 1822
- Detector SKIP rows: 4276
- Support coverage: 1.000000
- Raw combined_w range: -49.912093 to 0.393239
- Candidate Brier: 0.249355
- Legacy unsafe clipped diagnostic Brier: 0.618695

## Adaptive July 10 challenge (secondary)

Not run: `invalid_comparator_avoided_additional_compute`.

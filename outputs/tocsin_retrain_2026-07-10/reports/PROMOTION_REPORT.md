# DO_NOT_PROMOTE

## Preflight

- Candidate `prompts_only_tocsin_v2`: VALID
- Comparator `prompts_only`: INVALID
- Comparator `prompts_specdetect`: INVALID

## Provenance status

- Provenance status: `VERIFIED_BYTES_LEGACY_IDENTITY_UNBOUND`
- Diagnostic allowed: `True`
- Promotion allowed: `False`
- Retrospective manifest: `outputs/tocsin_retrain_2026-07-10/manifests/retrospective-cache-provenance.json` SHA-256 `bdc8576acac8d2b2cecfdcd6fc425eea984741116821316147fa35ec7dc7bd38`
- Unbound fields: `modal_output_lineage`, `modal_output_sha256`, `actual_base_identity`, `huggingface_model_revisions`, `modal_image_digest`
- Candidate integrity manifest: `outputs\tocsin_retrain_2026-07-10\manifests\candidate-integrity.json` SHA-256 `5e55416397af3052deda4e6f9e0a59ff25a96f2109b378e587098c4e52ffbed4`
  - `configs/.profile_prompts_only_tocsin_v2.yaml` SHA-256 `b99b36e91b19c6e5229b478715a58d7bd207b6172e4ff362221a1487fcf8fce3` (382 bytes)
  - `models/calib2000/profiles/prompts_only_tocsin_v2/conformal.json` SHA-256 `4ac78138983bd433a381c872547a2d191f1c49f5cf63b992db6c70fcc343a323` (509 bytes)
  - `models/calib2000/profiles/prompts_only_tocsin_v2/detector_calibration.json` SHA-256 `6a1fc800faf14e196158876b546253618c27c818b97967b114158d6374c673be` (24120 bytes)
  - `models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops` SHA-256 `41b34b70a290a2d13ae36255237f53dfd1d47ec80a14845bf020a0305d82720a` (9640540 bytes)
  - `models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops.hmac` SHA-256 `f4342c510ba43e550422b529548eea3d14c8b0da3e4b582d42052247d4f8a251` (64 bytes)
  - `models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops.meta.json` SHA-256 `7679dd0911979a782cd171e92ff87d121a63cef1fd155b30e05dfa603ff6bd99` (3533 bytes)
- Slice `hard_human` (296 unique hashes):
  - Dataset `data\_hard_human_clean.jsonl` SHA-256 `cb1cb45b17c884e1fdd95d43b62969858ed09998c84a213503758722332f2142`
  - Cache `outputs\tocsin_retrain_2026-07-10\caches\_hard_human_clean.features.jsonl` SHA-256 `d2b4d27c64b626fcf414ea60c893deedcf96de799a8f0ae3e4949b8a35597791`
  - Manifest `outputs\tocsin_retrain_2026-07-10\manifests\_hard_human_clean.cache.json` SHA-256 `d76f8137871e8b4946c545d87d9e94dfe97b455256c98d5fa9fccf7af27a1a85`
- Slice `humanized` (502 unique hashes):
  - Dataset `data\_humanized_eval.jsonl` SHA-256 `6d3a946fa88c386807def5b90f7a727516308f909210eecd61f89a93e503680c`
  - Cache `outputs\tocsin_retrain_2026-07-10\caches\_humanized_eval.features.jsonl` SHA-256 `a1dab063ea76835ec39e159b85c6b6513990f2da51d803989090cace1011dc67`
  - Manifest `outputs\tocsin_retrain_2026-07-10\manifests\_humanized_eval.cache.json` SHA-256 `fb5b78af913ed4272146ce4d67a5689d0040442262fd32ae3ed2bc2cca8c9126`
- Slice `ood_m4` (200 unique hashes):
  - Dataset `data\ood_m4_test.jsonl` SHA-256 `d074bc99480cca4868a1e991b70f05b67727a61ca67a51485a5a303d7fde2db3`
  - Cache `outputs\tocsin_retrain_2026-07-10\caches\ood_m4_test.features.jsonl` SHA-256 `6ebf3d6d4329bcae9fab529806a7e12ae49834f27da44716a2c9a24538a46594`
  - Manifest `outputs\tocsin_retrain_2026-07-10\manifests\ood_m4_test.cache.json` SHA-256 `b61f630a79c6b0dd7c0bcef62faa4da303454f6bcddf4d090e676bc924835be0`
- Slice `ood_mage` (200 unique hashes):
  - Dataset `data\ood_mage_test.jsonl` SHA-256 `393c533e8bec02270ccf9f272397b75201e1673b50ea51b3e0f17c2feae5a31f`
  - Cache `outputs\tocsin_retrain_2026-07-10\caches\ood_mage_test.features.jsonl` SHA-256 `bb9b4bdd21452872b74c7634ee07215412ea75c553a9bfda9adb188070c6d4c3`
  - Manifest `outputs\tocsin_retrain_2026-07-10\manifests\ood_mage_test.cache.json` SHA-256 `89fcfdbbc8b6cfb3b2a6ba1f50c91cfc0fc8f3ee4ae2bc07d7c314583c6ed226`
- Slice `ood_raid` (200 unique hashes):
  - Dataset `data\ood_raid_test.jsonl` SHA-256 `a4d967f3adbf4e4e11c10d0b45acafbd23c2ddd31bdc86d05e9d78efbe73b23d`
  - Cache `outputs\tocsin_retrain_2026-07-10\caches\ood_raid_test.features.jsonl` SHA-256 `b3a9358c8b8d1fcda638022f6cb2429bd18afd77109d1a7a20a7cde83ba26c92`
  - Manifest `outputs\tocsin_retrain_2026-07-10\manifests\ood_raid_test.cache.json` SHA-256 `1e7e6647447b2c8220e83ae15a052226bd04877cecaa839b53bbc155d8211e3d`
- Slice `prompt_test` (6098 unique hashes):
  - Dataset `data\prompts_v5_test.jsonl` SHA-256 `30dd89ce7f50ed2fca8f0019ade08bb10e78ec21d870a9c34259e635d9635a3b`
  - Cache `outputs\tocsin_retrain_2026-07-10\caches\prompts_v5_test.features.jsonl` SHA-256 `22d83b9c60e6215bf27747a53cb5d59c55a4356181c6598d032618f215209755`
  - Manifest `outputs\tocsin_retrain_2026-07-10\manifests\prompts_v5_test.cache.json` SHA-256 `6cc248328b8aef7a3dd44aa8bb86f199fbf25df7098140bac3d805f25da23fca`

## Automatic promotion gates

| Status | Gate ID | Scope | Profile | Slice | Observed | Operator | Reference | Detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PASS | `preflight.comparator_identity` | preflight | NA | NA | true | is | true |  |
| FAIL | `preflight.provenance` | preflight | NA | NA | VERIFIED_BYTES_LEGACY_IDENTITY_UNBOUND | == | promotion-grade provenance distinct from cairn.tocsin-cache-retrospective.v1/VERIFIED_BYTES_LEGACY_IDENTITY_UNBOUND | UNPROMOTABLE_PROVENANCE: schema=cairn.tocsin-cache-retrospective.v1; status=VERIFIED_BYTES_LEGACY_IDENTITY_UNBOUND; promotion_allowed=False; unbound_fields=['modal_output_lineage', 'modal_output_sha256', 'actual_base_identity', 'huggingface_model_revisions', 'modal_image_digest'] |
| PASS | `preflight.candidate_artifacts` | preflight | prompts_only_tocsin_v2 | NA | true | is | true |  |
| FAIL | `preflight.comparator.prompts_only.artifacts` | preflight | prompts_only | NA | false | is | true | INVALID_COMPARATOR prompts_only:<br>smoke calibration_status=degraded, expected calibrated* |
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
| NOT_EVALUATED | `comparator.prompts_only.prompt_test.all_ebm` | comparator | prompts_only | prompt_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.prompt_test.finite` | comparator | prompts_only | prompt_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.prompt_test.in_unit_interval` | comparator | prompts_only | prompt_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.prompt_test.max_ebm_reconstruction_error` | comparator | prompts_only | prompt_test | NA | <= | 1e-09 | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.humanized_hard.all_ebm` | comparator | prompts_only | humanized_hard | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.humanized_hard.finite` | comparator | prompts_only | humanized_hard | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.humanized_hard.in_unit_interval` | comparator | prompts_only | humanized_hard | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.humanized_hard.max_ebm_reconstruction_error` | comparator | prompts_only | humanized_hard | NA | <= | 1e-09 | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_m4_test.all_ebm` | comparator | prompts_only | ood_m4_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_m4_test.finite` | comparator | prompts_only | ood_m4_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_m4_test.in_unit_interval` | comparator | prompts_only | ood_m4_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_m4_test.max_ebm_reconstruction_error` | comparator | prompts_only | ood_m4_test | NA | <= | 1e-09 | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_raid_test.all_ebm` | comparator | prompts_only | ood_raid_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_raid_test.finite` | comparator | prompts_only | ood_raid_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_raid_test.in_unit_interval` | comparator | prompts_only | ood_raid_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_raid_test.max_ebm_reconstruction_error` | comparator | prompts_only | ood_raid_test | NA | <= | 1e-09 | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_mage_test.all_ebm` | comparator | prompts_only | ood_mage_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_mage_test.finite` | comparator | prompts_only | ood_mage_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_mage_test.in_unit_interval` | comparator | prompts_only | ood_mage_test | NA | is | true | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_mage_test.max_ebm_reconstruction_error` | comparator | prompts_only | ood_mage_test | NA | <= | 1e-09 | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.prompt_test.brier` | comparator | prompts_only | prompt_test | NA | <= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.prompt_test.log_loss` | comparator | prompts_only | prompt_test | NA | <= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.prompt_test.balanced_accuracy` | comparator | prompts_only | prompt_test | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.prompt_test.auroc` | comparator | prompts_only | prompt_test | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.prompt_test.human_fpr` | comparator | prompts_only | prompt_test | NA | <= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.humanized_hard.auroc` | comparator | prompts_only | humanized_hard | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.humanized_hard.tpr_at_fpr_05` | comparator | prompts_only | humanized_hard | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.humanized_hard.tpr_at_fpr_10` | comparator | prompts_only | humanized_hard | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_m4_test.auroc` | comparator | prompts_only | ood_m4_test | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_raid_test.auroc` | comparator | prompts_only | ood_raid_test | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_only.ood_mage_test.auroc` | comparator | prompts_only | ood_mage_test | NA | >= | NA | comparator artifacts are invalid |
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
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.brier` | comparator | prompts_specdetect | prompt_test | NA | <= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.log_loss` | comparator | prompts_specdetect | prompt_test | NA | <= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.balanced_accuracy` | comparator | prompts_specdetect | prompt_test | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.auroc` | comparator | prompts_specdetect | prompt_test | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.prompt_test.human_fpr` | comparator | prompts_specdetect | prompt_test | NA | <= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.humanized_hard.auroc` | comparator | prompts_specdetect | humanized_hard | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.humanized_hard.tpr_at_fpr_05` | comparator | prompts_specdetect | humanized_hard | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.humanized_hard.tpr_at_fpr_10` | comparator | prompts_specdetect | humanized_hard | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_m4_test.auroc` | comparator | prompts_specdetect | ood_m4_test | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_raid_test.auroc` | comparator | prompts_specdetect | ood_raid_test | NA | >= | NA | comparator artifacts are invalid |
| NOT_EVALUATED | `comparator.prompts_specdetect.ood_mage_test.auroc` | comparator | prompts_specdetect | ood_mage_test | NA | >= | NA | comparator artifacts are invalid |

## Detector calibration

- Eligible rows: 1822
- Detector SKIP rows: 4276
- Support coverage: 1.000000
- Raw combined_w range: -49.912093 to 0.393239
- Candidate Brier: 0.249355
- Legacy unsafe clipped diagnostic Brier: 0.618695

## Adaptive July 10 challenge (secondary)

Not run: `invalid_comparator_avoided_additional_compute`.

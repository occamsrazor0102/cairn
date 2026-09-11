# Profile screen — generated tables

Generated 2026-09-11T05:28:09+00:00 from `buckeye_test_prompts.jsonl` (n=16, phases [1, 2], commit `b5e685a1b8eb`). Every number is a pipeline output; the input carries no ground-truth labels.

## Profiles run

| profile | mode | calibration_status | detector calibration | specdetect | mean p(LLM) | RED | AMBER | YELLOW | GREEN | UNCERTAIN | MIXED | conf [LLM] | conf [human] | conf abstain | own cascade -> phase 2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `master_minus_arxiv` | replay | degraded | loaded-incompatible | no | 0.699 | 7 | 5 | 0 | 1 | 3 | 0 | 12 | 1 | 3 | 16/16 |
| `master_minus_arxiv_robust_v1` | replay | degraded | loaded-incompatible | no | 0.685 | 3 | 7 | 0 | 0 | 6 | 0 | 10 | 0 | 6 | 16/16 |
| `master_minus_arxiv_robust_v2` | replay | degraded | loaded-incompatible | yes | 0.453 | 2 | 0 | 0 | 1 | 13 | 0 | 2 | 1 | 13 | 16/16 |
| `master_minus_arxiv_robust_v2a_ablated` | replay | degraded | loaded-incompatible | yes | 0.446 | 2 | 0 | 0 | 1 | 13 | 0 | 2 | 1 | 13 | 16/16 |
| `master_minus_arxiv_robust_v2b_cascade` | replay | degraded | loaded-incompatible | yes | 0.922 | 14 | 2 | 0 | 0 | 0 | 0 | 16 | 0 | 0 | 16/16 |
| `master_minus_arxiv_robust_v2c_lowint` | replay | degraded | loaded-incompatible | yes | 0.553 | 3 | 0 | 0 | 1 | 12 | 0 | 3 | 1 | 12 | 16/16 |
| `master_minus_arxiv_robust_v2d_gated` | replay | degraded | loaded-incompatible | yes | 0.340 | 1 | 0 | 0 | 4 | 11 | 0 | 1 | 4 | 11 | 16/16 |
| `master_minus_arxiv_robust_v2e_no_specdetect` | replay | degraded | loaded-incompatible | yes | 0.116 | 0 | 0 | 0 | 14 | 2 | 0 | 0 | 14 | 2 | 16/16 |
| `master_minus_arxiv_robust_v2f_pllm_only` | replay | degraded | loaded-incompatible | yes | 0.151 | 1 | 0 | 0 | 14 | 1 | 0 | 1 | 14 | 1 | 16/16 |
| `master_minus_arxiv_robust_v2g_clean_control` | replay | degraded | loaded-incompatible | yes | 0.118 | 0 | 0 | 0 | 14 | 2 | 0 | 0 | 14 | 2 | 16/16 |
| `master_minus_arxiv_robust_v2h_clean_full` | replay | degraded | loaded-incompatible | yes | 0.459 | 2 | 0 | 0 | 1 | 13 | 0 | 2 | 1 | 13 | 16/16 |
| `master_minus_arxiv_robust_v2i_v1_calibrator` | replay | degraded | loaded-incompatible | yes | 0.126 | 0 | 0 | 0 | 14 | 2 | 0 | 0 | 14 | 2 | 16/16 |

## p(LLM) per sample x profile

| sample | domain | words | router | `master_minus_arxiv` | `master_minus_arxiv_robust_v1` | `master_minus_arxiv_robust_v2` | `master_minus_arxiv_robust_v2a_ablated` | `master_minus_arxiv_robust_v2b_cascade` | `master_minus_arxiv_robust_v2c_lowint` | `master_minus_arxiv_robust_v2d_gated` | `master_minus_arxiv_robust_v2e_no_specdetect` | `master_minus_arxiv_robust_v2f_pllm_only` | `master_minus_arxiv_robust_v2g_clean_control` | `master_minus_arxiv_robust_v2h_clean_full` | `master_minus_arxiv_robust_v2i_v1_calibrator` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MEC-01 | Mechanical Engineers | 414 | prose | 0.692 | 0.602 | 0.279 | 0.224 | 0.970 | 0.428 | 0.136 | 0.041 | 0.040 | 0.043 | 0.289 | 0.050 |
| MEC-02 | Mechanical Engineers | 203 | prose | 0.792 | 0.704 | 0.339 | 0.340 | 0.978 | 0.299 | 0.215 | 0.237 | 0.275 | 0.245 | 0.347 | 0.276 |
| MEC-03 | Mechanical Engineers | 430 | prose | 0.735 | 0.642 | 0.638 | 0.636 | 0.967 | 0.846 | 0.532 | 0.039 | 0.152 | 0.041 | 0.634 | 0.048 |
| MEC-04 | Mechanical Engineers | 437 | prose | 0.704 | 0.585 | 0.238 | 0.232 | 0.954 | 0.467 | 0.213 | 0.047 | 0.044 | 0.048 | 0.242 | 0.055 |
| MEC-05 | Mechanical Engineers | 498 | prose | 0.504 | 0.712 | 0.044 | 0.047 | 0.725 | 0.045 | 0.027 | 0.037 | 0.036 | 0.038 | 0.048 | 0.045 |
| MEC-06 | Mechanical Engineers | 213 | prose | 0.763 | 0.718 | 0.383 | 0.379 | 0.761 | 0.295 | 0.213 | 0.032 | 0.031 | 0.033 | 0.389 | 0.035 |
| MEC-07 | Mechanical Engineers | 355 | prose | 0.824 | 0.720 | 0.647 | 0.659 | 0.986 | 0.794 | 0.542 | 0.061 | 0.170 | 0.064 | 0.652 | 0.073 |
| MEC-08 | Mechanical Engineers | 432 | prose | 0.707 | 0.619 | 0.465 | 0.455 | 0.978 | 0.761 | 0.285 | 0.041 | 0.113 | 0.042 | 0.476 | 0.050 |
| CIV-01 | Civil Engineers | 203 | prose | 0.770 | 0.713 | 0.308 | 0.311 | 0.960 | 0.252 | 0.175 | 0.040 | 0.039 | 0.041 | 0.317 | 0.052 |
| CIV-02 | Civil Engineers | 230 | prose | 0.877 | 0.855 | 0.624 | 0.632 | 0.988 | 0.645 | 0.417 | 0.046 | 0.128 | 0.047 | 0.632 | 0.050 |
| CIV-03 | Civil Engineers | 395 | mixed | 0.957 | 0.989 | 0.998 | 0.996 | 0.999 | 0.998 | 0.992 | 0.942 | 0.982 | 0.942 | 0.998 | 0.951 |
| CIV-04 | Civil Engineers | 471 | prose | 0.237 | 0.400 | 0.259 | 0.226 | 0.732 | 0.383 | 0.128 | 0.043 | 0.040 | 0.044 | 0.268 | 0.051 |
| CIV-05 | Civil Engineers | 524 | prose | 0.777 | 0.742 | 0.341 | 0.333 | 0.974 | 0.735 | 0.287 | 0.101 | 0.089 | 0.103 | 0.352 | 0.121 |
| CIV-06 | Civil Engineers | 200 | prose | 0.842 | 0.830 | 0.603 | 0.591 | 0.987 | 0.755 | 0.447 | 0.046 | 0.127 | 0.048 | 0.608 | 0.051 |
| CIV-07 | Civil Engineers | 236 | prose | 0.694 | 0.673 | 0.280 | 0.282 | 0.957 | 0.395 | 0.213 | 0.062 | 0.057 | 0.064 | 0.285 | 0.068 |
| CIV-08 | Civil Engineers | 249 | prose | 0.304 | 0.458 | 0.800 | 0.790 | 0.841 | 0.759 | 0.619 | 0.035 | 0.097 | 0.037 | 0.802 | 0.043 |

## Label per sample x profile

| sample | domain | words | router | `master_minus_arxiv` | `master_minus_arxiv_robust_v1` | `master_minus_arxiv_robust_v2` | `master_minus_arxiv_robust_v2a_ablated` | `master_minus_arxiv_robust_v2b_cascade` | `master_minus_arxiv_robust_v2c_lowint` | `master_minus_arxiv_robust_v2d_gated` | `master_minus_arxiv_robust_v2e_no_specdetect` | `master_minus_arxiv_robust_v2f_pllm_only` | `master_minus_arxiv_robust_v2g_clean_control` | `master_minus_arxiv_robust_v2h_clean_full` | `master_minus_arxiv_robust_v2i_v1_calibrator` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MEC-01 | Mechanical Engineers | 414 | prose | AMBER | UNCERTAIN | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | GREEN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| MEC-02 | Mechanical Engineers | 203 | prose | RED | AMBER | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | UNCERTAIN | UNCERTAIN | UNCERTAIN | UNCERTAIN | UNCERTAIN |
| MEC-03 | Mechanical Engineers | 430 | prose | AMBER | UNCERTAIN | UNCERTAIN | UNCERTAIN | RED | RED | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| MEC-04 | Mechanical Engineers | 437 | prose | AMBER | UNCERTAIN | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| MEC-05 | Mechanical Engineers | 498 | prose | UNCERTAIN | AMBER | GREEN | GREEN | AMBER | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN |
| MEC-06 | Mechanical Engineers | 213 | prose | RED | AMBER | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| MEC-07 | Mechanical Engineers | 355 | prose | RED | AMBER | UNCERTAIN | UNCERTAIN | RED | RED | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| MEC-08 | Mechanical Engineers | 432 | prose | AMBER | UNCERTAIN | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-01 | Civil Engineers | 203 | prose | RED | AMBER | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | GREEN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-02 | Civil Engineers | 230 | prose | RED | RED | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-03 | Civil Engineers | 395 | mixed | UNCERTAIN | RED | RED | RED | RED | RED | RED | UNCERTAIN | RED | UNCERTAIN | RED | UNCERTAIN |
| CIV-04 | Civil Engineers | 471 | prose | GREEN | UNCERTAIN | UNCERTAIN | UNCERTAIN | AMBER | UNCERTAIN | GREEN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-05 | Civil Engineers | 524 | prose | RED | AMBER | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-06 | Civil Engineers | 200 | prose | RED | RED | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-07 | Civil Engineers | 236 | prose | AMBER | AMBER | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-08 | Civil Engineers | 249 | prose | UNCERTAIN | UNCERTAIN | RED | RED | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | RED | GREEN |

## Conformal set per sample x profile

| sample | domain | words | router | `master_minus_arxiv` | `master_minus_arxiv_robust_v1` | `master_minus_arxiv_robust_v2` | `master_minus_arxiv_robust_v2a_ablated` | `master_minus_arxiv_robust_v2b_cascade` | `master_minus_arxiv_robust_v2c_lowint` | `master_minus_arxiv_robust_v2d_gated` | `master_minus_arxiv_robust_v2e_no_specdetect` | `master_minus_arxiv_robust_v2f_pllm_only` | `master_minus_arxiv_robust_v2g_clean_control` | `master_minus_arxiv_robust_v2h_clean_full` | `master_minus_arxiv_robust_v2i_v1_calibrator` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MEC-01 | Mechanical Engineers | 414 | prose | LLM | human+LLM | human+LLM | human+LLM | LLM | human+LLM | human | human | human | human | human+LLM | human |
| MEC-02 | Mechanical Engineers | 203 | prose | LLM | LLM | human+LLM | human+LLM | LLM | human+LLM | human+LLM | human+LLM | human+LLM | human+LLM | human+LLM | human+LLM |
| MEC-03 | Mechanical Engineers | 430 | prose | LLM | human+LLM | human+LLM | human+LLM | LLM | LLM | human+LLM | human | human | human | human+LLM | human |
| MEC-04 | Mechanical Engineers | 437 | prose | LLM | human+LLM | human+LLM | human+LLM | LLM | human+LLM | human+LLM | human | human | human | human+LLM | human |
| MEC-05 | Mechanical Engineers | 498 | prose | human+LLM | LLM | human | human | LLM | human | human | human | human | human | human | human |
| MEC-06 | Mechanical Engineers | 213 | prose | LLM | LLM | human+LLM | human+LLM | LLM | human+LLM | human+LLM | human | human | human | human+LLM | human |
| MEC-07 | Mechanical Engineers | 355 | prose | LLM | LLM | human+LLM | human+LLM | LLM | LLM | human+LLM | human | human | human | human+LLM | human |
| MEC-08 | Mechanical Engineers | 432 | prose | LLM | human+LLM | human+LLM | human+LLM | LLM | human+LLM | human+LLM | human | human | human | human+LLM | human |
| CIV-01 | Civil Engineers | 203 | prose | LLM | LLM | human+LLM | human+LLM | LLM | human+LLM | human | human | human | human | human+LLM | human |
| CIV-02 | Civil Engineers | 230 | prose | LLM | LLM | human+LLM | human+LLM | LLM | human+LLM | human+LLM | human | human | human | human+LLM | human |
| CIV-03 | Civil Engineers | 395 | mixed | human+LLM | LLM | LLM | LLM | LLM | LLM | LLM | human+LLM | LLM | human+LLM | LLM | human+LLM |
| CIV-04 | Civil Engineers | 471 | prose | human | human+LLM | human+LLM | human+LLM | LLM | human+LLM | human | human | human | human | human+LLM | human |
| CIV-05 | Civil Engineers | 524 | prose | LLM | LLM | human+LLM | human+LLM | LLM | human+LLM | human+LLM | human | human | human | human+LLM | human |
| CIV-06 | Civil Engineers | 200 | prose | LLM | LLM | human+LLM | human+LLM | LLM | human+LLM | human+LLM | human | human | human | human+LLM | human |
| CIV-07 | Civil Engineers | 236 | prose | LLM | LLM | human+LLM | human+LLM | LLM | human+LLM | human+LLM | human | human | human | human+LLM | human |
| CIV-08 | Civil Engineers | 249 | prose | human+LLM | human+LLM | LLM | LLM | LLM | human+LLM | human+LLM | human | human | human | LLM | human |

## Own-cascade escalation to phase 2 (counterfactual)

| sample | domain | words | router | `master_minus_arxiv` | `master_minus_arxiv_robust_v1` | `master_minus_arxiv_robust_v2` | `master_minus_arxiv_robust_v2a_ablated` | `master_minus_arxiv_robust_v2b_cascade` | `master_minus_arxiv_robust_v2c_lowint` | `master_minus_arxiv_robust_v2d_gated` | `master_minus_arxiv_robust_v2e_no_specdetect` | `master_minus_arxiv_robust_v2f_pllm_only` | `master_minus_arxiv_robust_v2g_clean_control` | `master_minus_arxiv_robust_v2h_clean_full` | `master_minus_arxiv_robust_v2i_v1_calibrator` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MEC-01 | Mechanical Engineers | 414 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| MEC-02 | Mechanical Engineers | 203 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| MEC-03 | Mechanical Engineers | 430 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| MEC-04 | Mechanical Engineers | 437 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| MEC-05 | Mechanical Engineers | 498 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| MEC-06 | Mechanical Engineers | 213 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| MEC-07 | Mechanical Engineers | 355 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| MEC-08 | Mechanical Engineers | 432 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| CIV-01 | Civil Engineers | 203 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| CIV-02 | Civil Engineers | 230 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| CIV-03 | Civil Engineers | 395 | mixed | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| CIV-04 | Civil Engineers | 471 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| CIV-05 | Civil Engineers | 524 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| CIV-06 | Civil Engineers | 200 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| CIV-07 | Civil Engineers | 236 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| CIV-08 | Civil Engineers | 249 | prose | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |

## Raw detector pass

Superset detectors: contrastive_lm, fast_detectgpt, fingerprint_vocab, instruction_density, mixed_boundary, nssi, preamble, prompt_structure, semantic_kinematics, specdetect, surprisal_dynamics, token_cohesiveness, voice_spec.

All enabled detectors were available.

## Replay verification

`master_minus_arxiv_robust_v1` re-run directly with `CairnPipeline` on 16 sample(s): max |p(LLM) diff| = 4.6e-07, max per-detector |p diff| = 0.0, labels match = True, conformal sets match = True, detector sets match = True.

## Profile availability

| profile | status | config | detail |
|---|---|---|---|
| `ablation_full` | unavailable | `configs/.ablation_full.yaml` | model: models/calib2000/ablation/full/ebm.pkl not found; conformal: models/calib2000/ablation/full/conformal.json not found; detector_calibration: models/calib2000/ablation/full/detector_calibration.json not found |
| `ablation_loo_anthropic` | unavailable | `configs/.ablation_loo_anthropic.yaml` | model: models/calib2000/ablation/loo_anthropic/ebm.pkl not found; conformal: models/calib2000/ablation/loo_anthropic/conformal.json not found; detector_calibration: models/calib2000/ablation/loo_anthropic/detector_calibration.json not found |
| `ablation_loo_deepseek` | unavailable | `configs/.ablation_loo_deepseek.yaml` | model: models/calib2000/ablation/loo_deepseek/ebm.pkl not found; conformal: models/calib2000/ablation/loo_deepseek/conformal.json not found; detector_calibration: models/calib2000/ablation/loo_deepseek/detector_calibration.json not found |
| `ablation_loo_gemini` | unavailable | `configs/.ablation_loo_gemini.yaml` | model: models/calib2000/ablation/loo_gemini/ebm.pkl not found; conformal: models/calib2000/ablation/loo_gemini/conformal.json not found; detector_calibration: models/calib2000/ablation/loo_gemini/detector_calibration.json not found |
| `ablation_loo_mistral` | unavailable | `configs/.ablation_loo_mistral.yaml` | model: models/calib2000/ablation/loo_mistral/ebm.pkl not found; conformal: models/calib2000/ablation/loo_mistral/conformal.json not found; detector_calibration: models/calib2000/ablation/loo_mistral/detector_calibration.json not found |
| `ablation_loo_moonshot` | unavailable | `configs/.ablation_loo_moonshot.yaml` | model: models/calib2000/ablation/loo_moonshot/ebm.pkl not found; conformal: models/calib2000/ablation/loo_moonshot/conformal.json not found; detector_calibration: models/calib2000/ablation/loo_moonshot/detector_calibration.json not found |
| `ablation_loo_openai` | unavailable | `configs/.ablation_loo_openai.yaml` | model: models/calib2000/ablation/loo_openai/ebm.pkl not found; conformal: models/calib2000/ablation/loo_openai/conformal.json not found; detector_calibration: models/calib2000/ablation/loo_openai/detector_calibration.json not found |
| `ablation_solo_anthropic` | unavailable | `configs/.ablation_solo_anthropic.yaml` | model: models/calib2000/ablation/solo_anthropic/ebm.pkl not found; conformal: models/calib2000/ablation/solo_anthropic/conformal.json not found; detector_calibration: models/calib2000/ablation/solo_anthropic/detector_calibration.json not found |
| `ablation_solo_deepseek` | unavailable | `configs/.ablation_solo_deepseek.yaml` | model: models/calib2000/ablation/solo_deepseek/ebm.pkl not found; conformal: models/calib2000/ablation/solo_deepseek/conformal.json not found; detector_calibration: models/calib2000/ablation/solo_deepseek/detector_calibration.json not found |
| `ablation_solo_gemini` | unavailable | `configs/.ablation_solo_gemini.yaml` | model: models/calib2000/ablation/solo_gemini/ebm.pkl not found; conformal: models/calib2000/ablation/solo_gemini/conformal.json not found; detector_calibration: models/calib2000/ablation/solo_gemini/detector_calibration.json not found |
| `ablation_solo_mistral` | unavailable | `configs/.ablation_solo_mistral.yaml` | model: models/calib2000/ablation/solo_mistral/ebm.pkl not found; conformal: models/calib2000/ablation/solo_mistral/conformal.json not found; detector_calibration: models/calib2000/ablation/solo_mistral/detector_calibration.json not found |
| `ablation_solo_moonshot` | unavailable | `configs/.ablation_solo_moonshot.yaml` | model: models/calib2000/ablation/solo_moonshot/ebm.pkl not found; conformal: models/calib2000/ablation/solo_moonshot/conformal.json not found; detector_calibration: models/calib2000/ablation/solo_moonshot/detector_calibration.json not found |
| `ablation_solo_openai` | unavailable | `configs/.ablation_solo_openai.yaml` | model: models/calib2000/ablation/solo_openai/ebm.pkl not found; conformal: models/calib2000/ablation/solo_openai/conformal.json not found; detector_calibration: models/calib2000/ablation/solo_openai/detector_calibration.json not found |
| `calib2000_mix` | unavailable | `configs/.calib2000_mix.yaml` | model: models/calib2000/mix/ebm.pkl not found; conformal: models/calib2000/mix/conformal.json not found; detector_calibration: models/calib2000/mix/detector_calibration.json not found |
| `calib2000_v2` | unavailable | `configs/.calib2000_v2.yaml` | model: models/calib2000/v2/ebm.pkl not found; conformal: models/calib2000/v2/conformal.json not found; detector_calibration: models/calib2000/v2/detector_calibration.json not found |
| `calib2000_v3` | unavailable | `configs/.calib2000_v3.yaml` | model: models/calib2000/v3/ebm.pkl not found; conformal: models/calib2000/v3/conformal.json not found; detector_calibration: models/calib2000/v3/detector_calibration.json not found |
| `calib2000_v4` | unavailable | `configs/.calib2000_v4.yaml` | model: models/calib2000/v4/ebm.pkl not found; conformal: models/calib2000/v4/conformal.json not found; detector_calibration: models/calib2000/v4/detector_calibration.json not found |
| `calib2000_v4_1` | unavailable | `configs/.calib2000_v4_1.yaml` | model: models/calib2000/v4_1/ebm.pkl not found; conformal: models/calib2000/v4_1/conformal.json not found; detector_calibration: models/calib2000/v4_1/detector_calibration.json not found |
| `calib2000_v5` | unavailable | `configs/.calib2000_v5.yaml` | model: models/calib2000/v5/ebm.pkl not found; conformal: models/calib2000/v5/conformal.json not found; detector_calibration: models/calib2000/v5/detector_calibration.json not found |
| `calib2000_v6` | unavailable | `configs/.calib2000_v6.yaml` | model: models/calib2000/v6/ebm.pkl not found; conformal: models/calib2000/v6/conformal.json not found; detector_calibration: models/calib2000/v6/detector_calibration.json not found |
| `loo_domain_arxiv` | unavailable | `configs/.loo_domain_arxiv.yaml` | model: models/calib2000/loo/domain_arxiv/ebm.pkl not found; conformal: models/calib2000/loo/domain_arxiv/conformal.json not found; detector_calibration: models/calib2000/loo/domain_arxiv/detector_calibration.json not found |
| `loo_domain_gsm8k` | unavailable | `configs/.loo_domain_gsm8k.yaml` | model: models/calib2000/loo/domain_gsm8k/ebm.pkl not found; conformal: models/calib2000/loo/domain_gsm8k/conformal.json not found; detector_calibration: models/calib2000/loo/domain_gsm8k/detector_calibration.json not found |
| `loo_domain_humaneval` | unavailable | `configs/.loo_domain_humaneval.yaml` | model: models/calib2000/loo/domain_humaneval/ebm.pkl not found; conformal: models/calib2000/loo/domain_humaneval/conformal.json not found; detector_calibration: models/calib2000/loo/domain_humaneval/detector_calibration.json not found |
| `loo_hc3_finance` | unavailable | `configs/.loo_hc3_finance.yaml` | model: models/calib2000/loo/hc3_finance/ebm.pkl not found; conformal: models/calib2000/loo/hc3_finance/conformal.json not found; detector_calibration: models/calib2000/loo/hc3_finance/detector_calibration.json not found |
| `loo_hc3_medicine` | unavailable | `configs/.loo_hc3_medicine.yaml` | model: models/calib2000/loo/hc3_medicine/ebm.pkl not found; conformal: models/calib2000/loo/hc3_medicine/conformal.json not found; detector_calibration: models/calib2000/loo/hc3_medicine/detector_calibration.json not found |
| `loo_hc3_open` | unavailable | `configs/.loo_hc3_open.yaml` | model: models/calib2000/loo/hc3_open/ebm.pkl not found; conformal: models/calib2000/loo/hc3_open/conformal.json not found; detector_calibration: models/calib2000/loo/hc3_open/detector_calibration.json not found |
| `loo_hc3_reddit` | unavailable | `configs/.loo_hc3_reddit.yaml` | model: models/calib2000/loo/hc3_reddit/ebm.pkl not found; conformal: models/calib2000/loo/hc3_reddit/conformal.json not found; detector_calibration: models/calib2000/loo/hc3_reddit/detector_calibration.json not found |
| `loo_hc3_wiki` | unavailable | `configs/.loo_hc3_wiki.yaml` | model: models/calib2000/loo/hc3_wiki/ebm.pkl not found; conformal: models/calib2000/loo/hc3_wiki/conformal.json not found; detector_calibration: models/calib2000/loo/hc3_wiki/detector_calibration.json not found |
| `loo_mage` | unavailable | `configs/.loo_mage.yaml` | model: models/calib2000/loo/mage/ebm.pkl not found; conformal: models/calib2000/loo/mage/conformal.json not found; detector_calibration: models/calib2000/loo/mage/detector_calibration.json not found |
| `loo_prompt_alpaca` | unavailable | `configs/.loo_prompt_alpaca.yaml` | model: models/calib2000/loo/prompt_alpaca/ebm.pkl not found; conformal: models/calib2000/loo/prompt_alpaca/conformal.json not found; detector_calibration: models/calib2000/loo/prompt_alpaca/detector_calibration.json not found |
| `loo_prompt_gsm8k` | unavailable | `configs/.loo_prompt_gsm8k.yaml` | model: models/calib2000/loo/prompt_gsm8k/ebm.pkl not found; conformal: models/calib2000/loo/prompt_gsm8k/conformal.json not found; detector_calibration: models/calib2000/loo/prompt_gsm8k/detector_calibration.json not found |
| `loo_prompt_magpie` | unavailable | `configs/.loo_prompt_magpie.yaml` | model: models/calib2000/loo/prompt_magpie/ebm.pkl not found; conformal: models/calib2000/loo/prompt_magpie/conformal.json not found; detector_calibration: models/calib2000/loo/prompt_magpie/detector_calibration.json not found |
| `loo_prompt_medqa` | unavailable | `configs/.loo_prompt_medqa.yaml` | model: models/calib2000/loo/prompt_medqa/ebm.pkl not found; conformal: models/calib2000/loo/prompt_medqa/conformal.json not found; detector_calibration: models/calib2000/loo/prompt_medqa/detector_calibration.json not found |
| `loo_prompt_mmlu` | unavailable | `configs/.loo_prompt_mmlu.yaml` | model: models/calib2000/loo/prompt_mmlu/ebm.pkl not found; conformal: models/calib2000/loo/prompt_mmlu/conformal.json not found; detector_calibration: models/calib2000/loo/prompt_mmlu/detector_calibration.json not found |
| `loo_prompt_slimorca` | unavailable | `configs/.loo_prompt_slimorca.yaml` | model: models/calib2000/loo/prompt_slimorca/ebm.pkl not found; conformal: models/calib2000/loo/prompt_slimorca/conformal.json not found; detector_calibration: models/calib2000/loo/prompt_slimorca/detector_calibration.json not found |
| `loo_raid_abstracts` | unavailable | `configs/.loo_raid_abstracts.yaml` | model: models/calib2000/loo/raid_abstracts/ebm.pkl not found; conformal: models/calib2000/loo/raid_abstracts/conformal.json not found; detector_calibration: models/calib2000/loo/raid_abstracts/detector_calibration.json not found |
| `lossfilter_exclude_easy` | unavailable | `configs/.lossfilter_exclude_easy.yaml` | model: models/calib2000/loss_filter/exclude_easy/ebm.pkl not found; conformal: models/calib2000/loss_filter/exclude_easy/conformal.json not found; detector_calibration: models/calib2000/loss_filter/exclude_easy/detector_calibration.json not found |
| `lossfilter_high_loss_top30` | unavailable | `configs/.lossfilter_high_loss_top30.yaml` | model: models/calib2000/loss_filter/high_loss_top30/ebm.pkl not found; conformal: models/calib2000/loss_filter/high_loss_top30/conformal.json not found; detector_calibration: models/calib2000/loss_filter/high_loss_top30/detector_calibration.json not found |
| `lossfilter_high_loss_top50` | unavailable | `configs/.lossfilter_high_loss_top50.yaml` | model: models/calib2000/loss_filter/high_loss_top50/ebm.pkl not found; conformal: models/calib2000/loss_filter/high_loss_top50/conformal.json not found; detector_calibration: models/calib2000/loss_filter/high_loss_top50/detector_calibration.json not found |
| `lossfilter_uncertain` | unavailable | `configs/.lossfilter_uncertain.yaml` | model: models/calib2000/loss_filter/uncertain/ebm.pkl not found; conformal: models/calib2000/loss_filter/uncertain/conformal.json not found; detector_calibration: models/calib2000/loss_filter/uncertain/detector_calibration.json not found |
| `cascade_default_with_specdetect` | not-calibrated | `configs/.profile_cascade_default_with_specdetect.yaml` | no fusion.model_path (heuristic profile) |
| `chat_prompts` | unavailable | `configs/.profile_chat_prompts.yaml` | model: models/calib2000/profiles/chat_prompts/ebm.pkl not found; conformal: models/calib2000/profiles/chat_prompts/conformal.json not found; detector_calibration: models/calib2000/profiles/chat_prompts/detector_calibration.json not found |
| `drop_top3_drags` | unavailable | `configs/.profile_drop_top3_drags.yaml` | model: models/calib2000/profiles/drop_top3_drags/ebm.pkl not found; conformal: models/calib2000/profiles/drop_top3_drags/conformal.json not found; detector_calibration: models/calib2000/profiles/drop_top3_drags/detector_calibration.json not found |
| `everything` | unavailable | `configs/.profile_everything.yaml` | model: models/calib2000/profiles/everything/ebm.pkl not found; conformal: models/calib2000/profiles/everything/conformal.json not found; detector_calibration: models/calib2000/profiles/everything/detector_calibration.json not found |
| `finance` | unavailable | `configs/.profile_finance.yaml` | model: models/calib2000/profiles/finance/ebm.pkl not found; conformal: models/calib2000/profiles/finance/conformal.json not found; detector_calibration: models/calib2000/profiles/finance/detector_calibration.json not found |
| `full_coverage` | not-calibrated | `configs/.profile_full_coverage.yaml` | no fusion.model_path (heuristic profile) |
| `full_coverage_with_specdetect` | not-calibrated | `configs/.profile_full_coverage_with_specdetect.yaml` | no fusion.model_path (heuristic profile) |
| `general_full` | unavailable | `configs/.profile_general_full.yaml` | model: models/calib2000/profiles/general_full/ebm.pkl not found; conformal: models/calib2000/profiles/general_full/conformal.json not found; detector_calibration: models/calib2000/profiles/general_full/detector_calibration.json not found |
| `general_hc3` | unavailable | `configs/.profile_general_hc3.yaml` | model: models/calib2000/profiles/general_hc3/ebm.pkl not found; conformal: models/calib2000/profiles/general_hc3/conformal.json not found; detector_calibration: models/calib2000/profiles/general_hc3/detector_calibration.json not found |
| `general_plus_medical` | unavailable | `configs/.profile_general_plus_medical.yaml` | model: models/calib2000/profiles/general_plus_medical/ebm.pkl not found; conformal: models/calib2000/profiles/general_plus_medical/conformal.json not found; detector_calibration: models/calib2000/profiles/general_plus_medical/detector_calibration.json not found |
| `general_plus_short` | unavailable | `configs/.profile_general_plus_short.yaml` | model: models/calib2000/profiles/general_plus_short/ebm.pkl not found; conformal: models/calib2000/profiles/general_plus_short/conformal.json not found; detector_calibration: models/calib2000/profiles/general_plus_short/detector_calibration.json not found |
| `general_plus_stem` | unavailable | `configs/.profile_general_plus_stem.yaml` | model: models/calib2000/profiles/general_plus_stem/ebm.pkl not found; conformal: models/calib2000/profiles/general_plus_stem/conformal.json not found; detector_calibration: models/calib2000/profiles/general_plus_stem/detector_calibration.json not found |
| `master_minus_arxiv` | available | `configs/.profile_master_minus_arxiv.yaml` |  |
| `master_minus_arxiv_clean` | unavailable | `configs/.profile_master_minus_arxiv_clean.yaml` | model: models/calib2000/profiles/master_minus_arxiv_clean/ebm.pkl not found; conformal: models/calib2000/profiles/master_minus_arxiv_clean/conformal.json not found; detector_calibration: models/calib2000/profiles/master_minus_arxiv_clean/detector_calibration.json not found |
| `master_minus_arxiv_robust_v1` | available | `configs/.profile_master_minus_arxiv_robust_v1.yaml` |  |
| `master_minus_arxiv_robust_v2` | available | `configs/.profile_master_minus_arxiv_robust_v2.yaml` |  |
| `master_minus_arxiv_robust_v2a_ablated` | available | `configs/.profile_master_minus_arxiv_robust_v2a_ablated.yaml` |  |
| `master_minus_arxiv_robust_v2b_cascade` | available | `configs/.profile_master_minus_arxiv_robust_v2b_cascade.yaml` |  |
| `master_minus_arxiv_robust_v2c_lowint` | available | `configs/.profile_master_minus_arxiv_robust_v2c_lowint.yaml` |  |
| `master_minus_arxiv_robust_v2d_gated` | available | `configs/.profile_master_minus_arxiv_robust_v2d_gated.yaml` |  |
| `master_minus_arxiv_robust_v2e_no_specdetect` | available | `configs/.profile_master_minus_arxiv_robust_v2e_no_specdetect.yaml` |  |
| `master_minus_arxiv_robust_v2f_pllm_only` | available | `configs/.profile_master_minus_arxiv_robust_v2f_pllm_only.yaml` |  |
| `master_minus_arxiv_robust_v2g_clean_control` | available | `configs/.profile_master_minus_arxiv_robust_v2g_clean_control.yaml` |  |
| `master_minus_arxiv_robust_v2h_clean_full` | available | `configs/.profile_master_minus_arxiv_robust_v2h_clean_full.yaml` |  |
| `master_minus_arxiv_robust_v2i_v1_calibrator` | available | `configs/.profile_master_minus_arxiv_robust_v2i_v1_calibrator.yaml` |  |
| `medical` | unavailable | `configs/.profile_medical.yaml` | model: models/calib2000/profiles/medical/ebm.pkl not found; conformal: models/calib2000/profiles/medical/conformal.json not found; detector_calibration: models/calib2000/profiles/medical/detector_calibration.json not found |
| `prompts_only` | unavailable | `configs/.profile_prompts_only.yaml` | model: models/calib2000/profiles/prompts_only/ebm.skops not found; conformal: models/calib2000/profiles/prompts_only/conformal.json not found; detector_calibration: models/calib2000/profiles/prompts_only/detector_calibration.json not found |
| `prompts_specdetect` | unavailable | `configs/.profile_prompts_specdetect.yaml` | model: models/calib2000/profiles/prompts_specdetect/ebm.skops not found |
| `short_form` | unavailable | `configs/.profile_short_form.yaml` | model: models/calib2000/profiles/short_form/ebm.pkl not found; conformal: models/calib2000/profiles/short_form/conformal.json not found; detector_calibration: models/calib2000/profiles/short_form/detector_calibration.json not found |
| `stem` | unavailable | `configs/.profile_stem.yaml` | model: models/calib2000/profiles/stem/ebm.pkl not found; conformal: models/calib2000/profiles/stem/conformal.json not found; detector_calibration: models/calib2000/profiles/stem/detector_calibration.json not found |
| `stem_with_prompts` | unavailable | `configs/.profile_stem_with_prompts.yaml` | model: models/calib2000/profiles/stem_with_prompts/ebm.pkl not found; conformal: models/calib2000/profiles/stem_with_prompts/conformal.json not found; detector_calibration: models/calib2000/profiles/stem_with_prompts/detector_calibration.json not found |
| `prompts_only` | unavailable | `configs/.prompts_only.yaml` | model: models/calib2000/prompts_only/ebm.pkl not found; conformal: models/calib2000/prompts_only/conformal.json not found; detector_calibration: models/calib2000/prompts_only/detector_calibration.json not found |
| `calibrated` | unavailable | `configs/calibrated.yaml` | model: models/fusion/hc3_finance_ebm.pkl not found; conformal: models/conformal/hc3_finance.json not found; detector_calibration: models/calibrators/hc3_finance.json not found |
| `calibrated_all` | unavailable | `configs/calibrated_all.yaml` | model: models/fusion/hc3_all_ebm.pkl not found; conformal: models/conformal/hc3_all.json not found; detector_calibration: models/calibrators/hc3_all.json not found |
| `calibrated_calib2000` | unavailable | `configs/calibrated_calib2000.yaml` | model: models/calib2000/full/ebm.pkl not found; conformal: models/calib2000/full/conformal.json not found; detector_calibration: models/calib2000/full/detector_calibration.json not found |
| `calibrated_full` | unavailable | `configs/calibrated_full.yaml` | model: models/fusion/superunion_ebm.pkl not found; conformal: models/conformal/superunion.json not found; detector_calibration: models/calibrators/superunion.json not found |
| `calibrated_matched_mixed` | unavailable | `configs/calibrated_matched_mixed.yaml` | model: models/fusion/matched_mixed_ebm.pkl not found; conformal: models/conformal/matched_mixed.json not found; detector_calibration: models/calibrators/matched_mixed.json not found |
| `calibrated_matched_single` | unavailable | `configs/calibrated_matched_single.yaml` | model: models/fusion/matched_single_ebm.pkl not found; conformal: models/conformal/matched_single.json not found; detector_calibration: models/calibrators/matched_single.json not found |
| `calibrated_regen` | unavailable | `configs/calibrated_regen.yaml` | model: models/fusion/regen_mixed_ebm.pkl not found; conformal: models/conformal/regen_mixed.json not found; detector_calibration: models/calibrators/regen_mixed.json not found |
| `calibrated_union` | unavailable | `configs/calibrated_union.yaml` | model: models/fusion/union_ebm.pkl not found; conformal: models/conformal/union.json not found; detector_calibration: models/calibrators/union.json not found |
| `default` | not-calibrated | `configs/default.yaml` | no fusion.model_path (heuristic profile) |
| `no-api` | not-calibrated | `configs/no-api.yaml` | no fusion.model_path (heuristic profile) |
| `production` | unavailable | `configs/production.yaml` | model: /app/models/ebm_v1.pkl not found; conformal: /app/models/conformal_cal.json not found; detector_calibration: /app/models/detector_calibration.json not found |
| `screening` | not-calibrated | `configs/screening.yaml` | no fusion.model_path (heuristic profile) |
| `strict` | not-calibrated | `configs/strict.yaml` | no fusion.model_path (heuristic profile) |


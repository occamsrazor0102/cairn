# Buckeye test prompts — phase 1+2 screen across every available calibrated profile

**Date:** 2026-09-11 · **Runner:** `scripts/run_profile_screen.py` · **Input:** the 16 prompts
in `CAIRN_Buckeye_Test_Prompts.xlsx` (8 Mechanical Engineering + 8 Civil Engineering scenario
prompts, 200–524 words), staged unchanged as `data/buckeye_test_prompts.jsonl` (git-ignored;
the sheet is the source of truth). **Ground truth:** none — the sheet carries no human/LLM
labels, so this is a *screen* (what does each profile say?), not a benchmark (how accurate is
it?). Generated tables with every number live in [`results.md`](results.md); machine-readable
outputs are listed at the end.

## TL;DR

1. **The verdict depends far more on which profile is loaded than on which prompt is scored.**
   No prompt receives the same label from all 12 profiles; the two production-grade profiles,
   `master_minus_arxiv` and `master_minus_arxiv_robust_v1`, agree on the label for only 4 of
   16 prompts. Per-prompt p(LLM) spans typically run from ~0.04 to ~0.97 across profiles.
2. **`master_minus_arxiv` and `robust_v1` lean LLM** (mean p(LLM) 0.70 and 0.69; honest
   conformal set `["LLM"]` on 12/16 and 10/16). **The `robust_v2` ablation family mostly
   abstains** (`UNCERTAIN`, conformal `["human","LLM"]` on 11–13/16), and the four variants
   whose EBM carries no SpecDetect spectral features (`v2e`, `v2f`, `v2g`) or reuses the v1
   calibrator (`v2i`) call **14/16 prompts human** (`GREEN`, conformal `["human"]`).
   `v2b_cascade` is the outlier at 14/16 `RED`. The v2 family is documented in
   `docs/benchmarks/robust_v2_validation/` as a regression that was never promoted; read it as
   research context, not a second opinion.
3. **CIV-03 is the only prompt every profile scores ≥ 0.94.** It is also the only prompt the
   router classifies as `mixed` rather than `prose`, so it is the only one on which the three
   prompt-domain detectors (`voice_spec`, `prompt_structure`, `instruction_density`) ran; their
   confidence features are the top EBM contributors on it for every profile. On the other 15
   prompts the router said `prose` and skipped those detectors by design.
4. **Phase-1 pattern detectors are silent on this set.** `preamble` matched nothing on any
   prompt, `fingerprint_vocab` fired only on MEC-02, and `nssi` is ~0 everywhere. Every verdict
   therefore rests on the phase-2 LM-scoring detectors, whose *calibrated* outputs differ by
   0.2–0.35 between the two calibrator generations (see "Detector-level view").
5. **Every profile ran in `calibration_status: degraded`.** All 12 shipped detector calibrators
   carry a legacy, unsigned `token_cohesiveness` (TOCSIN) entry that the current pipeline
   rejects fail-closed (`calibrator_signature_mismatch`), so TOCSIN contributed nothing and its
   EBM features were mean-imputed. This is the repository's current state for these artifacts
   (see `docs/superpowers/specs/2026-07-10-tocsin-calibration-retrain-design.md` and
   `outputs/tocsin_retrain_2026-07-10/reports/PROMOTION_REPORT.md`), not something this run
   introduced — but it does mean these numbers are not the profiles' as-trained behaviour.
6. **Restricting the cascade to phases 1+2 changed nothing relative to production.** Under each
   profile's own short-circuit thresholds the phase-1 aggregate (0.48–0.74) would have
   escalated all 16 prompts to phase 2 and on to phase 3 — but every phase-3 detector
   (`perturbation`, `contrastive_gen`, `dna_gpt`, `lapd`) ships disabled in all 12 profiles, so
   the production cascade would have produced exactly these results.

## What was run

- **Profiles.** All 85 YAML files under `configs/` (dot-prefixed research profiles included)
  were loaded and classified. 12 declare fusion artifacts that are all present on disk and were
  run; 66 declare artifacts that are not in this checkout (the `models/` tree is git-ignored
  except the `master_minus_arxiv` family) and were skipped; 7 are heuristic profiles with no
  fusion artifacts. Notably **`prompts_only` / `prompts_specdetect` — the profiles meant for
  prompt detection — could not be run** (`ebm.skops` is not tracked). Full table:
  [`profiles.json`](profiles.json) / bottom of [`results.md`](results.md).
- **Phase restriction.** For every profile the cascade thresholds were overlaid in memory so
  phase 2 always fires and phase 3 never does, and the phase-3/4 detectors were disabled
  explicitly. The router stays in the loop exactly as in production (it decides which phase-1
  detectors run per text). No API-backed detector was enabled; `task_description.auto_mode`
  stayed `off`; nothing left the machine.
- **One raw pass, twelve replays.** Detector outputs are calibrator-independent, so the 13
  phase-1/2 detectors ran once per prompt under an uncalibrated superset config (default +
  SpecDetect, which the v2 family enables) and were cached with `cairn.feature_cache`; each
  profile then replayed that cache through its own detector calibrator, EBM, conformal wrapper
  and decision engine via `CachedPipeline` (config-dict support added for this). The runner
  refuses to replay a profile whose detector config differs from the superset and would run it
  directly instead; all 12 were exact matches.
- **Verification.** `master_minus_arxiv_robust_v1` was additionally re-run end-to-end with
  `CairnPipeline` on all 16 prompts: max |Δ p(LLM)| = 4.6e-07 (JSON rounding), labels and
  conformal sets identical on 16/16, per-detector p(LLM) identical.
- **Runtime.** CPU only (4 cores), `torch 2.14.0+cpu`, `transformers 5.17.0`,
  `interpret 0.7.8`, `skops 0.14.0`. Scoring LMs as configured: `gpt2` (surprisal, Fast-DetectGPT,
  SpecDetect), `gpt2` + `gpt2-medium` (Binoculars), `facebook/bart-base` (TOCSIN),
  `all-MiniLM-L6-v2` (semantic kinematics). Raw pass ≈ 3 min (37 s for the first prompt including
  model load, 5–13 s each after); each profile replay ≈ 0.1 s; whole run incl. verification 308 s. The committed `results.json` is a regeneration from the cached raw pass (`n_reused: 16`, 145 s) after the workbook writer changed; every number is identical to the first pass.
- **Local-only fix.** The tracked `master_minus_arxiv/ebm.pkl` is a skops archive with no
  `.hmac` sidecar, and `load_model` refuses such artifacts, which silently drops that profile to
  naive fusion on a fresh checkout. The sidecar was generated locally with the repo's default
  HMAC key (git-ignored) so the profile could run calibrated. This is worth fixing in the repo.

## Per-profile summary

Short names: `base` = `master_minus_arxiv`, `v1` = `master_minus_arxiv_robust_v1`,
`v2*` = `master_minus_arxiv_robust_v2*`. "conf" columns count the honest binary conformal set
(α = 0.05); "abstain" = `["human","LLM"]`.

| profile | status | SpecDetect in EBM | mean p(LLM) | RED | AMBER | GREEN | UNCERTAIN | conf [LLM] | conf [human] | abstain |
|---|---|---|---|---|---|---|---|---|---|---|
| `base` | degraded | 9 columns, detector off (imputed) | 0.699 | 7 | 5 | 1 | 3 | 12 | 1 | 3 |
| `v1` | degraded | no | 0.685 | 3 | 7 | 0 | 6 | 10 | 0 | 6 |
| `v2` | degraded | 9 columns | 0.453 | 2 | 0 | 1 | 13 | 2 | 1 | 13 |
| `v2a_ablated` | degraded | 9 columns | 0.446 | 2 | 0 | 1 | 13 | 2 | 1 | 13 |
| `v2b_cascade` | degraded | 9 columns | 0.922 | 14 | 2 | 0 | 0 | 16 | 0 | 0 |
| `v2c_lowint` | degraded | 9 columns (interactions=5) | 0.553 | 3 | 0 | 1 | 12 | 3 | 1 | 12 |
| `v2d_gated` | degraded | 9 columns | 0.340 | 1 | 0 | 4 | 11 | 1 | 4 | 11 |
| `v2e_no_specdetect` | degraded | none (dropped) | 0.116 | 0 | 0 | 14 | 2 | 0 | 14 | 2 |
| `v2f_pllm_only` | degraded | 3 columns (p_llm only) | 0.151 | 1 | 0 | 14 | 1 | 1 | 14 | 1 |
| `v2g_clean_control` | degraded | none (dropped) | 0.118 | 0 | 0 | 14 | 2 | 0 | 14 | 2 |
| `v2h_clean_full` | degraded | 9 columns | 0.459 | 2 | 0 | 1 | 13 | 2 | 1 | 13 |
| `v2i_v1_calibrator` | degraded | none (v1 schema + v1 calibrator) | 0.126 | 0 | 0 | 14 | 2 | 0 | 14 | 2 |

No profile produced `YELLOW` or `MIXED`. "SpecDetect in EBM" counts the `specdetect_*`
columns in each profile's EBM artifact (`_cairn_feature_names`). The detector itself is
enabled in every v2-family config and disabled in `base`/`v1`; the tracked `base` EBM
nevertheless carries the nine columns, so they are mean-imputed on every input it scores.
The four EBMs without SpecDetect's spectral columns (`v2e`, `v2f`, `v2g`, `v2i`) are exactly
the four that call 14/16 prompts human.

## p(LLM) per prompt × profile

| prompt | words | router | base | v1 | v2 | v2a | v2b | v2c | v2d | v2e | v2f | v2g | v2h | v2i |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MEC-01 | 414 | prose | 0.692 | 0.602 | 0.279 | 0.224 | 0.970 | 0.428 | 0.136 | 0.041 | 0.040 | 0.043 | 0.289 | 0.050 |
| MEC-02 | 203 | prose | 0.792 | 0.704 | 0.339 | 0.340 | 0.978 | 0.299 | 0.215 | 0.237 | 0.275 | 0.245 | 0.347 | 0.276 |
| MEC-03 | 430 | prose | 0.735 | 0.642 | 0.638 | 0.636 | 0.967 | 0.846 | 0.532 | 0.039 | 0.152 | 0.041 | 0.634 | 0.048 |
| MEC-04 | 437 | prose | 0.704 | 0.585 | 0.238 | 0.232 | 0.954 | 0.467 | 0.213 | 0.047 | 0.044 | 0.048 | 0.242 | 0.055 |
| MEC-05 | 498 | prose | 0.504 | 0.712 | 0.044 | 0.047 | 0.725 | 0.045 | 0.027 | 0.037 | 0.036 | 0.038 | 0.048 | 0.045 |
| MEC-06 | 213 | prose | 0.763 | 0.718 | 0.383 | 0.379 | 0.761 | 0.295 | 0.213 | 0.032 | 0.031 | 0.033 | 0.389 | 0.035 |
| MEC-07 | 355 | prose | 0.824 | 0.720 | 0.647 | 0.659 | 0.986 | 0.794 | 0.542 | 0.061 | 0.170 | 0.064 | 0.652 | 0.073 |
| MEC-08 | 432 | prose | 0.707 | 0.619 | 0.465 | 0.455 | 0.978 | 0.761 | 0.285 | 0.041 | 0.113 | 0.042 | 0.476 | 0.050 |
| CIV-01 | 203 | prose | 0.770 | 0.713 | 0.308 | 0.311 | 0.960 | 0.252 | 0.175 | 0.040 | 0.039 | 0.041 | 0.317 | 0.052 |
| CIV-02 | 230 | prose | 0.877 | 0.855 | 0.624 | 0.632 | 0.988 | 0.645 | 0.417 | 0.046 | 0.128 | 0.047 | 0.632 | 0.050 |
| CIV-03 | 395 | mixed | 0.957 | 0.989 | 0.998 | 0.996 | 0.999 | 0.998 | 0.992 | 0.942 | 0.982 | 0.942 | 0.998 | 0.951 |
| CIV-04 | 471 | prose | 0.237 | 0.400 | 0.259 | 0.226 | 0.732 | 0.383 | 0.128 | 0.043 | 0.040 | 0.044 | 0.268 | 0.051 |
| CIV-05 | 524 | prose | 0.777 | 0.742 | 0.341 | 0.333 | 0.974 | 0.735 | 0.287 | 0.101 | 0.089 | 0.103 | 0.352 | 0.121 |
| CIV-06 | 200 | prose | 0.842 | 0.830 | 0.603 | 0.591 | 0.987 | 0.755 | 0.447 | 0.046 | 0.127 | 0.048 | 0.608 | 0.051 |
| CIV-07 | 236 | prose | 0.694 | 0.673 | 0.280 | 0.282 | 0.957 | 0.395 | 0.213 | 0.062 | 0.057 | 0.064 | 0.285 | 0.068 |
| CIV-08 | 249 | prose | 0.304 | 0.458 | 0.800 | 0.790 | 0.841 | 0.759 | 0.619 | 0.035 | 0.097 | 0.037 | 0.802 | 0.043 |

## Label per prompt × profile

| prompt | base | v1 | v2 | v2a | v2b | v2c | v2d | v2e | v2f | v2g | v2h | v2i |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MEC-01 | AMBER | UNCERTAIN | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | GREEN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| MEC-02 | RED | AMBER | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | UNCERTAIN | UNCERTAIN | UNCERTAIN | UNCERTAIN | UNCERTAIN |
| MEC-03 | AMBER | UNCERTAIN | UNCERTAIN | UNCERTAIN | RED | RED | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| MEC-04 | AMBER | UNCERTAIN | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| MEC-05 | UNCERTAIN | AMBER | GREEN | GREEN | AMBER | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN |
| MEC-06 | RED | AMBER | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| MEC-07 | RED | AMBER | UNCERTAIN | UNCERTAIN | RED | RED | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| MEC-08 | AMBER | UNCERTAIN | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-01 | RED | AMBER | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | GREEN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-02 | RED | RED | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-03 | UNCERTAIN | RED | RED | RED | RED | RED | RED | UNCERTAIN | RED | UNCERTAIN | RED | UNCERTAIN |
| CIV-04 | GREEN | UNCERTAIN | UNCERTAIN | UNCERTAIN | AMBER | UNCERTAIN | GREEN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-05 | RED | AMBER | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-06 | RED | RED | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-07 | AMBER | AMBER | UNCERTAIN | UNCERTAIN | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | UNCERTAIN | GREEN |
| CIV-08 | UNCERTAIN | UNCERTAIN | RED | RED | RED | UNCERTAIN | UNCERTAIN | GREEN | GREEN | GREEN | RED | GREEN |

`UNCERTAIN` here always means the conformal set was non-singleton (`abstention_reason:
conformal_set`); the point estimate is still in the p(LLM) table. The conformal set is
router-conditional: for the `prose` cell the profiles need p(LLM) above ~0.54 (`base`), ~0.65
(`v1`, `v2b`) or ~0.69–0.77 (the rest of the v2 family) to drop `human` from the set, and
below ~0.25 (`base`), ~0.35 (`v1`, `v2b`) or ~0.19–0.21 (the rest) to drop `LLM` — which is
why the v2 family abstains on most of this set while `base` commits.
For the `mixed` cell (CIV-03) `human` stays in the set up to p(LLM) ≈ 0.95–0.99, which is why
`base`, `v2e`, `v2g` and `v2i` abstain on CIV-03 at 0.94–0.96 while the others call it `RED`.

## Per-prompt consensus

| prompt | domain | p(LLM) min / median / max over 12 profiles | conf [LLM] | conf [human] | abstain | `base` | `v1` |
|---|---|---|---|---|---|---|---|
| MEC-01 | Mechanical | 0.04 / 0.25 / 0.97 | 2 | 5 | 5 | AMBER (0.69) | UNCERTAIN (0.60) |
| MEC-02 | Mechanical | 0.21 / 0.32 / 0.98 | 3 | 0 | 9 | RED (0.79) | AMBER (0.70) |
| MEC-03 | Mechanical | 0.04 / 0.63 / 0.97 | 3 | 4 | 5 | AMBER (0.73) | UNCERTAIN (0.64) |
| MEC-04 | Mechanical | 0.04 / 0.23 / 0.95 | 2 | 4 | 6 | AMBER (0.70) | UNCERTAIN (0.58) |
| MEC-05 | Mechanical | 0.03 / 0.05 / 0.73 | 2 | 9 | 1 | UNCERTAIN (0.50) | AMBER (0.71) |
| MEC-06 | Mechanical | 0.03 / 0.34 / 0.76 | 3 | 4 | 5 | RED (0.76) | AMBER (0.72) |
| MEC-07 | Mechanical | 0.06 / 0.65 / 0.99 | 4 | 4 | 4 | RED (0.82) | AMBER (0.72) |
| MEC-08 | Mechanical | 0.04 / 0.46 / 0.98 | 2 | 4 | 6 | AMBER (0.71) | UNCERTAIN (0.62) |
| CIV-01 | Civil | 0.04 / 0.28 / 0.96 | 3 | 5 | 4 | RED (0.77) | AMBER (0.71) |
| CIV-02 | Civil | 0.05 / 0.63 / 0.99 | 3 | 4 | 5 | RED (0.88) | RED (0.86) |
| CIV-03 | Civil | 0.94 / 0.99 / 1.00 | 8 | 0 | 4 | UNCERTAIN (0.96) | RED (0.99) |
| CIV-04 | Civil | 0.04 / 0.23 / 0.73 | 1 | 6 | 5 | GREEN (0.24) | UNCERTAIN (0.40) |
| CIV-05 | Civil | 0.09 / 0.34 / 0.97 | 3 | 4 | 5 | RED (0.78) | AMBER (0.74) |
| CIV-06 | Civil | 0.05 / 0.60 / 0.99 | 3 | 4 | 5 | RED (0.84) | RED (0.83) |
| CIV-07 | Civil | 0.06 / 0.28 / 0.96 | 3 | 4 | 5 | AMBER (0.69) | AMBER (0.67) |
| CIV-08 | Civil | 0.04 / 0.54 / 0.84 | 4 | 4 | 4 | UNCERTAIN (0.30) | UNCERTAIN (0.46) |

Reading across the two production-grade profiles only: they agree on CIV-02 and CIV-06 (`RED`),
CIV-07 (`AMBER`) and CIV-08 (`UNCERTAIN`). `base` scores higher than `v1` on 12/16 prompts;
`v1` is higher on MEC-05, CIV-03, CIV-04 and CIV-08. There is no systematic MEC-vs-CIV
difference in either profile (group means within 0.04 of each other, in opposite directions).

## Detector-level view

Raw detector scores are calibrator-independent, so they are the same for every profile; the
table shows what the 13 phase-1/2 detectors actually saw. `-` = skipped by the router
(`prompt_structure`, `voice_spec`, `instruction_density` only run on `prompt`/`mixed` text);
`semantic_kinematics` is `heuristic_suppress`ed in every profile and contributes only its
`sk_*` kinematics signals to the EBM.

| prompt | router | preamble | fingerprint | nssi | prompt_structure | voice_spec | instr_density | surprisal | binoculars ratio | fast_detectgpt curvature | tocsin (dropped) | mixed_boundary | specdetect energy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MEC-01 | prose | 0 | 0 | 0 | - | - | - | 0.853 | 1.057 | -0.144 | -0.452 | 0 | 11.12 |
| MEC-02 | prose | 0 | 4.93 | 0.129 | - | - | - | 0.853 | 1.050 | -0.096 | -0.114 | 0 | 11.36 |
| MEC-03 | prose | 0 | 0 | 0 | - | - | - | 1.306 | 1.050 | -0.608 | -1.333 | 0 | 13.82 |
| MEC-04 | prose | 0 | 0 | 0 | - | - | - | 0.756 | 1.057 | -0.072 | -0.150 | 0 | 11.00 |
| MEC-05 | prose | 0 | 0 | 0.013 | - | - | - | 0.810 | 1.035 | 0.019 | 0.015 | 0 | 10.85 |
| MEC-06 | prose | 0 | 0 | 0 | - | - | - | 0.941 | 1.036 | -0.078 | -0.093 | 0 | 11.47 |
| MEC-07 | prose | 0 | 0 | 0 | - | - | - | 1.080 | 1.072 | -0.177 | -0.321 | 0 | 12.35 |
| MEC-08 | prose | 0 | 0 | 0 | - | - | - | 0.849 | 1.059 | -0.073 | -0.126 | 0 | 12.41 |
| CIV-01 | prose | 0 | 0 | 0 | - | - | - | 0.656 | 1.029 | -0.112 | -0.156 | 0 | 11.32 |
| CIV-02 | prose | 0 | 0 | 0.038 | - | - | - | 0.722 | 1.034 | -0.237 | -0.392 | 0 | 12.39 |
| CIV-03 | mixed | 0 | 0 | 0 | 0.203 | 0.506 | 0.253 | 0.994 | 1.051 | -0.541 | -1.494 | 0 | 12.56 |
| CIV-04 | prose | 0 | 0 | 0 | - | - | - | 0.812 | 1.051 | -0.072 | -0.114 | 0 | 11.11 |
| CIV-05 | prose | 0 | 0 | 0 | - | - | - | 0.633 | 1.056 | 0.080 | 0.044 | 0 | 10.51 |
| CIV-06 | prose | 0 | 0 | 0 | - | - | - | 0.712 | 1.042 | -0.147 | -0.213 | 0 | 13.21 |
| CIV-07 | prose | 0 | 0 | 0 | - | - | - | 0.683 | 1.052 | -0.074 | -0.087 | 0 | 11.50 |
| CIV-08 | prose | 0 | 0 | 0 | - | - | - | 1.048 | 1.019 | -0.389 | -1.155 | 0 | 12.52 |

What the profiles do with those scores:

- **Two calibrator generations, two readings of the same raw scores.** `base`, `v1` and `v2i`
  use the older per-detector isotonic tables; the rest of the v2 family uses the May-2026
  retrain. Mean calibrated p(LLM) over the 16 prompts: `surprisal_dynamics` 0.85–0.89 (old) vs
  0.55 (new); `contrastive_lm` 0.41–0.50 vs 0.30; `mixed_boundary` 0.89–0.92 vs 0.69. The raw
  signal is identical; the calibrators disagree about what it means for this kind of text.
- **Top EBM contributors.** `base`/`v1`: `binoculars_p_llm` (15/16 prompts) and the
  `surprisal_diversity × ps_confidence` interaction (12/16). v2 family with SpecDetect features:
  `surprisal_confidence` (10–14/16) and SpecDetect spectral features (`spectral_entropy`,
  `total_energy`, band interactions). `v2e`/`v2f`/`v2g`/`v2i`: Binoculars (`binoculars_p_llm`
  or `binoculars_ratio`) and `nssi_confidence`. In other words the v2 EBMs largely re-anchor on SpecDetect and detector
  *confidence* features rather than on detector p(LLM) values.
- **Fast-DetectGPT curvature is negative (human-leaning, heuristic p ≈ 0.15) on 14/16 prompts**
  and is not covered by any calibrator; it never appears among the top contributors.
  Binoculars ratios sit in a
  tight 1.02–1.07 band across all 16 prompts, i.e. the detector barely separates them.
- **TOCSIN** raw `combined_w` ranged −1.49 to +0.04 but was discarded on every prompt by every
  profile (legacy calibrator, see TL;DR 5), so its columns in the EBM were filled with training
  means.

## Caveats — how (not) to read this

- **No accuracy claim is possible.** Without labels, agreement between profiles is the only
  internal consistency signal, and it is poor. If the intent is to find out whether these
  prompts are LLM-written, the sheet needs ground truth (or a matched human-written control set)
  before any of these numbers can be scored.
- **These are prose profiles applied to prompts.** The `master_minus_arxiv` family is trained on
  responses/prose (HC3, OOD prose, domain corpora); the router agrees, classifying 15/16 prompts
  as `prose`. The prompt-domain profile (`prompts_only`) is the documented pick for
  human-vs-LLM *prompt* detection and is exactly the one that cannot be run from this checkout.
  `docs/benchmarks/lighthouse_prompts_calibration.md` also documents that `prompts_only`
  p(LLM) on long tax prompts (150–760 words) tracked *length* rather than authorship — the
  Buckeye prompts sit in the same 200–524-word range, so that caveat would apply there too.
- **`degraded` is not cosmetic.** The shipped calibrators pre-date the faithful TOCSIN rebuild;
  until the `master_minus_arxiv` family is retrained (the design spec covers `prompts_only`
  only), production runs of these profiles carry the same TOCSIN gap as this screen.
- **SpecDetect ran on `gpt2` at inference while its calibrator was fitted on GPT-J spectra**
  (documented in `configs/.profile_master_minus_arxiv_robust_v2.yaml`) — one more reason the
  v2 family's numbers should be read as research output.
- **Ten of the twelve profiles are documented regressions** (`robust_v2` and `v2a`–`v2i`;
  `docs/benchmarks/robust_v2_validation/VERDICT.md`, `FOLLOWUP_VERDICT.md`). They were run
  because the request was "each calibration profile available", and because their spread is
  itself informative about how fragile the verdicts are — not because any of them is a
  candidate answer.

## Reproduce

```bash
pip install -e ".[fusion,tier2]"           # torch, transformers, interpret, skops ...
# stage the sheet as JSONL (id, domain, task, environment, word_count, text) under data/
python scripts/run_profile_screen.py \
    --dataset data/buckeye_test_prompts.jsonl \
    --phases 1,2 \
    --out-dir docs/benchmarks/buckeye_prompts \
    --verify-direct master_minus_arxiv_robust_v1 \
    --xlsx
```

`--dry-run` prints profile availability and the superset detector set without running any
detector. `--profiles a,b` restricts the run; `--phases 1` or `1,2,3` change the cascade window;
the raw feature cache (`data/<dataset>_features_p12.jsonl`) is reused on re-runs when the
superset config is unchanged.

## Files

| file | contents |
|---|---|
| `results.md` | auto-generated full-width tables (profile summary, four matrices, verification, availability) |
| `results.json` | everything: run metadata + versions, sample metadata (ids, hashes, excerpts), profile availability, raw-pass timings, per profile × prompt determinations with per-detector rows (raw score, calibrated p(LLM), confidence, calibrator flags, scalar signals), verification diff |
| `results_long.csv` | one row per profile × prompt (router, p(LLM), label, conformal set, top features, own-cascade counterfactuals, reason) |
| `detectors_long.csv` | one row per profile × prompt × detector |
| `profiles.json` | all 85 config files with artifact status |
| `results.xlsx` | static export: Summary, p_llm / label / conformal_set / own_cascade_phase2 matrices, both long tables, profiles, run metadata |

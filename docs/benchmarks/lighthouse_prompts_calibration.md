# Project Lighthouse — `prompts_only` calibration probe

**Date:** 2026-06-26
**Profile:** `prompts_only` (calibrated EBM + isotonic + split-conformal)
**Corpus:** 40 **known-LLM** tax-help prompts — Project Lighthouse detection
screens, Form A (A1–A20) and Form B (B1–B20). Saved as a labelled regression
slice at `data/lighthouse_prompts_eval.jsonl` (all `label: 1`).
**Repro:** `python scripts/_lighthouse_screen_cairn.py`
(needs `CAIRN_ALLOW_LEGACY_PICKLE=1` only until the artifact migration below is
committed; `ebm.skops` loads without the flag).

## Result — only ~30% of known-LLM prompts are flagged

| Metric | Count |
|---|---|
| p(LLM) ≥ 0.5 (point-estimate LLM call) | **12 / 40** (30%) |
| Honest conformal set `["LLM"]` (confident LLM) | **10 / 40** (25%) |
| Conformal abstains `["human","LLM"]` | 27 / 40 |
| Conformal `["human"]` (**confident miss**) | **3 / 40** — A10, B8, B17 |
| mean p(LLM) | 0.365 · median **0.124** |

## The verdict is a function of prompt *register*

The 40 items split 10/10/20 by surface register, and that split predicts the
score almost perfectly:

| Style | n | p≥0.5 | mean p(LLM) | conf `["LLM"]` | conf `["human"]` |
|---|---|---|---|---|---|
| **Preamble** ("Certainly! / Of course! Here is the prompt…") | 10 | **10/10** | **0.968** | 9 | 0 |
| **Formal 1st-person** ("I am \[Name\], I need…") | 10 | 1/10 | 0.172 | 1 | **3** |
| **Casual / natural** (lowercase, typos, or fluent human voice) | 20 | 1/20 | 0.159 | 0 | 0 |

- **Preamble tier** (A4, A7, A11, A14, A16, B1, B4, B9, B14, B19) → 0.91–1.00.
  The `preamble` detector fires at 1.00 and the EBM rides it to RED. This is
  the only register the profile reliably catches.
- **Everything else slips.** Of 30 non-preamble items, only **A6** (0.963 —
  `instruction_density` spiked to 0.89) and **A13** (0.755 — longest item, 294
  words) clear 0.5. The other 28 sit at p(LLM) 0.06–0.36 (read as human).
- **Confident misses:** A10, B8, B17 — all clean "I am \[Name\]…" prompts —
  get a singleton `["human"]` conformal set at p≈0.057.

**Mechanism.** The `preamble` detector plus a single `instruction_density` hit
do *all* the catching. `surprisal_dynamics` is pinned ~0.77 on nearly every
item regardless of truth (non-discriminative here; the EBM correctly discounts
it). So `prompts_only` behaves as a **surface-marker detector**: strip the
assistant preamble and write casual or clean-formal, and it evades. (Compare
`perturbation_tocsin_longform_saturation.md` — same "one detector carries a
near-constant, non-discriminative signal" shape, on the prose side.)

## Two artifact/field bugs found while running this

1. **`prompts_only/ebm.pkl` was a pre-skops legacy pickle** (git-ignored, no
   HMAC sidecar) — the loader refused it without `CAIRN_ALLOW_LEGACY_PICKLE=1`,
   and the first run *silently* fell back to flat ~0.65 heuristic numbers.
   Migrated non-destructively to `ebm.skops` (+ `.hmac` + `.meta.json`),
   verified prediction-identical to the pickle on all 40 items
   (`scripts/_migrate_prompts_only_skops.py`); config now points at the skops
   artifact. The original `.pkl` is left in place.
2. **`Determination.calibration_status` always reported `"heuristic"`** —
   `decision.py::decide()` never set it, so a fully calibrated EBM+conformal
   run mislabelled its own determination (and `report.py`) as heuristic, even
   though `/health` said `calibrated`. Fixed in `pipeline.py` (stamp the
   pipeline status onto the determination at both `decide()` sites); regression
   test in `tests/test_calibration_status_propagation.py`.

## Retrain recommendation — do **not** retrain on these 40

These 40 are an excellent *eval/regression* slice but a poor *training*
increment: the `prompts_only` corpus is ~30K rows, so +40 humanized examples is
~0.1% of the data and will not move the EBM's decision boundary — the
"carefully-humanized" and clean-formal classes need **corpus-scale** coverage
(hundreds–thousands of provenance-clean humanized/clean-formal LLM prompts)
before a retrain is worth running. Firing a full feature-cache → EBM-fit →
conformal-recalibrate → OOD-validate pass for +0.1% data is a no-op at best and
risks the known "retrained profile underperforms baseline" trap.

**Path to actually close the blind spot:**
1. Collect humanized + clean-formal LLM prompts at scale (the two missed
   registers), provenance-clean, balanced against the existing human prompts.
2. Re-cache features and retrain `prompts_only` (or a `prompts_robust` variant).
3. Gate on **this** slice (`data/lighthouse_prompts_eval.jsonl`) plus the
   existing prompt held-outs — track flag-rate per `style` so a future model
   that only learns the preamble tell again is caught.

---

## Follow-up: 31-variant graded benchmark — **length dominates** (2026-06-26)

Ran the companion graded benchmark (62 items: 31 raw + 31 tiered transforms,
each with a blind 3-lens human panel score, 0=def-AI … 100=def-human) through
the same calibrated `prompts_only`. All 62 are AI-generated. Slice saved at
`data/lighthouse_benchmark_eval.jsonl`; runner `scripts/_lighthouse_benchmark_cairn.py`.

| tier | blind human-ness | mean words | CAIRN p(LLM) | flag p≥.5 |
|---|---|---|---|---|
| T3 "blatant LLM" | ~7 | 602 | 0.961 | 10/10 |
| T2 "performed human" | ~50 | 457 | 0.943 | 11/11 |
| raw variants | 11–57 (mean 40) | 218 | 0.271 | 7/31 |
| T1 "genuine" (restraint rewrite) | ~75 | 240 | 0.376 | 2/10 |

**Headline: `corr(word_count, p_llm) = +0.887`** — p(LLM) tracks length far more
than the human-ness axis (Pearson `p_llm` vs blind score = −0.43, Spearman
−0.25, both largely a length proxy). Proof it's length not tier: the two *raw*
items that happen to be long — Expanded 1 (428 w → 0.96) and Expanded 2
(384 w → 0.93) — score like T2/T3 despite no humanization transform, while
short items (raw + T1, ~220–240 w) sit ~0.3 regardless of tier.

The seductive misread — "CAIRN catches the T2 discriminator 11/11, where weak
human reviewers fail" — is **mostly a length artifact**: T2/T3 were transformed
into verbose figure-dumping prose (~450–600 w); T1 genuine was rewritten
*shorter* with restraint (~240 w). This does not contradict the Form A/B
finding — those casual items were short. **Unified mechanism: `prompts_only`
p(LLM) on tax prompts is driven by (1) the preamble override and (2) raw
length**, not robust stylistic detection. Consequence: a long detailed prompt
from a real human would be flagged; a short AI prompt with no preamble passes.

**Implication for "use for calibration":** before any retrain, the eval/gate
must **length-match** human vs LLM (or include `log_word_count` as a controlled
covariate), otherwise length leakage will masquerade as detection skill. The
actionable fix is to down-weight / decorrelate the length features for the
prompt domain and re-validate on a length-balanced set — not to celebrate the
T2 catch rate.

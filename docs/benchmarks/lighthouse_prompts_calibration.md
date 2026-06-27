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
prompt domain and re-validate on a length-balanced set, not to celebrate the
T2 catch rate.

---

## Length-balanced control — CAIRN *does* discriminate in-distribution (2026-06-26)

Scored the full held-out `prompts_v5_test` (6,068 balanced prompts) through the
prompts_only EBM via the feature cache (`CachedPipeline`, fusion only, no GPT-J;
100% cache coverage). This isolates real content discrimination from length.

| measure | AUC |
|---|---|
| **Natural** (full set; human median 69 w, LLM 41 w) | **0.841** |
| Length-only (word count alone) | 0.244 -> **0.756** in the human-longer direction |
| **Within-bin** (length held ~constant), size-weighted | **0.697** |
| Matched subsample (identical class length dists) | 0.654 |
| `corr(word_count, p_llm)` | **-0.399** |

within-bin AUC by length: 5-15 w **0.69**, 15-30 **0.78**, 30-50 0.63, 50-80
0.68, 80-130 0.69, 130-260 0.73, **260+ 0.57** (near chance).

**This corrects the "no genuine signal" reading above.** In-distribution,
`corr(words,p_llm) = -0.399` (longer -> more human) is the **opposite sign** of
Lighthouse's +0.887. The `word_count` EBM shape function is **non-monotonic**:
"longer->human" across the training range (humans write longer prompts; length
is a legitimately strong in-distribution signal, AUC 0.756), but it **flips and
saturates to "longer->LLM" (+2.10) in the sparse OOD-long tail** (>260 w = only
1.6% of the test set). The Lighthouse tax prompts (150-760 w) sit entirely in
that tail, so their verdict is set by the saturated length feature, and content
discrimination there is near chance (260+ bin AUC 0.57).

**Corrected takeaway.** prompts_only is **not** signal-free: it has modest but
real content discrimination (content-only AUC ~0.65-0.70) that **degrades with
length** and **breaks (sign-flips + saturates) on OOD-long prompts**. "Length
dominates" is specific to the OOD-long regime the Lighthouse set probes; it is
not the in-distribution behaviour.

**Actionable.** (1) Guard length features against OOD extrapolation: prompts
beyond the training length range should not inherit the saturated `word_count`
contribution (clamp, or add an `is_oversized` gate). (2) Any prompt-domain
retrain must include **long prompts in both classes** so the `word_count` shape
cannot extrapolate to a sign-flip. (3) Report content-only (within-bin / length-
matched) AUC alongside headline AUC so length leakage cannot masquerade as skill.
Eval: `scripts/_lengthbalanced_eval_cairn.py`.

---

## Fix shipped: OOD-length guard `fusion.word_count_clamp` (2026-06-26)

Implemented the guard from fix (1) above. `FeatureAssembler` now takes an
optional `word_count_clamp`; when set it caps `word_count` before the length
features (`word_count` / `log_word_count` / `is_*`) are computed, so prompts
longer than the training range cannot inherit the EBM's saturated long-tail
length contribution. Config-gated (`fusion.word_count_clamp`), plumbed through
`EBMFusion` and `pipeline._build_fusion`; **default None = no change for any
other profile**. Set to **200** in `.profile_prompts_only.yaml` (upper edge of
the dense prompt distribution; the `word_count` shape turns positive ~245 w and
saturates to +2.1 above ~340 w). Tests: `tests/test_word_count_clamp.py`; full
suite 798 passed.

**In-distribution: no harm** (held-out prompts_v5_test, via cache):

| measure | no clamp | clamp 200 |
|---|---|---|
| Natural AUC | 0.841 | 0.834 |
| Within-bin (weighted) | 0.697 | 0.697 |
| Matched AUC | 0.654 | 0.652 |
| 260+ bin AUC | 0.566 | **0.597** |

**OOD-long: length leakage removed** (62-item graded benchmark, all LLM, all >200 w):

| tier | p(LLM) no clamp | p(LLM) clamp 200 |
|---|---|---|
| T3 blatant (602 w) | 0.961 (10/10 flagged) | 0.237 (0/10) |
| T2 performed (457 w) | 0.943 (11/11) | 0.155 (0/11) |
| raw (218 w) | 0.271 (7/31) | 0.116 (0/31) |
| T1 genuine (240 w) | 0.376 (2/10) | 0.178 (0/10) |
| **flagged p≥0.5** | **30/62** | **0/62** |

Lighthouse Form A/B (mostly <200 w): mean 0.365 -> 0.339; the short
preamble-marked items are unaffected (still caught, p ~0.95-1.00) — the clamp
only binds on the >200 w items.

**Honest read of the result.** The clamp is a **correctness fix, not a detection
upgrade**. It removes the *length-leakage false confidence* — the same mechanism
that would have flagged a long *human* prompt — and preserves in-distribution
skill (AUC 0.834). It does NOT make CAIRN catch long LLM prompts: the 62 benchmark
items now score ~0.1-0.24 because, with length neutralised, CAIRN's content
discrimination on long prompts is ~chance (within-bin 260+ AUC ~0.60). Catching
long LLM prompts on genuine content tells still requires a retrain with long
prompts in **both** classes. What the clamp buys is that prompts_only will no
longer flag a long human submission purely for being long.

### Retrain feasibility audit — the long-prompt retrain is DATA-BLOCKED

Counted long prompts by class across the prompt corpora:

| corpus | human ≥300 w | LLM ≥300 w | human ≥400 w | human ≥500 w |
|---|---|---|---|---|
| prompts_only_train (10,366/class) | 13 | 131 | 0 | 0 |
| full human pool (15,245) | 26 | — | 1 | 0 |

The entire 15,245-prompt human corpus has **26** prompts ≥300 w, **1** ≥400 w,
**0** ≥500 w. The Lighthouse prompts are 400–760 w — there is **no human
training data where they live**. The only long examples that exist are LLM
(~10:1 at ≥300 w), which is exactly why the EBM learned the saturating
"long→LLM" shape. A length-balanced long-prompt retrain is therefore
**infeasible** with available data, and a naive retrain (adding the long LLM
benchmark prompts) would *reinforce* the bias. The clamp is the operative fix;
the only route to genuine long-prompt content discrimination is **collecting
real long human prompts at scale** (a data effort — synthetic "human" data is
self-defeating). Audit not scripted (one-off count); reproduce via word-count
histograms over `data/prompts_human.jsonl` / `data/prompts_llm.jsonl`.

### Stage 1 operating-point measurement — ensemble caps ~0.39 TPR (2026-06-26)

Target: TPR >= 0.70 at 5-10% FPR (screening) on the 102 humanized LLM prompts
(Lighthouse A+B + 62 graded) vs human prompts. CPU-only via cache
(`scripts/_stage1_cpu.py`, `_stage1_cpu_clampoff.py`):

| operating point | AUC | TPR@5% | TPR@10% | TPR@20% |
|---|---|---|---|---|
| clamp ON (shipped) | 0.247 (inverted) | 0.11 | 0.11 | 0.11 |
| clamp OFF (recall) | 0.489 (~chance) | 0.36 | 0.39 | 0.42 |

Length split (clamp off, @10% FPR): >=300 w -> TPR 1.00; <300 w -> TPR 0.19.
The entire clamp-off TPR is the length trick catching the 25 long prompts; on
the 77 short/casual prompts the ensemble is ~chance, and that length signal
would false-positive long humans. **The current heuristic-detector + EBM
ensemble cannot reach 0.70** — content discrimination on humanized prompts is
~chance (they were built to evade surface tells). The clamp must be gated OFF
for recall (it inverts to AUC 0.247), but even off it tops out ~0.39. This is a
FEATURE ceiling (the detectors don't capture humanized-AI signal), so better
fusion can't close it — Stage 2 needs a stronger (neural) detector + humanized
training data.

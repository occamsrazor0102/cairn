# Detector slate 2026-08-26 -- scoping + design

Date: 2026-08-26
Status: implemented, code-only (`enabled: false`). Awaiting a local
feature-cache rebuild and profile retrain before any calibrated profile
changes.
Parents: `2026-05-07-market-leadership-roadmap.md` (§B2 new detector
heads), `2026-05-07-post-honest-calibration-roadmap.md` (§A3
hidden-reasoning sub-detector).

## Goal

Add three stdlib-only detectors, each aimed at a vulnerability the
benchmarks already measure, without repeating the two integration
failures that preceded them.

The three vulnerabilities:

1. **`summarization_attack` AUROC 0.4167 -- below random.**
   `docs/benchmarks/attack_coverage.md` flags this as the one cell in the
   per-attack grid where the ensemble systematically calls LLM text
   human. Summarize-then-re-expand defeats both the register detectors
   and the likelihood detectors.
2. **`verbosity=low` short-form costs 15-24 AUROC points**
   (`docs/SESSION_RECAP.md`). Every LM-backed detector carries a
   `min_text_length` floor of 50-60 words, so short input falls back to
   the six Phase-1 lexical detectors -- four of which sit at chance on
   `docs/benchmarks/calib2000/per_detector_auroc.csv`.
3. **OpenAI hidden reasoning is ~6 AUROC points stealthier** than other
   frontier providers (0.9358 vs Gemini 0.9961), per the
   matched-provider eval in `README.md`.

### Acceptance gates

Measured locally after the retrain, per detector:

| detector | gate | abort |
|---|---|---|
| `compression_profile` | `summarization_attack` per-attack AUROC improves over 0.4167 and clears 0.55 | no improvement on the summarization cell -> disable, keep as EBM-only features |
| `typographic_signature` | `verb_low` slice improves, or in-domain AUROC on samples < 150 words improves >= 0.02 | regression > 0.01 on any full-length slice -> disable |
| `reasoning_artifact` | OpenAI per-provider AUROC >= 0.96 **AND** Gemini drops <= 0.02 (the A3 over-fit guard) | Gemini drops > 0.02 -> the detector learned one provider's UX, roll back |

Slate-wide, from `docs/retrain-with-new-detectors.md`: in-domain >= 0.94,
RAID >= 0.73, MAGE >= 0.60, no per-attack regression > 0.02.

## Scope correction from the roadmaps

§B2 lists six candidate heads and ranks `binoculars_v2` and
`encoder_head` first. Both are deferred here, for one reason: **the
binding constraint on this project is not detector quality, it is
integration.**

- **LAPD** never reproduced its paper (0.351 raw against a claimed
  0.852) and shipped inert, leaving orphan `lapd_*` columns behind
  (`docs/benchmarks/lapd_reproducibility_abort.md`).
- **SpecDetect** beat the entire ten-detector ensemble standalone --
  RAID 0.855 against 0.784 -- and still regressed the fused model
  (`docs/benchmarks/robust_v2_validation/`). Two root causes were
  eventually isolated:
  1. `_HEURISTIC_TOTAL_ENERGY_TO_P_LLM` was anchored over a guessed band
     (0.5 to 5.0) while the corpus median sat at 5.75, so **70.9% of
     samples saturated to `p_llm = 0.05`** and isotonic calibration
     could not recover information the heuristic had already destroyed.
  2. `token_cohesiveness` drew its sample positions from the **global
     `random` module**, so identical text scored differently across
     cache builds -- silently miscalibrating production against its own
     training distribution since v1.

A detector that is strong standalone and unshippable in fusion is worth
nothing. So the slate is chosen for integration safety first and signal
second: three detectors that are pure functions of their input, need no
optional dependency, and emit scores that **cannot saturate a heuristic
table** because they are bounded in `[0, 1]` by construction.

Once the slate lands and retrains cleanly, the heavier heads become a
much lower-risk proposition on top of a harness that has been shown to
work.

## The detectors

### `compression_profile` -- Phase 1, trivial, stdlib

Redundancy as a Kolmogorov-complexity proxy. Summarize-then-re-expand
pads a compressed semantic core back to length; that inflates
restatement without changing register, which is why the register and
likelihood detectors both miss it. No current detector measures
redundancy.

```
signals
  zlib_ratio         compressed/raw under zlib level 9   (32 KB window)
  lzma_ratio         same under lzma preset 6            (long-range)
  mattr_100          moving-average type-token ratio, window 100
  distinct_4         |set(4-grams)| / |4-grams|
  max_ngram_repeat   top 4-gram repeat count / n-gram count
  redundancy         0.6*(1 - mattr_100) + 0.4*(1 - distinct_4)
```

`p_llm` is derived from `redundancy` alone. The two compression ratios
are deliberately **excluded** from the composite: compression is
strongly length-dependent (short UTF-8 text compresses to ~0.65-0.75,
long text to ~0.30-0.40), so any fixed rescale band saturates for one
length regime -- exactly the SpecDetect failure. They ship as EBM
features instead, where `log_word_count` / `is_short` / `is_medium` /
`is_long` already exist to model that interaction.

Spans: `kind="repetition"` on the top five 4-grams occurring 3+ times,
filtered to disjoint offsets (sliding 4-grams overlap by construction,
and the SPA reports dropped overlaps in the legend).

### `typographic_signature` -- Phase 1, trivial, stdlib

Formatting habits. Every term is a **style ratio** -- which convention
was chosen where both were available -- not a per-1000-words rate.
Rates are the length trap: one em-dash in a 30-word note is 33 per 1000
words and saturates any fixed normalizer, while the same dash in a
3000-word essay is 0.33.

```
signals
  dash_style_ratio            typographic dashes / all dashes
  curly_quote_ratio           curly quotes / all quotes
  fancy_bullet_ratio          glyph bullets / all bullets
  oxford_comma_ratio          serial-comma coordinations / all coordinations
  heading_title_case          title-cased headings / all headings
  typography_score            mean of the five
```

Each ratio reports `0.0` until it has `_MIN_EVENTS_FOR_RATIO = 3`
observations. `0.0` means "no evidence", never "human" -- absence of a
habit is not itself a signal, and the EBM sees the raw counts anyway.

**Constrained to signals that survive `normalize_text`.** This is a hard
constraint, not a stylistic one. `cairn/pipeline.py` normalizes before
any detector runs, and `cairn/feature_cache.py` keys the cache on
`sha256(normalize_text(text))` -- so a detector reading raw text would
break the cache key, and two raw inputs normalizing to the same string
would collide and share cached detector output. Verified against
`normalize_text` directly:

| signal | fate |
|---|---|
| em/en dash, curly quotes, glyph bullets, Oxford comma, heading case | survives |
| ellipsis form | destroyed -- NFKC folds U+2026 to `...` |
| double space after a period | destroyed -- whitespace collapse |
| homoglyphs, zero-width chars, ligatures | destroyed |

Only the surviving set is implemented.

### `reasoning_artifact` -- Phase 2, cheap, stdlib

Implements roadmap §A3. Surface traces of hidden chain-of-thought:
deliberation phrasing carried out of the trace, self-correction, clauses
that announce an intent instead of performing it, enumeration
scaffolding, and register uniformity.

```
signals
  deliberation_density        per 1000 words
  self_correction_density     per 1000 words
  intent_density              per 1000 words
  enum_scaffold_density       per 1000 words
  register_uniformity         1 - (sentence-length CV / 0.75), clamped [0,1]
  artifact_score              mean of the five clamped terms
```

Phase 2 rather than Phase 1 despite costing nothing: it is a register
judgement over longer prose, and Phase 1 detectors feed
`cascade.aggregate_p_llm`, which decides whether phases 2 and 3 run at
all. Spans use `kind="reasoning"` -- **the first Phase-2 detector to
emit spans**; the six existing emitters are all Phase 1.

## Cascade-interaction note

`compression_profile` and `typographic_signature` join Phase 1, so they
enter the confidence-weighted `aggregate_p_llm` that gates phases 2 and
3. Both therefore cap their confidence deliberately low (0.80 and 0.65)
so neither can drive a premature short-circuit on the strength of a
coarse heuristic. The retrain must check cascade-phase distribution, not
only AUROC: a slate that improves AUROC while short-circuiting more
samples away from the phase-2 detectors has not actually improved the
system.

## Integration touch points

| File | Change |
|---|---|
| `cairn/detectors/compression_profile.py` | new |
| `cairn/detectors/typographic_signature.py` | new |
| `cairn/detectors/reasoning_artifact.py` | new |
| `cairn/cascade.py` | two added to `PHASE1_DETECTORS`, one to `PHASE2_DETECTORS` |
| `configs/default.yaml` | three `detectors.*` blocks, all `enabled: false` |
| `cairn/fusion/ebm.py` | 27 new columns (102 -> 129); strictly additive |
| `cairn/gui/static/index.html` | `.span-{repetition,reasoning,typography}` CSS, legend chips, `KIND_LABELS` |
| `tests/test_detectors/test_*.py` | new, 56 cases |
| `tests/test_ebm_missingness.py` | triplet list extended (also adds the long-missing `lapd` / `specdetect`) |
| `README.md` | phase table, detector count, span-emitter list |
| `docs/retrain-with-new-detectors.md` | v11 cache + slate gates |

No `cairn/pipeline.py::detector_availability` entry is needed: all three
are stdlib, so registry-load alone is a correct availability signal.
Verified -- all three report `available=True, requires=[]`.

Feature prefixes are the **detector ids**, not new hand-chosen aliases.
The older detectors alias (`contrastive_lm` -> `binoculars`,
`token_cohesiveness` -> `tocsin`, `prompt_structure` -> `ps`, ...) and
that mapping has to be maintained by hand in
`tests/test_ebm_schema_coverage.py::_ALIAS`. Using ids keeps the three
out of that map entirely.

## Verification performed

Runnable in a cloud session, and run:

- `pytest -q` -> **1179 passed**, up from 1123 on the same tree without
  the slate. The 16 remaining failures are identical before and after
  (`git stash` A/B): they are missing optional deps (`torch` /
  `transformers` for `lapd` / `specdetect`) and gitignored local data
  files. **Zero new failures.**
- Schema additivity proved by diffing `ASSEMBLER_FEATURE_NAMES` against
  `HEAD`: 102 -> 129 columns, **nothing removed**. `EBMFusion.__init__`
  raises only for columns an artifact needs and the assembler lacks, so
  existing artifacts keep loading and ignore the 27 new columns as
  `extra_in_assembler` until retrained.
- End-to-end `CairnPipeline.analyze_detailed` with all three enabled:
  routes, fires, fuses, and emits five span kinds -- all of them styled
  and labelled in the SPA.
- Each detector: determinism (repeat-call equality), bounded signals,
  SKIP contract, heuristic-table monotonicity, and a table-spans-the-
  full-range assertion that makes the SpecDetect saturation failure
  unreachable.

**Not verified, and not verifiable here:** any accuracy claim. The
corpora and feature caches are gitignored and local-only, and there is
no GPU or Modal token in a cloud session. Every number in the
acceptance-gate table above is a target to be measured locally, not a
result.

## Out of scope

| candidate | disposition |
|---|---|
| `encoder_head` | Biggest projected MAGE lever (§B2: +9-14) but needs a supervised training run and a torch-checkpoint artifact story -- `cairn/fusion/serialization.py` is skops+HMAC, built for sklearn. Defer to a local session. |
| `boundary_segmenter` | Replaces the dead `mixed_boundary` (0.507) and would add sub-sentence span evidence. Needs M4GT-Bench labels and the same infrastructure as `encoder_head`. Sequence after it. |
| `watermark_probe` | Cheap and near-deterministic when it fires, but positive-only semantics need care: the EBM mean-imputes NaN, so "did not fire" must not read as "human". Worth doing next. |
| obfuscation / evasion artifacts | Blocked on the cache-key invariant above, not on the detector. Needs a cache key that covers pre-normalization text. |
| `glimpse` | Would send the **submission** to a provider, colliding with the `contrastive_gen` privacy invariant in CLAUDE.md (task description only). Would need the same opt-in and leak-warning treatment as `task_description.auto_mode: llm`. Not recommended. |
| SpecDetect re-anchoring | Its heuristic table is a known root cause and the empirical percentiles are already recorded (p1/p25/p50/p75/p95/p99 = 3.0/4.8/5.8/6.9/9.0/10.9). High-value local job: one table edit plus a retrain. Orthogonal to this slate. |
| `style_cpc`, AMR §C5, LAPD revival, SpecDetect++ Track B | Research-stage; unchanged. |

## Known adjacent defects (not fixed here)

Found while reviewing, recorded so they are not lost:

- `semantic_kinematics`' suppress branch returns `SKIP`, and
  `FeatureAssembler` drops SKIP results -- so its `sk_*` columns never
  reach the EBM. One-line fix (`SKIP` -> `UNCERTAIN`, mirroring the
  `surprisal_dynamics` suppression path) already named as a fast-follow
  in `docs/retrain-with-new-detectors.md:148`.
- `surprisal_dynamics` reads `config["model_name"]` where every other
  tier-2 detector reads `config["model"]`, and `default.yaml` exposes
  neither.
- The `lapd_*` orphan columns remain in the assembler. Measured harmless
  (v2g ~= v2e in the robust_v2 grid), and removing a column would brick
  artifacts that carry it, so leaving them is correct -- but they should
  be dropped at the next schema-breaking retrain.

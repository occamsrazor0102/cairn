# CAIRN — Calibration & Training Pipeline

This doc covers the end-to-end flow that turns raw labeled text into
a deployable calibrated profile: corpus generation, feature caching,
per-detector calibration, EBM fusion training, conformal calibration,
and the analysis tooling (profile grid, LOO, loss-filter) used to
pick the production profile.

For an overview of the resulting profiles and which to deploy, see
[`docs/configuration.md`](configuration.md). For benchmark numbers,
see [`docs/benchmarks/README.md`](benchmarks/README.md).

## Overview — the four-stage pipeline

```
1. Corpus generation
   scripts/regenerate_llm_outputs.py   — multi-provider HC3 fan-out
   scripts/regen_domain_calib.py       — domain corpora (arxiv/gsm8k/...)
   scripts/download_*_samples.py       — OOD + prompts
                                          ▼
                       data/calib2000_v6_train.jsonl  (~94K rows)
                                          ▼
2. Feature caching (one-time, GPU-friendly)
   scripts/cache_pipeline_features.py
   scripts/modal_cache_features.py     — T4 offload via Modal
                                          ▼
                       data/all_features_v6.jsonl     (sha256-keyed)
                                          ▼
3. Profile training (per-profile, fast — feature cache reads <1ms)
   scripts/calibrate_detectors.py      — per-detector isotonic
   scripts/train_fusion.py             — EBM with interactions=15
   scripts/calibrate_conformal.py      — split-conformal, alpha=0.05
                                          ▼
                       models/calib2000/profiles/<name>/
                                          ▼
4. Evaluation + analysis
   scripts/run_benchmark.py            — single profile vs single test
   scripts/run_profile_grid.py         — N profiles × M tests
   scripts/run_per_source_loo.py       — leave-one-source-out matrix
   scripts/run_loss_filter.py          — point-level loss filtering
                                          ▼
                       docs/benchmarks/calib2000/
```

## Stage 1 — Corpus generation

### Multi-provider HC3 corpus (calib2000)

`scripts/regenerate_llm_outputs.py` regenerates HC3 questions through
any of six current frontier providers, preserving the original
human-side answer:

| Provider | Model | Notes |
|----------|-------|-------|
| Anthropic | `claude-sonnet-4-6` | `temperature=0.7` |
| OpenAI | `gpt-5.4-mini` (Responses API) | reasoning effort + verbosity sweep |
| Google | `gemini-3-flash-preview` | thinking_level MEDIUM (HIGH was costly) |
| Mistral | `mistral-medium-latest` | `temperature=0.89` |
| DeepSeek | `deepseek-chat` | openai-compat provider |
| Moonshot | `kimi-k2.6` | openai-compat provider |

The full v6 master training set is ~12K HC3 question-pairs (~5
subdomains × 400 prompts × 6 providers, with one provider 2 short).
Rows are tagged `hc3_finance`, `hc3_medicine`, `hc3_open`,
`hc3_reddit`, `hc3_wiki` for downstream source filtering.

Per-provider matched comparison (same prompts answered by every
provider) is cheaper than chasing more questions — see the
[per-provider results](benchmarks/README.md#per-provider-detection-difficulty-matched-comparison-eval).

### Domain corpora

`scripts/regen_domain_calib.py` fans the same prompt across all six
providers for math/code/scientific corpora:

- `domain_arxiv` — arXiv abstracts (provenance: pre-2021,
  pre-LLM-era for human authenticity)
- `domain_gsm8k` — GSM8K math word problems
- `domain_humaneval` — HumanEval coding prompts
- (`prompt_medqa`, `prompt_mmlu`, `prompt_bbh`, `prompt_gpqa` for
  prompts side)

The `--all-providers` mode runs every prompt through every provider
in parallel and uses resume logic (reads existing `source_id`s) to
recover from API quota walls.

### OOD validation samples

`scripts/download_ood_samples.py` pulls samples from three external
benchmarks for held-out validation:

- **M4** (`HuggingFaceH4/m4`) — multilingual + multidomain
- **RAID** (`liamdugan/raid`) — adversarial transformations
- **MAGE** (`yaful/MAGE`) — model-attribution benchmark

These are never used for training; only for OOD evaluation. The MAGE
sampler initially used the wrong label convention (`label=1=human`
not `label=0=human`); fixed by spot-checking known sources like
`cmv_human`. The output JSONLs live at `data/ood_*_test.jsonl`.

### Prompt detection samples

`scripts/download_prompt_samples.py` builds the prompt corpus
(distinct from response detection — these are *prompts* a human or
LLM wrote, not responses). Provenance discipline matters: human
prompts come from pre-LLM-era datasets to avoid contamination
(HC3 / HH-RLHF / BBH / TruthfulQA / GPQA / MedQA — all dated to the
pre-LLM era). LLM-authored prompts come from Alpaca / WizardLM /
Magpie / SlimOrca synthetic prompt datasets.

The corpus includes reasoning prompts (BBH, GPQA, GSM8K) and
"stump-the-model" style adversarial prompts in the mix.

## Stage 2 — Feature caching

### Why cache

Profile-grid + LOO experiments retrain the same EBM 13–16 times on
overlapping subsets of the same corpus. Without caching, each
retrain re-runs the full detector cascade (~1–2 hrs per profile);
with caching, retrain time drops to ~3 minutes.

### `cairn/feature_cache.py::CachedPipeline`

Drop-in replacement for `CairnPipeline`. Keys detector outputs by
`sha256(normalize_text(input))` and serves cached `LayerResult`
lists from a JSONL file. The cache is **calibrator-independent**:
it stores raw detector outputs (`raw_score`, `signals`, `spans`),
so per-detector calibration is applied at fusion-time and the
same cache feeds all 13+ profile variants.

### Building the cache

```bash
# Local: ~1–2 hrs depending on corpus size
python scripts/cache_pipeline_features.py \
    --dataset data/calib2000_v6_train.jsonl \
    --config configs/default.yaml \
    --out data/all_features_v6.jsonl

# Modal T4 GPU offload — faster for tier-2 (perturbation, contrastive_lm)
python scripts/modal_cache_features.py \
    --dataset data/calib2000_v6_train.jsonl \
    --out data/all_features_v6.jsonl
```

Modal usage requires `modal` + `python-dotenv` locally and a
`.env` with provider keys. The Modal function uses a persistent
HF cache volume so model downloads are amortized across runs.

## Stage 3 — Profile training

Three steps per profile, all cache-fed and fast (~3 minutes total):

```bash
# 1. Per-detector isotonic calibration on the training split
python scripts/calibrate_detectors.py \
    --dataset data/calib2000_v6_train.jsonl \
    --config configs/.profile_master_minus_arxiv.yaml \
    --out models/calib2000/profiles/master_minus_arxiv/detector_calibration.json \
    --features-cache data/all_features_v6.jsonl

# 2. EBM fusion with length-aware features and interactions=15
python scripts/train_fusion.py \
    --dataset data/calib2000_v6_train.jsonl \
    --config configs/.profile_master_minus_arxiv.yaml \
    --out models/calib2000/profiles/master_minus_arxiv/ebm.pkl \
    --features-cache data/all_features_v6.jsonl

# 3. Split-conformal wrapper on a disjoint calibration split
python scripts/calibrate_conformal.py \
    --dataset data/calib2000_v2_calib.jsonl \
    --config configs/.profile_master_minus_arxiv.yaml \
    --alpha 0.05 \
    --out models/calib2000/profiles/master_minus_arxiv/conformal.json \
    --features-cache data/all_features_v6.jsonl
```

The profile YAML just points at the three artifact paths:

```yaml
_profile: profile_master_minus_arxiv
_extends: default
fusion:
  model_path: models/calib2000/profiles/master_minus_arxiv/ebm.pkl
  conformal_path: models/calib2000/profiles/master_minus_arxiv/conformal.json
  detector_calibration_path: models/calib2000/profiles/master_minus_arxiv/detector_calibration.json
```

### Length-aware features

Added during this calibration round to give the EBM more capacity
for short-form text:

```python
"log_word_count": math.log1p(float(word_count)),
"is_short":  1.0 if word_count < 150 else 0.0,
"is_medium": 1.0 if 150 <= word_count < 500 else 0.0,
"is_long":   1.0 if word_count >= 500 else 0.0,
```

`interactions=15` (was 5) was raised so the EBM can capture
pairwise effects — `surprisal_dynamics × is_short` ranks among the
top contributors on short-form OOD samples.

## Stage 4 — Analysis tooling

### `scripts/run_profile_grid.py`

Trains N profiles by source-tag regex filtering and benchmarks each
on M test sets. Output: `docs/benchmarks/calib2000/profile_grid_summary.csv`
with one row per profile and four metrics (AUROC, ECE, TPR@1%FPR,
F1) per test. Resumable via `--skip-existing`.

### `scripts/run_per_source_loo.py`

Leave-one-source-out matrix: for each source-tag prefix in the
master corpus, train a profile that EXCLUDES that source, benchmark
on the standard 9 test sets, compute Δ vs the `everything`
baseline. Sources whose removal *helps* are training drags;
sources whose removal *hurts* are critical.

This is what surfaced `domain_arxiv` as the dominant drag (Δ +0.026
mean across 9 tests, +0.124 on RAID, +0.094 on verb_low) and led
directly to the `master_minus_arxiv` production profile.

### `scripts/run_loss_filter.py`

Point-level filtering by per-row model loss. Four strategies tested:
`high_loss_top30`, `high_loss_top50`, `uncertain` (boundary p_llm in
[0.3, 0.7]), `exclude_easy`. **Negative result** — all four
underperformed the `everything` baseline; high-loss strategies
inverted to AUROC < 0.5 because training only on hard cases without
diverse easy ones produces an anti-correlated model. Per-source LOO
is the better lever.

### `scripts/per_detector_auroc.py`

Computes per-detector AUROC across multiple datasets to surface
which detectors carry empirical signal. Findings:

- **Useless across the board** (~0.50 AUROC):
  `token_cohesiveness`, `mixed_boundary`, `preamble`,
  `fingerprint_vocab` — kept enabled but the EBM weights them low
- **Inverts on OOD**: `surprisal_dynamics` flipped sign on M4 / RAID
  in some cells — the EBM captures this via interactions but it's
  a signal worth watching
- **Strongest contributors**: `contrastive_lm`, `instruction_density`,
  `nssi`, the length features

## Reproducing a profile from scratch

```bash
# 1. Build training corpus (slow — ~12 hours fan-out across 6 providers)
python scripts/regenerate_llm_outputs.py \
    --providers anthropic openai gemini mistral deepseek moonshot \
    --subset all --max-per-class 400 \
    --out data/calib2000_v6_train.jsonl

# 2. Cache detector features (slow once — ~1–2 hours; or use Modal)
python scripts/modal_cache_features.py \
    --dataset data/calib2000_v6_train.jsonl \
    --out data/all_features_v6.jsonl

# 3. Train profile (fast — ~3 minutes)
python scripts/calibrate_detectors.py --dataset data/calib2000_v6_train.jsonl \
    --config configs/.profile_master_minus_arxiv.yaml \
    --out models/calib2000/profiles/master_minus_arxiv/detector_calibration.json \
    --features-cache data/all_features_v6.jsonl
python scripts/train_fusion.py --dataset data/calib2000_v6_train.jsonl \
    --config configs/.profile_master_minus_arxiv.yaml \
    --out models/calib2000/profiles/master_minus_arxiv/ebm.pkl \
    --features-cache data/all_features_v6.jsonl
python scripts/calibrate_conformal.py --dataset data/calib2000_v2_calib.jsonl \
    --config configs/.profile_master_minus_arxiv.yaml --alpha 0.05 \
    --out models/calib2000/profiles/master_minus_arxiv/conformal.json \
    --features-cache data/all_features_v6.jsonl

# 4. Benchmark
python scripts/run_benchmark.py \
    --dataset data/calib2000_full_test.jsonl \
    --config configs/.profile_master_minus_arxiv.yaml \
    --out docs/benchmarks/calib2000/profiles/master_minus_arxiv/in_domain.json \
    --features-cache data/all_features_v6.jsonl
```

Steps 1+2 are the expensive part; steps 3+4 are cheap and can be
re-run for any profile experiment.

## Lessons learned

A few things this calibration round surfaced that aren't obvious from
the code:

- **More data isn't always better.** The `everything` baseline
  (full v6, ~94K rows) lost 0.05–0.12 AUROC on OOD vs HC3-only
  because arxiv abstracts contaminated the prose-detection signal.
- **LOO finds drags; loss-filter doesn't help.** Point-level loss
  is too noisy a signal for EBM training; source-level LOO was the
  effective lever.
- **`verbosity=low` is a primary attack surface.** OpenAI Responses
  API at `verbosity=low` drops detector AUROC by 15–24 points on
  most profiles. The length-aware features partially counteract
  this; `master_minus_arxiv` reaches AUROC 0.89 on `verb_low` (best
  of any profile).
- **Provenance matters for prompts.** Using post-LLM-era datasets as
  "human" prompts contaminates training — the model learns to
  detect *contamination*, not authorship. Pre-2022 datasets only.
- **Per-provider matched eval beats more prompts.** Detection
  difficulty varies by provider (Gemini AUROC 0.996; OpenAI 0.936
  with reasoning hidden). For coverage of a new provider, regenerate
  *the same* prompts rather than adding new ones.

## Feedback corpus export

`cairn/history.py::HistoryStore.record_feedback` writes reviewer-confirmed
labels to a `feedback_log` table in the history sqlite — one row per
"this submission was actually human / actually LLM" annotation made
through the GUI. By itself this table is just an audit trail; the
`scripts/export_feedback_corpus.py` script bridges those rows back into
the calibration pipeline as `EvalSample` JSONL.

### Why most operators see zero training-ready feedback rows

The privacy default is `gui.history.retain_text: false` (hash-only
submission storage). Under that default, `submissions.text` is `NULL`
for every row, so feedback annotations cannot be replayed as labeled
training data — only as an audit signal. To collect feedback usable
for retraining, set `gui.history.retain_text: true` in the operator
config *before* the feedback is recorded. Toggling it later does not
recover already-dropped text.

This is an intentional privacy/utility tradeoff. The exporter honours
either choice without complaint, but emits a stderr warning whenever
text-null rows land in the output so the gap is visible:

```
WARNING: 14 rows have text=null because retain_text was disabled at
write time. These rows cannot be used for retraining.
```

### CLI

```bash
# Whole-history export — audit + (any) training-ready rows.
python scripts/export_feedback_corpus.py \
    --history-path ~/.cairn/history.sqlite3 \
    --out data/feedback_2026-05.jsonl

# Monthly window for the C2 retrain runner.
python scripts/export_feedback_corpus.py \
    --history-path ~/.cairn/history.sqlite3 \
    --out data/feedback_2026-05.jsonl \
    --since 2026-05-01 \
    --until 2026-06-01 \
    --profile master_minus_arxiv

# Drop privacy-redacted rows entirely — output is training-ready.
python scripts/export_feedback_corpus.py \
    --history-path ~/.cairn/history.sqlite3 \
    --out data/feedback_2026-05.jsonl \
    --include-text-only
```

The output JSONL matches the `EvalSample` schema in
`cairn/evaluation/dataset.py` and loads via `load_dataset` without
schema errors. Each row carries `tier="feedback"` and a `metadata`
dict with `source: "feedback_log"`, `recorded_at`, `profile`, and
`reviewer_notes` for downstream stratification.

### Feeding feedback into the training pipeline

For the monthly retrain (Track C2 in
[`docs/superpowers/specs/2026-05-07-post-honest-calibration-roadmap.md`](superpowers/specs/2026-05-07-post-honest-calibration-roadmap.md)),
concatenate the feedback file into the master training corpus and
re-run Stages 2–3:

```bash
# 1. Export this month's feedback (training-ready).
python scripts/export_feedback_corpus.py \
    --history-path ~/.cairn/history.sqlite3 \
    --out data/feedback_2026-05.jsonl \
    --since 2026-05-01 --until 2026-06-01 \
    --include-text-only

# 2. Concatenate to the master training corpus. De-dup by id is the
#    monthly runner's job; the export itself is idempotent only
#    against the current sqlite contents.
cat data/feedback_2026-05.jsonl >> data/calib2000_v6_train.jsonl

# 3. Re-cache features for the new rows + retrain the profile.
python scripts/cache_pipeline_features.py \
    --dataset data/calib2000_v6_train.jsonl \
    --config configs/default.yaml \
    --out data/all_features_v6.jsonl
python scripts/calibrate_detectors.py ...   # as in Stage 3 above
python scripts/train_fusion.py ...
python scripts/calibrate_conformal.py ...
```

The retraining step is gated on the in-domain / OOD / `verb_low`
benchmarks not regressing — see Track C2 of the post-honest-calibration
roadmap for the promotion gate, rollback path, and retrain failure log.

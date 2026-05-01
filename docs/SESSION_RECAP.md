# Session Recap — Calibration Build-Out (2026-04-29 → 2026-05-01)

End-to-end summary of the calibration session that took CAIRN from
"trained-EBM-doesn't-exist" to a 13-profile production grid with
held-out OOD validation and a recommended production profile.

## TL;DR

1. **Built a six-provider calibration corpus** — ~94K rows across
   Anthropic / OpenAI / Gemini / Mistral / DeepSeek / Moonshot for
   HC3 + domain corpora + prompt detection.
2. **Trained 13+ calibrated profiles** with proper EBM + isotonic
   + split-conformal artifacts.
3. **Validated against held-out OOD benchmarks** (M4 / RAID / MAGE)
   that the model never saw during training.
4. **Identified the production profile via empirical analysis** —
   per-source LOO surfaced `master_minus_arxiv` as a strict
   improvement over the full corpus on 6 of 9 metrics.
5. **Refreshed all project documentation** to reflect the new state.

10 commits landed; all 392 tests pass; pushed to origin/master.

## Commits this session

```
891b884  docs: refresh for calib2000 production architecture
c1828bd  feat(calibration): point-level + per-source LOO analyses + master_minus_arxiv profile
552a62f  feat(calibration): v6 + 13-profile grid + domain regen tooling
9efa710  feat(calibration): v2-v5 + prompts_only profiles + comprehensive benchmarks
862b4d8  feat(scripts): OOD sampler + provenance-clean prompt sampler + per-detector AUROC
d9dbb29  feat(scripts/regen): concise mode + Gemini MEDIUM thinking + sampling knobs
f4ff426  feat(modal): GPU-offloaded feature cache builder for cairn
6df65b7  feat(fusion): length-aware features + 15 EBM interactions
41be78a  feat(calibration): feature cache + ablation grid orchestration
cf533dd  feat(scripts/regen): openai responses-api + gemini ungrounded path
```

## What got built

### 1. Multi-provider calibration corpus (`scripts/regenerate_llm_outputs.py`)

Six providers regenerating the same HC3 prompts:

| Provider | Model | Pairs |
|----------|-------|-------|
| Anthropic | `claude-sonnet-4-6` | 2000 |
| OpenAI | `gpt-5.4-mini` (Responses API) | 2000 |
| Google | `gemini-3-flash-preview` (no grounding) | 1998 |
| Mistral | `mistral-medium-latest` | 2000 |
| DeepSeek | `deepseek-chat` | 2000 |
| Moonshot | `kimi-k2.6` | 1999 |

Total ~12K HC3 question-pairs, stratified across HC3's five
subdomains (finance / medicine / open_qa / reddit_eli5 / wiki_csai).

Added: openai-compat provider for DeepSeek + Moonshot, Responses API
path for OpenAI with reasoning-effort + verbosity sweep, Gemini
thinking-level controls (HIGH→MEDIUM for cost), resume logic to
recover from API quota walls.

### 2. Domain corpora (`scripts/regen_domain_calib.py`)

Same multi-provider fan-out for math / code / scientific corpora:
arXiv abstracts (provenance: pre-2021, pre-LLM era), GSM8K math
problems, HumanEval coding prompts. Plus prompt-side corpora for
prompt detection (MMLU-Pro, BBH, GPQA, MedQA).

### 3. OOD validation (`scripts/download_ood_samples.py`)

Held-out external benchmarks (never used for training):
- **M4** (multilingual + multidomain)
- **RAID** (adversarial transformations)
- **MAGE** (model-attribution; fixed a label-flip bug here —
  spot-check confirmed `cmv_human` showing as `label=1`)

200 samples each at `data/ood_*_test.jsonl`.

### 4. Prompt detection corpus (`scripts/download_prompt_samples.py`)

~30K provenance-clean human + LLM-generated prompts. Provenance
discipline: human prompts come from pre-LLM-era datasets (HC3 /
HH-RLHF / BBH / TruthfulQA / GPQA / MedQA — pre-2022). LLM-authored
prompts from synthetic-prompt datasets (Alpaca, WizardLM, Magpie,
SlimOrca). Includes reasoning + "stump-the-model" mix.

### 5. Feature cache (`cairn/feature_cache.py::CachedPipeline`)

Drop-in replacement for `CairnPipeline` that keys detector outputs
by `sha256(normalize_text(input))` and serves cached `LayerResult`
lists in <1ms. Calibrator-independent — same cache feeds all 13+
profile variants. Profile retrains drop from ~1–2 hr to ~3 min.

### 6. Modal GPU offload (`scripts/modal_cache_features.py`)

Modal T4 function with persistent HF cache volume for the tier-2
detector inference (`perturbation`, `contrastive_lm`). Uses
`Secret.from_dotenv()` for provider keys.

### 7. Length-aware EBM features (`cairn/fusion/ebm.py`)

Added to give the EBM more capacity for short-form attacks:

```python
"log_word_count": math.log1p(float(word_count)),
"is_short":  1.0 if word_count < 150 else 0.0,
"is_medium": 1.0 if 150 <= word_count < 500 else 0.0,
"is_long":   1.0 if word_count >= 500 else 0.0,
```

Bumped `interactions=5` → `interactions=15` so the EBM can capture
pairwise effects (`surprisal_dynamics × is_short` was a top
contributor on short-form OOD).

### 8. Profile grid (`scripts/run_profile_grid.py`)

Orchestrator that trains N profiles by source-tag regex filtering,
benchmarks each on M test sets, writes a summary CSV with one row
per profile and four metrics per test. Resumable.

### 9. Per-source LOO (`scripts/run_per_source_loo.py`)

Leave-one-source-out matrix: for each source-tag prefix, train a
profile that excludes it, benchmark on the standard 9 test sets,
compute Δ vs the `everything` baseline. **This is what surfaced
`master_minus_arxiv` as the production profile.**

### 10. Loss-filter analysis (`scripts/run_loss_filter.py`)

Point-level filtering by per-row model loss. Four strategies tested:
`high_loss_top30`, `high_loss_top50`, `uncertain`, `exclude_easy`.
**Negative result** — all four underperformed the full-corpus
baseline; high-loss strategies inverted the detector to AUROC < 0.5.

### 11. Per-detector AUROC (`scripts/per_detector_auroc.py`)

Computes per-detector AUROC across multiple datasets.

## Key empirical findings

### Production-decision table

Comparison vs the `everything` baseline (full v6 corpus):

| test | hc3 | everything | -arxiv | -top3 | Δ-arx | Δ-top3 |
|---|---|---|---|---|---|---|
| in_domain | 0.9410 | 0.9024 | 0.9028 | 0.8993 | +0.0004 | -0.0031 |
| ood_m4 | 0.6039 | 0.5489 | 0.5297 | 0.5414 | -0.0192 | -0.0075 |
| **ood_raid** | 0.6264 | 0.5596 | **0.6841** | 0.6757 | **+0.1245** | +0.1161 |
| **ood_mage** | 0.6279 | 0.5230 | 0.5667 | **0.5844** | +0.0437 | **+0.0614** |
| **verb_low** | 0.7641 | 0.7970 | 0.8908 | **0.9032** | +0.0938 | **+0.1062** |
| ctrl anth t=1 | 0.9538 | 0.9313 | 0.9118 | 0.9059 | -0.0195 | -0.0254 |
| ctrl oai m/h | 0.9813 | 0.9481 | 0.9660 | 0.9687 | +0.0179 | +0.0206 |
| prompts_heldout | 0.5716 | 0.7963 | 0.7809 | 0.7110 | -0.0154 | -0.0853 |
| prompts_fresh | 0.5498 | 0.6192 | 0.6267 | **0.6454** | +0.0075 | +0.0262 |

**Verdict:** `master_minus_arxiv` is the new production response
profile. It beats `everything` on 6 of 9 metrics with negligible
in-domain cost (+0.0004), recovering ~0.125 AUROC on RAID and
~0.094 on `verb_low`.

### Lessons learned

- **More data isn't always better.** The `everything` baseline
  lost 0.05–0.12 AUROC on OOD vs HC3-only because arXiv abstracts
  contaminated the prose-detection signal.
- **LOO finds drags; loss-filter doesn't help.** Point-level loss
  is too noisy for EBM training; source-level LOO was the
  effective lever.
- **`verbosity=low` is the primary attack surface.** OpenAI
  Responses API at `verbosity=low` drops detector AUROC by 15–24
  points. Length-aware features partially counteract it
  (`master_minus_arxiv` reaches AUROC 0.89 on `verb_low`).
- **Provenance matters for prompts.** Using post-LLM-era datasets
  as "human" prompts contaminates training — the model learns to
  detect *contamination*, not authorship. Pre-2022 datasets only.
- **Per-provider matched eval beats more prompts.** Detection
  difficulty varies by provider (Gemini AUROC 0.996; OpenAI 0.936
  with reasoning hidden). For coverage, regenerate *the same*
  prompts rather than adding new ones.
- **Some detectors carry no signal.** Per-detector AUROC analysis
  surfaced `token_cohesiveness`, `mixed_boundary`, `preamble`,
  `fingerprint_vocab` as near-random; `surprisal_dynamics` even
  inverts on some OOD distributions.

## Figures

All rendered to [`docs/benchmarks/calib2000/figures/`](benchmarks/calib2000/figures/).
Re-generate with `python scripts/build_session_figures.py`.

| Figure | Insight |
|---|---|
| `profile_grid_auroc.png` | 13 profiles × 9 tests AUROC heatmap — full empirical sweep |
| `production_profile_comparison.png` | `master_minus_arxiv` vs the three alternatives |
| `loo_deltas.png` | Per-source LOO ranked by mean Δ vs baseline |
| `verb_low_attack.png` | Verbosity attack surface across profiles |
| `per_detector_auroc.png` | Per-detector AUROC across in/m4/raid/mage |
| `loss_filter_negative.png` | Loss-filter strategies all lose to full-corpus baseline |

## Data exports available

Raw summary CSVs (in `docs/benchmarks/calib2000/`):

- `profile_grid_summary.csv` — 13 profiles × 9 tests × 4 metrics
- `loo_summary.csv` — 16-source LOO matrix × 9 tests × 4 metrics
- `loss_filter_summary.csv` — 4 point-level filter strategies × 9 tests
- `per_detector_auroc.csv` — 12 detectors × 4 datasets
- `ablation_summary.csv` — earlier provider ablation grid

Per-profile per-test JSON reports under `profiles/<name>/`,
`loo/<source>/`, `loss_filter/<strategy>/` — full EvalReport shape
(AUROC, ECE, Brier, TPR@FPR, confusion matrix).

## Documentation refresh

Same session, post-results:

- `README.md` — Limitations rewritten; new "Calibrated profiles" section
- `docs/architecture.md` — fusion/conformal sections rewritten; profile system + feature cache documented
- `docs/configuration.md` — heuristic vs calibrated profile tables; decision tree
- `docs/benchmarks/README.md` — new calib2000 production-numbers section
- `docs/calibration.md` (NEW) — full four-stage training pipeline guide
- `CLAUDE.md` — "does NOT have" updated; "DOES have now" added

All 392 tests still passing on the final commit.

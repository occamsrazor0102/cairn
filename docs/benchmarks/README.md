# Benchmarks

This directory is the landing zone for metrics reports produced by
`scripts/run_benchmark.py`. Results are tracked here so anyone can see
how the current pipeline performs on public data — not just the
synthetic fixtures bundled with the repo.

## Status

The current production-grade calibration is the **calib2000 v6**
corpus + profile family — see [the calib2000 results](#calib2000-six-provider-master-corpus--profile-grid)
section below for the production numbers and
[`docs/calibration.md`](../calibration.md) for the training pipeline.
The earlier per-corpus profiles (`calibrated`, `calibrated_all`,
`calibrated_regen`, `calibrated_union`, `calibrated_full`) remain in
`configs/` for reproducibility but are superseded.

### History — early calibrated benchmarks

First *calibrated* benchmark landed: `hc3_finance_calibrated.json` —
150-sample test split of HC3 finance, evaluated with the EBM fusion +
split-conformal wrapper trained on the disjoint train/calib slices.
Same test split was also evaluated in heuristic mode for comparison
(`hc3_finance_heuristic_baseline.json`).

| metric           | heuristic | calibrated | Δ |
|------------------|-----------|------------|---|
| AUROC            | 0.7264    | **0.9165** | +0.190 |
| ECE              | 0.3751    | **0.0498** | -0.325 |
| Brier            | 0.3469    | **0.1190** | -0.228 |
| TPR @ FPR 1%     | 0.1200    | **0.4267** | +0.307 |
| accuracy @ p≥0.5 | 0.5067    | **0.8467** | +0.340 |
| F1 @ p≥0.5       | 0.0263    | **0.8369** | +0.811 |

The headline is ECE: the calibrated stack is now genuinely calibrated
(under 5% expected calibration error) rather than emitting hand-picked
probabilities. The F1 jump from 0.026 to 0.84 reflects that the
heuristic stack ranks correctly but doesn't predict positive at the
default threshold — exactly the diagnosis the earlier reviews called
out.

Earlier smoke benchmark (`hc3_finance_screening_smoke.json`, 100
items, screening profile / Tier-1 only) is preserved alongside for
historical comparison.

The CI pipeline still runs the 100-sample smoke benchmark on every
push to master and uploads the result as an artifact.

### Broad calibration: HC3 `all`

Second calibrated profile landed (`configs/calibrated_all.yaml`),
trained on HC3's full `all` subset — 4000 balanced samples drawn
across finance, medicine, open-QA, reddit-eli5, and wiki-csai. The
profile is a generalist; the finance-only profile (`calibrated`)
remains as a domain specialist.

Three test slices, same trained profile:

| slice                                     | AUROC | ECE   | Brier | TPR@1%FPR | F1@0.5 |
|-------------------------------------------|-------|-------|-------|-----------|--------|
| hc3_all test (broad, in-distribution, n=600)   | **0.9524** | 0.0297 | 0.0875 | 0.5200 | 0.8731 |
| hc3_finance test via all (narrow, in-distribution, n=150) | 0.9009 | 0.1132 | 0.1462 | 0.4533 | 0.8072 |
| hc3_medicine full subset (cross-genre check, n=1000) | 0.9117 | 0.0680 | 0.1033 | **0.6100** | 0.8806 |

Generalist-vs-specialist on the 150-sample finance test set:

| profile                | AUROC  | ECE    | F1@0.5 |
|------------------------|--------|--------|--------|
| `calibrated` (finance) | 0.9165 | 0.0498 | 0.8369 |
| `calibrated_all`       | 0.9009 | 0.1132 | 0.8072 |

The specialist wins narrowly on its home domain; the generalist is
sharper on the broad mix and holds up well on medicine (a domain
present in training but only ~20% of samples). Pick the profile that
matches your deployment's input distribution.

### Regen calibration: modern LLM outputs

Third calibrated profile landed (`configs/calibrated_regen.yaml`),
trained on a regenerated HC3 corpus — same questions and same human
answers as HC3 all, but with the LLM-side answers freshly produced
by current models (Claude Sonnet 4.6 for q0–499; an OpenAI saved
Prompt template at v3 for q500–999). 999 question-pairs total.

The point is to test whether modern-LLM patterns differ enough from
HC3's 2023 ChatGPT that a detector calibrated against modern
outputs generalizes back to legacy ChatGPT-2023 — and they do.

| slice                                  | AUROC  | ECE    | Brier  | TPR@1%FPR | F1@0.5 |
|----------------------------------------|--------|--------|--------|-----------|--------|
| regen_mixed test (in-dist, n=302)      | **0.9879** | 0.0369 | 0.0423 | **0.8146** | **0.9536** |
| hc3_finance test via regen (n=150)     | 0.7363 | 0.2812 | 0.2992 | 0.0267    | 0.4528 |
| hc3_medicine via regen (n=1000)        | 0.9001 | 0.3171 | 0.2951 | 0.1900    | 0.4299 |

In-distribution numbers are the strongest of any profile shipped so
far (AUROC 0.99, F1 0.95). But generalization to 2023 ChatGPT
collapses: recall drops to 0.28-0.32 at threshold 0.5, meaning the
detector misses 2/3 of legacy ChatGPT samples. Precision stays high
(0.77-0.90) — when the regen profile DOES predict LLM, it's right;
it just misses too many.

**Read:** modern-LLM writing patterns are meaningfully different
from 2023 ChatGPT. Detectors should be calibrated against the
distribution they'll actually face. For maximum coverage, train on
the union of regen and HC3-all (concatenate before splitting) — see
docs/byo_dataset.md (Path 4 + multi-provider section) for the
recipe.

### Three-profile comparison on a stable test set

Using the 150-sample finance test set as a single yardstick:

| profile          | AUROC  | ECE    | F1@0.5 |
|------------------|--------|--------|--------|
| `calibrated`     | 0.9165 | 0.0498 | 0.8369 |
| `calibrated_all` | 0.9009 | 0.1132 | 0.8072 |
| `calibrated_regen` | 0.7363 | 0.2812 | 0.4528 |

`calibrated` and `calibrated_all` were both trained on HC3 (2023
ChatGPT) and do well on this 2023-ChatGPT test set. `calibrated_regen`
trained on modern LLM output and is poor at catching old-ChatGPT
text. None of these profiles is universally best — they reflect
their training distributions.

### Union calibration: legacy + modern combined

Fourth calibrated profile landed (`configs/calibrated_union.yaml`),
trained on the union of HC3 all (2023 ChatGPT, 4000 samples) and
the regenerated HC3 corpus (Sonnet 4.6 + OpenAI template v3, 1998
samples) — 5998 samples total. The point is to test whether
training on both distributions restores the cross-distribution
coverage the regen-only profile lost.

It does:

| slice                                   | AUROC  | ECE    | Brier  | TPR@1%FPR | F1@0.5 |
|-----------------------------------------|--------|--------|--------|-----------|--------|
| union test (in-dist, n=902)             | 0.9667 | **0.0209** | 0.0745 | 0.6364 | 0.8985 |
| regen test via union (n=302)            | **0.9914** | 0.0647 | 0.0515 | **0.8609** | 0.9333 |
| hc3_finance test via union (n=150)      | 0.9031 | 0.0877 | 0.1385 | 0.4800 | 0.8344 |
| hc3_medicine via union (n=1000)         | 0.9162 | 0.0803 | 0.0982 | 0.7000 | 0.8754 |

**Recall on legacy ChatGPT slices jumped from 0.28-0.32 (regen-only)
to 0.85-0.91 (union)** — the cross-distribution gap closes without
sacrificing modern-LLM performance (regen test AUROC stays at
0.99). This is the strongest production-ready profile so far.

### Four-profile comparison on the finance test set

| profile          | AUROC  | ECE    | F1@0.5 |
|------------------|--------|--------|--------|
| `calibrated`     | 0.9165 | 0.0498 | 0.8369 |
| `calibrated_all` | 0.9009 | 0.1132 | 0.8072 |
| `calibrated_regen` | 0.7363 | 0.2812 | 0.4528 |
| `calibrated_union` | 0.9031 | 0.0877 | 0.8344 |

The union profile matches `calibrated_all` on legacy ChatGPT (its
home-distribution-overlap weight) AND beats every other profile on
the modern-LLM slice. Recommended default for general deployments;
keep `calibrated` if your inputs are exclusively finance prose, and
`calibrated_regen` only if you need maximum sharpness on modern LLM
output and don't care about legacy.

### Super-union (`calibrated_full`): all 3 frontier labs + legacy

Fifth calibrated profile landed (`configs/calibrated_full.yaml`),
trained on the union of HC3 all (2000 question-pairs of 2023
ChatGPT) plus regenerated outputs from all three frontier labs:
* Anthropic Claude Sonnet 4.6 (q0-499, 499 pairs)
* OpenAI Responses + saved template v3 (q500-999, 500 pairs)
* Google Gemini 3 Flash + GoogleSearch (q1000-1499, 500 pairs)

Total 3499 question-pairs — the most provider-diverse calibration
corpus in this project.

Quad-eval:

| slice                                     | AUROC  | ECE    | Brier  | TPR@1%FPR | F1@0.5 |
|-------------------------------------------|--------|--------|--------|-----------|--------|
| superunion test (in-dist, n=1052)         | 0.9720 | 0.0230 | 0.0688 | **0.7186** | 0.9040 |
| regen test via full (n=302)               | 0.9875 | 0.0607 | 0.0570 | 0.8278    | 0.9216 |
| hc3_finance test via full (n=150)         | 0.9035 | 0.0864 | 0.1341 | 0.4800    | 0.8199 |
| hc3_medicine via full (n=1000)            | 0.9107 | 0.0739 | 0.1027 | 0.5280    | 0.8716 |

Head-to-head with the 2-provider union profile:

| metric (in-dist test)  | calibrated_union | calibrated_full |
|------------------------|------------------|-----------------|
| AUROC                  | 0.9667           | **0.9720**      |
| ECE                    | **0.0209**       | 0.0230          |
| F1@0.5                 | 0.8985           | **0.9040**      |
| TPR@1%FPR              | 0.6364           | **0.7186**      |

The biggest win from adding Gemini is **TPR@1%FPR** — at a strict
false-positive budget, the super-union catches significantly more
LLM samples. AUROC only nudges up because both profiles already
near-saturate ranking quality. The slight regression on the modern
regen-test slice (0.9914 → 0.9875) is the expected cost of dilution:
training spreads across more provider distributions, so any single
distribution gets slightly less specialized capacity.

Both `calibrated_union` and `calibrated_full` are production-ready.
Pick `calibrated_full` for maximum provider coverage (recommended
default); pick `calibrated_union` if you don't deploy against Gemini
output.

### Per-provider detection difficulty (matched-comparison eval)

Generated a controlled per-provider corpus by regenerating the same
500 HC3 questions (q1500-1999) with each of the three frontier labs:
identical prompts, identical human-side rows, only the LLM-side
differs. Evaluated each provider's slice through `calibrated_full`:

| provider                                 | n   | AUROC  | ECE    | TPR@1%FPR | F1@0.5 | recall |
|------------------------------------------|-----|--------|--------|-----------|--------|--------|
| Anthropic Sonnet 4.6                     | 1000 | 0.9672 | 0.0819 | 0.7760    | 0.8945 | 0.9580 |
| OpenAI Responses + saved template v3     | 1000 | **0.9358** | 0.0871 | **0.6520** | **0.8485** | **0.8740** |
| Gemini 3 Flash + GoogleSearch            | 998  | 0.9961 | 0.1197 | 0.9419    | 0.9138 | 0.9980 |

**Headlines:**
* **Gemini is nearly trivial to detect** (AUROC 0.996, recall 0.998)
  — GoogleSearch grounding's citation patterns and verbose markdown
  structure are giveaways.
* **OpenAI's reasoning template is the stealthiest** (AUROC 0.936) —
  hidden reasoning produces more naturalistic surface output. ~6 AUROC
  points behind Gemini, and at 1% FPR the detector catches only 65% of
  OpenAI samples vs 94% of Gemini.
* **Sonnet sits in the middle** at AUROC 0.967.

Read: a single calibrated detector is heterogeneously good across
modern LLMs; if you face a known target distribution, calibrate
specifically against it. The matched corpus is reusable for future
provider releases.

### Mixed-vs-single training experiment

Tested whether training on the SAME prompts answered by multiple
providers generalizes better than training on more diverse prompts
from a single provider. Both arms use:
* Same 349 train + 74 calib questions (q1500-1999)
* Same total sample count (698 train rows = 349 humans + 349 LLMs)
* Same model architecture (EBM + split-conformal)
* Same identical 304-row held-out test set (76 questions × 1 human
  + 3 LLMs each)

Only difference: LLM-side composition.
* `calibrated_matched_mixed` — LLM rotated across Anthropic / OpenAI /
  Gemini per training question (~117 from each)
* `calibrated_matched_single` — all 349 LLM samples from Anthropic
  Sonnet 4.6

Result on the identical held-out test:

| metric                 | matched_mixed | matched_single | Δ                          |
|------------------------|---------------|----------------|----------------------------|
| AUROC                  | **0.9804**    | 0.9722         | +0.008                     |
| ECE                    | **0.0394**    | 0.0875         | **-0.048 (cuts in half)**  |
| Brier                  | **0.0497**    | 0.0671         | -0.017                     |
| TPR@1%FPR              | **0.7061**    | 0.4868         | **+0.219 (22 pp gain)**    |
| F1@0.5                 | **0.9483**    | 0.9379         | +0.010                     |

The biggest wins for the multi-provider arm are **ECE (cut in half:
0.0875 → 0.0394) and TPR@1%FPR (+22 percentage points)** — calibration
quality and strict-FP-budget recall. AUROC and F1 are close because
both profiles already discriminate well at typical thresholds.

**Read: multi-provider exposure beats single-provider specialization
for cross-provider generalization, holding sample count constant.**
Adding a third LLM family to training data delivered a meaningful
production-relevant gain — at 1% false-positive budget, the
multi-provider profile catches 71% of held-out LLM samples vs only
49% for the single-provider profile, despite both being trained on
identical question pools.

Practical implication: when adding new model coverage, you don't
need proportionally more training data — substituting some single-
provider samples for multi-provider samples improves coverage at no
sample-count cost. Diversity is doing real work.

## calib2000 — six-provider master corpus + profile grid

The latest calibration round trained 13+ profiles against a master
training corpus of ~94K rows spanning six providers (Anthropic /
OpenAI / Gemini / Mistral / DeepSeek / Moonshot), HC3's five
subdomains, OOD samples from M4 / RAID / MAGE, domain prose
(arXiv, GSM8K, HumanEval), and ~30K provenance-clean prompts.

Full numbers in `docs/benchmarks/calib2000/`:

- `profile_grid_summary.csv` — 13 profiles × 9 test sets × 4 metrics
- `loo_summary.csv` — 16-source leave-one-out matrix × 9 tests
- `loss_filter_summary.csv` — 4 point-level filtering strategies
- `per_detector_auroc.csv` — per-detector AUROC across datasets
- `profiles/<name>/` — per-profile per-test JSON reports
- `loo/<source>/` — per-LOO per-test JSON reports

### Production profile selection

The 13-profile grid plus per-source LOO surfaced two top profiles:

| profile | in_domain | ood_raid | ood_mage | verb_low | prompts_heldout |
|---------|-----------|----------|----------|----------|-----------------|
| `general_hc3` | **0.9410** | 0.6264 | 0.6279 | 0.7641 | 0.5716 |
| `everything` (full v6) | 0.9024 | 0.5596 | 0.5230 | 0.7970 | 0.7963 |
| **`master_minus_arxiv`** | 0.9028 | **0.6841** | 0.5667 | **0.8908** | 0.7809 |
| `drop_top3_drags` | 0.8993 | 0.6757 | **0.5844** | **0.9032** | 0.7110 |
| `prompts_only` | 0.8737 | 0.6173 | 0.5023 | 0.5741 | **0.8414** |

**Key findings:**

- **More data isn't always better.** `everything` (full v6, ~94K
  rows) lost 0.05–0.12 AUROC on OOD vs HC3-only because arxiv
  abstracts contaminated the prose-detection signal.
- **`master_minus_arxiv` is the production response profile.**
  Beats `everything` on 6/9 tests with negligible in-domain cost
  (+0.0004): RAID +0.125, verb_low +0.094, MAGE +0.044. The Δ
  predicted by the LOO matrix held empirically to within 0.001.
- **`drop_top3_drags` goes further on some metrics** (verb_low
  0.903, MAGE 0.584) but loses 0.085 on `prompts_heldout` because
  Magpie was a major prompt source — `master_minus_arxiv` is the
  better single-profile pick.
- **Use `general_hc3` for in-domain prose**, `master_minus_arxiv`
  for cross-domain / short-form, `prompts_only` for prompt
  detection.

### OOD validation (held-out, never trained on)

Held-out samples from three external benchmarks:

| benchmark | n | best profile | AUROC |
|-----------|---|--------------|-------|
| **M4** (multilingual + multidomain) | varies | `general_full` | 0.6391 |
| **RAID** (adversarial transformations) | varies | `master_minus_arxiv` | **0.6841** |
| **MAGE** (model-attribution) | varies | `drop_top3_drags` | **0.5844** |

OOD numbers are substantially lower than in-domain by design —
these test transfer to genres/distributions the model has never
seen. RAID's adversarial transforms (paraphrase, repetition,
synonym-swap) and MAGE's model-attribution mix are deliberately
hostile.

### Verbosity sweep — primary attack surface

OpenAI Responses API at `verbosity=low` drops detector AUROC by
15–24 points on most profiles. The length-aware EBM features
(`is_short`, `log_word_count`) added in this round partially
counteract this:

| profile | verb_low AUROC |
|---------|----------------|
| `everything` | 0.7970 |
| `general_hc3` | 0.7641 |
| `general_full` | 0.7491 |
| `prompts_only` | 0.5741 |
| **`master_minus_arxiv`** | **0.8908** |
| **`drop_top3_drags`** | **0.9032** |

The `verb_low` test set is itself constructed from sweep cells run
through the full provider matrix — it represents a real attack
profile, not a synthetic adversary.

### Per-source leave-one-out (top drags)

The 16-source LOO matrix surfaced which sources hurt or help when
removed. Top drags (removal *helps* mean AUROC across 9 tests):

| excluded source | n_excluded | mean Δ vs everything | best Δ on |
|-----------------|------------|---------------------|-----------|
| `domain_arxiv` | 23,182 | **+0.026** | RAID (+0.125), verb_low (+0.094) |
| `domain_gsm8k` | 23,090 | +0.011 | (mixed) |
| `prompt_magpie` | 4,003 | +0.005 | verb_low (+0.020) |
| `hc3_reddit` | 3,299 | +0.004 | (mixed) |

Top critical (removal *hurts*): all five HC3 subdomains hurt
in_domain when removed by 0.005–0.015 — keep them. `prompt_*`
sources hurt prompts_heldout/fresh by 0.02–0.10 when removed —
keep for prompt detection.

## Reproducing the calibrated benchmark

```bash
# 1. Pull HC3 finance and stratified-split into train/calib/test.
python scripts/download_hc3.py --subset finance --out data/hc3_finance.jsonl --max-per-class 500
python scripts/split_dataset.py --in data/hc3_finance.jsonl --out-dir data --prefix hc3_finance

# 2. Fit per-detector isotonic calibration, EBM fusion, conformal wrapper.
python scripts/calibrate_detectors.py --dataset data/hc3_finance_train.jsonl --config configs/default.yaml    --out models/calibrators/hc3_finance.json
python scripts/train_fusion.py        --dataset data/hc3_finance_train.jsonl --config configs/default.yaml    --out models/fusion/hc3_finance_ebm.pkl
python scripts/calibrate_conformal.py --dataset data/hc3_finance_calib.jsonl --config configs/calibrated.yaml --alpha 0.10 --out models/conformal/hc3_finance.json

# 3. Evaluate on the held-out test split with the calibrated config.
python scripts/run_benchmark.py --dataset data/hc3_finance_test.jsonl --config configs/calibrated.yaml --out docs/benchmarks/hc3_finance_calibrated.json
```

To run the same recipe on your own corpus instead of HC3, see
[`docs/byo_dataset.md`](../byo_dataset.md).

## HC3 — Human ChatGPT Comparison Corpus

[Hugging Face link](https://huggingface.co/datasets/Hello-SimpleAI/HC3)

One-time install:

```bash
pip install datasets
```

Download a subset:

```bash
python scripts/download_hc3.py \
    --subset finance \
    --out data/hc3_finance.jsonl \
    --max-per-class 200
```

Subsets that ship with HC3: `all`, `finance`, `medicine`, `open_qa`,
`reddit_eli5`, `wiki_csai`. `--max-per-class` caps each of human /
LLM to keep runs manageable.

Run the pipeline against it:

```bash
python scripts/run_benchmark.py \
    --dataset data/hc3_finance.jsonl \
    --config configs/default.yaml \
    --out docs/benchmarks/hc3_finance.json \
    --progress
```

The output JSON has the full EvalReport shape (AUROC, ECE, Brier,
TPR@FPR, per-tier, per-attack, confusion at threshold 0.5 + derived
precision/recall/F1).

Smoke benchmark (100 items, a minute or two):

```bash
python scripts/run_benchmark.py \
    --dataset data/hc3_finance.jsonl \
    --config configs/screening.yaml \
    --out docs/benchmarks/hc3_finance_smoke.json \
    --limit 100
```

`screening` profile keeps it to Tier-1 detectors only, so no
transformers download is required.

## RAID — Robust AI Detector benchmark

[Hugging Face link](https://huggingface.co/datasets/liamdugan/raid)

RAID isn't included in `scripts/download_*`; convert manually:

```python
from datasets import load_dataset
from cairn.evaluation.dataset import EvalSample, save_dataset

raid = load_dataset("liamdugan/raid", split="train[:1000]")
samples = [
    EvalSample(
        id=f"raid_{i:06d}",
        text=row["generation"],
        label=0 if row["model"] == "human" else 1,
        tier=row.get("attack") or "raid_base",
        attack_name=row.get("attack"),
        attack_category=row.get("domain"),
    )
    for i, row in enumerate(raid)
]
save_dataset(samples, "data/raid_smoke.jsonl")
```

Then the same `run_benchmark.py` invocation as for HC3.

## Submitting results

After a benchmark run, commit the output JSON to this directory. The
file name should include the dataset name, the profile, and whether
it's a smoke run:

```
docs/benchmarks/
  hc3_finance_default.json
  hc3_finance_screening_smoke.json
  raid_default.json
```

Include the git SHA in the commit message so results can be pinned to
a code version. Do not overwrite historical results — add a dated
suffix if you need to re-run the same combination.

## Interpreting the numbers

In heuristic mode (no trained fusion / conformal):

- **AUROC** is the most meaningful number — it measures ranking ability
  independent of the threshold. Values above ~0.85 suggest the
  ensemble is discriminating; values near 0.5 mean the detector
  stack is adding noise at best.
- **ECE** and **Brier** reflect calibration quality. In heuristic mode
  these will be bad — the per-detector p(LLM) mappings are
  hand-picked. Don't read them as "the system's calibration"; they're
  what's left over after the guesses.
- **TPR@1%FPR** is the operationally relevant number if you care about
  minimising false positives (e.g. academic integrity use).
- **Confusion matrix** at threshold 0.5 is diagnostic, not the
  operating point — tune `decision.*_threshold` in the config to the
  working threshold you actually want.

After training a fusion model and a conformal wrapper, re-run every
benchmark and replace the entries here. At that point ECE and Brier
become meaningful.

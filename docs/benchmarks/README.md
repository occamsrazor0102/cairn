# Benchmarks

This directory is the landing zone for metrics reports produced by
`scripts/run_benchmark.py`. Results are tracked here so anyone can see
how the current pipeline performs on public data — not just the
synthetic fixtures bundled with the repo.

## Status

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
from beet.evaluation.dataset import EvalSample, save_dataset

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

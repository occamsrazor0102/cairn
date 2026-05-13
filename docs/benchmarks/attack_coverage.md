# Attack Coverage — A2 robust_v1 vs pre-A2 master_minus_arxiv

Status table for the A2 (adversarial-aware) sprint. Compares the
production `master_minus_arxiv` profile (pre-A2) against the new
`master_minus_arxiv_robust_v1` profile trained with the
`adversarial_raid_v1` augmentation (12,408 non-arxiv-derived
adversarial rows appended to the 70,502-row master_minus_arxiv base).

Spec source: `docs/superpowers/specs/2026-05-07-post-honest-calibration-roadmap.md`
§ A2. Spec gates: per-attack no-regression > 0.02 vs pre-A2 baseline;
RAID AUROC >= 0.73; MAGE AUROC >= 0.60.

All numbers from `data/calib2000_full_test.jsonl --limit 100` unless
otherwise noted. RAID / MAGE use the full 1,000-row OOD slices at
`data/ood_raid.jsonl` and `data/ood_mage.jsonl`.

## Aggregate

| Slice | Pre-A2 AUROC | A2 robust_v1 AUROC | Delta | Spec gate | Status |
|---|---|---|---|---|---|
| In-domain (no attack) | 0.9459 | 0.9579 | +0.0120 | >= 0.94 | PASS |
| RAID (OOD) | 0.6841 | 0.7835 | +0.0994 | >= 0.73 | PASS |
| MAGE (OOD) | 0.5667 | 0.6591 | +0.0924 | >= 0.60 | PASS |

Both OOD benchmarks gained ~0.09 absolute AUROC -- the A2 augmentation
closed exactly the structural gap the spec targeted. In-domain
performance improved slightly rather than regressing.

## Per-attack on calib2000_full_test, n=100

### Pre-A2 baseline set (5 attacks)

Source: `docs/benchmarks/calib2000/per_attack_auroc.csv` (pre-A2) and
`docs/benchmarks/calib2000/per_attack_auroc_robust_v1.csv` (A2).

| Attack | Pre-A2 | A2 robust_v1 | Delta vs pre-A2 |
|---|---|---|---|
| (none) | 0.9459 | 0.9579 | +0.0120 |
| synonym_swap | 0.9463 | 0.9547 | +0.0084 |
| inject_typos | 0.9439 | 0.9543 | +0.0104 |
| casualize | 0.9335 | 0.9351 | +0.0016 |
| mix_human | 0.9487 | 0.9571 | +0.0084 |

Every attack in the original 5-attack set improved vs pre-A2. No
regression > 0.02; the spec's per-attack gate is met cleanly.

### Expanded transform set (A2 step 2 additions)

Source: `docs/benchmarks/calib2000/per_attack_auroc_robust_v1_expanded.csv`.
Pre-A2 has no baseline for these -- they were introduced in A2 step 2.

| Attack | A2 robust_v1 AUROC | Delta vs robust_v1 baseline (0.9551) | Notes |
|---|---|---|---|
| burstiness_injection | 0.9307 | -0.0244 | pure-Python; minor degradation |
| emoji_noise | 0.9363 | -0.0188 | pure-Python; minor degradation |
| grammar_normalize | 0.9579 | +0.0028 | LM-driven (Haiku) |
| style_transfer | 0.9587 | +0.0036 | LM-driven (Haiku) |
| **summarization_attack** | **0.4167** | **-0.5385** | **VULNERABILITY -- below random** |

#### summarization_attack vulnerability

This attack rewrites a sample by asking the LM to summarize then
re-expand. Under the robust_v1 profile, AUROC collapsed to 0.4167 --
below the 0.5 random-classifier floor, meaning the detector now
systematically misclassifies LLM-summarized-then-re-expanded text AS
HUMAN.

Hypotheses (not yet investigated):

- The augmentation included 1,400 summarization_attack training rows
  labeled `1` (LLM). The summary-then-expand output's surface features
  are so dissimilar from typical LLM register that the EBM may have
  learned a counterproductive mapping. The fix likely needs more rows
  for this attack class, or class-conditional feature engineering.
- Summaries are short, and short text is intrinsically harder to
  classify; the detector may be falling back on a length prior.

Not a regression vs pre-A2 (the original profile didn't measure this
attack); however, it is a real attack vector that should not ship to
production without mitigation. Recommended follow-up:

1. Per-source per-attack AUROC breakdown to confirm the failure mode
   isn't isolated to one provider's summary style.
2. Increase summarization_attack augmentation count from 1,400 to
   ~5,000 in a v2 retrain.
3. Treat short-output transforms as their own router class for
   conformal calibration so the threshold can adapt.

### Deferred

| Attack | Reason |
|---|---|
| translation_pivot | NLLB-200 (~1.5 GB) download not yet present locally; defer to a follow-on Modal run |
| humanizer_tools | Commercial-API skeleton; requires `CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1` and a vendor account |

Both attacks are listed in the spec; their numbers will be added when
the gates fire. Not blocking for the A2 sprint completion since both
are explicitly opt-in.

## Reproduction recipe

```
# 1. Generate adversarial corpus
python scripts/build_raid_aware_corpus.py \
    --max-per-provider-per-attack 200 \
    --include-prompt-attacks --provider anthropic \
    --concurrency 32

# 2. Append non-arxiv rows to the master_minus_arxiv training base
cp models/calib2000/profiles/master_minus_arxiv/train.jsonl \
   data/master_minus_arxiv_robust_v1_train.jsonl
grep -v '"source_id": "arxiv' data/adversarial_raid_v1.jsonl \
    >> data/master_minus_arxiv_robust_v1_train.jsonl

# 3. Cache detector features for the new rows on Modal T4
modal run scripts/modal_cache_features.py \
    --dataset-path data/adversarial_raid_v1.jsonl \
    --out-path data/adversarial_raid_v1_features.jsonl

# 4. Concatenate features
cat data/all_features_v6.jsonl data/adversarial_raid_v1_features.jsonl \
    > data/all_features_v7.jsonl

# 5. Train the new profile
python scripts/calibrate_detectors.py \
    --dataset data/master_minus_arxiv_robust_v1_train.jsonl \
    --config configs/.profile_master_minus_arxiv_robust_v1.yaml \
    --features-cache data/all_features_v7.jsonl \
    --out models/calib2000/profiles/master_minus_arxiv_robust_v1/detector_calibration.json

python scripts/train_fusion.py \
    --dataset data/master_minus_arxiv_robust_v1_train.jsonl \
    --config configs/.profile_master_minus_arxiv_robust_v1.yaml \
    --features-cache data/all_features_v7.jsonl \
    --out models/calib2000/profiles/master_minus_arxiv_robust_v1/ebm.pkl

python scripts/calibrate_conformal.py \
    --dataset data/calib2000_v6_calib.jsonl \
    --config configs/.profile_master_minus_arxiv_robust_v1.yaml \
    --features-cache data/all_features_v7.jsonl \
    --out models/calib2000/profiles/master_minus_arxiv_robust_v1/conformal.json \
    --alpha 0.05
```

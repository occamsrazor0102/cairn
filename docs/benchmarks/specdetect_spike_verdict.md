# SpecDetect spike verdict -- PASS

Date: 2026-05-17

The base SpecDetect detector (Luo et al., arXiv 2508.11343) cleared
both spec gates from
`docs/superpowers/specs/2026-05-17-specdetect-design.md` on a
dual-slice Modal T4 spike. Calibrator retrain is greenlit.

## What we tested

- **Implementation:** `cairn/detectors/specdetect.py` -- DFT total
  energy + per-band fractions + low/mid ratio + spectral entropy
  computed via `numpy.fft` on zero-mean per-token log-probs under a
  proxy LM. Track A (no contrastive sampling).
- **Proxy LM:** `EleutherAI/gpt-j-6b` at fp16 (model loaded once, T4
  16 GB). The first launch OOM'd at fp32 (24 GB); the fp16 fix is in
  the spike commit chain.
- **Slices:**
  - `data/ood_raid.jsonl` (1000 rows, label-balanced 500/500)
  - `data/ood_mage.jsonl` (1000 rows, label-balanced 500/500) -- used
    as the second control after the planned XSum slice couldn't be
    sourced cleanly (MAGE jsonl carries no subdomain tags).

## Spike runs

| Attempt | GPU | Outcome | Cost |
|---|---|---|---|
| 1 | T4 16 GB | GPT-J-6B at fp32 OOM during model load (~24 GB needed, 14.6 GB free after load attempt) | ~$0.05 |
| 2 | T4 16 GB | fp16 fix applied. 1,770 samples scored across both slices in ~355 s | ~$0.10 |

Total Modal spend: **~$0.15** (GPT-J-6B is cached in `cairn-hf-cache`
for future spikes, no re-download).

## Headline numbers

### RAID (n=993 after `too_short` skip)

| Feature | AUROC raw | AUROC flipped |
|---|---|---|
| **`p_llm` (heuristic)** | **0.8549** | 0.1451 |
| `dft_total_energy` | 0.1167 | **0.8833** |
| `dft_low_band_energy` | **0.7424** | 0.2576 |
| `dft_low_mid_ratio` | **0.7343** | 0.2657 |
| `dft_mid_band_energy` | 0.3204 | 0.6796 |
| `dft_high_band_energy` | 0.3603 | 0.6397 |
| `dft_spectral_entropy` | 0.3251 | 0.6749 |

`dft_total_energy` raw 0.117 / flipped 0.883 is the same finding, two
ways: human text has HIGHER spectral energy (paper's claim
reproduces). The heuristic `_HEURISTIC_TOTAL_ENERGY_TO_P_LLM` table is
right-signed -- `p_llm` raw 0.855 confirms.

### MAGE (n=777 after `too_short` skip)

| Feature | AUROC raw | AUROC flipped |
|---|---|---|
| **`p_llm` (heuristic)** | **0.7110** | 0.2890 |
| `dft_total_energy` | 0.2431 | **0.7569** |
| `dft_low_band_energy` | **0.6051** | 0.3949 |
| `dft_low_mid_ratio` | **0.6035** | 0.3965 |
| `dft_mid_band_energy` | 0.4110 | 0.5890 |
| `dft_high_band_energy` | 0.4639 | 0.5361 |
| `dft_spectral_entropy` | 0.4449 | 0.5551 |

Same directional pattern as RAID; MAGE is just harder (consistent
with prior CAIRN benchmarks).

## Comparison to robust_v1 baseline

| Slice | robust_v1 ensemble (10 detectors + EBM) | SpecDetect alone |
|---|---|---|
| RAID | 0.7835 | **0.8549** |
| MAGE | 0.6591 | **0.7110** |

SpecDetect alone beats the entire `master_minus_arxiv_robust_v1`
ensemble on both OOD slices. Adding SpecDetect to the EBM should
genuinely move the needle on the next retrain.

## Spec gate evaluation

- RAID `>= 0.78`: **PASS** (0.8549)
- MAGE `>= 0.60`: **PASS** (0.7110)
- implementation_ok (departure from random in at least one slice):
  **PASS** (both depart by 0.07+ from 0.5 raw, much more flipped)
- `overall_greenlight: true`

## Per-sample compute

- RAID: 0.21 s/sample on T4 with GPT-J-6B fp16
- MAGE: 0.19 s/sample

Both well below the Phase-2 budget. Far cheaper than the original
LAPD detector's 8 s/sample with the Falcon-7B pair.

## Skips

- RAID: 7 rows skipped (`too_short`, < 50 words)
- MAGE: 223 rows skipped (`too_short`)

MAGE has many short rows. The 50-word floor is conservative; could be
lowered (with corresponding `_MIN_TOKENS_FOR_FFT` adjustment) to
recover more MAGE coverage. Current floor matches the rest of CAIRN's
Phase-2 detectors.

## Next step

Per the design spec's greenlight branch:

1. Cache features on the full 110k training corpus
   (`data/calib2000_v6_train.jsonl`, now 110,488 rows after the
   user-flagged additions). Modal T4 with GPT-J-6B fp16, expected
   ~$1-2 and ~30-45 minutes.
2. Concatenate with existing feature cache.
3. Retrain `master_minus_arxiv_robust_v2` with the 6 new SpecDetect
   features in the EBM input.
4. Validate on RAID + MAGE + per-attack vs robust_v1.
5. Targeted check: does the new EBM close the math-heavy LLM solution
   false-negative we surfaced on the user's NINJ1 step-by-step
   sample? Re-run the user's two batches through robust_v2 and
   compare to robust_v1's 2/3 score.

## Track B (SpecDetect++) status

Still deferred per the scoping note. The base detector at 0.855 RAID
AUROC is strong; adding 100-sample contrastive normalization for a
+0.04 AUC bump (paper's claim on different datasets) is not worth the
~100x compute increase unless robust_v2 shows specific deficiencies
the contrastive standardization would address.

# SpecDetect / SpecDetect++ -- scoping note

Date: 2026-05-17
Status: pre-design (scoping). Successor to `2026-05-17-lapd-detector-scoping.md`
after LAPD aborted per `docs/benchmarks/lapd_reproducibility_abort.md`.
Go/no-go decision required before design + implement.

## Paper

Luo, Zhang, Wang, Zou, Lin, Meng, Zhang. **"SpecDetect: Simple, Fast,
and Training-Free Detection of LLM-Generated Text via Spectral
Analysis."** arXiv [2508.11343](https://arxiv.org/abs/2508.11343),
accepted at AAAI 2026.

- Code: <https://github.com/luohaitong/SpecDetect> -- **populated**
  (Python 3.8, PyTorch 2.0, `py_scripts/` + `shell_scripts/`,
  requirements.txt). Reference implementation exists; from-equations
  port risk is much lower than LAPD's empty-repo situation.
- License: not yet confirmed on the repo home page; check before any
  code vendoring.

## Mechanism

Two variants. Both treat the **per-token log-probability sequence**
under a proxy LM as a 1-D signal and analyze its frequency-domain
energy.

### Base SpecDetect

For text `x = (x_1, ..., x_n)` scored by proxy LM `M_theta`:

```
l_i = log P_theta(x_{i+1} | x_<=i)         # per-token log-prob
l'(x) = l(x) - mu_l                          # zero-mean signal
X_k   = SUM_m  l'_m * exp(-j 2*pi k m / n)   # global DFT
S(x)  = SUM_k  |X_k|^2                       # total spectral energy
```

The headline finding: **human text shows significantly higher DFT
total energy** than LLM text. Single scalar `S(x)` is the detector.
FFT complexity is `O(n log n)`.

### SpecDetect++ (the "++" enhancement)

Adds z-normalization using contrastive resampling. For the analyzed
text `x`:

```
For i in 1..N:                               # N = 100 by default
    x_tilde_i ~ M_theta(. | x)               # resample under the same LM
    S_tilde_i = base_specdetect(x_tilde_i)
mu_S    = mean(S_tilde)
sigma_S = std(S_tilde)
S++(x)  = (S(x) - mu_S) / sigma_S
```

The standardization makes the score invariant to text length / domain
shifts in the proxy LM's distribution.

### Proxy LM

Black-box scenario (CAIRN's case): **GPT-J-6B**. White-box uses the
known source model. GPT-J-6B is Apache-2.0, ~6 GB fp16 -- single-GPU
T4 fits comfortably without quantization.

## Headline numbers (averaged across XSum / WritingPrompts / Reddit)

| Method | Avg AUC | Notes |
|---|---|---|
| Lastde | 0.8776 | prior baseline (Yang et al.) |
| SpecDetect (base) | 0.8875 | DFT total energy only |
| Lastde++ | 0.9246 | prior SOTA |
| **SpecDetect++** | **0.9259** | DFT + 100-sample z-norm |

**SpecDetect++ runs "nearly twice as fast" as Lastde++** despite
matching/beating its AUC. The base SpecDetect is faster still and
outperforms Lastde on the same datasets.

The paper does **NOT** test on RAID, M4, MAGE, or HC3 -- we cannot
predict CAIRN-slice numbers without measuring.

Paraphrasing robustness (Figure 6): SpecDetect++ "matches SOTA"; base
SpecDetect outperforms competitors. No specific adversarial robustness
claim like LAPD's 99% AUROC under 1%-token attacks.

## Why CAIRN-relevant

1. **Truly orthogonal signal axis.** No current CAIRN detector uses
   frequency-domain log-prob analysis. EBM fusion gains proportional
   to signal orthogonality -- this should land non-trivial weight even
   with modest standalone AUC.
2. **Single-model footprint.** GPT-J-6B alone, no pair load. ~6 GB
   resident, fits on T4 (or even cheap consumer GPU). Much cheaper to
   spike than LAPD's 14-GB Falcon pair.
3. **Reference impl exists.** Significant de-risk vs LAPD's empty
   repo. Can pattern-match against `py_scripts/` rather than porting
   from equations.
4. **Two integration scopes available.** Base SpecDetect = one forward
   pass through proxy LM + FFT (cheap, Phase-2 candidate). SpecDetect++
   adds 100 generation calls (expensive, Phase-3 only).
5. **Potential coverage for the math-heavy LLM solution false-negative
   class** we surfaced in session_handoff_2026-05-17.md: Binoculars
   dropped from 0.85-0.92 on prose-y LLM text to 0.412 on math-heavy
   step text. Spectral analysis of the *shape* of the log-prob curve
   (rather than its mean) may catch what Binoculars misses on
   structurally-uniform LLM reasoning.

## Two-track integration plan

### Track A -- Base SpecDetect only (Phase 2)

Lower risk, faster integration:

| Day | Scope |
|---|---|
| 1 | `cairn/detectors/specdetect.py` -- single proxy LM load (`EleutherAI/gpt-j-6b`), DFT total energy via `numpy.fft` or `torch.fft`. Lazy `transformers` import; SKIP when missing. Reference: `py_scripts/` in the GH repo. |
| 2 | Cascade Phase 2. Config block (`detectors.specdetect`). EBM schema entries (`specdetect_p_llm`, `_missing`, `_confidence`). Tests with mocked log-prob arrays. README phase-table update. |
| 3 | Modal T4 spike on `data/ood_raid.jsonl` (1000 rows) + 200-row HC3 slice. Expected cost ~$0.30-0.50. Spec gate: AUROC >= 0.78 on RAID to greenlight calibrator retrain. |

Net: **~3 days**. Single-model load, no quantization gymnastics, no
contrastive sampling. Should be much smoother than the LAPD path.

### Track B -- Add SpecDetect++ z-norm later

Conditional follow-on once base SpecDetect is in:

| Day | Scope |
|---|---|
| 4 | Add `n_contrastive_samples` config knob to `specdetect.py`. When > 0, generate samples from the same proxy LM and z-normalize. |
| 5 | Phase-3 fallback for ambiguous-middle samples only (cascade naturally caps the compute hit at ~10-20% of traffic). Modal spike to confirm the +0.04 AUC delta from the paper. |

Conditional on Track A clearing its spec gate.

## Cascade placement

- **Base SpecDetect:** Phase 2. One forward pass through GPT-J-6B
  (~0.3-0.5 s on a T4 for a 500-token input). DFT is trivial. Latency
  fits the Phase-2 slot alongside `contrastive_lm` and `surprisal_dynamics`.
- **SpecDetect++:** Phase 3 only. 100 contrastive generations multiply
  cost by ~100x. Should only run for ambiguous samples.

## Dependencies / extras impact

- `transformers`, `torch` -- already in `tier2`.
- `numpy` or `torch.fft` -- already vendored.
- GPT-J-6B weights are **not vendored** (6 GB). Lazy-download via
  `transformers.AutoModelForCausalLM`. Documented in detector
  docstring + `models/README.md`.
- No new extras group needed.

Base install stays tiny per CLAUDE.md invariant -- detector degrades
to SKIP when transformers/torch missing.

## Risk register

1. **AUC numbers in paper are on short-form text (XSum, WritingPrompts,
   Reddit).** CAIRN's `master_minus_arxiv` profile targets long-form
   prose. SpecDetect on long text may behave differently -- the DFT
   energy scales with length, and the zero-mean step changes character
   for long texts with mixed registers. **Mitigation:** spike measures
   AUROC binned by word-count on RAID.
2. **100-sample contrastive cost (SpecDetect++).** Generating 100
   samples per analyzed text from a 6B model is ~2-3 seconds per
   sample x 100 = ~5 minutes wall clock per analyzed text on a T4.
   **Mitigation:** ship Track A (base) only; Track B contingent on
   measured delta and willingness to absorb the cost in Phase 3.
3. **Paper not tested on RAID / M4 / MAGE.** We're flying blind on
   CAIRN's OOD slices. **Mitigation:** the spike answers this in a
   day at low cost.
4. **GPT-J-6B vs newer LMs.** Paper anchors black-box detection on
   GPT-J-6B (released 2021). Modern LLMs trained against current
   instruct data may have a different relationship to GPT-J's
   surprisal landscape. **Mitigation:** measure first, swap proxy LM
   if needed (`Mistral-7B` or `Pythia-6.9B` are plausible alternatives).
5. **Possible reproducibility gap as with LAPD.** Reference impl
   exists but we still need to verify it reproduces on our test set.
   **Mitigation:** spike measures AUROC against paper-tested datasets
   (XSum, Reddit) AS A CONTROL before running on RAID. If we can't
   reproduce 0.88 on XSum, we abort early like LAPD.

## Recommendation

**Go on Track A (base SpecDetect)** with the explicit spike gate:

- Spike on `data/ood_raid.jsonl` AND a small XSum control slice.
- Spec gate: XSum AUROC must be within `+/- 0.02` of paper's 0.88 to
  validate the implementation, AND RAID AUROC must be `>= 0.78` to
  justify the calibrator retrain.
- If XSum reproduces but RAID is below the gate: abort the SpecDetect
  integration, but document the implementation as a debuggable
  reference (different from LAPD's "failed to reproduce" outcome).
- If XSum fails to reproduce: investigate before claiming abort -- we
  may have an implementation bug that LAPD's spike couldn't surface
  because LAPD's code didn't exist.

**Defer Track B (SpecDetect++)** until Track A clears the gate. The
+0.04 AUC delta from contrastive sampling is small compared to the
~10x compute cost; it should be an optional Phase-3 enhancement, not
the default integration.

## Cost projection

| Stage | Estimate |
|---|---|
| Implementation (Track A, 3 days) | $0 (local dev) |
| Modal T4 spike (1000 RAID + 200 XSum, base SpecDetect, single GPT-J-6B load) | ~$0.30-0.50 |
| Modal feature-cache pass on full 110k training corpus (T4) | ~$0.50-1.00 |
| Calibrator retrain (`master_minus_arxiv_robust_v2`) | $0 (local CPU, fast on cached features) |

**Total expected spike + integration: ~$1-2**, vs LAPD's $2.60 abort.

## Open questions for the design phase

- Does base SpecDetect produce a single scalar feature or also feed
  per-bin energies into the EBM? Paper's "DFT total energy" suggests
  scalar, but per-band energies (low/mid/high frequency) might be a
  richer EBM input.
- STFT signal: the paper mentions L=20, hop=10 STFT but the headline
  detector is DFT-only. Worth ablating later whether STFT-derived
  features add anything.
- Proxy LM choice: GPT-J-6B vs Pythia-6.9B vs a smaller model.
  Smaller proxy = faster spike turnaround.
- Calibration target: re-train just `detector_calibration.json` for
  the new detector, or retrain the EBM too? Retraining the EBM is
  cheap given the cached features pipeline.

## Next step

If go-ahead:

1. Draft `docs/superpowers/specs/2026-05-17-specdetect-design.md`.
2. Implement Track A (`cairn/detectors/specdetect.py` + tests +
   cascade/config wiring + EBM schema).
3. Run the dual-control spike (RAID + XSum) on Modal T4.
4. Conditional on spike: cache features on full corpus, retrain
   `master_minus_arxiv_robust_v2`, validate per-attack + RAID + MAGE.

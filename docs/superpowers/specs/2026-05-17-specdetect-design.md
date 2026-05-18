# SpecDetect detector -- design spec

Date: 2026-05-17
Status: design (post-scoping go-ahead). Track A only (base SpecDetect).
Parent: `2026-05-17-specdetect-plus-plus-scoping.md`.

## Goal

Add a `specdetect` Phase-2 detector that exposes **frequency-domain
features of the per-token log-probability sequence** to CAIRN's EBM,
based on Luo et al., arXiv 2508.11343.

Acceptance gates (dual-control spike):

1. **XSum control gate** -- AUROC within `+/- 0.02` of paper's 0.88
   on a 200-row XSum slice. Validates the implementation.
2. **RAID spec gate** -- AUROC `>= 0.78` on `data/ood_raid.jsonl`
   (1000 rows). Justifies the calibrator retrain.

If XSum reproduces but RAID fails: document and abort. If XSum fails:
debug; do not declare abort until reproduction is validated.

## Scope correction from scoping note

Re-reading the paper's "DFT total energy" definition through Parseval's
identity:

```
sum_k |X_k|^2 = n * sum_m |l'_m|^2 = n^2 * variance(log-probs)
```

So the paper's headline scalar feature is mathematically equivalent to
`n * surprisal_std^2`. CAIRN's `surprisal_dynamics` detector already
computes `surprisal_std`. **A naive port that exposes only DFT total
energy would be redundant** with what the EBM already has.

The novel signal lies in **per-frequency-band** structure, not the
total. The detector exposes:

- `dft_total_energy` -- for direct paper comparison + EBM benchmarking
- `dft_low_band_energy` -- bottom 25% of bins (slow / DC-near drift)
- `dft_mid_band_energy` -- middle 50% of bins
- `dft_high_band_energy` -- top 25% of bins (rapid token-to-token variation)
- `dft_low_mid_ratio` -- discriminates slow-drift vs jittery profiles
- `dft_spectral_entropy` -- entropy over the normalized power spectrum

These are orthogonal to `surprisal_std` because they redistribute the
total variance into bins by frequency. Two texts with identical
surprisal_std can have very different spectral shapes.

## Class signature

```python
# cairn/detectors/specdetect.py
class SpecDetectDetector:
    id = "specdetect"
    domain = "prose"
    compute_cost = "moderate"

    def __init__(self): ...
    def analyze(self, text: str, config: dict) -> LayerResult: ...
    def calibrate(self, labeled_data: list) -> None: ...

DETECTOR = SpecDetectDetector()
```

Auto-registers via `cairn/detectors/__init__.py`'s discovery loop.

## Config schema

```yaml
detectors:
  specdetect:
    # SpecDetect (Luo et al., arXiv 2508.11343). Frequency-domain
    # analysis of per-token log-probabilities under a proxy LM. Requires
    # `cairn[tier2]`. Defaults to gpt2 (small, matches surprisal_dynamics);
    # override to EleutherAI/gpt-j-6b for paper-comparison runs.
    enabled: false
    weight: 0.85
    model: "gpt2"
    max_tokens: 512
```

## Signals schema

```jsonc
{
  "n_tokens": 387,
  "dft_total_energy": 1234.5,         // sum |X_k|^2 / n^2 (normalized by length^2)
  "dft_low_band_energy": 0.42,        // fraction of energy in 0-25% bins
  "dft_mid_band_energy": 0.31,        // fraction of energy in 25-75% bins
  "dft_high_band_energy": 0.27,       // fraction of energy in 75-100% bins
  "dft_low_mid_ratio": 1.35,          // low_band / mid_band
  "dft_spectral_entropy": 2.84,       // entropy of normalized spectrum
  "model": "gpt2",
  "device": "cuda",
  "truncated": false
}
```

Band fractions sum to 1.0 by construction. EBM gets `dft_total_energy`
+ 3 band fractions + ratio + entropy = 6 features.

## Compute path

```
text --tokenize--> input_ids
                    |
                forward(M_theta) ---> log P(x_{t+1} | x_<=t)  shape (T-1,)
                    |
            zero-mean ---> l'(x) = log P - mean(log P)
                    |
            torch.fft.fft(l'(x))
                    |
            |X_k|^2 --binning--> low/mid/high band energies + spectral entropy
                    |
            ratio + heuristic table --> p_llm
```

One forward pass through the proxy LM. FFT is trivial (numpy or
torch.fft). No contrastive sampling for Track A.

## Heuristic p_llm mapping (pre-calibration)

The paper says: **lower DFT total energy = more LLM-like**. Mapping
table anchored on what we know empirically about surprisal-std
distributions (LLM ~ 1.5, human ~ 3.0):

```python
_HEURISTIC_TOTAL_ENERGY_TO_P_LLM = [
    (0.5,  0.92),    # very low energy -> strongly LLM
    (1.0,  0.78),
    (1.5,  0.55),
    (2.0,  0.30),
    (3.0,  0.10),
    (5.0,  0.05),    # high energy -> strongly human
]
```

Per CLAUDE.md: `_HEURISTIC_` prefix preserved. Real calibration
replaces this in the next isotonic pass.

## SKIP conditions

| Condition | `signals.skipped` value |
|---|---|
| `len(text.split()) < 50` | `"too_short"` |
| `not _HAS_TORCH` | `"torch_unavailable"` (hint: `cairn[tier2]`) |
| Model load raises | `"model_load_failed"` (include exception type) |
| All forward passes raise (OOM, etc.) | `"scoring_failed"` |
| `n_tokens < 16` (FFT needs reasonable length) | `"too_few_tokens_for_fft"` |

## Cascade placement

**Phase 2** (`PHASE2_DETECTORS`). One forward pass through GPT-2 (~50ms
on T4) or GPT-J-6B (~0.5s on T4). FFT is negligible. Same compute slot
as `surprisal_dynamics` and `contrastive_lm`.

## Test plan (`tests/test_detectors/test_specdetect.py`)

Pattern mirrors `test_lapd.py` and `test_perturbation.py`. Mocked
log-probs so no GPU/weights needed in CI.

1. `test_specdetect_registered`
2. `test_specdetect_placed_in_phase2`
3. `test_specdetect_skips_short_text`
4. `test_specdetect_skip_when_torch_unavailable`
5. `test_specdetect_skip_when_too_few_tokens`
6. `test_specdetect_uniform_log_probs_have_low_energy` -- mocked
   near-constant log-probs -> low total energy -> high p_llm
7. `test_specdetect_volatile_log_probs_have_high_energy` -- mocked
   varied log-probs -> high total energy -> low p_llm
8. `test_specdetect_band_fractions_sum_to_one`
9. `test_specdetect_spectral_entropy_higher_for_flat_spectrum`
10. `test_specdetect_signals_schema`
11. `test_heuristic_total_energy_table_is_monotone_decreasing`

## Modal spike (`scripts/modal_spike_specdetect.py`)

Dual-control:

- `data/ood_raid.jsonl` (1000 rows, stratified -- same fix as LAPD)
- A 200-row XSum slice (download via `datasets` library or use a
  small bundled fixture)

Steps:

1. Load GPT-J-6B once (for paper comparison) -- ~$0.30 amortized
2. Run base SpecDetect on each row
3. Compute AUROC of `dft_total_energy` vs label on both datasets
4. Also report per-band-energy AUROCs to surface which features
   discriminate best on each slice
5. Write `docs/benchmarks/calib2000/specdetect_spike.json`

GPU: T4 (16 GB) -- GPT-J-6B at fp16 is ~12 GB, fits with headroom.
Timeout: 60 min. Cost: ~$0.50-1.00 total.

## Integration touch points

| File | Change |
|---|---|
| `cairn/detectors/specdetect.py` | New file |
| `cairn/cascade.py` | `PHASE2_DETECTORS = (..., "specdetect")` |
| `configs/default.yaml` | New `specdetect:` block, default `enabled: false` |
| `cairn/pipeline.py` | Add `"specdetect"` to transformers+torch set |
| `cairn/fusion/ebm.py` | `specdetect_p_llm`, `_missing`, `_confidence` + 5 signal-level keys (band fractions, ratio, entropy) |
| `tests/test_detectors/test_specdetect.py` | New file |
| `scripts/modal_spike_specdetect.py` | New file |
| `README.md` | Phase-2 detector table update |
| `docs/benchmarks/specdetect_spike.md` | After spike: results + verdict |

## Risks acknowledged from scoping note

Spec defers to scoping note's risk register. Design-time mitigations:

1. **DFT total energy redundancy with surprisal_std** -- mitigated by
   exposing per-band features and spectral entropy, not just total.
2. **Paper-untested datasets (RAID/M4/MAGE)** -- dual-control spike
   answers this with XSum reproduction check.
3. **GPT-J-6B vs newer LMs** -- config knob; default to GPT-2 for
   cost, override to GPT-J-6B for paper reproduction.

## Out of scope for this design

- SpecDetect++ contrastive normalization (Track B, deferred).
- STFT (local windowed) features. Worth ablating once base lands.
- Proxy-LM swap experiments. Default is GPT-2 (matches surprisal_dynamics);
  paper-compare runs use GPT-J-6B.

# LAPD detector — design spec

Date: 2026-05-17
Status: design (post-scoping go-ahead, pre-spike).
Parent: `2026-05-17-lapd-detector-scoping.md`.

## Goal

Add a `lapd` Phase-3 detector that computes the
**Log-likelihood Alignment Preference Discrepancy** (Wu et al.,
arXiv 2604.16923) over a paired (aligned, base) LM and feeds its
standardized score into the EBM-fusion + conformal pipeline.

Acceptance gates:

1. **Spike gate** — Falcon-7B + Falcon-7B-Instruct (4-bit quantized,
   Modal T4) reproduces RAID AUROC `≥ 0.82` on `data/ood_raid.jsonl`
   (paper claims 0.852). Below 0.78 → abort.
2. **Retrain gate** — `master_minus_arxiv_robust_v2` with LAPD as
   the 11th calibrated detector clears the three A2 gates from
   `2026-05-07-post-honest-calibration-roadmap.md` § A2:
   in-domain `≥ 0.94`, RAID `≥ 0.73`, MAGE `≥ 0.60`. Plus no
   per-attack regression `> 0.02` vs `master_minus_arxiv_robust_v1`.

## Class signature

```python
# cairn/detectors/lapd.py
class LAPDDetector:
    id = "lapd"
    domain = "universal"
    compute_cost = "expensive"

    def __init__(self): ...
    def analyze(self, text: str, config: dict) -> LayerResult: ...
    def calibrate(self, labeled_data: list) -> None: ...

DETECTOR = LAPDDetector()
```

Auto-registers via `cairn/detectors/__init__.py`'s discovery loop —
no manual wiring beyond:

1. `PHASE3_DETECTORS` in `cairn/cascade.py` (add `"lapd"`).
2. `configs/default.yaml::detectors::lapd` block (default
   `enabled: false`, like `perturbation`).
3. `pipeline.detector_availability` gate — add `lapd` to the
   `transformers + torch` block at line 96.
4. Bump README Phase-3 table per CLAUDE.md invariant.

## Config schema

```yaml
detectors:
  lapd:
    enabled: false                 # opt-in; heavy compute
    weight: 0.95                   # only used in naive fusion mode
    # Aligned (instruction-tuned) model. Used as φ.
    aligned_model: "tiiuae/falcon-7b-instruct"
    # Base (non-aligned) model. Used as θ. MUST share tokenizer with aligned.
    base_model: "tiiuae/falcon-7b"
    # 4-bit quantization to fit Falcon-7B pair on a single T4 (≈ 8 GB resident).
    quantize_4bit: true
    # Number of perturbation samples for the standardizer μ̃/σ̃.
    # Paper uses 100; 50 is our default until the spike measures otherwise.
    n_perturbations: 50
    perturbation_rate: 0.10        # fraction of tokens replaced for std estimate
    max_tokens: 512                # truncate input to keep one forward pass cheap
    seed: 42
    device: null                   # null → auto: cuda > cpu
```

## Signals schema

```jsonc
{
  "raw_s": 12.34,                  // S(x) before standardization
  "s_lapd": 1.87,                  // S_LAPD(x) = (S - μ̃) / σ̃
  "mu_tilde": 8.12,                // perturbation-sampled mean
  "sigma_tilde": 2.25,             // perturbation-sampled std
  "n_tokens": 387,
  "n_perturbations": 50,
  "model_pair": "tiiuae/falcon-7b-instruct + tiiuae/falcon-7b",
  "quantized": true,
  "device": "cuda",
  "truncated": false               // true if input exceeded max_tokens
}
```

No `spans` array (LAPD is a scalar over the whole text — no positional
evidence to highlight).

## Compute path

```
text ── tokenize ──► input_ids
                      │
                      ▼
       ┌──────────────┴───────────────┐
       │                              │
  forward(φ) ── log P_φ(x_t)    forward(θ) ── log P_θ(x_t)
       │                              │
       └──── Δ_t = log P_φ - log P_θ ─┘
                      │
                      ▼
       I_φ_t = -log P_φ(x_t)
       S = Σ I_φ_t · Δ_t
                      │
                      ▼
    perturbation loop × n:
       randomly mask 10% of tokens
       recompute S' under fresh forward passes
       collect S' samples → μ̃, σ̃
                      │
                      ▼
       S_LAPD = (S - μ̃) / σ̃
```

**One forward pass per perturbation per model =
`2 × (1 + n_perturbations)` forward passes per sample.** With
`n_perturbations = 50` that's 102 forwards per sample. On Falcon-7B
4-bit on a T4 (~80 ms/forward), ≈ 8 s per sample. Above the paper's
quoted `0.58 s` (the paper likely used a smaller standardizer-sample
budget). Spike measures `n_perturbations ∈ {20, 50, 100}` and picks
the smallest that preserves AUROC within 0.005 of the n=100 reference.

## Heuristic p_llm mapping (pre-calibration)

`_HEURISTIC_S_LAPD_TO_P_LLM` table — same shape as
`perturbation._HEURISTIC_Z_TO_P_LLM`. Standardized S_LAPD is roughly
N(0, 1) under human authorship and shifted positive under LLM
authorship per the paper's distribution plots. Hand-picked anchors:

```python
_HEURISTIC_S_LAPD_TO_P_LLM = [
    (-2.0, 0.02), (-1.0, 0.05), (0.0, 0.10),
    (0.5, 0.30), (1.0, 0.55),
    (2.0, 0.78), (3.0, 0.90), (5.0, 0.97),
]
```

Per CLAUDE.md naming invariant: this is `_HEURISTIC_`, not
`_CALIBRATION_`. Real isotonic calibration replaces this once the
calibrator pass runs on the full features cache.

## SKIP conditions

| Condition | `signals.skipped` value |
|---|---|
| `len(text.split()) < 100` | `"too_short"` |
| `not _HAS_TORCH` | `"torch_unavailable"` (hint: `cairn[tier2]`) |
| `not _has_bitsandbytes()` when `quantize_4bit: true` and no full-precision fallback | `"bitsandbytes_unavailable"` |
| Model load raises | `"model_load_failed"` (include exception type) |
| Tokenizer mismatch between aligned and base | `"tokenizer_mismatch"` (include both tokenizer hashes) |
| All forward passes raise (OOM, etc.) | `"scoring_failed"` |
| Perturbation `σ̃ < 1e-4` | `"insufficient_perturbation_variance"` |

SKIP returns `confidence = 0.0` and `p_llm = 0.5` — consistent with
other detectors. Per CLAUDE.md "Detector failures surface, not
swallow," every SKIP carries enough in `signals` to diagnose
post-hoc.

## Cascade placement

**Phase 3 (`PHASE3_DETECTORS`).** Reasoning:

- Compute cost (~8 s/sample at default settings) puts it well above
  Phase-2 thresholds.
- Two-model GPU load is the heaviest single-detector footprint in
  the cascade — should only run when phase 1+2 already left the
  sample ambiguous.
- Per `cairn/cascade.py::should_run_phase3`, runs when
  `0.15 < aggregate_p_llm < 0.80` from phases 1+2 — typically
  10-20 % of traffic under `master_minus_arxiv` cascade.

## Test plan (`tests/test_detectors/test_lapd.py`)

Structural — mocked log-prob fixtures so the suite does NOT need
14 GB of weights. Pattern mirrors `tests/test_detectors/test_perturbation.py`.

1. `test_lapd_registered` — `lapd` appears in `get_all_detectors()`.
2. `test_lapd_placed_in_phase3` — `lapd in PHASE3_DETECTORS`.
3. `test_lapd_skips_short_text` — < 100 words → SKIP with `"too_short"`.
4. `test_lapd_skip_when_torch_unavailable` — patch `_HAS_TORCH = False`,
   expect SKIP with `"torch_unavailable"` hint mentioning `tier2`.
5. `test_lapd_skip_on_tokenizer_mismatch` — patch in mock tokenizers
   with different vocab hashes, expect SKIP.
6. `test_lapd_sign_on_llm_like_input` — patch log-prob to simulate
   aligned-model preference (LLM-like): expect `s_lapd > 0` and
   `p_llm > 0.5`. Uses synthetic per-token log-prob arrays with known
   integral.
7. `test_lapd_sign_on_human_like_input` — patch log-probs to be
   near-equal between φ and θ: expect `|s_lapd| < 0.5` and
   `p_llm ≈ 0.1`.
8. `test_lapd_perturbation_variance_floor` — `σ̃ < 1e-4` → SKIP
   with `"insufficient_perturbation_variance"`.
9. `test_lapd_signals_schema` — round-trip the full signals dict,
   assert every documented key is present and types are right.
10. `test_lapd_truncation_flag` — input longer than `max_tokens` must
    set `signals.truncated = True`.

No test downloads weights or calls real `transformers`. Coverage
target: ≥ 90 % of the new module's lines via mocked paths.

## Modal spike script

`scripts/modal_spike_lapd.py`:

- Image: `modal.Image.debian_slim().pip_install("transformers", "torch",
  "accelerate", "bitsandbytes", "numpy", "scikit-learn")`.
- Volume `model-cache` (read-only after first prefetch) holds the
  pre-downloaded Falcon-7B pair.
- GPU: `T4`, timeout `60 min`.
- Steps:
  1. Load `data/ood_raid.jsonl` (1000 rows, label + text).
  2. Run `LAPDDetector.analyze` per row with `n_perturbations=50`.
  3. Compute AUROC of `s_lapd` vs label, also AUROC of `p_llm`.
  4. Sweep `n_perturbations ∈ {20, 50, 100}` to find smallest budget
     that holds within 0.005 of the n=100 AUROC.
  5. Write `docs/benchmarks/calib2000/lapd_spike_raid.{csv,json}`.
- Estimated cost: ~$3-5 on T4 (downloads cached after first run).

## Integration touch points (post-spike, conditional on gate)

| File | Change |
|---|---|
| `cairn/detectors/lapd.py` | New file — implementation. |
| `cairn/cascade.py` | `PHASE3_DETECTORS = (..., "lapd")`. |
| `configs/default.yaml` | New `lapd:` block under `detectors`. |
| `cairn/pipeline.py` | Add `"lapd"` to the `transformers + torch` set at line 96. |
| `pyproject.toml` | `tier3` extras gain `bitsandbytes` (optional). |
| `README.md` | Phase-3 detector table — add `lapd`. |
| `docs/benchmarks/attack_coverage.md` | New section for robust_v2 results. |
| `scripts/modal_spike_lapd.py` | New file — spike runner. |
| `scripts/modal_cache_features.py` | No change (registry-driven). |
| `tests/test_detectors/test_lapd.py` | New file — 10 tests above. |
| `models/calib2000/profiles/master_minus_arxiv_robust_v2/` | Retrain artifacts. |
| `configs/.profile_master_minus_arxiv_robust_v2.yaml` | New profile YAML. |
| `.gitignore` | Allowlist for robust_v2 artifacts. |

13 touch points. Comparable to the A2 sprint (12 touch points).

## Risks acknowledged from scoping note

Spec defers to the scoping note's risk register. The two design-time
mitigations baked in:

1. **No reference implementation** → from-equations implementation
   gets six structural tests with mocked log-probs (#6, #7 above
   verify sign-of-S behavior).
2. **VRAM footprint** → 4-bit quantization is the default and ABS
   AUROC delta vs full-precision is measured in the spike before
   the retrain phase.

## Out of scope for this design

- Heritable feature support for the EBM (LAPD is a single scalar +
  derived signals — same shape as `perturbation` from the EBM's
  perspective). Adding `s_lapd` to the cached features schema is a
  registry-driven side effect of step 3, not a contract change.
- Mondrian-conformal v3 routing class refinement — open question in
  scoping note. Defer to a follow-on once we see the per-router
  threshold drift on the robust_v2 retrain.
- LiSCP / DNA-DetectLLM / SpecDetect++ — separate scoping notes.

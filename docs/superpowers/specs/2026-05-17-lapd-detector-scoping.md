# LAPD detector — scoping note

Date: 2026-05-17
Status: pre-design (scoping). No code yet. Go/no-go decision required
before design + implement.

## Paper

Wu, Huang, Hu, Chen, Wu, Xia, Zou. **"Alignment Imprint: Zero-Shot
AI-Generated Text Detection via Provable Preference Discrepancy."**
arXiv [2604.16923](https://arxiv.org/abs/2604.16923), April 2026.

Acronym: **LAPD = Log-likelihood Alignment Preference Discrepancy.**

Code repo announced at <https://github.com/creator-xi/LAPD> — verified
empty on 2026-05-17 ("Coming soon."). **No reference implementation
exists.** Any CAIRN integration is from-equations.

## Mechanism

Zero-shot, **model-pair** detector. Requires two LMs:

- φ = aligned (instruction-tuned) model — e.g. `Llama2-7B-Instruct`
- θ = base model — e.g. `Llama2-7B`

Per token `x_t` in the input:

```
I_φ(x_t) = -log P_φ(x_t | x_<t)            # self-information from aligned model
Δ(x_t)   =  log P_φ(x_t | x_<t)
         -  log P_θ(x_t | x_<t)            # alignment imprint
```

Aggregate raw score:

```
S(x) = Σ_t  I_φ(x_t) · Δ(x_t)
```

Standardize via conditional-independence perturbation sampling:

```
S_LAPD(x) = ( S(x) - μ̃_x ) / σ̃_x
```

LLM-generated text → larger Δ (the aligned model assigns much higher
probability than the base model). Human text → small Δ.
Information-weighting suppresses high-entropy positions where Δ is
noisy.

Ablations confirm the method across `Falcon-7B`, `GPT-J-6B`,
`Llama3.1-8B` pairs.

## Headline numbers (from the paper)

| Slice | LAPD AUROC | Comparable CAIRN today |
|---|---|---|
| M4 | 0.880 | 0.659 (`master_minus_arxiv`, recent) |
| RAID | 0.852 | 0.7835 (`master_minus_arxiv_robust_v1`) |
| DetectRL multi-LLM | 0.972 | not measured |
| DetectRL multi-domain | 0.961 | not measured |
| Cross-domain (XSum / WritingPrompts / Arxiv on GPT-4 / Gemini / Claude) | 0.995 avg | not directly comparable |

TPR @ 0.5 % FPR: **92.27 %** average vs Binoculars 66.67 %,
DNA-DetectLLM 59.56 %, Fast-DetectGPT 51.79 %. This is the headline
"low-FPR profile" CAIRN's framing prizes.

Adversarial robustness — 1 % token insertion / deletion / replacement
on GPT-4 Turbo text:

| Attack | LAPD AUROC | Fast-DetectGPT AUROC |
|---|---|---|
| insertion | 0.992 | 0.848 |
| deletion | 0.990 | 0.909 |
| replacement | 0.991 | 0.846 |

LAPD's adversarial drop is `< 0.01`; Fast-DetectGPT drops `~ 0.10–0.15`.

Per-sample cost: **0.58 s** at batch size 1 (paper does not list
hardware). Comparable to Fast-DetectGPT (0.56 s); faster than
DNA-DetectLLM (0.77 s).

## Why CAIRN-relevant

1. **Direct RAID delta of +0.069 absolute AUROC** (0.7835 → 0.852 if
   the paper's number reproduces on our RAID slice). That's `>2×` the
   A2 sprint's RAID gain from `7,000` Haiku calls of adversarial
   augmentation.
2. **Low-FPR profile (TPR @ 0.5% FPR ≈ 92 %)** matches CAIRN's
   conservative-classifier framing better than any current detector.
   Fits naturally into Mondrian-conformal v3 router class.
3. **Novel signal axis** — model-pair preference discrepancy. Today
   CAIRN has no detector that uses paired aligned/base models. EBM
   fusion benefits proportionally to the orthogonality of added
   signals.
4. **Adversarial robustness** is the A2 follow-on the spec asks for.
   Native LAPD robustness ≈ 0.99 means we can stop chasing one-off
   attack augmentation for the most common perturbations.

## Integration cost — preliminary estimate

| Day | Scope |
|---|---|
| 1 | `cairn/detectors/lapd.py` from paper equations. Lazy-import `transformers`. Pick a permissive-license model pair (`falcon-7B` ablation pair preferred — no Meta gate). Cascade Phase 3 (heavy-compute slot). Unit tests with fixture pairs against synthetic data with known-sign expectations. |
| 2 | `pyproject.toml`: `tier3` extras already includes `transformers`; model files are 14 GB combined fp16. Add 4-bit quantization path via `bitsandbytes` (optional extra). Add `pipeline.detector_availability` runtime check (model weights present? GPU available?). Wire to `configs/default.yaml::detectors`. |
| 3 | Modal feature-cache pass on the existing 110 k training corpus (~$3-5 on T4 with quantized models; ~$15 unquantized on A10). Concat into `data/all_features_v8.jsonl`. Retrain `master_minus_arxiv_robust_v2`. Validate on RAID + MAGE + per-attack. Update `docs/benchmarks/attack_coverage.md`. |

Net: **3 working days for integration + retrain + validation**.
Comparable to the A2 sprint that just landed.

## Cascade placement

Phase 3 (deep-but-slow). LAPD's `0.58 s` per sample × 7B-pair memory
footprint puts it firmly in the "phase 3 heavy-compute" slot alongside
`semantic_kinematics` and `perturbation`. **Not** an early-cascade
detector — never run unless phase 1 / 2 leave the sample ambiguous.

## Dependencies / extras impact

- `transformers` — already in `tier2` / `tier3`.
- `torch` — already optional.
- New: `bitsandbytes` (4-bit quantization) — `tier3` extras, optional
  unless GPU memory `< 16 GB`.
- Model weights — **not vendored**. Download instructions in detector
  docstring + `models/README.md`. Falcon-7B + Falcon-7B-Instruct are
  Apache-2.0 (permissive). Llama-2 pair requires Meta gate — avoid for
  the first cut.

Base install must stay tiny per CLAUDE.md — model-pair load is opt-in
through `cairn[tier3]` and the detector degrades with a clear SKIP
signal when neither weights nor GPU are present.

## Risk register

1. **No reference implementation.** Code repo is `Coming soon.` Risk:
   from-equations port has a sign error, dropped term, or off-by-one
   that silently degrades performance. **Mitigation:** build unit tests
   from synthetic distributions where the expected sign of `S` is
   known (LLM-style outputs → positive `Δ`, human → ~zero). Validate
   against the paper's reported numbers on a 100-sample HC3 sample as
   a gate before retraining.
2. **Two-model VRAM footprint.** 14 GB fp16 on Falcon-7B pair, 28 GB
   on Llama-2 pair. Modal T4 (16 GB) needs 4-bit quantization
   (`bitsandbytes`). Quantization could attenuate the alignment
   imprint signal. **Mitigation:** measure quantized vs full-precision
   LAPD AUROC on a small dev slice before committing to the quantized
   path.
3. **Attack scope.** Paper's robustness section tests 1 % token
   insertion/deletion/replacement. CAIRN's A2 vulnerability was
   `summarization_attack` (text rewritten via summarize-then-re-expand)
   — a different attack class. LAPD may or may not help with
   summarization rewrites. **Mitigation:** include summarization_attack
   in the validation suite. Do not claim LAPD fixes the A2
   vulnerability until measured.
4. **License lock-in on Llama pairs.** If we ship a profile that
   depends on `Llama-2`, users behind the Meta gate cannot run it.
   **Mitigation:** Falcon-7B pair as primary; document Llama-2 as an
   optional uplift for users who already have the weights.
5. **The "preference discrepancy" assumes the target model is aligned
   on the same task as φ.** If a user's text was produced by, e.g., a
   heavily RLHF'd Claude Opus 4.7 but our φ is Falcon-7B-Instruct, the
   discrepancy might attenuate. **Mitigation:** measure cross-model
   transfer on the calib2000 corpus before claiming generality.
6. **Cost of two model loads in production.** Phase 3 is reached only
   for ambiguous-middle samples (≈ 10-20 % of traffic under
   `master_minus_arxiv` cascade). 14 GB resident is still a real
   deployment ask. **Mitigation:** make LAPD opt-in via an extras
   group; the default `cairn gui` install does not load it.

## Recommendation

**Go.** Conditional on a 1-day spike to confirm the paper's RAID 0.852
number reproduces on our `data/ood_raid.jsonl` with a quantized
Falcon-7B pair. If the spike confirms `≥ 0.82 AUROC`, proceed with the
3-day integration. If the spike returns `< 0.78` (worse than current
robust_v1), abort and pivot to SpecDetect++ (next in the queue, far
cheaper signal).

## Open questions for the design phase

- Which detector calibration profile does LAPD plug into — does
  `master_minus_arxiv_robust_v2` retrain solo, or does it pair with
  SpecDetect++ in a single retrain? (Combined retrain is cheaper if
  both detectors land same week.)
- Does LAPD need its own Mondrian-conformal router class? The very
  high TPR @ 0.5 % FPR profile may pull conformal thresholds further
  from the cluster of existing detectors.
- Cache key: feature-cache should hash the model-pair identifier into
  the key alongside text — re-running with a different pair must not
  reuse stale feature rows.

## Next step

If go-ahead is given:

1. Draft `docs/superpowers/specs/2026-05-17-lapd-detector-design.md`
   (full design spec — class signature, signals schema, error modes,
   tests).
2. Run the 1-day reproducibility spike.
3. Proceed with Day 1-3 plan above.

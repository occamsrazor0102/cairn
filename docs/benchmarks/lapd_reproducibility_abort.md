# LAPD reproducibility -- ABORT

Date: 2026-05-17

The LAPD (Log-likelihood Alignment Preference Discrepancy) detector
scoped in [`2026-05-17-lapd-detector-scoping.md`](../superpowers/specs/2026-05-17-lapd-detector-scoping.md)
and designed in [`2026-05-17-lapd-detector-design.md`](../superpowers/specs/2026-05-17-lapd-detector-design.md)
**does not reproduce the paper's reported numbers on our stack** at the
scope tested. Following the scoping note's explicit abort branch, the
integration is dropped. The next session pivots to **SpecDetect++**
(arXiv 2508.11343).

## What we tested

- **Paper:** Wu et al., "Alignment Imprint: Zero-Shot AI-Generated Text
  Detection via Provable Preference Discrepancy", arXiv
  [2604.16923](https://arxiv.org/abs/2604.16923), April 2026.
- **Paper claim:** RAID AUROC 0.852; TPR @ 0.5% FPR averaging 92.27%.
- **Our pair:** `tiiuae/falcon-7b-instruct` + `tiiuae/falcon-7b`
  (Apache-2.0; Falcon ablation pair documented in the paper).
- **Slice:** `data/ood_raid.jsonl` (1,000 rows; label-balanced 500/500).
- **Implementation:** `cairn/detectors/lapd.py` -- from-equations port
  per design spec (paper's code repo `creator-xi/LAPD` is still
  "Coming soon" as of 2026-05-17).

## Spike runs

| Attempt | GPU | Outcome | Cost |
|---|---|---|---|
| 1 | T4 16 GB | `load_in_4bit` deprecated in `transformers.from_pretrained` -- fail at model-load API | ~$0.10 |
| 2 | T4 16 GB | After API fix: `device_map="auto"` validated CPU offload of second model, refused by bitsandbytes 4-bit | ~$0.20 |
| 3 | A10G 24 GB | After `device_map={"": 0}`, fp16, `low_cpu_mem_usage`: OOM during second model's `caching_allocator_warmup` (~14 GB peak per model) | ~$0.30 |
| 4 | A100-40GB | Models loaded; sweep `n_perturbations=20` completed on 1000 rows in 2,567 s; **AUROC 0.4343** | ~$1.80 |
| 5 | A100-40GB | Stratified 100-sample sub-spike to test sign-flip hypothesis; n=92 after `too_short` skips; **AUROC 0.351 (flipped 0.649)** | ~$0.20 |

Total Modal spend: **~$2.60**.

## Result

The headline number:

| Metric | Value |
|---|---|
| AUROC (p_llm, raw orientation) | **0.351** |
| AUROC (p_llm, sign-flipped) | **0.649** |
| AUROC (s_lapd, raw) | **0.351** |
| AUROC (s_lapd, sign-flipped) | **0.649** |
| Paper RAID AUROC | 0.852 |
| Spec gate threshold | 0.82 |
| Spec abort threshold | 0.78 |

Even under the most generous sign-flip interpretation, our reproduction
sits **17 absolute AUROC points below the paper's number** and **below
the 0.78 abort threshold from the scoping note**. The failure is
methodological, not numerical-precision-related: even the larger full-
sweep run (n=904 unstratified) returned AUROC 0.4343.

## Diagnostic notes

- **EETQ bypass warning:** Modal log surfaced
  `"You are loading your model using eetq but no linear modules were
  found in your model"`, indicating bitsandbytes 4-bit quantization was
  not applied (the first model held ~13.71 GiB resident -- fp16
  footprint, not 4-bit). However, full precision should *improve*
  detection signal, not degrade it, so this is not the source of the
  AUROC gap.
- **Sweep-1 skipped:** 96/1000 rows skipped as `too_short`, all
  likely `label=0` (human, since they cluster at the start of the file).
- **Sub-spike stratification:** raw `data/ood_raid.jsonl` is sorted by
  label (rows 0-499 are human, 500-999 are LLM); a flat 100-row cap
  yields a single-class subset and undefined AUROC. The spike script
  now interleaves classes before applying `max_samples`.

## Probable root causes (untested hypotheses)

1. **Perturbation strategy differs from the paper.** Our standardizer
   uses random word-swap perturbations drawn from the input's own
   vocabulary (mirrors `cairn/detectors/perturbation.py`). The paper
   describes "conditional-independence sampling" with explicit
   `E[I_phi(x_t) * Delta(x_t)]` expectations; the math implies a
   structured per-token resampling we did not implement. If mu/sigma
   are off, `S_LAPD` is a poor signal regardless of model precision.
2. **Heuristic anchor mismatch.** `_HEURISTIC_S_LAPD_TO_P_LLM` was
   anchored to a hand-picked Gaussian under human authorship -- if
   the actual `S_LAPD` distribution shifts in the opposite direction
   (we observed negative correlation with `label=1`), the p_llm
   mapping inverts.
3. **Model-pair drift.** The paper's primary experiments are on
   `Llama2-7B / Llama2-7B-Instruct`; Falcon-7B is one of three ablation
   pairs. The "alignment imprint" may be markedly weaker on
   Falcon-instruct relative to Llama-instruct (different RLHF
   procedure, different instruct dataset).
4. **RAID slice differences.** We tested on `data/ood_raid.jsonl`
   (1000-row selection from RAID); the paper's RAID number may be
   on a different subset or with different filtering.

## What remains on disk

The implementation, design spec, scoping note, and spike script all
remain committed. They are structurally sound -- they could be
revived if someone reimplements the paper's exact conditional-
independence perturbation. Files:

- `cairn/detectors/lapd.py`
- `tests/test_detectors/test_lapd.py` (18 cases, all green)
- `scripts/modal_spike_lapd.py`
- `configs/default.yaml` (`detectors.lapd:` block, `enabled: false`)
- `cairn/cascade.py` (`lapd` in `PHASE3_DETECTORS`)
- `cairn/fusion/ebm.py` (`lapd_p_llm`/`_missing`/`_confidence` keys)
- `docs/superpowers/specs/2026-05-17-lapd-detector-{scoping,design}.md`

We are not deleting any of this -- `enabled: false` keeps it inert in
production but available for a future revival attempt.

## Next

Per the scoping note's abort branch: pivot to **SpecDetect++** (arXiv
2508.11343). Cheap, orthogonal signal; matches CAIRN's early-cascade
slot. New session opens with that scoping note.

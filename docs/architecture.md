# CAIRN — Architecture

This doc describes the runtime pipeline, the detector contract, and how
cascade / fusion / decision compose to produce a verdict.

## Pipeline

```
                ┌──────────┐
  text in ─────►│  Router  │─── {prompt | prose | mixed | insufficient}
                └──────────┘
                     │
                     ▼
             ┌────────────────┐
             │ CascadeScheduler│
             └────────────────┘
                     │
                     ▼
   Phase 1 (always) ─► Phase 2 (if inconclusive) ─► Phase 3 (if still inconclusive)
     trivial/cheap       moderate (statistical)       expensive (LLM-backed)
        │                      │                             │
        └──────────────────────┴─────────────────────────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │   EBM Fusion    │
                             └─────────────────┘
                                      │
                                      ▼
                      ┌────────────────────────────┐
                      │ Conformal prediction set    │
                      └────────────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────┐
                           │  DecisionEngine  │
                           └──────────────────┘
                                      │
                                      ▼
                              Determination
```

`cairn/pipeline.py::CairnPipeline.analyze(text)` wires these together and
returns a `Determination` object (defined in `cairn/contracts.py`).

## Core contracts

`cairn/contracts.py` defines three dataclasses plus a detector protocol:

### `LayerResult`
One detector's output.
```python
layer_id: str
domain: "prompt" | "prose" | "universal"
raw_score: float
p_llm: float          # calibrated P(LLM), 0.0-1.0
confidence: float     # detector's trust in its own output
signals: dict         # detector-specific key/value evidence
determination: str    # per-detector verdict (RED/AMBER/...)
attacker_tiers: list[str]
compute_cost: "trivial" | "cheap" | "moderate" | "expensive"
min_text_length: int
spans: list[dict]     # optional character-level evidence
```

### `FusionResult`
Aggregated across all layers.
```python
p_llm: float
confidence_interval: (lo, hi)
prediction_set: list[str]
feature_contributions: dict[str, float]
top_contributors: list[(name, value)]
```

### `Determination`
The final verdict returned by `pipeline.analyze`.
```python
label: str
p_llm: float
confidence_interval: (lo, hi)
prediction_set: list[str]
reason: str
top_features: list[(name, value)]
override_applied: bool
detectors_run: list[str]
cascade_phases: list[int]
mixed_report: dict | None
layer_results: list[LayerResult]
feature_contributions: dict[str, float]
```

### `Detector` protocol
```python
class Detector(Protocol):
    id: str
    domain: str
    compute_cost: str
    def analyze(self, text: str, config: dict) -> LayerResult: ...
    def calibrate(self, labeled_data: list) -> None: ...
```

## Router

`cairn/router.py` classifies incoming text into one of four domains:
`prompt`, `prose`, `mixed`, `insufficient`. The result decides which
detectors are *recommended* versus *skipped* — e.g. prompt-shaped
detectors (`instruction_density`, `voice_spec`, `prompt_structure`) are
routed around when the input is clearly prose.

Routing is conservative: when confidence is low, all detectors run.

## Cascade

`cairn/cascade.py::CascadeScheduler` orchestrates four phases with
short-circuit logic driven by configurable thresholds
(`cascade.phase1_short_circuit_{high,low}`, etc.):

| Phase | Detectors | When it runs |
|------:|-----------|--------------|
| 1 | preamble, fingerprint_vocab, prompt_structure, voice_spec, instruction_density, nssi | always |
| 2 | surprisal_dynamics, contrastive_lm, token_cohesiveness, mixed_boundary | if aggregated Phase 1 p(LLM) falls in the uncertain band (between `phase1_short_circuit_low` and `phase1_short_circuit_high`; defaults 0.10 and 0.85) |
| 3 | perturbation, contrastive_gen, dna_gpt | if aggregated Phase 1+2 is still uncertain |
| 4 | cross_similarity | batch-only (requires multiple submissions). `contributor_graph` is invoked separately via `pipeline.analyze_contributors`, not through the normal cascade. |

`phase3_always_run: true` forces every detector regardless of
intermediate confidence — useful for evaluation.

## Fusion

`cairn/fusion/ebm.py::EBMFusion` has two modes, selected by config:

- **Naive weighted mean (heuristic mode).** If no trained model
  artifact is loaded (controlled by `fusion.model_path` in the
  active config), the fusion returns a weighted mean of per-detector
  `p_llm` values and reports per-detector deviations from prior
  (`(p_llm - 0.5) * confidence`) as pseudo-contributions. The
  `default` / `screening` / `strict` profiles run in this mode.
  `/health` reports `calibration_status: "heuristic"` and the GUI
  shows a banner.
- **EBM (calibrated mode).** When a fitted Explainable Boosting
  Machine is loaded via `fusion.model_path`, the fusion builds a
  feature vector and returns a fused `p_llm` plus additive feature
  contributions. The feature vector includes per-detector `p_llm`,
  detector confidence, plus length-aware features
  (`log_word_count`, `is_short` / `is_medium` / `is_long`) so the
  EBM can route differently for short vs long inputs. The EBM
  trains with `interactions=15` to capture pairwise effects.

### Conformal prediction

`cairn/fusion/conformal.py::ConformalWrapper` is a true split-conformal
wrapper trained by `scripts/calibrate_conformal.py` on a held-out
calibration set. With `alpha=0.05` it provides marginal coverage on
in-distribution test data — the binary `conformal_set` field on
`FusionResult` is honest under exchangeability assumptions; the
`prediction_set` field carries the cosmetic severity-band mapping for
UI display. Coverage degrades on OOD data (MAGE, RAID, unseen
providers); the prediction set widens but the coverage guarantee
holds only inasmuch as the calibration distribution matches.

### Per-detector calibration

Each detector ships a `_HEURISTIC_*` raw-score → `p_llm` table used
when no calibration is loaded. Calibrated profiles override these via
`cairn.calibration.DetectorCalibrator` — per-detector isotonic
regression fit by `scripts/calibrate_detectors.py` on a labeled
training set. `DetectorCalibrator.has(layer_id)` gates whether the
isotonic transform applies; otherwise the heuristic table is used.

## Profile system

Profiles in `configs/` come in two flavors:

- **Heuristic profiles** (`default`, `screening`, `strict`,
  `no-api`, `production`) — no `fusion.model_path`; all detectors
  use their `_HEURISTIC_*` tables; fusion is the naive weighted mean.
- **Calibrated profiles** (`calibrated_*`, `profile_*`) — set
  `fusion.model_path`, `fusion.conformal_path`,
  `fusion.detector_calibration_path` to artifacts under
  `models/calib2000/`. These are produced by the training pipeline
  documented in [`docs/calibration.md`](calibration.md).

`RuntimeContext.switch_profile()` hot-swaps between any two profiles
without restarting the process; the EBM/calibrator/conformal artifacts
are loaded into a new `CairnPipeline` outside the context lock and
swapped atomically.

## Feature cache (training-only)

`cairn/feature_cache.py::CachedPipeline` is a drop-in replacement for
`CairnPipeline` used by the training scripts. It keys detector outputs
by `sha256(normalize_text(input))` and serves cached `LayerResult`
lists in <1ms, bypassing all detector inference. This is what makes
profile-grid and LOO experiments tractable — once
`scripts/cache_pipeline_features.py` (or
`scripts/modal_cache_features.py` for GPU offload) writes
`data/all_features_v6.jsonl`, every subsequent
`scripts/calibrate_detectors.py` / `train_fusion.py` /
`run_benchmark.py` invocation reads from cache instead of re-running
the pipeline.

The cache is calibrator-independent: it stores raw detector outputs
(`raw_score`, signals, spans), and the per-detector calibration is
applied at fusion-time, not at cache-write-time. Same cache feeds all
13+ profile variants.

## Decision engine

`cairn/decision.py::DecisionEngine` maps `FusionResult` → `Determination`:

1. **Override rules** — if a detector returned a `CRITICAL` severity
   signal (today: `preamble` only), short-circuit to RED with
   `override_applied=True`.
2. **Abstention** — if the conformal prediction set spans too many
   severity bands (`decision.abstention.max_prediction_set_size`),
   abstain to UNCERTAIN.
3. **Threshold mapping** — fused `p_llm` → RED / AMBER / YELLOW /
   GREEN via `red_threshold`, `amber_threshold`, `yellow_threshold`.

## Evaluation

`cairn/evaluation/` is separable from the live pipeline:

- `dataset.py` — JSONL loader / saver / builder for `EvalSample` lists
- `metrics.py` — pure functions for AUROC, ECE, Brier, TPR@FPR,
  confusion_at_threshold
- `runner.py` — `run_eval(pipeline, dataset) -> EvalReport`;
  `eval_report_to_dict` for UI-safe JSON serialization
- `ablation.py` — leave-one-out detector ablation
- `fairness.py` — per-group metric breakdown
- `robustness.py` — per-attack breakdown (requires adversarial transforms)

## Monitoring

`cairn/monitoring/drift.py::DriftMonitor` tracks feature histograms,
population p(LLM), and (if reviewer labels are recorded) calibration
via ECE. Baselines are rebuilt from the history store on demand.

`cairn/monitoring/meta_detector.py::MetaDetector` records per-detector
agreement with reviewer feedback to flag degradation over time.

## Runtime shells

All three shells consume the same `index.html` and the same RPC
surface:

| Shell | Backend | Transport |
|-------|---------|-----------|
| `cairn gui` | stdlib HTTP in `cairn/gui/server.py` | `fetch()` |
| `cairn serve` | FastAPI in `cairn.api.create_app` | `fetch()` |
| Tauri desktop | `cairn.sidecar` subprocess | stdio JSON-RPC + `invoke()` |

The frontend picks `fetch` vs `invoke` via `window.__TAURI_INTERNALS__`
detection, so one `index.html` works everywhere.

`cairn/runtime.py::RuntimeContext` holds `{pipeline, profile, config}`
and provides a thread-safe `switch_profile()` so the UI can hot-swap
the active config without restarting the process. The new pipeline is
built outside the lock; the assignment is atomic; a failed switch
leaves the old context untouched.

## Privacy and provenance

- `cairn/privacy/store.py::PrivacyStore` — feature-vector store with
  optional raw-text vault and an access log.
- `cairn/privacy/retention.py` — purge-by-age with a `--confirm` gate.
- `cairn/provenance/chain.py::AuditChain` — hash-chained JSONL where
  each entry's `_hash` incorporates the previous entry's hash. Any
  tamper breaks the chain and `validate()` surfaces it. Wired into
  `CairnPipeline.analyze` as a best-effort append (default off; opt in
  via `audit.enabled: true`). Each appended record carries:

  ```json
  {
    "submission_id": "...",
    "text_hash": "sha256:...",
    "profile": "master_minus_arxiv",
    "p_llm": 0.84,
    "determination": "AMBER",
    "calibration_status": "calibrated",
    "model_artifact_hash": "sha256:...",
    "timestamp": "2026-05-07T23:54:00Z"
  }
  ```

  `calibration_status` records one of `"heuristic"`, `"fusion-only"`,
  `"calibrated"`, or `"degraded"` (with the same
  `getattr(conformal, "calibrated", False)` guard so a loaded-but-unfit
  conformal does not falsely promote records).

  The submission text itself is never written — only `text_hash`
  (`cairn/privacy/hashing.py::hash_text`, which NFKC-normalizes,
  collapses whitespace, and lowercases before SHA-256, so two
  submissions that differ only in casing or whitespace map to the
  same hash). This holds even when `gui.history.retain_text=true`.
  Validate with `cairn audit verify`. A failed chain-append never
  fails analyze: same best-effort discipline as history and drift
  writes.

## History store

`cairn/history.py::HistoryStore` is a SQLite-backed queryable log that
sits alongside PrivacyStore and AuditChain. Every analyze / batch /
feedback call writes through it so the History, Batch, Monitoring, and
Evaluation tabs can filter, sort, paginate, and export across the
entire submission history without re-running the pipeline.

Schema lives in `cairn/history.py::SCHEMA`. Two tables: `submissions`
(one row per analyze) and `feedback_log` (0..N per submission).

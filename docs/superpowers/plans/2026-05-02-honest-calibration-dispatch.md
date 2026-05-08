# CAIRN Honest Calibration Story — Subagent Dispatch Plan

> **⚠ SUPERSEDED — see [`2026-05-02-honest-calibration-unified.md`](2026-05-02-honest-calibration-unified.md)** for the canonical merged plan (this dispatch structure + the original plan's concrete TDD code, ten phases plus integration). This document is preserved for provenance.

> **Earlier note:** this plan superseded [`2026-05-02-honest-calibration.md`](2026-05-02-honest-calibration.md), which is also preserved for snippet provenance.

## Operating rule

Each phase should be handled by one fresh subagent. Every subagent gets a narrow mission, modifies only the listed files, writes tests first where practical, runs targeted tests, then runs a broader suite before handoff.

No subagent should mutate `master` directly. Every phase should land as its own branch or PR. Any phase that touches trained artifacts must log the dataset path, artifact path, metrics before/after, and whether the artifact was actually promoted.

## Phase 0 baseline (recorded 2026-05-02)

- Branch: `claude/pedantic-meninsky-32f39e` (worktree). Parent repo at `C:\Users\occam\My Projects\LLM Detector Reboot\`.
- Pytest baseline: **392 passed / 0 failed / 0 skipped / 36 warnings** in 102s under `pytest -x -q`. Python 3.14.3, pytest 9.0.2.
- Installed: `interpret==0.7.8`, `scikit-learn==1.8.0`, `numpy==2.4.4`, `torch==2.11.0`, `transformers==5.5.0`, `sentence-transformers==5.3.0`.
- **NOT installed** (gates downstream phases): `skops` (Phase 7), `faiss-cpu` (Phase 8).
- **Profile config naming gotcha:** the recommended profile configs are dot-prefixed/hidden:
  - `configs/.profile_master_minus_arxiv.yaml` (prose)
  - `configs/.profile_prompts_only.yaml` (prompts)
  Many other profiles also dot-prefixed under `configs/.profile_*.yaml`. Use `ls -a configs/` to enumerate. Do not write `configs/master_minus_arxiv.yaml` in any dispatch prompt — that path does not exist.
- **`data/` and `models/` live in the parent repo, not in this worktree.** Phases that read corpora or load trained artifacts (Phase 4, 5, 6, 9) must run from the parent repo OR pass absolute paths OR symlink. Configs reference relative paths so cwd matters. Canonical paths:
  - Production EBM: `<parent>/models/calib2000/profiles/master_minus_arxiv/ebm.pkl` (also `.../prompts_only/ebm.pkl`)
  - Production conformal: `<parent>/models/calib2000/profiles/master_minus_arxiv/conformal.json`
  - Production detector calibration: `<parent>/models/calib2000/profiles/master_minus_arxiv/detector_calibration.json`
  - Canonical training corpus: `<parent>/data/calib2000_v6_train.jsonl`
  - Feature cache: `<parent>/data/all_features_v6.jsonl`
  - OOD validation: `<parent>/data/ood_m4_test.jsonl`, `<parent>/data/ood_raid_test.jsonl`, `<parent>/data/ood_mage_test.jsonl`
- Current artifacts use plain `pickle` (`.pkl`), not skops. Phase 7 will need a one-time migration.

## Dependency graph

```text
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 5
                    └─► Phase 4 ──► Phase 6 ──► Phase 9
                              └────► Phase 7
                    └─────────────► Phase 8
```

Recommended dispatch waves:

- **Wave 1:** Phase 0, then Phase 1.
- **Wave 2:** Phase 2 and Phase 4 in parallel after Phase 1.
- **Wave 3:** Phase 3 after Phase 2. Phase 6 and Phase 7 after Phase 4. Phase 8 can also run after Phase 1.
- **Wave 4:** Phase 5 after Phases 3 and 4.
- **Wave 5:** Phase 9 after Phases 4 and 6.

---

# Phase 0 — Pre-flight and repo baseline

## Subagent mission

Establish a clean baseline before any calibration, schema, model, or detector changes.

## Files

No intended code changes unless a trivial environment note is needed.

## Tasks

Confirm the active branch is not `master`.

Confirm the baseline test suite status with:

```bash
pytest -x -q
```

Record the exact number of passing tests, failures, skips, warnings, Python version, and installed extras.

Confirm these files exist:

```text
cairn/fusion/conformal.py
cairn/fusion/ebm.py
cairn/fusion/training.py
cairn/decision.py
cairn/contracts.py
cairn/router.py
cairn/pipeline.py
cairn/evaluation/runner.py
cairn/evaluation/dataset.py
cairn/detectors/cross_similarity.py
scripts/train_fusion.py
scripts/calibrate_conformal.py
```

Locate the canonical training dataset used for the currently shipped EBM artifacts. Record dataset path, artifact path, config profile, and held-out benchmark path.

## Acceptance gate

Subagent returns:

```text
branch:
python_version:
pytest_baseline:
canonical_training_dataset:
current_ebm_artifact:
current_conformal_artifact:
current_detector_calibration_artifact:
benchmark_reference:
known_environment_gaps:
```

No merge needed unless documentation was updated.

---

# Phase 1 — Decision-engine coverage accounting

## Subagent mission

Make the decision engine honest when conformal coverage does not apply. Highest-priority blast-radius fix because it changes user-facing truth without requiring retraining.

## Why this phase comes first

Later conformal improvements depend on the decision engine handling all conformal set states correctly. The current logic correctly handles non-singleton sets, but the audit identifies the empty conformal set as a missing abstention path. An empty set is not "confident." It is "neither class fits."

## Files

Modify:

```text
cairn/decision.py
cairn/contracts.py
tests/test_decision.py
tests/test_decision_abstention.py
configs/default.yaml
configs/*.yaml as needed
```

## Required changes

Add `conformal_empty` to `Determination.abstention_reason`.

Add `coverage_guarantee_applies: bool = True` to `Determination`.

Treat `fusion.conformal_set == []` as `UNCERTAIN`, with `abstention_reason="conformal_empty"`.

Set `coverage_guarantee_applies=False` whenever a hard override bypasses normal conformal decisioning, especially:

```text
CRITICAL preamble override
MIXED authorship override
```

Move the hardcoded mixed-boundary threshold from `0.60` into config as:

```yaml
decision:
  mixed_threshold: 0.60
```

## Tests to add

- Empty conformal set yields `label == "UNCERTAIN"`, `abstention_reason == "conformal_empty"`, reason mentions OOD or α-coverage.
- CRITICAL preamble override sets `label == "RED"`, `override_applied == True`, `coverage_guarantee_applies == False`.
- Normal conformal decision path keeps `coverage_guarantee_applies == True`.
- `decision.mixed_threshold` is configurable and no longer hardcoded.

## Acceptance gate

```bash
pytest tests/test_decision.py tests/test_decision_abstention.py -v
pytest -x -q
```

Subagent returns:

```text
fields_added:
abstention_paths_added:
config_keys_added:
tests_added:
pytest_result:
schema_compatibility_notes:
```

## Dispatch prompt

```text
Implement Phase 1 of the CAIRN Honest Calibration Story. Make cairn/decision.py honest about
conformal coverage. Add empty conformal-set abstention, add coverage_guarantee_applies to
Determination, mark override paths as coverage_guarantee_applies=False, and move the
mixed-boundary threshold into decision.mixed_threshold. Write tests first in tests/test_decision.py
and tests/test_decision_abstention.py. Preserve existing behavior except where the audit explicitly
requires honesty fixes. Run targeted tests and pytest -x -q before handoff.
```

---

# Phase 2 — Class-conditional Mondrian conformal

## Subagent mission

Upgrade conformal prediction from one global marginal threshold to one threshold per class: human and LLM.

## Files

Modify:

```text
cairn/fusion/conformal.py
scripts/calibrate_conformal.py
tests/test_conformal.py
```

Create `tests/test_conformal.py` if it does not already exist.

## Required changes

Change `ConformalWrapper.calibrate(scores, labels)` so it computes:

```text
threshold_per_class[0] for human
threshold_per_class[1] for LLM
```

Add:

```python
threshold_for(cls: int) -> float | None
```

Change `predict_set(p_llm)` to use class-specific thresholds.

Bump artifact schema:

```text
binary-split-conformal-v1  # old global threshold
binary-split-conformal-v2  # new class-conditional thresholds
```

Maintain backward compatibility:

```text
v1 loads as threshold_per_class = {0: old_threshold, 1: old_threshold}
```

Update `scripts/calibrate_conformal.py` so it prints both thresholds after calibration.

## Tests to add

- Calibration stores separate class thresholds.
- Approximate per-class coverage on synthetic data.
- Too few calibration points in either class raises a clear error.
- Save/load roundtrip for v2.
- v1 artifact load compatibility.

## Acceptance gate

```bash
pytest tests/test_conformal.py -v
pytest -x -q
```

Subagent returns:

```text
new_schema:
backward_compatibility:
thresholds_printed:
tests_added:
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 2 of the CAIRN Honest Calibration Story. Upgrade cairn/fusion/conformal.py to
class-conditional Mondrian conformal: one threshold for human, one threshold for LLM. Add
threshold_for(cls), update predict_set, bump artifact schema to binary-split-conformal-v2, and
keep v1 artifacts readable by lifting the global threshold into both classes. Update
scripts/calibrate_conformal.py to print both thresholds. Add tests for per-class thresholds,
coverage, too-few-class samples, v2 roundtrip, and v1 load compatibility.
```

---

# Phase 3 — Router-conditional Mondrian conformal

## Subagent mission

Extend conformal calibration from class-only to `(router_domain, class)` thresholds, with safe fallback to class-only thresholds when a router/class cell has too little data.

## Files

Modify:

```text
cairn/fusion/conformal.py
cairn/fusion/ebm.py
cairn/pipeline.py
scripts/calibrate_conformal.py
tests/test_conformal.py
tests/test_fusion.py
tests/test_pipeline.py
```

## Required changes

Extend calibration signature:

```python
calibrate(scores, labels, router_classes=None)
```

Always compute class-only thresholds as fallback.

Optionally compute thresholds for:

```text
(prompt, human), (prompt, LLM)
(prose, human), (prose, LLM)
(mixed, human), (mixed, LLM)
(insufficient, human), (insufficient, LLM)
```

Use a minimum per-cell sample threshold:

```python
_MIN_PER_CELL = 30
```

Add:

```python
threshold_for(cls: int, router_class: str | None = None)
predict_set(p_llm: float, router_class: str | None = None)
```

Bump schema to `binary-split-conformal-v3`. Maintain compatibility with v1 and v2.

Plumb `RouterDecision.domain` through:

```text
pipeline → EBMFusion.fuse(...) → _maybe_conformal(...) → ConformalWrapper.predict_set(...)
```

Update `scripts/calibrate_conformal.py` to call `pipeline.analyze_detailed(...)` so it can collect router domain cleanly.

## Tests to add

- Per-router thresholds are stored when cells are sufficiently populated.
- Fallback to class-only threshold when a router/class cell is underpowered.
- `EBMFusion` passes `router_class` into conformal prediction.
- Pipeline passes `RouterDecision.domain`.

**Important correction to the uploaded draft:** avoid a test that expects `threshold_for(1, "prose")` to differ when there are no LLM prose samples in the synthetic setup. Build synthetic data with both labels represented in each router class.

## Acceptance gate

```bash
pytest tests/test_conformal.py tests/test_fusion.py tests/test_pipeline.py -v
pytest -x -q
```

Subagent returns:

```text
new_schema:
fallback_policy:
minimum_cell_size:
router_domains_supported:
tests_added:
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 3 of the CAIRN Honest Calibration Story. Extend ConformalWrapper to
router-conditional Mondrian conformal. Add per-(router_class, label) thresholds when each cell has
enough calibration points, with fallback to class-only thresholds. Bump schema to v3 while
preserving v1/v2 load compatibility. Plumb RouterDecision.domain from pipeline into EBMFusion and
ConformalWrapper.predict_set. Update calibrate_conformal.py to collect router domains using
analyze_detailed. Add tests for per-router thresholds, fallback behavior, fusion plumbing, and
pipeline plumbing. Build synthetic test data with both labels present in each router class.
```

---

# Phase 4 — EBM missingness, detector confidence, and schema hardening

## Subagent mission

Strengthen EBM inputs so the model can distinguish "detector absent" from "detector neutral," and "detector fired weakly" from "detector fired confidently."

## Files

Modify:

```text
cairn/fusion/ebm.py
cairn/fusion/training.py
tests/test_fusion.py
tests/test_ebm_schema_coverage.py
```

## Required changes

Add `<detector>_missing` features.

Add `<detector>_confidence` features.

Replace fixed `0.5` NaN imputation with column means computed during training and stored on the artifact:

```python
model._cairn_feature_means
```

Use stored means at inference. Keep fallback `0.5` imputation only for legacy artifacts without `_cairn_feature_means`.

Promote artifact schema mismatch from warning to `RuntimeError`.

## Tests to add

- Assembler emits missingness and confidence features.
- Present detector → `missing == 0.0`, `confidence == detector confidence`.
- Absent detector → `missing == 1.0`, `confidence == NaN`.
- Training stores column means.
- Inference uses stored column means.
- Artifact feature schema mismatch raises `RuntimeError`.

## Retraining requirement

Retrain EBM on the canonical dataset identified in Phase 0. Compare old vs new artifact on the same held-out benchmark. Do not promote the new artifact unless the subagent reports AUROC, ECE, Brier, TPR@1%FPR, F1 deltas. If performance worsens, keep the code change but do not swap production artifact.

## Acceptance gate

```bash
pytest tests/test_fusion.py tests/test_ebm_schema_coverage.py -v
pytest -x -q
```

Subagent returns:

```text
features_added:
artifact_schema_change:
legacy_artifact_behavior:
metrics_before:
metrics_after:
artifact_promoted: true/false
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 4 of the CAIRN Honest Calibration Story. Add detector missingness indicators and
detector confidence features to FeatureAssembler. Replace fixed 0.5 NaN imputation with stored
column means from train_ebm. Use stored means at inference with legacy fallback. Make extra
features expected by an EBM artifact a RuntimeError instead of a warning. Add tests for assembler
output, column means, inference imputation, and artifact schema mismatch. Retrain on the canonical
dataset from Phase 0 and report before/after metrics before promoting any artifact.
```

---

# Phase 5 — Honest confidence intervals and monotonic constraints

## Subagent mission

Stop reporting hardcoded confidence intervals when calibrated conformal information exists. Add monotonic constraints for features whose meaning should only increase LLM probability.

## Files

Modify:

```text
cairn/fusion/ebm.py
cairn/fusion/training.py
tests/test_fusion.py
```

## Required changes

When conformal is calibrated, derive CI half-width from conformal thresholds:

```text
ci_halfwidth = max(
    threshold_for(human, router_class),
    threshold_for(LLM, router_class)
)
```

When no conformal wrapper exists in naive mode, report maximum uncertainty:

```python
confidence_interval = (0.0, 1.0)
```

Add monotonic constraints during EBM training for:

```text
preamble_severity
preamble_n_matches
fingerprint_hits_per_1000
fingerprint_bigram_hits
nssi_n_signals_active
nssi_formulaic_density
ps_meta_design_hits
```

Verify the correct `interpret-ml` argument name for the installed version before committing. If version support is inconsistent, pin the required version in `pyproject.toml`.

## Tests to add

- EBM fuse uses conformal threshold to compute CI.
- Naive uncalibrated fusion reports `(0.0, 1.0)`.
- Training passes monotonic constraints aligned to feature names.

## Retraining requirement

Retrain after monotonic constraints. Promote only if accuracy does not regress materially:

```text
AUROC drop must be < 0.005
TPR@1%FPR drop must be reviewed
ECE/Brier should not worsen materially
```

## Acceptance gate

```bash
pytest tests/test_fusion.py -v
pytest -x -q
```

Subagent returns:

```text
ci_policy:
monotonic_features:
interpret_ml_version:
metrics_before:
metrics_after:
artifact_promoted: true/false
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 5 of the CAIRN Honest Calibration Story. Replace hardcoded EBM ±0.10 confidence
intervals with conformal-threshold-derived intervals when conformal is calibrated. In naive
uncalibrated mode, report CI=(0,1). Add monotonic constraints in train_ebm for physically monotone
fingerprint/preamble features. Verify interpret-ml support, add tests, retrain, and report
before/after benchmark metrics before artifact promotion.
```

---

# Phase 6 — Leave-one-LLM-out training and per-source evaluation

## Subagent mission

Expose generator-family generalization instead of hiding it inside aggregate AUROC.

## Files

Modify:

```text
scripts/train_fusion.py
cairn/evaluation/runner.py
tests/test_evaluation/test_runner.py
```

## Required changes

Add `--holdout-source <source_name>` to `scripts/train_fusion.py`:

```text
training set = all samples whose source != holdout source
test set = all samples whose source == holdout source
```

Fail loudly if the holdout source matches zero samples.

Add `per_source` metrics to `EvalReport`. Add `per_source` serialization in `eval_report_to_dict`:

```json
{
  "per_source": {
    "openai": {"auroc": 0.91, "ece": 0.08},
    "anthropic": {"auroc": 0.96, "ece": 0.05}
  }
}
```

Optionally add `per_attack_category` if `EvalSample.attack_category` is already populated.

## Tests to add

- `eval_report_to_dict` includes `per_source`.
- Per-source metrics are computed only for samples with source metadata.
- Holdout source split rejects empty holdout.

## Acceptance gate

```bash
pytest tests/test_evaluation/test_runner.py -v
pytest -x -q
```

Smoke holdout:

```bash
python scripts/train_fusion.py \
  --dataset <canonical>.jsonl \
  --config <profile>.yaml \
  --out /tmp/holdout_test.skops \
  --holdout-source <known_source>
```

Subagent returns:

```text
holdout_source_tested:
train_count:
holdout_count:
per_source_metrics_added:
tests_added:
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 6 of the CAIRN Honest Calibration Story. Add --holdout-source to
scripts/train_fusion.py for leave-one-generator-family-out validation. Extend
cairn/evaluation/runner.py so EvalReport and eval_report_to_dict include per_source metrics. Add
tests proving per_source serialization works and the holdout split rejects empty source matches.
Run a smoke holdout training job on a known source and report train/holdout counts.
```

---

# Phase 7 — Safer model serialization and artifact provenance

## Subagent mission

Replace unsafe pickle-first artifact handling with `skops`, while keeping legacy pickle artifacts readable during migration.

## Files

Modify:

```text
cairn/fusion/training.py
scripts/train_fusion.py
pyproject.toml
tests/test_fusion.py
```

## Required changes

Add `skops` to the fusion optional dependency.

Change `save_model` to use `skops.io.dump`.

Change `load_model` to:

```text
detect skops/zip artifact
load via skops with explicit trusted types
otherwise load legacy pickle with DeprecationWarning
```

Attach artifact metadata:

```python
model._cairn_metadata = {
    "trained_at": ...,
    "n_train": ...,
    "n_features": ...,
    "label_balance": {"0": ..., "1": ...},
    "dataset_path": ...,
    "schema_version": "ebm-v1"
}
```

Update `scripts/train_fusion.py` to pass `dataset_path`.

## Tests to add

- `save_model` writes skops/zip-style artifact.
- `load_model` reads skops artifact.
- `load_model` still reads legacy pickle artifact and warns.
- `train_ebm` attaches provenance metadata.

## Acceptance gate

```bash
pip install -e ".[fusion]"
pytest tests/test_fusion.py -v
pytest -x -q
```

Subagent returns:

```text
serialization_format:
legacy_pickle_behavior:
metadata_fields:
skops_trusted_types:
tests_added:
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 7 of the CAIRN Honest Calibration Story. Replace model save/load in
cairn/fusion/training.py with skops serialization. Keep legacy pickle load support with a
DeprecationWarning. Add artifact provenance metadata including trained_at, n_train, n_features,
label balance, dataset path, and schema version. Add skops to pyproject.toml. Update
scripts/train_fusion.py to pass dataset_path. Add tests for skops roundtrip, legacy pickle
fallback, warning behavior, and metadata.
```

---

# Phase 8 — Retrieval-based cross-similarity

## Subagent mission

Upgrade `cross_similarity` from within-batch similarity to retrieval against a persistent historical index.

## Files

Create:

```text
cairn/index/__init__.py
cairn/index/retrieval.py
tests/test_index/test_retrieval.py
```

Modify:

```text
cairn/detectors/cross_similarity.py
cairn/history.py
pyproject.toml
tests for cross_similarity/history as needed
```

## Required changes

Add optional dependency group:

```toml
retrieval = [
  "sentence-transformers>=2.7",
  "faiss-cpu>=1.7"
]
```

Implement `RetrievalIndex` with `add(doc_id, text)`, `query(text, k=5) -> list[tuple[str, float]]`, `save()`, `load()`. Use normalized sentence-transformer embeddings and FAISS inner product search.

Update `cross_similarity` single-text path:

```text
if retrieval_index exists:
    query historical index
    emit retrieval_max_similarity
    emit retrieval_top_id
    emit retrieval_n_hits_above_threshold
else:
    preserve existing SKIP/fallback behavior
```

Wire history ingestion so new submissions can update the retrieval index.

Cold-start: empty index → SKIP with reason `"empty_retrieval_index"`.

## Tests to add

- Index add/query returns expected nearest neighbor.
- Index persists across instances.
- Cross-similarity uses retrieval index when provided.
- History-recorded submission becomes retrievable, if the index is configured.

**Important: keep retrieval tests behind `pytest.importorskip` so the embedding model is never silently downloaded during normal CI runs.**

## Acceptance gate

```bash
pip install -e ".[retrieval]"
pytest tests/test_index/test_retrieval.py -v
pytest tests/test_detectors/ -v -k cross_similarity
pytest -x -q
```

Subagent returns:

```text
index_backend:
embedding_model:
persistence_path:
cold_start_behavior:
history_ingest_behavior:
tests_added:
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 8 of the CAIRN Honest Calibration Story. Add a retrieval index module using
sentence-transformers and FAISS. Refactor cross_similarity so it can query a persistent
historical vector index for near-neighbor matches, while preserving existing batch/cold-start
behavior. Add optional retrieval dependencies. Wire history ingestion to update the index when
configured. Add tests for add/query, persistence, cross_similarity retrieval use, and history
ingest behavior. Keep all retrieval tests behind pytest.importorskip so the embedding model is
not silently downloaded during normal CI runs.
```

---

# Phase 9 — RAID benchmark in CI

## Subagent mission

Add a recurring external benchmark gate that reports TPR at a strict false-positive budget.

## Files

Create:

```text
scripts/run_raid_benchmark.py
.github/workflows/raid.yml
```

Modify only if needed:

```text
cairn/evaluation/metrics.py
pyproject.toml
```

Data handling depends on RAID license. Do not commit benchmark data unless redistribution is permitted. If not permitted, commit a downloader plus SHA256 manifest.

## Required changes

Benchmark script accepts `--raid-csv`, `--config`, `--out`, `--fpr-ceiling 0.01`. Loads RAID samples, runs `CairnPipeline`, uses `run_eval`, serializes `eval_report_to_dict(include_predictions=False)`, adds headline TPR@FPR<=0.01, writes JSON report, prints headline metric.

GitHub Actions workflow: manual dispatch, weekly cron, release tags. Upload JSON report artifact. **Use a calibrated profile, not heuristic profile**, unless the workflow name explicitly says heuristic smoke.

## Tests / smoke checks

Run locally on a small subset first. Confirm script prints `RAID TPR@FPR<=0.01: <value>`. Confirm JSON artifact includes:

```json
{
  "headline": {
    "metric": "TPR@FPR<=0.01",
    "value": ...
  }
}
```

## Acceptance gate

```bash
python scripts/run_raid_benchmark.py \
  --raid-csv <raid_subset>.csv \
  --config <calibrated_profile>.yaml \
  --out /tmp/raid_report.json
```

Subagent returns:

```text
raid_source:
data_license_status:
sample_count:
profile_used:
headline_tpr_at_1pct_fpr:
artifact_path:
workflow_status:
```

## Dispatch prompt

```text
Implement Phase 9 of the CAIRN Honest Calibration Story. Add scripts/run_raid_benchmark.py to run
CAIRN against a RAID subset and report TPR@FPR<=0.01, using the existing evaluation runner. Add a
GitHub Actions workflow that runs on release tags, weekly cron, and manual dispatch, then uploads
the RAID JSON report artifact. Verify RAID data licensing before committing data; otherwise
implement download plus SHA256 manifest. Use a calibrated profile by default — not configs/default.yaml.
```

---

# Integration phase — Final audit pass

## Subagent mission

Verify that all phases compose cleanly and that the product tells the same truth everywhere: CLI, GUI, API, sidecar, reports, history, monitoring, and benchmark docs.

## Files

Likely modify:

```text
README.md
docs/calibration.md
docs/benchmarks/README.md
docs/configuration.md
docs/deployment.md
cairn/report.py
cairn/gui/static/index.html
cairn/api.py
cairn/sidecar.py
tests/test_report.py
tests/test_http_api.py
tests/test_rpc_sidecar.py
```

## Required checks

Every report should expose:

```text
fusion_mode
calibration_status
conformal_set
conformal_alpha
abstention_reason
coverage_guarantee_applies
detector_errors
warnings
```

Override reports should visibly caveat: "coverage guarantee does not apply to this override path".

Heuristic mode should be impossible to mistake for calibrated mode.

API `/health` should distinguish:

```text
heuristic
fusion-only
calibrated
calibrated-router-conditional
```

Docs should say which profile is recommended for: prose, prompt detection, short-form/OOD, batch similarity, heuristic screening only.

## Acceptance gate

```bash
pytest -x -q
```

Smoke call through each runtime: `cairn gui --profile <calibrated profile>`, `cairn serve --profile <calibrated profile>`, `cairn-sidecar`.

Subagent returns:

```text
runtime_surfaces_checked:
schema_fields_present:
docs_updated:
remaining_known_gaps:
pytest_result:
```

---

# Deferred backlog

## Neyman-Pearson threshold calibration

Create `scripts/calibrate_np_threshold.py` choosing the operating threshold that maximizes TPR under a specified FPR ceiling. Args: `--dataset`, `--config`, `--fpr-ceiling 0.01`, `--out-threshold-config`.

## PSI drift metric

Add Population Stability Index alongside KL divergence in monitoring. Trigger: `PSI > 0.2` = significant drift.

## Ground-truth behavioral fixtures

Create `tests/fixtures/ground_truth_50.jsonl` and a regression test asserting minimum behavior at controlled FPR.

## Post-hoc isotonic calibration on EBM

Consider exposing both `p_llm_raw` and `p_llm_calibrated`. Use calibrated score for decisioning and conformal; raw score for feature explanation continuity.

---

# Subagent return template

Every subagent should return this exact summary:

```text
Phase:
Branch:
Files changed:
Tests added:
Tests run:
Result:
Metrics before:
Metrics after:
Artifacts created:
Artifacts promoted:
Backward compatibility:
Risks / caveats:
Recommended next phase:
```

# Highest-risk corrections to the prior draft (carried forward)

- Phase 3 synthetic router test must ensure both labels exist in each router class. Otherwise the expected threshold comparison can fail for the wrong reason.
- The RAID workflow should use a calibrated profile, not `configs/default.yaml`, unless the intent is explicitly a heuristic smoke benchmark.
- Retrieval phase tests must be behind `pytest.importorskip` so embedding models are not silently downloaded during normal CI runs.
- The raw-text/history privacy interaction should be revisited during integration. Purging the privacy vault does not necessarily purge SQLite history text if history retention remains enabled.

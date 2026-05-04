# CAIRN Honest Calibration Story — Unified Implementation Plan

> **This plan supersedes** [`2026-05-02-honest-calibration.md`](2026-05-02-honest-calibration.md) and [`2026-05-02-honest-calibration-dispatch.md`](2026-05-02-honest-calibration-dispatch.md). It merges the dispatch plan's phase structure with the original plan's concrete TDD task content.
>
> **For agentic workers:** Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Each phase is sized for a single fresh subagent. Steps use `- [ ]` checkboxes for tracking.

## Goal

Make CAIRN's "calibrated probability" UX claim honest end-to-end: replace marginal split-conformal with class- and router-stratified Mondrian conformal, fix the empty-set / override holes in the decision engine, give the EBM honest missingness and confidence features, hold out by LLM family at training time, replace pickle with skops + metadata, replace pairwise word-shingle cross-similarity with a retrieval index, and wire RAID into CI as the first external benchmark.

## Operating rule

Each phase is handled by one fresh subagent. Each subagent gets a narrow mission, modifies only the listed files, writes tests first where practical, runs targeted tests, then runs `pytest -x -q` before handoff.

No subagent mutates `master` directly. Every phase lands as its own branch or PR. Any phase touching trained artifacts logs the dataset path, artifact path, metrics before/after, and whether the artifact was actually promoted.

## Phase 0 baseline (recorded 2026-05-02)

- Branch: `claude/pedantic-meninsky-32f39e` (worktree). Parent repo at `C:\Users\occam\My Projects\LLM Detector Reboot\`.
- Pytest baseline: **392 passed / 0 failed / 0 skipped / 36 warnings** in 102 s under `pytest -x -q`. Python 3.14.3, pytest 9.0.2.
- Installed: `interpret==0.7.8`, `scikit-learn==1.8.0`, `numpy==2.4.4`, `torch==2.11.0`, `transformers==5.5.0`, `sentence-transformers==5.3.0`.
- **NOT installed** (gates downstream phases): `skops` (Phase 7), `faiss-cpu` (Phase 8).
- **Profile config naming gotcha:** the recommended profile configs are dot-prefixed/hidden:
  - `configs/.profile_master_minus_arxiv.yaml` (prose)
  - `configs/.profile_prompts_only.yaml` (prompts)
  Use `ls -a configs/` to enumerate. Do not write `configs/master_minus_arxiv.yaml` in any dispatch prompt — that path does not exist.
- **`data/` and `models/` live in the parent repo, not in this worktree.** Phases 4, 5, 6, 9 must run from the parent repo OR pass absolute paths OR symlink. Configs reference relative paths so cwd matters. Canonical paths:
  - Production EBM: `<parent>/models/calib2000/profiles/master_minus_arxiv/ebm.pkl` (also `.../prompts_only/ebm.pkl`)
  - Production conformal: `<parent>/models/calib2000/profiles/master_minus_arxiv/conformal.json`
  - Production detector calibration: `<parent>/models/calib2000/profiles/master_minus_arxiv/detector_calibration.json`
  - Canonical training corpus: `<parent>/data/calib2000_v6_train.jsonl`
  - Feature cache: `<parent>/data/all_features_v6.jsonl`
  - OOD validation: `<parent>/data/ood_m4_test.jsonl`, `<parent>/data/ood_raid_test.jsonl`, `<parent>/data/ood_mage_test.jsonl`
- Current artifacts use plain `pickle` (`.pkl`). Phase 7 will need a one-time migration.

## Dependency graph

```text
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 5
                    └─► Phase 4 ──► Phase 6 ──► Phase 9
                    │         └────► Phase 7
                    │         └────► Phase 10  (also needs Phase 8)
                    └─────────────► Phase 8
```

**Phase 10** needs both Phase 4 (assembler `_missing`/`_confidence` pattern) and Phase 8 (the `cairn[retrieval]` extra that pulls in `sentence-transformers`). It is the only phase with two real parents in the graph.

**Recommended dispatch waves:**
- Wave 1: Phase 0, then Phase 1.
- Wave 2: Phase 2 and Phase 4 in parallel after Phase 1.
- Wave 3: Phase 3 after Phase 2. Phase 6 and Phase 7 after Phase 4. Phase 8 can also run after Phase 1.
- Wave 4: Phase 5 after Phases 3 and 4.
- Wave 5: Phase 9 (after Phases 4 and 6) and Phase 10 (after Phases 4 and 8) in parallel.

## Stop-here checkpoints

- After **Phase 1**: ship as `feat(decision): honest coverage accounting` — no retrain, ~50 LOC.
- After **Phase 3**: ship as `feat(conformal): Mondrian per-class + per-router thresholds`.
- After **Phase 4**: ship as `feat(ebm): missingness-aware fusion v7` — first retrain.
- After **Phase 6**: ship as `feat(eval): leave-one-LLM-out + per-source reporting`.
- After **Phase 9**: full audit priority list 1–7 closed.
- After **Phase 10**: ship as `feat(detectors): semantic_kinematics — embedding-trajectory macro-feature`. Survives paraphrase and burstiness-prompt attacks that defeat token-level features.

## Subagent return template

Every subagent returns this exact summary at handoff:

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

---

# Phase 0 — Pre-flight and repo baseline

## Subagent mission

Establish a clean baseline before any calibration, schema, model, or detector changes. **Read-only investigation. No code changes, no commits.**

## Files

No intended code changes.

## Implementation tasks

### Task 0.1: Confirm branch and pytest baseline

- [ ] **Step 1: Confirm the active branch is not `master`**

Run: `git branch --show-current`
Expected: any branch name other than `master`.

- [ ] **Step 2: Run pytest under `-x` and capture summary**

Run: `pytest -x -q`
Record: passed, failed, skipped, warnings, runtime, first-failure id (if any).

- [ ] **Step 3: If `-x` bailed, get full counts**

Run: `pytest --tb=no -q`
Record full pass/fail/skip count.

### Task 0.2: Confirm required files exist

- [ ] **Step 1: Glob each path and report missing**

Files to check:
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

### Task 0.3: Locate canonical artifacts and dataset

- [ ] **Step 1: Inspect production profile configs**

Run: `ls -a configs/ | grep profile`
Read the profile configs to extract `fusion.model_path`, `fusion.conformal_path`, `fusion.detector_calibration_path`.

- [ ] **Step 2: Verify canonical training corpus exists in parent repo**

Run: `ls -lh ../../../data/calib2000_v6_train.jsonl ../../../data/all_features_v6.jsonl ../../../data/ood_*_test.jsonl 2>&1 | head -10`
(Adjust relative path depending on worktree depth.)

### Task 0.4: Record Python version and installed extras

- [ ] **Step 1: Capture interpreter info**

Run: `python --version && python -c "import sys; print(sys.executable)"`

- [ ] **Step 2: Check key extras**

Run: `python -m pip list 2>/dev/null | grep -iE "interpret|skops|sentence-transformers|faiss|numpy|scikit|torch|transformers"`

## Acceptance gate

Subagent returns the populated baseline block. No merge needed unless documentation is updated.

```text
branch:
python_version:
python_interpreter:
pytest_baseline:
  passed:
  failed:
  skipped:
  warnings:
  runtime_seconds:
required_files_present: [yes/no per file]
canonical_training_dataset:
current_ebm_artifact:
current_conformal_artifact:
current_detector_calibration_artifact:
ood_validation_sets:
installed_extras: [yes/no per extra]
known_environment_gaps:
```

## Dispatch prompt

```text
You are dispatched as Phase 0 of the CAIRN Honest Calibration Story (Unified Plan). This is a
pre-flight baseline phase: read-only investigation, no code changes, no commits, no installs
unless required for pytest to run at all.

Run pytest -x -q and capture passed/failed/skipped/warnings/runtime. If -x bails, also run
pytest --tb=no -q. Confirm the 12 required files exist (list in the plan). Locate the canonical
training corpus, currently shipped EBM/conformal/detector-calibration artifacts, OOD validation
sets. Record Python version, interpreter path, and which of these are installed: interpret,
skops, sentence-transformers, faiss-cpu, numpy, scikit-learn, torch, transformers. Return the
populated baseline block from the plan and stop.
```

---

# Phase 1 — Decision-engine coverage accounting

## Subagent mission

Make the decision engine honest when conformal coverage does not apply. Highest-priority blast-radius fix because it changes user-facing truth without requiring retraining.

## Why this phase comes first

Later conformal improvements depend on the decision engine handling all conformal-set states correctly. The current logic correctly handles non-singleton sets but treats an empty set as "fall through to threshold," which is wrong: an empty conformal set means "neither hypothesis conforms to α-coverage" — the strongest OOD signal we can produce.

## Files

**Modify:** `cairn/decision.py`, `cairn/contracts.py`, `configs/default.yaml`, existing `tests/test_decision.py`, `tests/test_decision_abstention.py`
**Create:** `tests/test_decision_coverage.py`

## Required changes (summary)

- Add `coverage_guarantee_applies: bool = True` to `Determination` and `FusionResult` (append, never reorder per CLAUDE.md).
- Add `conformal_empty` to `Determination.abstention_reason` literal.
- Treat `fusion.conformal_set == []` as `UNCERTAIN` with `abstention_reason="conformal_empty"` and `coverage_guarantee_applies=False`.
- Set `coverage_guarantee_applies=False` on CRITICAL preamble and MIXED authorship overrides.
- Move the hardcoded mixed-boundary `0.60` threshold into `decision.mixed.min_mixed_probability` (the config knob already exists; `decision.py:28` ignores it).

## Implementation tasks

### Task 1.1: Add `coverage_guarantee_applies` to contracts

- [ ] **Step 1: Write the failing test** in `tests/test_contracts.py`:

```python
def test_determination_default_coverage_flag():
    from cairn.contracts import Determination
    d = Determination(
        label="GREEN", p_llm=0.1, confidence_interval=(0.0, 0.2),
        prediction_set=["GREEN"], reason="ok", top_features=[],
        override_applied=False, detectors_run=[], cascade_phases=[], mixed_report=None,
    )
    assert d.coverage_guarantee_applies is True


def test_fusion_result_default_coverage_flag():
    from cairn.contracts import FusionResult
    fr = FusionResult(p_llm=0.5, confidence_interval=(0.4, 0.6), prediction_set=["YELLOW"],
                     feature_contributions={}, top_contributors=[])
    assert fr.coverage_guarantee_applies is True
```

- [ ] **Step 2: Run, verify fails** — `pytest tests/test_contracts.py::test_determination_default_coverage_flag -xvs`
- [ ] **Step 3: Append the field to both dataclasses** in `cairn/contracts.py` (append-only, never reorder):

```python
@dataclass
class FusionResult:
    # ... existing fields unchanged ...
    fusion_mode: str = "naive"
    coverage_guarantee_applies: bool = True


@dataclass
class Determination:
    # ... existing fields unchanged ...
    calibration_status: str = "heuristic"
    coverage_guarantee_applies: bool = True
```

Extend the abstention literal:

```python
abstention_reason: Literal["conformal_set", "prediction_set", "mixed_authorship", "conformal_empty"] | None = None
```

- [ ] **Step 4: Run, verify passes** — `pytest tests/test_contracts.py -xvs`
- [ ] **Step 5: Commit**

```bash
git add cairn/contracts.py tests/test_contracts.py
git commit -m "feat(contracts): add coverage_guarantee_applies + conformal_empty abstention reason"
```

### Task 1.2: Empty conformal set is the strongest abstention signal

- [ ] **Step 1: Write the failing test** in `tests/test_decision_coverage.py`:

```python
import pytest
from cairn.contracts import FusionResult, LayerResult
from cairn.decision import DecisionEngine


def _engine():
    cfg = {"decision": {"red_threshold": 0.75, "amber_threshold": 0.50,
                        "yellow_threshold": 0.25,
                        "abstention": {"enabled": True, "max_prediction_set_size": 3},
                        "mixed": {"enabled": True, "min_mixed_probability": 0.60}}}
    return DecisionEngine(cfg)


def _fr(p, conformal_set):
    return FusionResult(
        p_llm=p, confidence_interval=(max(0, p - 0.1), min(1, p + 0.1)),
        prediction_set=["YELLOW"], feature_contributions={}, top_contributors=[],
        conformal_set=conformal_set, conformal_alpha=0.1, fusion_mode="ebm",
    )


def test_empty_conformal_set_yields_uncertain_with_reason():
    eng = _engine()
    det = eng.decide(_fr(0.42, []), layer_results=[])
    assert det.label == "UNCERTAIN"
    assert det.abstention_reason == "conformal_empty"
    assert det.coverage_guarantee_applies is False
    assert ("out-of-distribution" in det.reason.lower()) or ("neither" in det.reason.lower())


def test_singleton_conformal_set_uses_that_label():
    eng = _engine()
    det = eng.decide(_fr(0.82, ["LLM"]), layer_results=[])
    assert det.label == "RED"
    assert det.coverage_guarantee_applies is True


def test_two_class_conformal_set_yields_uncertain():
    eng = _engine()
    det = eng.decide(_fr(0.5, ["human", "LLM"]), layer_results=[])
    assert det.label == "UNCERTAIN"
    assert det.abstention_reason == "conformal_set"
    assert det.coverage_guarantee_applies is True
```

- [ ] **Step 2: Run, verify fails** — `pytest tests/test_decision_coverage.py -xvs`
- [ ] **Step 3: Replace the abstention block in `DecisionEngine.decide`**:

```python
        abstention_reason = None
        coverage_applies = True
        if self._abstain and fusion.conformal_set is not None:
            if len(fusion.conformal_set) == 0:
                label = "UNCERTAIN"
                reason = (
                    "Conformal set is empty — neither hypothesis is conformal at "
                    f"α={fusion.conformal_alpha or 0:.2f}. Input is out-of-distribution "
                    "relative to calibration; route to human review."
                )
                abstention_reason = "conformal_empty"
                coverage_applies = False
            elif len(fusion.conformal_set) >= 2:
                label = "UNCERTAIN"
                reason = (
                    f"Conformal set is non-singleton ({', '.join(fusion.conformal_set)}); "
                    f"detector cannot rule out either class at α={fusion.conformal_alpha or 0:.2f}."
                )
                abstention_reason = "conformal_set"
            else:
                label = self._p_llm_to_label(fusion.p_llm)
                reason = self._generate_reason(label, fusion)
        elif self._abstain and fusion.conformal_set is None and len(fusion.prediction_set) >= self._max_pred_set:
            label = "UNCERTAIN"
            reason = f"Prediction set spans {len(fusion.prediction_set)} severity bands ({', '.join(fusion.prediction_set)}). Human review recommended."
            abstention_reason = "prediction_set"
        else:
            label = self._p_llm_to_label(fusion.p_llm)
            reason = self._generate_reason(label, fusion)
```

Update the trailing `return Determination(...)` to pass `coverage_guarantee_applies=coverage_applies`.

- [ ] **Step 4: Run, verify passes** — `pytest tests/test_decision_coverage.py -xvs`
- [ ] **Step 5: Commit**

```bash
git add cairn/decision.py tests/test_decision_coverage.py
git commit -m "fix(decision): empty conformal set → UNCERTAIN with coverage flag off"
```

### Task 1.3: Overrides mark coverage as not applying

- [ ] **Step 1: Write the failing test**:

```python
def test_preamble_critical_override_marks_coverage_off():
    eng = _engine()
    crit = LayerResult(
        layer_id="preamble", domain="universal", raw_score=1.0, p_llm=0.95,
        confidence=0.95, signals={"severity": "CRITICAL"}, determination="RED",
        attacker_tiers=[], compute_cost="trivial", min_text_length=0,
    )
    det = eng.decide(_fr(0.95, ["LLM"]), layer_results=[crit])
    assert det.label == "RED"
    assert det.override_applied is True
    assert det.coverage_guarantee_applies is False


def test_mixed_override_marks_coverage_off():
    eng = _engine()
    mb = LayerResult(
        layer_id="mixed_boundary", domain="universal", raw_score=0.7, p_llm=0.7,
        confidence=0.7, signals={"mixed_probability": 0.85, "n_boundaries": 3,
                                 "segment_determinations": []},
        determination="AMBER", attacker_tiers=[], compute_cost="moderate", min_text_length=0,
    )
    det = eng.decide(_fr(0.5, ["human", "LLM"]), layer_results=[mb])
    assert det.label == "MIXED"
    assert det.coverage_guarantee_applies is False
```

- [ ] **Step 2: Run, verify fails** — `pytest tests/test_decision_coverage.py -xvs`
- [ ] **Step 3: Add `coverage_guarantee_applies=False` to override returns** — both the preamble CRITICAL `return Determination(...)` and the MIXED `return Determination(...)` constructors.
- [ ] **Step 4: Run, verify passes** — `pytest tests/test_decision_coverage.py -xvs`
- [ ] **Step 5: Commit**

```bash
git add cairn/decision.py tests/test_decision_coverage.py
git commit -m "feat(decision): mark coverage_guarantee_applies=False on overrides"
```

### Task 1.4: MIXED threshold sourced from config

- [ ] **Step 1: Write the failing test**:

```python
def test_mixed_threshold_respects_config_lower():
    cfg = {"decision": {"red_threshold": 0.75, "amber_threshold": 0.50,
                        "yellow_threshold": 0.25,
                        "abstention": {"enabled": True, "max_prediction_set_size": 3},
                        "mixed": {"enabled": True, "min_mixed_probability": 0.30}}}
    eng = DecisionEngine(cfg)
    mb = LayerResult(
        layer_id="mixed_boundary", domain="universal", raw_score=0.4, p_llm=0.4,
        confidence=0.5, signals={"mixed_probability": 0.40, "n_boundaries": 2,
                                 "segment_determinations": []},
        determination="AMBER", attacker_tiers=[], compute_cost="moderate", min_text_length=0,
    )
    det = eng.decide(_fr(0.5, ["LLM"]), layer_results=[mb])
    assert det.label == "MIXED"


def test_mixed_threshold_respects_config_disabled():
    cfg = {"decision": {"red_threshold": 0.75, "amber_threshold": 0.50,
                        "yellow_threshold": 0.25,
                        "abstention": {"enabled": True, "max_prediction_set_size": 3},
                        "mixed": {"enabled": False, "min_mixed_probability": 0.60}}}
    eng = DecisionEngine(cfg)
    mb = LayerResult(
        layer_id="mixed_boundary", domain="universal", raw_score=0.7, p_llm=0.7,
        confidence=0.7, signals={"mixed_probability": 0.85, "n_boundaries": 3,
                                 "segment_determinations": []},
        determination="AMBER", attacker_tiers=[], compute_cost="moderate", min_text_length=0,
    )
    det = eng.decide(_fr(0.82, ["LLM"]), layer_results=[mb])
    assert det.label == "RED"
```

- [ ] **Step 2: Run, verify fails** — FAIL because `decision.py:28` hardcodes `0.60`.
- [ ] **Step 3: Read the config and respect both knobs** — in `DecisionEngine.__init__`:

```python
        mixed = decision.get("mixed", {})
        self._mixed_enabled = bool(mixed.get("enabled", True))
        self._mixed_threshold = float(mixed.get("min_mixed_probability", 0.60))
```

Update the `mixed_boundary` scan in `decide`:

```python
            if result.layer_id == "mixed_boundary":
                mixed_prob = float(result.signals.get("mixed_probability", 0.0) or 0.0)
                if (
                    self._mixed_enabled
                    and mixed_prob >= self._mixed_threshold
                    and result.determination != "SKIP"
                ):
                    mixed_report = {
                        "mixed_probability": mixed_prob,
                        "n_boundaries": result.signals.get("n_boundaries", 0),
                        "segment_determinations": result.signals.get("segment_determinations", []),
                    }
```

- [ ] **Step 4: Run, verify passes** — `pytest tests/test_decision_coverage.py -xvs`
- [ ] **Step 5: Commit**

```bash
git add cairn/decision.py tests/test_decision_coverage.py
git commit -m "fix(decision): MIXED threshold + enable flag honour config"
```

### Task 1.5: Run full pytest

Run: `pytest -x -q`
Expected: ≥392 pass.

## Acceptance gate

```bash
pytest tests/test_decision.py tests/test_decision_abstention.py tests/test_decision_coverage.py -v
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
Implement Phase 1 of the CAIRN Honest Calibration Story (Unified Plan). Make cairn/decision.py
honest about conformal coverage. Tasks (each TDD-style: failing test → implement → re-run → commit):

1.1 Add coverage_guarantee_applies (default True) to FusionResult and Determination in
    cairn/contracts.py. Append-only — do not reorder fields. Extend the abstention_reason literal
    with "conformal_empty".
1.2 In cairn/decision.py, treat fusion.conformal_set == [] as UNCERTAIN with
    abstention_reason="conformal_empty" and coverage_guarantee_applies=False. The reason string
    must mention "out-of-distribution" or "neither".
1.3 Set coverage_guarantee_applies=False on the CRITICAL-preamble override return and the MIXED
    override return.
1.4 Move the hardcoded mixed-boundary 0.60 threshold into decision.mixed.min_mixed_probability
    (the config knob already exists in configs/default.yaml). Also honour decision.mixed.enabled.

Write tests first in tests/test_decision_coverage.py (full code in the plan). Run targeted tests
and pytest -x -q before handoff. Return the Phase 1 acceptance block.
```

---

# Phase 2 — Class-conditional Mondrian conformal

## Subagent mission

Upgrade conformal prediction from one global marginal threshold to one threshold per class (human, LLM).

## Why it matters

A global threshold can preserve marginal coverage while hiding class-specific failure. False positives and false negatives are not symmetric in this domain — class-conditional Mondrian is the right minimum standard (Vovk 2003; Romano, Sesia & Candès NeurIPS 2020).

## Files

**Modify:** `cairn/fusion/conformal.py`, `scripts/calibrate_conformal.py`
**Create:** `tests/test_fusion_conformal_mondrian.py`

## Required changes (summary)

- `ConformalWrapper.calibrate(scores, labels)` computes `_threshold_per_class[0]` and `_threshold_per_class[1]`.
- New API: `threshold_for(cls)`, property `threshold_per_class`.
- `predict_set(p_llm)` consults class-specific thresholds when present.
- Bump artifact schema to `binary-mondrian-conformal-v2`. v1 still loads (lifted to `{0: t, 1: t}`).
- `scripts/calibrate_conformal.py` prints both thresholds.

## Implementation tasks

### Task 2.1: Per-class thresholds in `calibrate`

- [ ] **Step 1: Write the failing test** in `tests/test_fusion_conformal_mondrian.py`:

```python
import math
import json
import numpy as np
import pytest
from pathlib import Path

from cairn.fusion.conformal import ConformalWrapper


def _mk_scores(n_human=200, n_llm=200, seed=0):
    rng = np.random.default_rng(seed)
    human = rng.beta(2, 8, n_human)
    llm = rng.beta(8, 2, n_llm)
    scores = np.concatenate([human, llm])
    labels = np.concatenate([np.zeros(n_human, dtype=int), np.ones(n_llm, dtype=int)])
    return scores, labels


def test_calibrate_populates_per_class_thresholds():
    w = ConformalWrapper(alpha=0.1)
    scores, labels = _mk_scores()
    w.calibrate(scores, labels)
    assert w.threshold_per_class is not None
    assert set(w.threshold_per_class.keys()) == {0, 1}


def test_per_class_calibration_rejects_class_with_too_few_points():
    w = ConformalWrapper(alpha=0.1)
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.beta(2, 8, 200), rng.beta(8, 2, 5)])
    labels = np.concatenate([np.zeros(200, dtype=int), np.ones(5, dtype=int)])
    with pytest.raises(ValueError, match="too few class-1 calibration points"):
        w.calibrate(scores, labels)
```

- [ ] **Step 2: Run, verify fails** — `pytest tests/test_fusion_conformal_mondrian.py -xvs`
- [ ] **Step 3: Implement** — replace `__init__` and `calibrate` in `cairn/fusion/conformal.py`:

```python
class ConformalWrapper:
    """Honest split-conformal for the binary LLM/human decision (Mondrian-capable)."""

    _MIN_PER_CLASS = 10

    def __init__(self, alpha: float = 0.05):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._alpha = alpha
        self._threshold: float | None = None
        self._threshold_per_class: dict[int, float] | None = None
        self._n_calibration: int = 0
        self._n_per_class: dict[int, int] | None = None

    @property
    def threshold_per_class(self) -> dict[int, float] | None:
        return self._threshold_per_class

    def threshold_for(self, cls: int) -> float | None:
        if self._threshold_per_class is None:
            return self._threshold
        return self._threshold_per_class.get(int(cls))

    def _quantile_for(self, nonconformity: np.ndarray) -> float:
        n = len(nonconformity)
        q_level = math.ceil((n + 1) * (1.0 - self._alpha)) / n
        q_level = min(q_level, 1.0)
        return float(np.quantile(nonconformity, q_level, method="higher"))

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if scores.shape != labels.shape:
            raise ValueError("scores and labels must have the same shape")
        if scores.ndim != 1:
            raise ValueError("scores and labels must be 1-D arrays")
        n = len(scores)
        if n < self._MIN_PER_CLASS:
            raise ValueError(f"too few calibration points ({n})")
        nonconformity = np.where(labels == 1, 1.0 - scores, scores)
        self._threshold = self._quantile_for(nonconformity)
        self._n_calibration = int(n)
        per_class: dict[int, float] = {}
        n_per_class: dict[int, int] = {}
        for cls in (0, 1):
            mask = labels == cls
            n_cls = int(mask.sum())
            if n_cls < self._MIN_PER_CLASS:
                raise ValueError(f"too few class-{cls} calibration points ({n_cls})")
            nc_cls = (1.0 - scores[mask]) if cls == 1 else scores[mask]
            per_class[cls] = self._quantile_for(nc_cls)
            n_per_class[cls] = n_cls
        self._threshold_per_class = per_class
        self._n_per_class = n_per_class
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/conformal.py tests/test_fusion_conformal_mondrian.py
git commit -m "feat(conformal): per-class Mondrian thresholds with too-few-points guard"
```

### Task 2.2: `predict_set` consumes per-class thresholds

- [ ] **Step 1: Write the failing test**:

```python
def test_predict_set_uses_per_class_thresholds():
    w = ConformalWrapper(alpha=0.1)
    scores, labels = _mk_scores()
    w.calibrate(scores, labels)
    assert w.threshold_per_class[0] != pytest.approx(w.threshold_per_class[1], abs=1e-6)
    assert w.predict_set(0.99) == ["LLM"]
    assert w.predict_set(0.01) == ["human"]
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Update `predict_set`**:

```python
    def predict_set(self, p_llm: float) -> list[str]:
        p_llm = max(0.0, min(1.0, float(p_llm)))
        if self._threshold is None:
            return list(_LABELS)
        if self._threshold_per_class is not None:
            out: list[str] = []
            nc = {0: p_llm, 1: 1.0 - p_llm}
            idx_to_label = {0: "human", 1: "LLM"}
            for cls in (0, 1):
                if nc[cls] <= self._threshold_per_class[cls]:
                    out.append(idx_to_label[cls])
            return out
        nc_named = {"human": p_llm, "LLM": 1.0 - p_llm}
        return [c for c in _LABELS if nc_named[c] <= self._threshold]
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/conformal.py tests/test_fusion_conformal_mondrian.py
git commit -m "feat(conformal): predict_set consumes per-class thresholds"
```

### Task 2.3: v2 schema save/load with v1 compatibility

- [ ] **Step 1: Write the failing test**:

```python
def test_save_and_load_round_trips_mondrian(tmp_path):
    w = ConformalWrapper(alpha=0.1)
    scores, labels = _mk_scores()
    w.calibrate(scores, labels)
    p = tmp_path / "cal.json"
    w.save(p)
    w2 = ConformalWrapper(alpha=0.5)
    w2.load(p)
    assert w2.threshold_per_class is not None
    assert w2.predict_set(0.5) == w.predict_set(0.5)


def test_load_v1_artifact_lifts_threshold_to_both_classes(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(
        '{"alpha": 0.05, "threshold": 0.42, "n_calibration": 100, '
        '"schema": "binary-split-conformal-v1"}'
    )
    w = ConformalWrapper()
    w.load(p)
    assert w.calibrated
    assert w.threshold_per_class == {0: 0.42, 1: 0.42}
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Implement v2 save/load** — replace `save` and `load`:

```python
    _SCHEMA_V1 = "binary-split-conformal-v1"
    _SCHEMA_V2 = "binary-mondrian-conformal-v2"

    def save(self, path: Path) -> None:
        if self._threshold is None:
            raise RuntimeError("ConformalWrapper.save: not calibrated")
        payload: dict = {
            "alpha": self._alpha,
            "threshold": self._threshold,
            "n_calibration": self._n_calibration,
            "schema": self._SCHEMA_V2,
        }
        if self._threshold_per_class is not None:
            payload["threshold_per_class"] = {
                str(k): v for k, v in self._threshold_per_class.items()
            }
        if self._n_per_class is not None:
            payload["n_per_class"] = {str(k): v for k, v in self._n_per_class.items()}
        Path(path).write_text(json.dumps(payload))

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text())
        schema = data.get("schema")
        if schema not in (None, self._SCHEMA_V1, self._SCHEMA_V2):
            raise ValueError(
                f"ConformalWrapper.load: unknown schema {schema!r}"
            )
        self._alpha = float(data["alpha"])
        self._threshold = float(data["threshold"])
        self._n_calibration = int(data.get("n_calibration", 0))
        tpc = data.get("threshold_per_class")
        if tpc:
            self._threshold_per_class = {int(k): float(v) for k, v in tpc.items()}
        elif schema == self._SCHEMA_V1 or schema is None:
            self._threshold_per_class = {0: self._threshold, 1: self._threshold}
        npc = data.get("n_per_class")
        self._n_per_class = (
            {int(k): int(v) for k, v in npc.items()} if npc else None
        )
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/conformal.py tests/test_fusion_conformal_mondrian.py
git commit -m "feat(conformal): persist v2 schema; lift v1 threshold to both classes on load"
```

### Task 2.4: `calibrate_conformal.py` prints per-class thresholds

- [ ] **Step 1: Update the script** — after `wrapper.save(args.out)`:

```python
    print(
        f"calibrated on {len(scores)} samples (alpha={args.alpha}); "
        f"per-class={wrapper.threshold_per_class}; "
        f"wrote {args.out}"
    )
```

- [ ] **Step 2: Smoke run** on a synthetic 40-row JSONL:

```bash
python scripts/calibrate_conformal.py --dataset /tmp/cal.jsonl --config configs/default.yaml --alpha 0.2 --out /tmp/cal.json
cat /tmp/cal.json | python -c "import json,sys; d=json.load(sys.stdin); print(d['schema'], d.get('threshold_per_class'))"
```

Expected: `binary-mondrian-conformal-v2 {'0': ..., '1': ...}`.

- [ ] **Step 3: Commit**

```bash
git add scripts/calibrate_conformal.py
git commit -m "feat(scripts): calibrate_conformal prints per-class thresholds + v2 schema"
```

### Task 2.5: Run full pytest

Run: `pytest -x -q`
Expected: ≥392 pass.

## Acceptance gate

```bash
pytest tests/test_fusion_conformal_mondrian.py -v
pytest -x -q
```

Subagent returns:

```text
new_schema: binary-mondrian-conformal-v2
backward_compatibility: v1 artifacts load (lifted to per-class)
thresholds_printed: per-class shown after calibration
tests_added:
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 2 of the CAIRN Honest Calibration Story (Unified Plan). Upgrade
cairn/fusion/conformal.py to class-conditional Mondrian conformal:

2.1 Compute threshold_per_class[0] (human) and threshold_per_class[1] (LLM); add threshold_for(cls)
    method and threshold_per_class property; raise on < _MIN_PER_CLASS samples in either class.
2.2 Update predict_set(p_llm) to consult per-class thresholds when present, falling back to the
    marginal threshold otherwise.
2.3 Bump save schema to binary-mondrian-conformal-v2; on load, accept v1 artifacts and lift the
    global threshold to threshold_per_class = {0: t, 1: t}.
2.4 Update scripts/calibrate_conformal.py to print both per-class thresholds.

Write tests first in tests/test_fusion_conformal_mondrian.py (full code in the plan). Run pytest
-x -q before handoff. Return the Phase 2 acceptance block.
```

---

# Phase 3 — Router-conditional Mondrian conformal

## Subagent mission

Extend conformal calibration from class-only to `(router_domain, class)` thresholds, with safe fallback to class-only when a router/class cell has too little data.

## Why it matters

Prompt vs. prose vs. mixed vs. insufficient inputs have different score distributions because different detector ensembles fire. A single LLM threshold for all domains is blunt. CAIRN already has a router; use it.

## Files

**Modify:** `cairn/fusion/conformal.py`, `cairn/fusion/ebm.py`, `cairn/pipeline.py`, `scripts/calibrate_conformal.py`
**Test:** `tests/test_fusion_conformal_mondrian.py`, `tests/test_fusion.py`, `tests/test_pipeline.py`

## Required changes (summary)

- `calibrate(scores, labels, router_classes=None)` — optional argument; build `(router_class, class)` thresholds when present.
- `_MIN_PER_CELL = 30`. Below this, the cell falls back to class-only.
- `threshold_for(cls, router_class=None)`, `predict_set(p_llm, router_class=None)`.
- Schema bump to `binary-split-conformal-v3`. v1 and v2 still load.
- Plumb `RouterDecision.domain` through `pipeline → EBMFusion.fuse(...) → _maybe_conformal(...) → predict_set(...)`.
- `scripts/calibrate_conformal.py` calls `pipeline.analyze_detailed(...)` to collect router domain.

## Implementation tasks

### Task 3.1: Per-router thresholds in `calibrate`

- [ ] **Step 1: Write the failing test**:

```python
def test_calibrate_with_router_classes_populates_per_router_thresholds():
    w = ConformalWrapper(alpha=0.1)
    rng = np.random.default_rng(0)
    n = 200
    scores = np.concatenate([
        rng.beta(2, 8, n), rng.beta(8, 2, n),
        rng.beta(2, 6, n), rng.beta(6, 2, n),
    ])
    labels = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)]).astype(int)
    routers = np.array(["prose"]*(2*n) + ["prompt"]*(2*n))
    w.calibrate(scores, labels, router_classes=routers)
    assert w.threshold_per_router is not None
    assert set(w.threshold_per_router.keys()) == {"prose", "prompt"}
    for rc in ("prose", "prompt"):
        assert set(w.threshold_per_router[rc].keys()) == {0, 1}


def test_underpowered_router_cell_falls_back_to_class_only():
    w = ConformalWrapper(alpha=0.1)
    rng = np.random.default_rng(0)
    n = 200
    scores = np.concatenate([rng.beta(2, 8, n), rng.beta(8, 2, n), rng.beta(2, 8, 10)])
    labels = np.concatenate([np.zeros(n), np.ones(n), np.zeros(10)]).astype(int)
    routers = np.array(["prose"]*(2*n) + ["mixed"]*10)
    w.calibrate(scores, labels, router_classes=routers)
    assert "mixed" not in (w.threshold_per_router or {})
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Extend `calibrate`** — in `cairn/fusion/conformal.py`, add to `__init__`:

```python
        self._threshold_per_router: dict[str, dict[int, float]] | None = None
        self._n_per_router: dict[str, int] | None = None
```

Add class constant and property:

```python
    _MIN_PER_CELL = 30

    @property
    def threshold_per_router(self) -> dict[str, dict[int, float]] | None:
        return self._threshold_per_router
```

Extend the signature and tail of `calibrate`:

```python
    def calibrate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        router_classes: np.ndarray | None = None,
    ) -> None:
        # ... existing class-only body unchanged ...
        if router_classes is not None:
            router_classes = np.asarray(router_classes)
            if router_classes.shape != labels.shape:
                raise ValueError("router_classes must match labels shape")
            per_router: dict[str, dict[int, float]] = {}
            n_per_router: dict[str, int] = {}
            for rc in np.unique(router_classes):
                rc_mask = router_classes == rc
                rc_total = int(rc_mask.sum())
                if rc_total < self._MIN_PER_CELL:
                    continue
                rc_class_thresholds: dict[int, float] = {}
                ok = True
                for cls in (0, 1):
                    inner = rc_mask & (labels == cls)
                    if int(inner.sum()) < self._MIN_PER_CLASS:
                        ok = False
                        break
                    nc_inner = (1.0 - scores[inner]) if cls == 1 else scores[inner]
                    rc_class_thresholds[cls] = self._quantile_for(nc_inner)
                if ok:
                    per_router[str(rc)] = rc_class_thresholds
                    n_per_router[str(rc)] = rc_total
            if per_router:
                self._threshold_per_router = per_router
                self._n_per_router = n_per_router
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/conformal.py tests/test_fusion_conformal_mondrian.py
git commit -m "feat(conformal): per-(router_class, class) Mondrian thresholds with cell fallback"
```

### Task 3.2: `predict_set` and `threshold_for` accept `router_class`

- [ ] **Step 1: Write the failing test**:

```python
def test_predict_set_with_router_class_uses_per_router_thresholds():
    w = ConformalWrapper(alpha=0.1)
    rng = np.random.default_rng(0)
    n = 200
    scores = np.concatenate([
        rng.beta(2, 8, n), rng.beta(8, 2, n),
        rng.beta(2, 6, n), rng.beta(6, 2, n),
    ])
    labels = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)]).astype(int)
    routers = np.array(["prose"]*(2*n) + ["prompt"]*(2*n))
    w.calibrate(scores, labels, router_classes=routers)
    s_prose = w.predict_set(0.5, router_class="prose")
    s_prompt = w.predict_set(0.5, router_class="prompt")
    assert isinstance(s_prose, list) and isinstance(s_prompt, list)
    s_unknown = w.predict_set(0.5, router_class="atlantis")
    assert isinstance(s_unknown, list)
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Update `threshold_for` and `predict_set`**:

```python
    def threshold_for(self, cls: int, router_class: str | None = None) -> float | None:
        if router_class is not None and self._threshold_per_router is not None:
            inner = self._threshold_per_router.get(str(router_class))
            if inner is not None:
                return inner.get(int(cls))
        if self._threshold_per_class is not None:
            return self._threshold_per_class.get(int(cls))
        return self._threshold

    def predict_set(self, p_llm: float, router_class: str | None = None) -> list[str]:
        p_llm = max(0.0, min(1.0, float(p_llm)))
        if self._threshold is None:
            return list(_LABELS)
        per_class: dict[int, float] | None = None
        if router_class is not None and self._threshold_per_router is not None:
            per_class = self._threshold_per_router.get(str(router_class))
        if per_class is None:
            per_class = self._threshold_per_class
        if per_class is not None:
            out: list[str] = []
            nc = {0: p_llm, 1: 1.0 - p_llm}
            idx_to_label = {0: "human", 1: "LLM"}
            for cls in (0, 1):
                if nc[cls] <= per_class[cls]:
                    out.append(idx_to_label[cls])
            return out
        nc_named = {"human": p_llm, "LLM": 1.0 - p_llm}
        return [c for c in _LABELS if nc_named[c] <= self._threshold]
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/conformal.py tests/test_fusion_conformal_mondrian.py
git commit -m "feat(conformal): predict_set + threshold_for accept router_class"
```

### Task 3.3: v3 schema save/load

- [ ] **Step 1: Write the failing test**:

```python
def test_v3_save_load_round_trips_router_thresholds(tmp_path):
    w = ConformalWrapper(alpha=0.1)
    rng = np.random.default_rng(0)
    n = 200
    scores = np.concatenate([
        rng.beta(2, 8, n), rng.beta(8, 2, n),
        rng.beta(2, 6, n), rng.beta(6, 2, n),
    ])
    labels = np.concatenate([np.zeros(n), np.ones(n), np.zeros(n), np.ones(n)]).astype(int)
    routers = np.array(["prose"]*(2*n) + ["prompt"]*(2*n))
    w.calibrate(scores, labels, router_classes=routers)
    p = tmp_path / "cal.json"
    w.save(p)
    payload = json.loads(p.read_text())
    assert payload["schema"] == "binary-split-conformal-v3"
    w2 = ConformalWrapper()
    w2.load(p)
    assert w2.threshold_per_router is not None
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Implement** — add to `ConformalWrapper`:

```python
    _SCHEMA_V3 = "binary-split-conformal-v3"
```

In `save`:

```python
        if self._threshold_per_router is not None:
            payload["threshold_per_router"] = {
                rc: {str(k): v for k, v in inner.items()}
                for rc, inner in self._threshold_per_router.items()
            }
            payload["n_per_router"] = dict(self._n_per_router or {})
            payload["schema"] = self._SCHEMA_V3
```

In `load`, accept v3 and extract:

```python
        if schema not in (None, self._SCHEMA_V1, self._SCHEMA_V2, self._SCHEMA_V3):
            raise ValueError(f"ConformalWrapper.load: unknown schema {schema!r}")
        # ... existing extraction ...
        tpr = data.get("threshold_per_router")
        if tpr:
            self._threshold_per_router = {
                rc: {int(k): float(v) for k, v in inner.items()}
                for rc, inner in tpr.items()
            }
        self._n_per_router = data.get("n_per_router") or None
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/conformal.py tests/test_fusion_conformal_mondrian.py
git commit -m "feat(conformal): v3 schema persists per-router thresholds"
```

### Task 3.4: Plumb router class from pipeline → fusion → conformal

- [ ] **Step 1: Write the failing test** in `tests/test_fusion.py`:

```python
def test_ebm_fuse_passes_domain_to_conformal():
    from cairn.fusion.ebm import EBMFusion
    from cairn.contracts import LayerResult

    captured: dict = {}

    class _FakeConformal:
        calibrated = True
        alpha = 0.1
        def predict_set(self, p, router_class=None):
            captured["router_class"] = router_class
            return ["LLM"]

    f = EBMFusion(model=None, conformal=_FakeConformal())
    fr = f.fuse(
        [LayerResult(layer_id="preamble", domain="universal", raw_score=0.9, p_llm=0.9,
                     confidence=0.8, signals={}, determination="RED", attacker_tiers=[],
                     compute_cost="trivial", min_text_length=0)],
        word_count=200, domain="prose",
    )
    assert captured["router_class"] == "prose"
    assert fr.conformal_set == ["LLM"]
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Update `_maybe_conformal` and both fusion paths** in `cairn/fusion/ebm.py`:

```python
    def _maybe_conformal(
        self, p_llm: float, router_class: str | None = None
    ) -> tuple[list[str] | None, float | None]:
        if self._conformal is None or not getattr(self._conformal, "calibrated", False):
            return None, None
        try:
            cs = self._conformal.predict_set(p_llm, router_class=router_class)
        except TypeError:
            cs = self._conformal.predict_set(p_llm)
        return cs, float(getattr(self._conformal, "alpha", 0.0))
```

In `fuse(...)`, pass `domain` to `_naive_fuse(results, domain="prose")` and `_ebm_fuse(results, word_count, domain)`. Replace each `self._maybe_conformal(p_llm)` with `self._maybe_conformal(p_llm, router_class=domain)`.

In `cairn/pipeline.py`, ensure the fusion call passes the router decision's domain:

```python
        rd = self._router.route(text)
        # ... existing ...
        fusion_result = self._fusion.fuse(layer_results, word_count=word_count, domain=rd.domain)
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/ebm.py cairn/pipeline.py tests/test_fusion.py
git commit -m "feat(fusion): plumb router domain to conformal predict_set"
```

### Task 3.5: `calibrate_conformal.py` collects router class via `analyze_detailed`

- [ ] **Step 1: Update the script body** after `samples = load_dataset(...)`:

```python
    scores: list[float] = []
    labels: list[int] = []
    router_classes: list[str] = []
    for s in tqdm(samples, desc="scoring", unit="sample"):
        det, _results, rd = pipeline.analyze_detailed(s.text)
        scores.append(float(det.p_llm))
        labels.append(int(s.label))
        router_classes.append(rd.domain)

    wrapper = ConformalWrapper(alpha=args.alpha)
    wrapper.calibrate(
        np.asarray(scores), np.asarray(labels),
        router_classes=np.asarray(router_classes),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wrapper.save(args.out)
    print(
        f"calibrated on {len(scores)} samples (alpha={args.alpha}); "
        f"per-class={wrapper.threshold_per_class}; "
        f"per-router={list((wrapper.threshold_per_router or {}).keys())}; "
        f"wrote {args.out}"
    )
```

- [ ] **Step 2: Smoke run**
- [ ] **Step 3: Commit**

```bash
git add scripts/calibrate_conformal.py
git commit -m "feat(scripts): calibrate_conformal collects router domain for v3 schema"
```

### Task 3.6: Run full pytest

Run: `pytest -x -q`
Expected: ≥392 pass.

## Acceptance gate

```bash
pytest tests/test_fusion_conformal_mondrian.py tests/test_fusion.py tests/test_pipeline.py -v
pytest -x -q
```

Subagent returns:

```text
new_schema: binary-split-conformal-v3
fallback_policy: under-MIN_PER_CELL → class-only; unknown router_class → class-only
minimum_cell_size: 30
router_domains_supported: prompt, prose, mixed, insufficient
tests_added:
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 3 of the CAIRN Honest Calibration Story (Unified Plan). Extend ConformalWrapper
to router-conditional Mondrian:

3.1 Per-(router_class, class) thresholds in calibrate when each cell has at least _MIN_PER_CELL=30
    samples; cells below the minimum fall back to class-only.
3.2 threshold_for(cls, router_class=None) and predict_set(p_llm, router_class=None) consult per-router
    first, then per-class.
3.3 Bump save schema to binary-split-conformal-v3 when per-router thresholds exist; preserve v1/v2.
3.4 Plumb RouterDecision.domain from pipeline into EBMFusion.fuse(..., domain=...) and onward into
    _maybe_conformal → predict_set.
3.5 Update scripts/calibrate_conformal.py to collect router domains via pipeline.analyze_detailed.

Build synthetic test data with BOTH labels in EACH router class (audit correction). Write tests
first; full code in the plan. Run pytest -x -q before handoff. Return the Phase 3 acceptance block.
```

---

# Phase 4 — EBM missingness, detector confidence, and schema hardening

## Subagent mission

Strengthen EBM inputs so the model can distinguish "detector absent" from "detector neutral," and "fired weakly" from "fired confidently."

## Why it matters

Existing assembler uses NaN for missing detectors; training/inference impute `0.5`. This conflates missingness with neutrality. In a cascade, missingness is informative — make it a feature.

## Files

**Modify:** `cairn/fusion/ebm.py`, `cairn/fusion/training.py`, `tests/test_fusion.py`, `tests/test_ebm_schema_coverage.py`
**Create:** `tests/test_ebm_missingness.py`

## Required changes (summary)

- Add `<detector>_missing` indicator features (1.0 absent / 0.0 present).
- Add `<detector>_confidence` features (NaN absent / detector confidence present).
- Replace fixed `0.5` NaN imputation with stored column means (`model._cairn_column_means`). Use stored means at inference; fall back to `0.5` only for legacy artifacts.
- Promote schema-mismatch from warning to `RuntimeError`.

## Implementation tasks

### Task 4.1: Per-detector `_missing` indicators

- [ ] **Step 1: Write the failing test** in `tests/test_ebm_missingness.py`:

```python
import math
import pytest
from cairn.contracts import LayerResult
from cairn.fusion.ebm import FeatureAssembler, ASSEMBLER_FEATURE_NAMES


def _lr(layer_id, p_llm=0.6, confidence=0.7, **signals):
    return LayerResult(
        layer_id=layer_id, domain="universal", raw_score=p_llm, p_llm=p_llm,
        confidence=confidence, signals=signals, determination="AMBER",
        attacker_tiers=[], compute_cost="trivial", min_text_length=0,
    )


def test_missing_indicator_per_detector():
    a = FeatureAssembler()
    vec = a.assemble([], word_count=200, domain="prose")
    assert vec["preamble_missing"] == 1.0
    assert vec["fingerprint_missing"] == 1.0
    assert vec["binoculars_missing"] == 1.0
    vec2 = a.assemble([_lr("preamble", p_llm=0.9, severity="HIGH", n_matches=2)],
                      word_count=200, domain="prose")
    assert vec2["preamble_missing"] == 0.0


def test_missing_indicator_in_canonical_feature_list():
    assert "preamble_missing" in ASSEMBLER_FEATURE_NAMES
    assert "fingerprint_missing" in ASSEMBLER_FEATURE_NAMES
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Implement** — in `FeatureAssembler.assemble`, add helper:

```python
        def missing(layer_id: str) -> float:
            return 0.0 if layer_id in by_id else 1.0
```

Emit `<detector>_missing` for each of: `preamble`, `fingerprint_vocab`, `prompt_structure`, `voice_spec`, `instruction_density`, `nssi`, `surprisal_dynamics`, `contrastive_lm`, `token_cohesiveness`, `perturbation`, `contrastive_gen`, `dna_gpt`, `mixed_boundary`, `cross_similarity`. Update `ASSEMBLER_FEATURE_NAMES` accordingly.

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/ebm.py tests/test_ebm_missingness.py
git commit -m "feat(ebm): per-detector _missing indicator features"
```

### Task 4.2: Per-detector `_confidence` features

- [ ] **Step 1: Write the failing test**:

```python
def test_confidence_per_detector():
    a = FeatureAssembler()
    vec = a.assemble([_lr("preamble", p_llm=0.9, confidence=0.85, severity="HIGH")],
                     word_count=200, domain="prose")
    assert vec["preamble_confidence"] == pytest.approx(0.85, abs=1e-6)
    assert math.isnan(vec["nssi_confidence"])
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Implement** — add helper:

```python
        def conf(layer_id: str) -> float:
            return float(by_id[layer_id].confidence) if layer_id in by_id else _NAN
```

Emit `<detector>_confidence` per detector.

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/ebm.py tests/test_ebm_missingness.py
git commit -m "feat(ebm): per-detector _confidence features for MoE-style gating"
```

### Task 4.3: Store column means on the artifact, use at inference

- [ ] **Step 1: Write the failing test**:

```python
def test_ebm_uses_stored_column_means_not_half():
    pytest.importorskip("interpret")
    from cairn.fusion.training import train_ebm
    a = FeatureAssembler()
    rows = []
    for i in range(40):
        if i < 20:
            r = _lr("contrastive_lm", p_llm=0.2, confidence=0.6, binoculars_ratio=1.0)
        else:
            r = _lr("contrastive_lm", p_llm=0.8, confidence=0.6, binoculars_ratio=0.7)
        rows.append((a.assemble([r], word_count=200, domain="prose"), int(i >= 20)))
    X = [r for r, _ in rows]
    y = [lab for _, lab in rows]
    model = train_ebm(X, y)
    means = getattr(model, "_cairn_column_means", None)
    assert means is not None
    assert means["binoculars_ratio"] == pytest.approx((1.0 + 0.7) / 2, abs=1e-6)
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Implement** in `cairn/fusion/training.py::train_ebm`:

```python
def train_ebm(X: list[dict], y: list[int], feature_names: list[str] | None = None) -> Any:
    if not _HAS_INTERPRET:
        raise ImportError("interpretML not installed. Run: pip install 'cairn[fusion]'")
    if feature_names is None:
        feature_names = list(X[0].keys())
    X_arr = np.array([[row.get(f, float("nan")) for f in feature_names] for row in X])
    col_means: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        col = X_arr[:, j]
        finite = col[~np.isnan(col)]
        col_means[name] = float(finite.mean()) if finite.size else 0.5
    for j, name in enumerate(feature_names):
        col = X_arr[:, j]
        col[np.isnan(col)] = col_means[name]
        X_arr[:, j] = col
    y_arr = np.array(y)
    model = ExplainableBoostingClassifier(
        feature_names=feature_names, max_rounds=300, interactions=15, random_state=42,
    )
    model.fit(X_arr, y_arr)
    model._cairn_feature_names = feature_names
    model._cairn_column_means = col_means
    return model
```

In `cairn/fusion/ebm.py::EBMFusion.__init__`:

```python
        self._column_means: dict[str, float] = {}
        if model is not None:
            cm = getattr(model, "_cairn_column_means", None)
            if cm:
                self._column_means = dict(cm)
```

In `_ebm_fuse`, replace global `0.5` imputation with per-column:

```python
        for j, name in enumerate(features):
            if np.isnan(X[0, j]):
                X[0, j] = self._column_means.get(name, 0.5)
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/training.py cairn/fusion/ebm.py tests/test_ebm_missingness.py
git commit -m "feat(ebm): per-column training means for inference imputation"
```

### Task 4.4: Schema mismatch is a hard error

- [ ] **Step 1: Write the failing test**:

```python
def test_ebm_raises_when_artifact_requires_unknown_features():
    from cairn.fusion.ebm import EBMFusion

    class _StubModel:
        _cairn_feature_names = ["preamble_p_llm", "imaginary_feature_xyz"]
        _cairn_column_means = {}

    with pytest.raises(RuntimeError, match="imaginary_feature_xyz"):
        EBMFusion(model=_StubModel())
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Promote warning to RuntimeError** in `EBMFusion.__init__`:

```python
                if extra_in_artifact:
                    raise RuntimeError(
                        "EBM artifact requires features the current assembler "
                        f"cannot produce: {sorted(extra_in_artifact)}. Either "
                        "retrain the EBM or update the assembler to produce them."
                    )
                if extra_in_assembler:
                    logger.info(
                        "Assembler produces features not in EBM artifact "
                        "(ignored until retrain): %s", sorted(extra_in_assembler),
                    )
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/ebm.py tests/test_ebm_missingness.py
git commit -m "fix(ebm): hard-error on artifact schema mismatch"
```

### Task 4.5: Retrain on canonical corpus + benchmark

- [ ] **Step 1: Confirm paths** per Phase 0:

```text
dataset:        <parent>/data/calib2000_v6_train.jsonl
features cache: <parent>/data/all_features_v6.jsonl
profile config: configs/.profile_master_minus_arxiv.yaml
```

- [ ] **Step 2: Retrain to versioned filename**:

```bash
python scripts/train_fusion.py \
  --dataset <parent>/data/calib2000_v6_train.jsonl \
  --config configs/.profile_master_minus_arxiv.yaml \
  --features-cache <parent>/data/all_features_v6.jsonl \
  --out <parent>/models/calib2000/profiles/master_minus_arxiv/ebm_v7.pkl
```

- [ ] **Step 3: Eval old vs new on OOD**:

```bash
python scripts/run_benchmark.py --config configs/.profile_master_minus_arxiv.yaml --dataset <parent>/data/ood_m4_test.jsonl --out /tmp/v6_m4.json
# Point config at ebm_v7.pkl, re-run to /tmp/v7_m4.json
```

Capture: AUROC, ECE, Brier, TPR@1%FPR, F1 deltas. Promote (swap config) **only** if no material regression.

- [ ] **Step 4: Smoke pytest**

```bash
pytest tests/test_pipeline.py tests/test_pipeline_detector_calibration.py -xvs
```

- [ ] **Step 5: Commit (artifact + config bump only if promoted)**

```bash
git add <parent>/models/calib2000/profiles/master_minus_arxiv/ebm_v7.pkl configs/.profile_master_minus_arxiv.yaml
git commit -m "chore(models): retrain EBM v7 with missingness/confidence features"
```

### Task 4.6: Run full pytest

Run: `pytest -x -q`
Expected: ≥392 pass.

## Acceptance gate

```bash
pytest tests/test_ebm_missingness.py tests/test_fusion.py tests/test_ebm_schema_coverage.py -v
pytest -x -q
```

Subagent returns:

```text
features_added: <list>
artifact_schema_change: ebm_v6 → ebm_v7 (column means stored)
legacy_artifact_behavior: legacy fallback to 0.5 imputation when _cairn_column_means absent
metrics_before: AUROC=, ECE=, Brier=, TPR@1%FPR=, F1=
metrics_after:  AUROC=, ECE=, Brier=, TPR@1%FPR=, F1=
artifact_promoted: true/false
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 4 of the CAIRN Honest Calibration Story (Unified Plan).

4.1 <detector>_missing (1.0 absent / 0.0 present) features in FeatureAssembler.
4.2 <detector>_confidence features (NaN absent / detector confidence present).
4.3 Replace fixed 0.5 NaN imputation in train_ebm + EBMFusion with stored model._cairn_column_means;
    legacy artifacts fall back to 0.5.
4.4 Promote artifact schema-mismatch from warning to RuntimeError.
4.5 Retrain on canonical corpus from Phase 0; eval old vs new on OOD m4/raid/mage. Report AUROC,
    ECE, Brier, TPR@1%FPR, F1 deltas. Promote (swap config) ONLY if no material regression.

Write tests first in tests/test_ebm_missingness.py (full code in the plan). Run pytest -x -q
before handoff. Return the Phase 4 acceptance block.
```

---

# Phase 5 — Honest CIs + monotonic constraints

## Subagent mission

Stop reporting hardcoded `±0.10` confidence intervals when calibrated conformal information exists. Add monotonic constraints for features whose meaning should only increase LLM probability.

## Files

**Modify:** `cairn/fusion/ebm.py`, `cairn/fusion/training.py`
**Test:** `tests/test_fusion.py`, `tests/test_ebm_missingness.py`

## Required changes (summary)

- When conformal is calibrated, derive CI half-width from `threshold_for(cls, router_class)` for the predicted class.
- Naive uncalibrated fusion reports `(0.0, 1.0)`.
- Add monotonic-up constraints for: `preamble_severity`, `preamble_n_matches`, `fingerprint_hits_per_1000`, `fingerprint_bigram_hits`, `nssi_n_signals_active`, `nssi_formulaic_density`, `idi_score`, `ps_meta_design_hits`. Detect interpret-ml support; on unsupported versions, log warning and proceed without.

## Implementation tasks

### Task 5.1: Conformal-derived CI when calibrated, wide CI when naive

- [ ] **Step 1: Write the failing test**:

```python
def test_naive_fusion_with_no_conformal_returns_wide_ci():
    from cairn.fusion.ebm import EBMFusion
    from cairn.contracts import LayerResult
    f = EBMFusion(model=None, conformal=None)
    fr = f.fuse(
        [LayerResult(layer_id="preamble", domain="universal", raw_score=0.9, p_llm=0.9,
                     confidence=0.5, signals={}, determination="RED", attacker_tiers=[],
                     compute_cost="trivial", min_text_length=0)],
        word_count=200, domain="prose",
    )
    assert fr.confidence_interval == (0.0, 1.0)


def test_ebm_fusion_ci_uses_conformal_threshold():
    import pytest
    from cairn.fusion.ebm import EBMFusion, ASSEMBLER_FEATURE_NAMES
    from cairn.contracts import LayerResult

    class _FakeConformal:
        calibrated = True
        alpha = 0.1
        def predict_set(self, p, router_class=None):
            return ["LLM"] if p > 0.5 else ["human"]
        def threshold_for(self, cls, router_class=None):
            return {0: 0.42, 1: 0.18}.get(cls)
        threshold_per_class = {0: 0.42, 1: 0.18}
        threshold_per_router = None

    class _FakeModel:
        _cairn_feature_names = list(ASSEMBLER_FEATURE_NAMES)
        _cairn_column_means = {n: 0.5 for n in _cairn_feature_names}
        def predict_proba(self, X):
            import numpy as np
            return np.array([[0.1, 0.9]])
        def explain_local(self, X):
            class _L:
                def data(self_inner, _):
                    return {"scores": [0.0] * len(_FakeModel._cairn_feature_names)}
            return _L()

    f = EBMFusion(model=_FakeModel(), conformal=_FakeConformal())
    fr = f.fuse(
        [LayerResult(layer_id="preamble", domain="universal", raw_score=0.9, p_llm=0.9,
                     confidence=0.5, signals={"severity": "HIGH", "n_matches": 1},
                     determination="RED", attacker_tiers=[], compute_cost="trivial",
                     min_text_length=0)],
        word_count=200, domain="prose",
    )
    assert fr.confidence_interval[0] == pytest.approx(0.72, abs=1e-6)
    assert fr.confidence_interval[1] == 1.0
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Add `_conformal_ci` and update both fusion paths**:

```python
    def _conformal_ci(self, p_llm: float, router_class: str | None = None) -> tuple[float, float] | None:
        if self._conformal is None or not getattr(self._conformal, "calibrated", False):
            return None
        cls = 1 if p_llm >= 0.5 else 0
        try:
            hw = self._conformal.threshold_for(cls, router_class=router_class)
        except (TypeError, AttributeError):
            try:
                hw = self._conformal.threshold_for(cls)
            except (TypeError, AttributeError):
                return None
        if hw is None:
            return None
        hw = float(hw)
        return (max(0.0, p_llm - hw), min(1.0, p_llm + hw))
```

In `_ebm_fuse`:

```python
        ci = self._conformal_ci(p_llm, router_class=domain) or (
            max(0.0, p_llm - 0.10), min(1.0, p_llm + 0.10)
        )
```

In `_naive_fuse`:

```python
        ci = self._conformal_ci(p_llm, router_class=domain) or (0.0, 1.0)
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/ebm.py tests/test_fusion.py
git commit -m "feat(fusion): CI from conformal threshold; (0,1) when naive"
```

### Task 5.2: Monotonic constraints in `train_ebm`

- [ ] **Step 1: Write the failing test** in `tests/test_ebm_missingness.py`:

```python
def test_train_ebm_applies_monotone_constraints():
    pytest.importorskip("interpret")
    a = FeatureAssembler()
    rows = []
    for i in range(50):
        if i < 25:
            lr = _lr("preamble", p_llm=0.05, confidence=0.5, severity="NONE", n_matches=0)
        else:
            lr = _lr("preamble", p_llm=0.95, confidence=0.5, severity="CRITICAL", n_matches=4)
        rows.append((a.assemble([lr], word_count=200, domain="prose"), int(i >= 25)))
    X = [r for r, _ in rows]
    y = [lab for _, lab in rows]
    from cairn.fusion.training import train_ebm
    model = train_ebm(X, y)
    constraints = getattr(model, "_cairn_monotone_constraints", None)
    if constraints is None:
        pytest.skip("interpret-ml without monotone_constraints support")
    feat_to_idx = {n: i for i, n in enumerate(model._cairn_feature_names)}
    for name in ("preamble_severity", "fingerprint_hits_per_1000", "nssi_n_signals_active"):
        idx = feat_to_idx[name]
        assert constraints[idx] == 1, f"expected monotone-up on {name}"
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Wire constraints** in `cairn/fusion/training.py`:

```python
_MONOTONE_UP_FEATURES = (
    "preamble_severity",
    "preamble_n_matches",
    "fingerprint_hits_per_1000",
    "fingerprint_bigram_hits",
    "nssi_n_signals_active",
    "nssi_formulaic_density",
    "idi_score",
    "ps_meta_design_hits",
)
```

In `train_ebm`:

```python
    monotone = [1 if name in _MONOTONE_UP_FEATURES else 0 for name in feature_names]
    try:
        model = ExplainableBoostingClassifier(
            feature_names=feature_names, max_rounds=300, interactions=15, random_state=42,
            monotone_constraints=monotone,
        )
    except TypeError:
        import interpret, logging
        logging.getLogger("cairn.fusion").warning(
            "monotone_constraints unsupported in interpret-ml=%s; training without",
            getattr(interpret, "__version__", "unknown"),
        )
        model = ExplainableBoostingClassifier(
            feature_names=feature_names, max_rounds=300, interactions=15, random_state=42,
        )
        monotone = None
    model.fit(X_arr, y_arr)
    model._cairn_feature_names = feature_names
    model._cairn_column_means = col_means
    if monotone is not None:
        model._cairn_monotone_constraints = monotone
    return model
```

- [ ] **Step 4: Run, verify passes** (or skip with version warning)
- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/training.py tests/test_ebm_missingness.py
git commit -m "feat(ebm): monotone-up constraints on fingerprint/preamble/nssi features"
```

### Task 5.3: Retrain + benchmark guardrail

- [ ] **Step 1: Retrain** to `ebm_v8.pkl`.
- [ ] **Step 2: Compare v7 vs v8** on OOD m4/raid/mage. Promotion guard:

```text
AUROC drop must be < 0.005
TPR@1%FPR drop must be reviewed (acceptable if interpretability gain documented)
ECE/Brier should not worsen materially
```

- [ ] **Step 3: Promote (swap config) only if guards pass; commit**.

### Task 5.4: Run full pytest

Run: `pytest -x -q`
Expected: ≥392 pass.

## Acceptance gate

```bash
pytest tests/test_fusion.py tests/test_ebm_missingness.py -v
pytest -x -q
```

Subagent returns:

```text
ci_policy: ebm uses conformal threshold; naive uses (0,1)
monotonic_features: <list>
interpret_ml_version: 0.7.8
metrics_before:
metrics_after:
artifact_promoted: true/false
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 5 of the CAIRN Honest Calibration Story (Unified Plan).

5.1 Replace hardcoded ±0.10 EBM CI with conformal-threshold-derived CI in cairn/fusion/ebm.py.
    Calibrated: half-width = ConformalWrapper.threshold_for(predicted_class, router_class).
    Naive: CI = (0.0, 1.0).
5.2 Add monotonic-up constraints in cairn/fusion/training.py for _MONOTONE_UP_FEATURES. Detect
    interpret-ml support; on unsupported versions, log a warning and proceed without.
5.3 Retrain to ebm_v8 and compare to v7 on OOD m4/raid/mage. Promote only if AUROC drop < 0.005.

Write tests first; full code in the plan. Run pytest -x -q before handoff. Return the Phase 5
acceptance block.
```

---

# Phase 6 — Leave-one-LLM-out training and per-source evaluation

## Subagent mission

Expose generator-family generalization instead of hiding it inside aggregate AUROC.

## Files

**Modify:** `scripts/train_fusion.py`, `cairn/evaluation/runner.py`, `cairn/evaluation/__init__.py`
**Test:** `tests/test_evaluation/test_runner.py`, `tests/test_eval_serialization.py`
**Create:** `tests/test_train_fusion_holdout.py`

## Required changes (summary)

- `--holdout-source <source>` flag in `train_fusion.py`. Train on samples whose source ≠ holdout; test on samples whose source == holdout. Empty split → SystemExit.
- `EvalReport.per_source: dict[str, dict]` — `{n_samples, auroc, brier}` per source slice (with `auroc_unavailable` for single-class slices).
- `eval_report_to_dict` surfaces `per_source`.

## Implementation tasks

### Task 6.1: `--holdout-source` CLI flag

- [ ] **Step 1: Write the failing test** in `tests/test_train_fusion_holdout.py`:

```python
import json
import subprocess
import sys
from pathlib import Path
import pytest


def _write_dataset(path: Path):
    rows = []
    for i in range(15):
        rows.append({"id": f"a{i}", "text": "Anthropic LLM sample. " * 30,
                     "label": 1, "source": "anthropic"})
    for i in range(15):
        rows.append({"id": f"o{i}", "text": "OpenAI LLM sample. " * 30,
                     "label": 1, "source": "openai"})
    for i in range(15):
        rows.append({"id": f"h{i}", "text": "Human writing sample. " * 30,
                     "label": 0, "source": "hc3"})
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_holdout_source_excludes_family_from_train(tmp_path):
    pytest.importorskip("interpret")
    ds = tmp_path / "ds.jsonl"
    _write_dataset(ds)
    out = tmp_path / "ebm.pkl"
    cfg = Path("configs/default.yaml")
    res = subprocess.run(
        [sys.executable, "scripts/train_fusion.py",
         "--dataset", str(ds), "--config", str(cfg),
         "--out", str(out), "--holdout-source", "openai"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert "trained on 30" in res.stdout
    assert "held out 15" in res.stdout
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Add the flag and the leave-one-source-out branch** in `scripts/train_fusion.py`:

```python
    ap.add_argument(
        "--holdout-source", default=None,
        help="LLM source to hold out entirely from training (zero-shot eval).",
    )
```

After featurize:

```python
    if args.holdout_source:
        idx_train = [i for i, s in enumerate(samples)
                     if (s.source or "").lower() != args.holdout_source.lower()]
        idx_test = [i for i, s in enumerate(samples)
                    if (s.source or "").lower() == args.holdout_source.lower()]
        if not idx_train or not idx_test:
            raise SystemExit(
                f"--holdout-source {args.holdout_source!r} produced an empty split "
                f"(train={len(idx_train)}, test={len(idx_test)})"
            )
        X_train = [X[i] for i in idx_train]
        y_train = [y[i] for i in idx_train]
        X_test = [X[i] for i in idx_test]
        y_test = [y[i] for i in idx_test]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=42
        )
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add scripts/train_fusion.py tests/test_train_fusion_holdout.py
git commit -m "feat(scripts): --holdout-source for leave-one-LLM-out training"
```

### Task 6.2: `per_source` in `EvalReport`

- [ ] **Step 1: Write the failing test**:

```python
def test_per_source_breakdown_in_report():
    from cairn.evaluation.dataset import EvalSample
    from cairn.evaluation.runner import run_eval

    class _FakePipeline:
        def analyze(self, text, task_metadata=None):
            from cairn.contracts import Determination
            p = 0.9 if "Anthropic" in text else 0.6
            return Determination(
                label="RED" if p > 0.75 else "AMBER",
                p_llm=p, confidence_interval=(p - 0.1, p + 0.1),
                prediction_set=["RED"], reason="", top_features=[],
                override_applied=False, detectors_run=[], cascade_phases=[],
                mixed_report=None,
            )

    samples = (
        [EvalSample(id=f"a{i}", text="Anthropic sample.", label=1, source="anthropic") for i in range(20)] +
        [EvalSample(id=f"o{i}", text="OpenAI sample.", label=1, source="openai") for i in range(20)] +
        [EvalSample(id=f"h{i}", text="Human writing.", label=0, source="hc3") for i in range(20)]
    )
    rep = run_eval(_FakePipeline(), samples)
    assert "anthropic" in rep.per_source
    assert rep.per_source["anthropic"]["n_samples"] == 20
    assert "auroc" in rep.per_source["anthropic"] or rep.per_source["anthropic"].get("auroc_unavailable")
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Add `per_source` to `EvalReport`** in `cairn/evaluation/runner.py`:

```python
@dataclass(frozen=True)
class EvalReport:
    predictions: list[dict]
    metrics: dict
    per_tier: dict[str, dict]
    n_samples: int
    config_hash: str
    failed_samples: list[dict] = field(default_factory=list)
    per_attack: dict[str, dict] = field(default_factory=dict)
    per_source: dict[str, dict] = field(default_factory=dict)
```

In `run_eval`:

```python
    by_source: dict[str, list[dict]] = defaultdict(list)
    for pred, sample in zip(predictions, kept):
        if sample.source:
            by_source[sample.source].append(pred)
    per_source: dict[str, dict] = {}
    for src, preds in by_source.items():
        scores = [p["p_llm"] for p in preds]
        labels_ = [p["label"] for p in preds]
        block: dict = {"n_samples": len(preds)}
        if len(set(labels_)) >= 2:
            block["auroc"] = auroc(scores, labels_)
            block["brier"] = brier(scores, labels_)
        else:
            block["auroc_unavailable"] = "single-class slice"
        per_source[src] = block
```

Pass `per_source=per_source` to `EvalReport(...)`.

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/evaluation/runner.py tests/test_evaluation/test_runner.py
git commit -m "feat(eval): per-source breakdown in EvalReport"
```

### Task 6.3: `eval_report_to_dict` surfaces `per_source`

- [ ] **Step 1: Write the failing test** in `tests/test_eval_serialization.py`:

```python
def test_eval_report_to_dict_includes_per_source():
    from cairn.evaluation.runner import EvalReport
    from cairn.evaluation import eval_report_to_dict
    rep = EvalReport(
        predictions=[], metrics={}, per_tier={}, n_samples=0,
        config_hash="abc", per_source={"anthropic": {"n_samples": 10, "auroc": 0.95}},
    )
    out = eval_report_to_dict(rep)
    assert "per_source" in out
    assert out["per_source"]["anthropic"]["auroc"] == 0.95
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Add `"per_source": dict(report.per_source)`** to the serializer's output dict.
- [ ] **Step 4: Commit**

```bash
git add cairn/evaluation/ tests/test_eval_serialization.py
git commit -m "feat(eval): per_source in serialized eval report"
```

### Task 6.4: Smoke holdout training run

- [ ] **Step 1: Run a small holdout job**:

```bash
python scripts/train_fusion.py \
  --dataset <parent>/data/calib2000_v6_train.jsonl \
  --config configs/.profile_master_minus_arxiv.yaml \
  --features-cache <parent>/data/all_features_v6.jsonl \
  --out /tmp/holdout_test.pkl \
  --holdout-source openai
```

Capture train count, holdout count.

### Task 6.5: Run full pytest

Run: `pytest -x -q`
Expected: ≥392 pass.

## Acceptance gate

```bash
pytest tests/test_train_fusion_holdout.py tests/test_evaluation/ tests/test_eval_serialization.py -v
pytest -x -q
```

Subagent returns:

```text
holdout_source_tested: openai
train_count:
holdout_count:
per_source_metrics_added: yes
tests_added:
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 6 of the CAIRN Honest Calibration Story (Unified Plan).

6.1 --holdout-source flag in scripts/train_fusion.py: train on source != holdout; test on
    source == holdout. SystemExit on empty split.
6.2 Add per_source: dict[str, dict] to EvalReport (n_samples, auroc, brier; auroc_unavailable
    when only one class).
6.3 Surface per_source in eval_report_to_dict.
6.4 Smoke-run a holdout training job on a known source.

Write tests first; full code in the plan. Run pytest -x -q before handoff. Return the Phase 6
acceptance block.
```

---

# Phase 7 — Safer model serialization + provenance metadata

## Subagent mission

Replace unsafe pickle-first artifact handling with `skops`, while keeping legacy pickle artifacts readable during migration.

## Files

**Modify:** `cairn/fusion/training.py`, `cairn/pipeline.py`, `scripts/train_fusion.py`, `pyproject.toml`
**Create:** `cairn/fusion/serialization.py`, `tests/test_fusion_serialization.py`

## Required changes (summary)

- Add `skops` to `[project.optional-dependencies].fusion`.
- New `save_model(model, path, *, metadata=None)` writes skops file + `.hmac` sidecar + `.meta.json`.
- New `load_model(path, *, verify_hmac=True, trusted=None)` returns `(model, metadata)`. Detects skops vs legacy pickle; pickle path emits `DeprecationWarning`.
- `train_fusion.py` writes provenance metadata.

## Implementation tasks

### Task 7.1: Add skops dep, write serialization module

- [ ] **Step 1: Write the failing test** in `tests/test_fusion_serialization.py`:

```python
import pytest


def test_save_and_load_round_trips_with_metadata(tmp_path):
    pytest.importorskip("interpret")
    pytest.importorskip("skops")
    import numpy as np
    from cairn.fusion.training import train_ebm
    from cairn.fusion.serialization import save_model, load_model

    X = [{"a": float(i % 5), "b": 1.0} for i in range(40)]
    y = [i % 2 for i in range(40)]
    model = train_ebm(X, y, feature_names=["a", "b"])
    out = tmp_path / "ebm.skops"
    save_model(model, out, metadata={"dataset_hash": "abc123", "n_samples": 40})
    loaded, metadata = load_model(out)
    assert metadata["dataset_hash"] == "abc123"
    Xt = np.array([[1.0, 1.0]])
    np.testing.assert_allclose(model.predict_proba(Xt), loaded.predict_proba(Xt), atol=1e-6)


def test_load_detects_tampering(tmp_path):
    pytest.importorskip("interpret")
    pytest.importorskip("skops")
    from cairn.fusion.training import train_ebm
    from cairn.fusion.serialization import save_model, load_model
    X = [{"a": float(i), "b": 1.0} for i in range(20)]
    y = [i % 2 for i in range(20)]
    model = train_ebm(X, y, feature_names=["a", "b"])
    out = tmp_path / "ebm.skops"
    save_model(model, out, metadata={"version": "v7"})
    sidecar = out.with_suffix(".skops.hmac")
    sidecar.write_text("0" * 64)
    with pytest.raises(ValueError, match="HMAC mismatch"):
        load_model(out, verify_hmac=True)


def test_legacy_pickle_load_warns_then_works(tmp_path):
    pytest.importorskip("interpret")
    import pickle
    from cairn.fusion.training import train_ebm
    from cairn.fusion.serialization import load_model
    X = [{"a": float(i), "b": 1.0} for i in range(20)]
    y = [i % 2 for i in range(20)]
    model = train_ebm(X, y, feature_names=["a", "b"])
    out = tmp_path / "legacy.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    with pytest.warns(DeprecationWarning, match="legacy pickle"):
        loaded, metadata = load_model(out, verify_hmac=False)
    assert loaded is not None
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Add skops to extras and create the module** — in `pyproject.toml`:

```toml
[project.optional-dependencies]
fusion = ["interpret>=0.5", "scikit-learn>=1.3", "skops>=0.10"]
```

Create `cairn/fusion/serialization.py`:

```python
from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any


_HMAC_KEY_ENV = "CAIRN_MODEL_HMAC_KEY"


def _hmac_key() -> bytes:
    return os.environ.get(_HMAC_KEY_ENV, "cairn-default-model-hmac-key").encode()


def _digest(payload: bytes) -> str:
    return hmac.new(_hmac_key(), payload, hashlib.sha256).hexdigest()


def save_model(model: Any, path: Path, *, metadata: dict | None = None) -> None:
    try:
        import skops.io as sio
    except ImportError as e:
        raise RuntimeError(
            "skops is required. Install with: pip install 'cairn[fusion]'"
        ) from e
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, path)
    payload = path.read_bytes()
    sidecar = path.with_suffix(path.suffix + ".hmac")
    sidecar.write_text(_digest(payload))
    if metadata is not None:
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _looks_like_skops(path: Path) -> bool:
    if path.suffix == ".skops":
        return True
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def load_model(
    path: Path, *, verify_hmac: bool = True, trusted: list[str] | None = None
) -> tuple[Any, dict]:
    path = Path(path)
    if not _looks_like_skops(path):
        warnings.warn(
            f"Loading legacy pickle artifact {path}; retrain with skops",
            DeprecationWarning, stacklevel=2,
        )
        with open(path, "rb") as f:
            return pickle.load(f), {}
    try:
        import skops.io as sio
    except ImportError as e:
        raise RuntimeError(
            "skops is required. Install with: pip install 'cairn[fusion]'"
        ) from e
    payload = path.read_bytes()
    if verify_hmac:
        sidecar = path.with_suffix(path.suffix + ".hmac")
        if not sidecar.exists():
            raise ValueError(f"HMAC sidecar missing for {path}")
        expected = sidecar.read_text().strip()
        actual = _digest(payload)
        if not hmac.compare_digest(expected, actual):
            raise ValueError(f"HMAC mismatch for {path} — artifact tampered")
    untrusted = sio.get_untrusted_types(file=path)
    if trusted is None:
        trusted = list(untrusted)
    model = sio.load(path, trusted=trusted)
    metadata: dict = {}
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    return model, metadata
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add pyproject.toml cairn/fusion/serialization.py tests/test_fusion_serialization.py
git commit -m "feat(fusion): skops + HMAC + provenance with legacy pickle fallback"
```

### Task 7.2: Wire skops into `train_fusion.py` with provenance

- [ ] **Step 1: Update the script** — capture metadata after `model = train_ebm(...)`:

```python
    import hashlib
    from datetime import datetime, timezone
    from cairn.fusion.serialization import save_model as save_model_skops
    dataset_bytes = args.dataset.read_bytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()[:16]
    metadata = {
        "dataset_path": str(args.dataset),
        "dataset_hash": dataset_hash,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "label_balance_train": sum(y_train) / max(1, len(y_train)),
        "label_balance_test": sum(y_test) / max(1, len(y_test)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "holdout_source": args.holdout_source,
        "interactions": 15,
        "max_rounds": 300,
        "schema_version": "ebm-v8",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model_skops(model, args.out, metadata=metadata)
```

- [ ] **Step 2: Update pipeline loader**:

```python
def _load_fusion_model(path: Path) -> Any:
    from cairn.fusion.serialization import load_model
    model, _meta = load_model(path)
    return model
```

- [ ] **Step 3: Run pytest** — `pytest -x -q`. Expected: ≥392 pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_fusion.py cairn/pipeline.py
git commit -m "feat(fusion): train_fusion writes skops artifact with provenance metadata"
```

### Task 7.3: Run full pytest

Run: `pytest -x -q`
Expected: ≥392 pass.

## Acceptance gate

```bash
pip install -e ".[fusion]"
pytest tests/test_fusion_serialization.py -v
pytest -x -q
```

Subagent returns:

```text
serialization_format: skops + .hmac + .meta.json
legacy_pickle_behavior: load works with DeprecationWarning
metadata_fields: dataset_path, dataset_hash, n_train, n_test, label_balance_*, trained_at, holdout_source, interactions, max_rounds, schema_version
skops_trusted_types: dynamically derived from sio.get_untrusted_types
tests_added:
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 7 of the CAIRN Honest Calibration Story (Unified Plan).

7.1 Add skops>=0.10 to pyproject.toml fusion extras. Create cairn/fusion/serialization.py with
    save_model(model, path, metadata=None) and load_model(path, verify_hmac=True, trusted=None).
    save_model writes the skops file plus .hmac sidecar (HMAC-SHA256 keyed by env
    CAIRN_MODEL_HMAC_KEY) plus optional .meta.json. load_model auto-detects skops (zip magic) vs
    legacy pickle and emits DeprecationWarning on the pickle path.
7.2 In scripts/train_fusion.py, capture provenance metadata (dataset_path, dataset_hash, n_train,
    n_test, label_balance_*, trained_at, holdout_source, interactions, max_rounds, schema_version)
    and write via save_model_skops. Update cairn/pipeline.py loader to use load_model.

Write tests first; full code in the plan. Run pytest -x -q before handoff. Return the Phase 7
acceptance block.
```

---

# Phase 8 — Retrieval-based cross-similarity

## Subagent mission

Upgrade `cross_similarity` from within-batch word-shingle Jaccard to retrieval against a persistent historical sentence-transformer + FAISS index.

## Files

**Create:** `cairn/index/__init__.py`, `cairn/index/retrieval.py`, `cairn/index/store.py`, `tests/test_index_retrieval.py`
**Modify:** `cairn/detectors/cross_similarity.py`, `cairn/pipeline.py`, `cairn/history.py` (optional ingest hook), `pyproject.toml`

## Required changes (summary)

- New `cairn[retrieval]` extra: `sentence-transformers>=2.7`, `faiss-cpu>=1.7`.
- `RetrievalIndex` with `add(sid, text)`, `query(sid_or_text, k=5)`. Uses normalized embeddings + FAISS inner product.
- `PersistentRetrievalIndex(db_path)` SQLite-backed durability. **Best-effort writes** — failed db ops log and continue (CLAUDE.md invariant).
- `cross_similarity` queries the injected index when present; cold-start → SKIP `"empty_retrieval_index"`.
- Pipeline lazily creates the index per `retrieval.enabled`, injects the handle, best-effort `add(sha256_prefix, text)` after analyze.
- **All retrieval tests behind `pytest.importorskip` so embedding models are NOT silently downloaded in normal CI.**

## Implementation tasks

### Task 8.1: `RetrievalIndex` (in-memory)

- [ ] **Step 1: Write the failing test** in `tests/test_index_retrieval.py`:

```python
import pytest


def test_retrieval_index_finds_near_duplicates():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from cairn.index.retrieval import RetrievalIndex
    idx = RetrievalIndex()
    idx.add("a", "The quick brown fox jumps over the lazy dog. " * 5)
    idx.add("b", "Dogs are loyal companions to humans throughout history. " * 5)
    idx.add("c", "The quick brown fox jumps over the lazy dog. " * 5)
    nbrs = idx.query("c", k=2)
    nbr_ids = [nid for nid, _ in nbrs if nid != "c"]
    assert nbr_ids[0] == "a"


def test_retrieval_index_empty_returns_no_neighbors():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from cairn.index.retrieval import RetrievalIndex
    idx = RetrievalIndex()
    idx.add("a", "Short text. " * 20)
    nbrs = idx.query("a", k=5)
    nbr_ids = [nid for nid, _ in nbrs if nid != "a"]
    assert nbr_ids == []
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Add the extras and module** — `pyproject.toml`:

```toml
[project.optional-dependencies]
retrieval = ["sentence-transformers>=2.7", "faiss-cpu>=1.7"]
```

`cairn/index/__init__.py`:

```python
from cairn.index.retrieval import RetrievalIndex
__all__ = ["RetrievalIndex"]
```

`cairn/index/retrieval.py`:

```python
from __future__ import annotations
import logging

logger = logging.getLogger("cairn.index")

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RetrievalIndex:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError as e:
            raise ImportError(
                "RetrievalIndex requires sentence-transformers + faiss-cpu. "
                "Install with: pip install 'cairn[retrieval]'"
            ) from e
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self._faiss = faiss
        self._index = faiss.IndexFlatIP(self._dim)
        self._ids: list[str] = []
        self._texts: dict[str, str] = {}

    def add(self, sid: str, text: str) -> None:
        emb = self._embed(text)
        self._index.add(emb)
        self._ids.append(sid)
        self._texts[sid] = text

    def query(self, sid_or_text: str, k: int = 5) -> list[tuple[str, float]]:
        text = self._texts.get(sid_or_text, sid_or_text)
        if self._index.ntotal == 0:
            return []
        emb = self._embed(text)
        k_eff = min(k, self._index.ntotal)
        scores, idx = self._index.search(emb, k_eff)
        out: list[tuple[str, float]] = []
        for s, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            out.append((self._ids[i], float(s)))
        return out

    def _embed(self, text: str):
        import numpy as np
        emb = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")
```

- [ ] **Step 4: Install + run** — `pip install -e ".[retrieval]" && pytest tests/test_index_retrieval.py -xvs`
- [ ] **Step 5: Commit**

```bash
git add cairn/index/ tests/test_index_retrieval.py pyproject.toml
git commit -m "feat(index): RetrievalIndex (sentence-transformer + FAISS)"
```

### Task 8.2: `PersistentRetrievalIndex` (SQLite, best-effort)

- [ ] **Step 1: Write the failing test**:

```python
def test_persistent_index_round_trips(tmp_path):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from cairn.index.store import PersistentRetrievalIndex
    db = tmp_path / "ix.db"
    idx = PersistentRetrievalIndex(db_path=db)
    idx.add("a", "The quick brown fox jumps over the lazy dog. " * 5)
    idx.add("b", "Dogs are loyal companions to humans. " * 5)
    idx.close()
    idx2 = PersistentRetrievalIndex(db_path=db)
    assert idx2.size == 2
    nbrs = idx2.query("a", k=2)
    nbr_ids = [n for n, _ in nbrs if n != "a"]
    assert nbr_ids[0] == "b"
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Implement** in `cairn/index/store.py`:

```python
from __future__ import annotations
import logging
import sqlite3
from pathlib import Path

from cairn.index.retrieval import RetrievalIndex

logger = logging.getLogger("cairn.index.store")


class PersistentRetrievalIndex(RetrievalIndex):
    def __init__(self, db_path: Path, **kw):
        super().__init__(**kw)
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._open_db()
        self._rebuild_from_db()

    @property
    def size(self) -> int:
        return len(self._ids)

    def _open_db(self) -> None:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS submissions ("
                " sid TEXT PRIMARY KEY, text TEXT NOT NULL, "
                " ts REAL DEFAULT (strftime('%s','now')))"
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.warning("PersistentRetrievalIndex: db open failed at %s", self._db_path, exc_info=True)
            self._conn = None

    def _rebuild_from_db(self) -> None:
        if self._conn is None:
            return
        try:
            cur = self._conn.execute("SELECT sid, text FROM submissions ORDER BY ts ASC")
            for sid, text in cur.fetchall():
                super().add(sid, text)
        except sqlite3.Error:
            logger.warning("PersistentRetrievalIndex: rebuild failed", exc_info=True)

    def add(self, sid: str, text: str) -> None:
        super().add(sid, text)
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO submissions(sid, text) VALUES(?, ?)",
                (sid, text),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.warning("PersistentRetrievalIndex: insert failed for %r", sid, exc_info=True)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/index/store.py tests/test_index_retrieval.py
git commit -m "feat(index): SQLite-backed PersistentRetrievalIndex"
```

### Task 8.3: Wire `cross_similarity` to the index

- [ ] **Step 1: Write the failing test** in `tests/test_detectors/test_cross_similarity.py`:

```python
def test_cross_similarity_uses_retrieval_index_when_present():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from cairn.detectors.cross_similarity import CrossSimilarityDetector
    from cairn.index.retrieval import RetrievalIndex

    rx = RetrievalIndex()
    rx.add("historical_llm_1", "Once upon a time the model produced text. " * 10)
    rx.add("historical_human_1", "I remember when I was a kid playing in the garden. " * 10)

    det = CrossSimilarityDetector()
    config = {
        "use_retrieval_index": True,
        "_index_handle": rx,
        "neighbor_score_threshold": 0.85,
    }
    text = "Once upon a time the model produced text. " * 10
    res = det.analyze(text, config)
    assert res.determination != "SKIP"
    assert res.signals["top_neighbor_id"] == "historical_llm_1"
    assert res.signals["top_neighbor_score"] > 0.85


def test_cross_similarity_falls_back_to_skip_when_no_index():
    from cairn.detectors.cross_similarity import CrossSimilarityDetector
    det = CrossSimilarityDetector()
    res = det.analyze("Some text. " * 30, {"use_retrieval_index": False})
    assert res.determination == "SKIP"
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Update `analyze`**:

```python
    def analyze(self, text: str, config: dict) -> LayerResult:
        use_index = bool(config.get("use_retrieval_index", False))
        index = config.get("_index_handle")
        if use_index and index is not None:
            return self._retrieval_analyze(text, config, index)
        return _skip(self, reason="single_text_mode_use_analyze_batch")

    def _retrieval_analyze(self, text: str, config: dict, index) -> LayerResult:
        threshold = float(config.get("neighbor_score_threshold", 0.85))
        neighbours = index.query(text, k=int(config.get("retrieval_k", 5)))
        if not neighbours:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0,
                signals={"reason": "empty_retrieval_index"},
                determination="SKIP", attacker_tiers=["A0", "A1", "A2"],
                compute_cost=self.compute_cost, min_text_length=0,
            )
        top_id, top_score = neighbours[0]
        p_llm = self._heuristic_p_llm(top_score, threshold)
        if p_llm >= 0.75: determination = "RED"
        elif p_llm >= 0.50: determination = "AMBER"
        elif p_llm >= 0.25: determination = "YELLOW"
        else: determination = "GREEN"
        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=float(top_score), p_llm=float(p_llm),
            confidence=0.7 if top_score >= threshold else 0.4,
            signals={
                "top_neighbor_id": top_id,
                "top_neighbor_score": round(float(top_score), 4),
                "neighbor_count": len(neighbours),
                "threshold": threshold,
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2"],
            compute_cost=self.compute_cost,
            min_text_length=0,
        )

    @staticmethod
    def _heuristic_p_llm(score: float, threshold: float) -> float:
        # _HEURISTIC_ score → p_llm map (CLAUDE.md naming convention).
        if score < threshold:
            return max(0.2, 0.5 * (score / max(threshold, 1e-6)))
        return min(0.95, 0.7 + (score - threshold) * (0.25 / max(1e-6, 1.0 - threshold)))
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/detectors/cross_similarity.py tests/test_detectors/test_cross_similarity.py
git commit -m "feat(cross_similarity): retrieval-index path with batch fallback"
```

### Task 8.4: Pipeline wiring (lazy index, inject, best-effort write)

- [ ] **Step 1: Write the failing test**:

```python
def test_pipeline_instantiates_index_when_configured(tmp_path):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    config = {
        "router": {"minimum_words": {"prompt": 30, "prose": 150}},
        "fusion": {},
        "decision": {"red_threshold": 0.75, "amber_threshold": 0.50, "yellow_threshold": 0.25,
                     "abstention": {"enabled": True, "max_prediction_set_size": 3},
                     "mixed": {"enabled": True, "min_mixed_probability": 0.60}},
        "detectors": {"cross_similarity": {"enabled": True, "use_retrieval_index": True}},
        "retrieval": {"enabled": True, "db_path": str(tmp_path / "ix.db")},
    }
    from cairn.pipeline import CairnPipeline
    p = CairnPipeline(config)
    assert getattr(p, "_retrieval_index", None) is not None
```

- [ ] **Step 2: Run, verify fails**
- [ ] **Step 3: Add lazy index instantiation in `CairnPipeline.__init__`**:

```python
        self._retrieval_index = None
        rcfg = (config or {}).get("retrieval", {})
        if rcfg.get("enabled"):
            try:
                from cairn.index.store import PersistentRetrievalIndex
                self._retrieval_index = PersistentRetrievalIndex(
                    db_path=rcfg.get("db_path", "data/retrieval.db"),
                )
            except ImportError:
                logger.warning(
                    "retrieval extra not installed; cross_similarity falls back to batch."
                )
            except Exception:
                logger.warning("retrieval index init failed; continuing without", exc_info=True)
```

In the analyze entrypoint, inject the handle:

```python
        if det_id == "cross_similarity" and self._retrieval_index is not None:
            det_config = {**det_config, "_index_handle": self._retrieval_index, "use_retrieval_index": True}
```

After analyze completes, best-effort add (privacy via hash):

```python
        if self._retrieval_index is not None:
            try:
                import hashlib
                sid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                self._retrieval_index.add(sid, text)
            except Exception:
                logger.warning("retrieval index add failed", exc_info=True)
```

- [ ] **Step 4: Run, verify passes**
- [ ] **Step 5: Commit**

```bash
git add cairn/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): wire RetrievalIndex into cross_similarity dispatch"
```

### Task 8.5: Run full pytest

Run: `pytest -x -q`
Expected: ≥392 pass.

## Acceptance gate

```bash
pip install -e ".[retrieval]"
pytest tests/test_index_retrieval.py -v
pytest tests/test_detectors/ -v -k cross_similarity
pytest -x -q
```

Subagent returns:

```text
index_backend: FAISS IndexFlatIP
embedding_model: sentence-transformers/all-MiniLM-L6-v2
persistence_path: SQLite (config retrieval.db_path)
cold_start_behavior: empty index → SKIP "empty_retrieval_index"
history_ingest_behavior: best-effort add(sha256_prefix, text) post-analyze
tests_added:
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 8 of the CAIRN Honest Calibration Story (Unified Plan).

8.1 Add cairn[retrieval] extra: sentence-transformers>=2.7, faiss-cpu>=1.7. Create
    cairn/index/retrieval.py::RetrievalIndex with add(sid, text) and query(sid_or_text, k).
8.2 Create cairn/index/store.py::PersistentRetrievalIndex(db_path) extending RetrievalIndex with
    SQLite durability. Best-effort writes (failed db ops log and continue).
8.3 Update cairn/detectors/cross_similarity.py: when config["use_retrieval_index"] AND
    config["_index_handle"], query the index. Empty index → SKIP "empty_retrieval_index".
8.4 Wire cairn/pipeline.py: lazy PersistentRetrievalIndex per config.retrieval.enabled, inject
    {"_index_handle": idx, "use_retrieval_index": True} into cross_similarity dispatch config,
    best-effort idx.add(sha256(text)[:16], text) after analyze.

CRITICAL: All retrieval tests must be behind pytest.importorskip("sentence_transformers") +
pytest.importorskip("faiss") so embedding models are NOT silently downloaded during normal CI runs.
Write tests first; full code in the plan. Run pytest -x -q before handoff. Return the Phase 8
acceptance block.
```

---

# Phase 9 — RAID benchmark in CI

## Subagent mission

Add a recurring external benchmark gate that reports TPR at a strict false-positive budget against the production-calibrated profile.

## Files

**Create:** `scripts/run_raid_benchmark.py`, `scripts/build_raid_fixture.py`, `tests/fixtures/raid_mini.jsonl` (50 rows), `.github/workflows/raid.yml`
**Modify (only if needed):** `cairn/evaluation/metrics.py`, `pyproject.toml`

## Required changes (summary)

- New `tpr_at_fpr_ceiling(scores, labels, ceiling=0.01)` returning `(tpr, threshold, achieved_fpr)`.
- Builder script (one-shot, manual; deterministic 50-row sample).
- Runner script accepting `--dataset`, `--config`, `--out`, `--fpr-ceiling`, `--min-tpr`. Loads samples, runs `CairnPipeline`, calls `run_eval`, computes headline `TPR@FPR<=0.01`, prints + writes JSON.
- Workflow on release tags + weekly cron + manual dispatch + PR-on-relevant-files. **Uses calibrated profile** `configs/.profile_master_minus_arxiv.yaml`, NOT `default.yaml`.
- Data-license aware: do not commit RAID CSV unless redistribution permitted; otherwise SHA256 manifest only.

## Implementation tasks

### Task 9.1: TPR-at-FPR-ceiling primitive

- [ ] **Step 1: Write the failing test** in `tests/test_evaluation/test_metrics.py`:

```python
def test_tpr_at_fpr_ceiling_returns_threshold_and_tpr():
    from cairn.evaluation.metrics import tpr_at_fpr_ceiling
    import numpy as np
    rng = np.random.default_rng(0)
    human = rng.beta(2, 8, 100).tolist()
    llm = rng.beta(8, 2, 100).tolist()
    scores = human + llm
    labels = [0] * 100 + [1] * 100
    tpr, threshold, achieved_fpr = tpr_at_fpr_ceiling(scores, labels, ceiling=0.01)
    assert 0.0 <= tpr <= 1.0
    assert 0.0 <= achieved_fpr <= 0.01 + 1e-6
    flagged_humans = sum(1 for s in human if s >= threshold) / len(human)
    assert flagged_humans <= 0.01 + 1e-6
```

- [ ] **Step 2: Implement and re-run** — add to `cairn/evaluation/metrics.py`:

```python
def tpr_at_fpr_ceiling(
    scores: list[float], labels: list[int], *, ceiling: float = 0.01
) -> tuple[float, float, float]:
    """Lowest threshold t s.t. FPR(t) <= ceiling; return (TPR, t, FPR)."""
    import numpy as np
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    human = s[y == 0]
    llm = s[y == 1]
    if human.size == 0 or llm.size == 0:
        raise ValueError("tpr_at_fpr_ceiling requires both classes")
    threshold = float(np.quantile(human, 1.0 - ceiling, method="higher"))
    achieved_fpr = float((human >= threshold).mean())
    tpr = float((llm >= threshold).mean())
    return tpr, threshold, achieved_fpr
```

- [ ] **Step 3: Commit**

```bash
git add cairn/evaluation/metrics.py tests/test_evaluation/test_metrics.py
git commit -m "feat(eval): tpr_at_fpr_ceiling primitive for Neyman-Pearson reporting"
```

### Task 9.2: RAID smoke fixture (50 rows)

- [ ] **Step 1: Write `scripts/build_raid_fixture.py`** (one-shot, manual; deterministic 50-row sample with seed=42).

```python
"""One-shot builder: pull a deterministic 50-row sample from the public RAID test CSV."""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", default=Path("tests/fixtures/raid_mini.jsonl"), type=Path)
    ap.add_argument("--n-per-class", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    with args.source.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row.get("label", 0))
            text = row.get("generation") or row.get("text") or ""
            if not text or len(text.split()) < 50:
                continue
            by_label[label].append({
                "id": f"raid_{row.get('id') or row.get('row_id')}",
                "text": text, "label": label,
                "source": row.get("model", "raid_human") or "raid_human",
                "attack_name": row.get("decoding") or row.get("attack"),
                "tier": "raid",
            })
    out_rows = (
        rng.sample(by_label[0], args.n_per_class) +
        rng.sample(by_label[1], args.n_per_class)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run once locally (manual)**

```bash
curl -L -o /tmp/raid-test.csv https://raid-bench.xyz/data/raid-test.csv
python scripts/build_raid_fixture.py --source /tmp/raid-test.csv --n-per-class 25
```

If unreachable, document the manual step and ask the user for the CSV.

- [ ] **Step 3: Commit (data only if license permits)**:

```bash
git add tests/fixtures/raid_mini.jsonl scripts/build_raid_fixture.py
git commit -m "feat(test): 50-row RAID smoke fixture and builder"
```

### Task 9.3: `run_raid_benchmark.py`

- [ ] **Step 1: Write the runner**:

```python
# scripts/run_raid_benchmark.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import load_dataset
from cairn.evaluation.metrics import tpr_at_fpr_ceiling
from cairn.evaluation.runner import run_eval
from cairn.pipeline import CairnPipeline


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--fpr-ceiling", type=float, default=0.01)
    ap.add_argument("--min-tpr", type=float, default=None,
                    help="Exit non-zero if TPR is below this floor (CI gate).")
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()

    pipeline = CairnPipeline.from_config_file(args.config)
    samples = load_dataset(args.dataset)
    report = run_eval(pipeline, samples, progress=True)

    scores = [p["p_llm"] for p in report.predictions]
    labels = [p["label"] for p in report.predictions]
    tpr, threshold, achieved_fpr = tpr_at_fpr_ceiling(scores, labels, ceiling=args.fpr_ceiling)
    summary = {
        "headline": {"metric": f"TPR@FPR<={args.fpr_ceiling}", "value": tpr},
        "n_samples": report.n_samples,
        "fpr_ceiling": args.fpr_ceiling,
        "threshold": threshold,
        "achieved_fpr": achieved_fpr,
        "auroc": report.metrics.get("auroc"),
        "per_source": report.per_source,
        "profile_used": str(args.config),
    }
    print(f"RAID TPR@FPR<={args.fpr_ceiling}: {tpr:.4f}")
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
    if args.min_tpr is not None and tpr < args.min_tpr:
        print(f"ERROR: TPR {tpr:.3f} below floor {args.min_tpr:.3f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke run against the calibrated profile**:

```bash
python scripts/run_raid_benchmark.py \
  --dataset tests/fixtures/raid_mini.jsonl \
  --config configs/.profile_master_minus_arxiv.yaml \
  --fpr-ceiling 0.01 \
  --out /tmp/raid_report.json
```

Expected: prints headline, JSON includes `headline.{metric, value}`.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_raid_benchmark.py
git commit -m "feat(scripts): run_raid_benchmark — TPR@FPR<=0.01 reporter"
```

### Task 9.4: GitHub Actions workflow

- [ ] **Step 1: Write the workflow** — `.github/workflows/raid.yml`:

```yaml
name: RAID benchmark

on:
  push:
    tags: ['v*']
  schedule:
    - cron: '0 5 * * 1'
  workflow_dispatch:
  pull_request:
    paths:
      - 'cairn/fusion/**'
      - 'cairn/detectors/**'
      - 'configs/.profile_master_minus_arxiv.yaml'
      - 'models/calib2000/**'
      - '.github/workflows/raid.yml'

jobs:
  raid-mini:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e '.[fusion]'
      - name: Run RAID-mini against calibrated profile
        run: |
          python scripts/run_raid_benchmark.py \
            --dataset tests/fixtures/raid_mini.jsonl \
            --config configs/.profile_master_minus_arxiv.yaml \
            --fpr-ceiling 0.01 \
            --min-tpr 0.40 \
            --out raid-mini-report.json
      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: raid-mini-report
          path: raid-mini-report.json
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/raid.yml
git commit -m "ci(raid): TPR@FPR<=0.01 gate against calibrated profile"
```

### Task 9.5: Run full pytest

Run: `pytest -x -q`
Expected: ≥392 pass.

## Acceptance gate

```bash
pytest tests/test_evaluation/test_metrics.py -v
python scripts/run_raid_benchmark.py \
  --dataset tests/fixtures/raid_mini.jsonl \
  --config configs/.profile_master_minus_arxiv.yaml \
  --out /tmp/raid_report.json
pytest -x -q
```

Subagent returns:

```text
raid_source: <URL or local CSV path>
data_license_status: <committed / SHA256 manifest only>
sample_count: 50
profile_used: configs/.profile_master_minus_arxiv.yaml
headline_tpr_at_1pct_fpr: <float>
artifact_path: raid-mini-report.json
workflow_status: created
```

## Dispatch prompt

```text
Implement Phase 9 of the CAIRN Honest Calibration Story (Unified Plan).

9.1 Add tpr_at_fpr_ceiling(scores, labels, ceiling=0.01) → (tpr, threshold, achieved_fpr) to
    cairn/evaluation/metrics.py.
9.2 scripts/build_raid_fixture.py: deterministic 50-row sample (seed=42). Verify RAID licensing
    before committing the JSONL; otherwise commit downloader plus SHA256 manifest.
9.3 scripts/run_raid_benchmark.py: load samples, run CairnPipeline.from_config_file, call run_eval,
    compute headline TPR@FPR<=0.01, print + write JSON with headline={metric, value}, per_source,
    auroc.
9.4 .github/workflows/raid.yml: manual dispatch + weekly cron + release tags + PR-on-relevant-files.
    CRITICAL: --config must be configs/.profile_master_minus_arxiv.yaml (calibrated profile, note
    the dot-prefix) — NOT configs/default.yaml.

Write tests; full code in the plan. Run pytest -x -q before handoff. Return the Phase 9 acceptance
block.
```

---

# Phase 10 — Semantic kinematics detector

## Subagent mission

Add a macro-feature detector that fingerprints the *topological shape of the argument* in embedding space. Sentence-to-sentence cosine-distance trajectory, plus its first/second/third derivatives (velocity, acceleration, jerk), distinguishes the human "drunken walk" pattern of episodic cognition from the LLM "frictionless gradient" pattern enforced by self-attention.

## Why it matters (theory)

Existing detectors are micro-features: token probability, vocabulary fingerprints, perplexity, syntax. A paraphraser (QuillBot, Undetectable.ai, StealthWriter) or a "write with high burstiness and typos"-style prompt scrambles micro-features. Tokens move; the trajectory of meaning between sentences is harder to disturb without breaking the underlying argument.

- **Human cognition is a drunken walk.** Episodic and associative — we linger on a specific detail, leap to an orthogonal metaphor, change subject. In high-dimensional semantic space the trajectory is jagged and fractal: high variance of sentence-to-sentence cosine distance, high jerk (third derivative), kinky path.
- **LLM "thought" is a frictionless gradient.** Self-attention mathematically tethers each next sentence to the preceding context window. Loss minimization → smooth interpolation between concepts. Low variance, low jerk, locally linear.

The macro-trajectory survives paraphrase because most paraphrasers operate sentence-locally; they don't reshape the global flow of meaning. They survive "burstiness" prompt-engineering because that controls token-level entropy, not embedding-level trajectory.

We don't need a large LM for this — a tiny embedding model (`sentence-transformers/all-MiniLM-L6-v2`, ~80 MB, milliseconds on CPU) is sufficient. The Phase 8 `cairn[retrieval]` extra already includes it.

## Files

**Create:**
- `cairn/detectors/semantic_kinematics.py`
- `tests/test_detectors/test_semantic_kinematics.py`

**Modify:**
- `cairn/cascade.py` (register in `PHASE2_DETECTORS`)
- `cairn/fusion/ebm.py::FeatureAssembler.assemble` (emit `semantic_kinematics_p_llm` / `_missing` / `_confidence` per Phase 4 pattern; plus signal-specific `sk_*` features)
- `configs/default.yaml` (`detectors.semantic_kinematics` block)
- `cairn/pipeline.py::detector_availability` (declare the `sentence-transformers` requirement so the Settings tab reports availability accurately)

## Required changes (summary)

- Detector class implementing the `Detector` protocol from `cairn/contracts.py`. Lazy-load `sentence-transformers/all-MiniLM-L6-v2` on first `analyze`. Sentence-segment via `cairn/normalizer.py` (regex fallback if no helper exists). Embed each sentence with `normalize_embeddings=True`. Compute trajectory features.
- Trajectory signals: `n_sentences`, `sk_velocity_mean`, `sk_velocity_std`, `sk_acceleration_mean`, `sk_acceleration_std`, `sk_jerk_mean`, `sk_jerk_std`. SKIP when text < 120 chars or < 5 sentences (trajectory is meaningless for short input).
- Heuristic `_HEURISTIC_*` mapping → `p_llm` per CLAUDE.md naming convention: low `velocity_std` + low `jerk_std` → high `p_llm`. Hand-tuned starting points only; replaced by EBM weights once retrained.
- Cascade: `PHASE2_DETECTORS`. Compute cost `moderate` (one embedding-pass per sentence; bounded by sentence count).
- EBM features added per the Phase 4 pattern (`semantic_kinematics_p_llm`, `_missing`, `_confidence`, plus signal-specific `sk_*` features).
- All tests behind `pytest.importorskip("sentence_transformers")` so CI without the retrieval extra still runs (matches Phase 8 convention).

## Implementation tasks

### Task 10.1: Detector skeleton + protocol compliance

**Files:**
- Create: `cairn/detectors/semantic_kinematics.py`, `tests/test_detectors/test_semantic_kinematics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detectors/test_semantic_kinematics.py
import pytest


def test_semantic_kinematics_skips_short_text():
    pytest.importorskip("sentence_transformers")
    from cairn.detectors.semantic_kinematics import SemanticKinematicsDetector
    det = SemanticKinematicsDetector()
    res = det.analyze("Just one sentence.", config={})
    assert res.determination == "SKIP"
    assert res.signals.get("reason") in ("too_short", "too_few_sentences")


def test_semantic_kinematics_emits_velocity_jerk_signals():
    pytest.importorskip("sentence_transformers")
    from cairn.detectors.semantic_kinematics import SemanticKinematicsDetector
    text = (
        "The river carved its way through limestone over millennia. "
        "Geologists call this karst topography. "
        "I once visited a karst cave system in Slovenia. "
        "It smelled like wet stone and old air. "
        "Anyway, the deeper question is whether erosion is fractal at every scale. "
        "Mandelbrot would have said yes. "
        "I'm less sure. "
        "The math gets tangled when you try to bound the dimension empirically."
    )
    det = SemanticKinematicsDetector()
    res = det.analyze(text, config={})
    assert res.determination != "SKIP"
    assert "sk_velocity_std" in res.signals
    assert "sk_jerk_std" in res.signals
    assert "n_sentences" in res.signals
    assert 0.0 <= res.p_llm <= 1.0


def test_semantic_kinematics_low_variance_text_yields_higher_p_llm():
    pytest.importorskip("sentence_transformers")
    from cairn.detectors.semantic_kinematics import SemanticKinematicsDetector
    smooth = (
        "Climate change is a pressing global issue. "
        "The pressing global issue requires immediate action. "
        "Immediate action must be coordinated internationally. "
        "International coordination requires shared frameworks. "
        "Shared frameworks enable consistent policy. "
        "Consistent policy drives measurable outcomes."
    )
    jagged = (
        "Climate change is real. "
        "My dog ate a bee yesterday and I felt bad for both of them. "
        "Anyway, the IPCC report is grim. "
        "Have you ever seen a mountain at dawn? "
        "We need carbon pricing. "
        "I should call my mother."
    )
    det = SemanticKinematicsDetector()
    r_smooth = det.analyze(smooth, config={})
    r_jagged = det.analyze(jagged, config={})
    assert r_smooth.p_llm > r_jagged.p_llm
```

- [ ] **Step 2: Run, verify fails**

Run: `pytest tests/test_detectors/test_semantic_kinematics.py -xvs`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the detector**

```python
# cairn/detectors/semantic_kinematics.py
"""Semantic-kinematics detector — trajectory shape of meaning in embedding space."""
from __future__ import annotations

from cairn.contracts import LayerResult


# _HEURISTIC_ priors per CLAUDE.md naming. Hand-tuned starting points only;
# the EBM picks up real weights once Phase 4/5 retrains include the new
# semantic_kinematics_* features.
_HEURISTIC_HUMAN_VELOCITY_STD = 0.18
_HEURISTIC_HUMAN_JERK_STD = 0.20
_HEURISTIC_SCALE = 6.0
_MIN_SENTENCES = 5
_MIN_TEXT_LENGTH = 120

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _segment(text: str) -> list[str]:
    try:
        from cairn.normalizer import split_sentences  # type: ignore[attr-defined]
        out = [s.strip() for s in split_sentences(text) if s.strip()]
        if out:
            return out
    except Exception:
        pass
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _cosine_distances(vecs) -> list[float]:
    import numpy as np
    out: list[float] = []
    for i in range(len(vecs) - 1):
        a = vecs[i]
        b = vecs[i + 1]
        out.append(float(1.0 - np.dot(a, b)))
    return out


def _kinematics(distances: list[float]) -> dict[str, float]:
    import numpy as np
    if len(distances) < 3:
        return {
            "velocity_mean": float(np.mean(distances)) if distances else 0.0,
            "velocity_std": 0.0,
            "acceleration_mean": 0.0, "acceleration_std": 0.0,
            "jerk_mean": 0.0, "jerk_std": 0.0,
        }
    v = np.asarray(distances, dtype=float)
    a = np.diff(v)
    j = np.diff(a)
    return {
        "velocity_mean": float(np.mean(v)),
        "velocity_std": float(np.std(v)),
        "acceleration_mean": float(np.mean(np.abs(a))),
        "acceleration_std": float(np.std(a)),
        "jerk_mean": float(np.mean(np.abs(j))) if len(j) else 0.0,
        "jerk_std": float(np.std(j)) if len(j) else 0.0,
    }


def _heuristic_p_llm(k: dict[str, float]) -> float:
    # Low std + low jerk → high p_llm.
    vz = max(0.0, _HEURISTIC_HUMAN_VELOCITY_STD - k["velocity_std"]) / max(_HEURISTIC_HUMAN_VELOCITY_STD, 1e-6)
    jz = max(0.0, _HEURISTIC_HUMAN_JERK_STD - k["jerk_std"]) / max(_HEURISTIC_HUMAN_JERK_STD, 1e-6)
    z = (vz + jz) / 2.0
    return float(min(1.0, z * (_HEURISTIC_SCALE / 6.0)))


class SemanticKinematicsDetector:
    id = "semantic_kinematics"
    domain = "prose"
    compute_cost = "moderate"

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SemanticKinematicsDetector requires sentence-transformers. "
                "Install with: pip install 'cairn[retrieval]'"
            ) from e
        self._model = SentenceTransformer(model_name)

    def analyze(self, text: str, config: dict) -> LayerResult:
        if len(text) < _MIN_TEXT_LENGTH:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0,
                signals={"reason": "too_short"},
                determination="SKIP", attacker_tiers=["A0", "A1", "A2"],
                compute_cost=self.compute_cost, min_text_length=_MIN_TEXT_LENGTH,
            )
        sentences = _segment(text)
        if len(sentences) < _MIN_SENTENCES:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0,
                signals={"reason": "too_few_sentences", "n_sentences": len(sentences)},
                determination="SKIP", attacker_tiers=["A0", "A1", "A2"],
                compute_cost=self.compute_cost, min_text_length=_MIN_TEXT_LENGTH,
            )
        embs = self._model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        distances = _cosine_distances(embs)
        k = _kinematics(distances)
        p_llm = _heuristic_p_llm(k)
        if p_llm >= 0.75: determination = "RED"
        elif p_llm >= 0.50: determination = "AMBER"
        elif p_llm >= 0.25: determination = "YELLOW"
        else: determination = "GREEN"
        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=k["velocity_std"],
            p_llm=p_llm,
            confidence=min(0.40 + len(sentences) / 50.0, 0.85),
            signals={
                "n_sentences": len(sentences),
                "sk_velocity_mean": round(k["velocity_mean"], 4),
                "sk_velocity_std": round(k["velocity_std"], 4),
                "sk_acceleration_mean": round(k["acceleration_mean"], 4),
                "sk_acceleration_std": round(k["acceleration_std"], 4),
                "sk_jerk_mean": round(k["jerk_mean"], 4),
                "sk_jerk_std": round(k["jerk_std"], 4),
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2"],
            compute_cost=self.compute_cost,
            min_text_length=_MIN_TEXT_LENGTH,
        )

    def calibrate(self, labeled_data: list) -> None:
        # Calibration is owned by the EBM after Phase 4/5 retrain.
        pass


class _LazySemanticKinematics:
    id = "semantic_kinematics"
    domain = "prose"
    compute_cost = "moderate"
    _instance = None

    def analyze(self, text: str, config: dict) -> LayerResult:
        if self._instance is None:
            self._instance = SemanticKinematicsDetector(
                model_name=(config or {}).get("model", _DEFAULT_MODEL),
            )
        return self._instance.analyze(text, config)

    def calibrate(self, labeled_data: list) -> None:
        pass


DETECTOR = _LazySemanticKinematics()
```

- [ ] **Step 4: Run, verify passes**

Run: `pytest tests/test_detectors/test_semantic_kinematics.py -xvs`
Expected: PASS (or skip if `sentence_transformers` missing).

- [ ] **Step 5: Commit**

```bash
git add cairn/detectors/semantic_kinematics.py tests/test_detectors/test_semantic_kinematics.py
git commit -m "feat(detectors): semantic_kinematics — embedding-trajectory macro-feature"
```

### Task 10.2: Register in cascade + config

- [ ] **Step 1: Add to `PHASE2_DETECTORS`** in `cairn/cascade.py`:

```python
PHASE2_DETECTORS = (..., "semantic_kinematics")
```

(Append; preserve existing membership and ordering.)

- [ ] **Step 2: Add config block** in `configs/default.yaml`:

```yaml
detectors:
  semantic_kinematics:
    enabled: true
    model: "sentence-transformers/all-MiniLM-L6-v2"
```

- [ ] **Step 3: Update `detector_availability`** in `cairn/pipeline.py`:

```python
        if det_id in {"semantic_kinematics"}:
            row["requires"] = ["sentence_transformers"]
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                row["reason"] = "requires cairn[retrieval] (sentence-transformers)"
                out.append(row)
                continue
```

- [ ] **Step 4: Cascade smoke test**

Run: `pytest tests/test_cascade.py tests/test_pipeline.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/cascade.py cairn/pipeline.py configs/default.yaml
git commit -m "feat(cascade): register semantic_kinematics in PHASE2 with availability gating"
```

### Task 10.3: Add EBM features per the Phase 4 pattern

- [ ] **Step 1: Write the failing test** — append to `tests/test_ebm_missingness.py`:

```python
def test_assembler_emits_semantic_kinematics_features():
    a = FeatureAssembler()
    vec_absent = a.assemble([], word_count=200, domain="prose")
    assert vec_absent["semantic_kinematics_missing"] == 1.0
    assert math.isnan(vec_absent["semantic_kinematics_confidence"])

    lr = _lr(
        "semantic_kinematics", p_llm=0.7, confidence=0.55,
        n_sentences=8, sk_velocity_mean=0.18, sk_velocity_std=0.04,
        sk_acceleration_mean=0.02, sk_acceleration_std=0.01,
        sk_jerk_mean=0.01, sk_jerk_std=0.005,
    )
    vec = a.assemble([lr], word_count=200, domain="prose")
    assert vec["semantic_kinematics_missing"] == 0.0
    assert vec["semantic_kinematics_p_llm"] == pytest.approx(0.7)
    assert vec["sk_velocity_std"] == pytest.approx(0.04)
    assert vec["sk_jerk_std"] == pytest.approx(0.005)


def test_semantic_kinematics_features_in_canonical_list():
    assert "semantic_kinematics_p_llm" in ASSEMBLER_FEATURE_NAMES
    assert "semantic_kinematics_missing" in ASSEMBLER_FEATURE_NAMES
    assert "semantic_kinematics_confidence" in ASSEMBLER_FEATURE_NAMES
    assert "sk_velocity_std" in ASSEMBLER_FEATURE_NAMES
    assert "sk_jerk_std" in ASSEMBLER_FEATURE_NAMES
```

- [ ] **Step 2: Run, verify fails**

Run: `pytest tests/test_ebm_missingness.py -xvs`
Expected: FAIL.

- [ ] **Step 3: Update assembler** — extend the `vec` dict in `FeatureAssembler.assemble`:

```python
            "semantic_kinematics_p_llm": p("semantic_kinematics"),
            "semantic_kinematics_missing": missing("semantic_kinematics"),
            "semantic_kinematics_confidence": conf("semantic_kinematics"),
            "sk_velocity_mean": sig("semantic_kinematics", "sk_velocity_mean"),
            "sk_velocity_std": sig("semantic_kinematics", "sk_velocity_std"),
            "sk_acceleration_mean": sig("semantic_kinematics", "sk_acceleration_mean"),
            "sk_acceleration_std": sig("semantic_kinematics", "sk_acceleration_std"),
            "sk_jerk_mean": sig("semantic_kinematics", "sk_jerk_mean"),
            "sk_jerk_std": sig("semantic_kinematics", "sk_jerk_std"),
```

Update `ASSEMBLER_FEATURE_NAMES` to include the new keys.

- [ ] **Step 4: Run, verify passes**

Run: `pytest tests/test_ebm_missingness.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/ebm.py tests/test_ebm_missingness.py
git commit -m "feat(ebm): assembler emits semantic_kinematics_* + sk_* features"
```

### Task 10.4: Optional retrain to ingest the new features

The detector lands and the existing v8 EBM continues to work because Phase 4.4's `RuntimeError` fires only on `extra_in_artifact` (artifact-requires-feature-the-assembler-cannot-produce), not `extra_in_assembler` (assembler-emits-feature-the-artifact-doesn't-need). The new features are simply ignored by v8 until the next retrain.

**Path A (eager):** retrain to `ebm_v9.pkl` now to ingest the new features.
**Path B (default, deferred):** roll up into the next regular retrain cycle.

- [ ] **Step 1 (Path A only): Retrain**

```bash
python scripts/train_fusion.py \
  --dataset <parent>/data/calib2000_v6_train.jsonl \
  --config configs/.profile_master_minus_arxiv.yaml \
  --features-cache <parent>/data/all_features_v6.jsonl \
  --out <parent>/models/calib2000/profiles/master_minus_arxiv/ebm_v9.pkl
```

- [ ] **Step 2 (Path A only): Compare v8 vs v9 on OOD m4/raid/mage**

Promotion guard same as Phase 5:
- AUROC drop must be < 0.005
- TPR@1%FPR drop reviewed (acceptable if anti-paraphrase robustness gain documented)
- ECE/Brier should not worsen

- [ ] **Step 3 (Path A only): Promote and commit**

```bash
git add <parent>/models/calib2000/profiles/master_minus_arxiv/ebm_v9.pkl configs/.profile_master_minus_arxiv.yaml
git commit -m "chore(models): retrain EBM v9 with semantic_kinematics features"
```

### Task 10.5: Run full pytest

Run: `pytest -x -q`
Expected: ≥392 pass.

## Acceptance gate

```bash
pip install -e ".[retrieval]"
pytest tests/test_detectors/test_semantic_kinematics.py -v
pytest tests/test_cascade.py tests/test_pipeline.py tests/test_ebm_missingness.py -v
pytest -x -q
```

Subagent returns:

```text
detector_id: semantic_kinematics
embedding_model: sentence-transformers/all-MiniLM-L6-v2
cascade_phase: 2
ebm_features_added: <list>
heuristic_priors:
  human_velocity_std:
  human_jerk_std:
  scale:
retrain_path_taken: A or B
metrics_before: AUROC=, TPR@1%FPR= (Path A only)
metrics_after:  AUROC=, TPR@1%FPR= (Path A only)
artifact_promoted: true/false (Path A only)
pytest_result:
```

## Dispatch prompt

```text
Implement Phase 10 of the CAIRN Honest Calibration Story (Unified Plan). Add a macro-feature
detector at cairn/detectors/semantic_kinematics.py that fingerprints sentence-to-sentence
trajectory shape in embedding space — variance/acceleration/jerk of cosine distances between
consecutive sentence embeddings.

Theory: human cognition is episodic (random-walk in semantic space — high variance, high jerk);
LLM self-attention enforces smooth interpolation (low variance, near-zero jerk). The macro-
trajectory survives paraphrase and burstiness-prompt attacks that defeat token-level features.

10.1 Detector class implementing Detector protocol; lazy-load
     sentence-transformers/all-MiniLM-L6-v2 (already in cairn[retrieval] from Phase 8). Segment
     via cairn.normalizer if available (regex fallback otherwise), embed sentences with
     normalize_embeddings=True, compute kinematics (velocity = sentence-pair cosine distances;
     acceleration = first diff; jerk = second diff). Heuristic _HEURISTIC_* priors: low
     velocity_std + low jerk_std → high p_llm. SKIP if text < 120 chars or < 5 sentences.
10.2 Register in cairn/cascade.py PHASE2_DETECTORS; add detectors.semantic_kinematics config
     block; extend cairn/pipeline.py::detector_availability to gate on sentence_transformers.
10.3 Extend FeatureAssembler.assemble (Phase 4 pattern): semantic_kinematics_p_llm, _missing,
     _confidence, plus signal-specific sk_velocity_mean/std, sk_acceleration_mean/std,
     sk_jerk_mean/std. Update ASSEMBLER_FEATURE_NAMES.
10.4 Path B by default: do not retrain immediately — Phase 4.4's RuntimeError fires only on
     extra_in_artifact, not extra_in_assembler, so the existing v8 EBM keeps working; new
     features are ingested at the next regular retrain. Path A (retrain to v9) is optional and
     gated by the same promotion guard as Phase 5.

CRITICAL: all tests behind pytest.importorskip("sentence_transformers"). The detector is
domain="prose" (skip on prompts). Run pytest -x -q before handoff. Return the Phase 10 acceptance
block.
```

---

# Integration phase — Final audit pass

## Subagent mission

Verify all phases compose cleanly and the product tells the same truth everywhere: CLI, GUI, API, sidecar, reports, history, monitoring, and benchmark docs.

## Files

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

Every report exposes:

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

Override reports visibly caveat: "coverage guarantee does not apply to this override path."

Heuristic mode is impossible to mistake for calibrated mode.

API `/health` distinguishes: `heuristic`, `fusion-only`, `calibrated`, `calibrated-router-conditional`.

Docs say which profile is recommended for: prose, prompt detection, short-form/OOD, batch similarity, heuristic screening only.

**Privacy/history note:** purging the privacy vault does not necessarily purge SQLite history text if history retention remains enabled. Document this and add a `cairn history purge` shortcut, or at least an explicit warning in the configuration docs. File an issue if not closed inline.

## Acceptance gate

```bash
pytest -x -q
```

Smoke each runtime: `cairn gui --profile <calibrated>`, `cairn serve --profile <calibrated>`, `cairn-sidecar`.

Subagent returns:

```text
runtime_surfaces_checked:
schema_fields_present:
docs_updated:
remaining_known_gaps:
pytest_result:
```

## Dispatch prompt

```text
Run the integration audit pass for the CAIRN Honest Calibration Story (Unified Plan). Verify the
fields fusion_mode, calibration_status, conformal_set, conformal_alpha, abstention_reason,
coverage_guarantee_applies, detector_errors, warnings appear in every report surface
(cairn/report.py, cairn/gui/static/index.html cairnApi, cairn/api.py, cairn/sidecar.py). Update
README, docs/calibration.md, docs/benchmarks/README.md, docs/configuration.md, docs/deployment.md
so heuristic vs calibrated vs calibrated-router-conditional are unambiguous. Add API /health
distinguishing those four calibration_status values. File an issue (or close inline) for the
privacy-vault vs history-text gap. Smoke each runtime. Run pytest -x -q. Return the integration
acceptance block.
```

---

# Deferred backlog

Should not block the main calibration sprint unless capacity is generous. Each item cross-references the audit section that motivated it (see Appendix A).

## Neyman-Pearson threshold calibration (Audit §8)

`scripts/calibrate_np_threshold.py` chooses the operating threshold that maximizes TPR under a specified FPR ceiling. Args: `--dataset`, `--config`, `--fpr-ceiling 0.01`, `--out-threshold-config`. Pair with Chow's rule (1970) for abstention — abstain when expected misclassification cost exceeds abstention cost. Maps onto the conformal-set-non-singleton check with explicit cost semantics.

## Adaptive Conformal Inference (Audit §1c)

`cairn/fusion/aci.py::AdaptiveConformal` wrapping `ConformalWrapper` with Gibbs & Candès 2021 online α-adjustment: `α_{t+1} = α_t + γ(target − err_t)`. Asymptotic coverage at `target_alpha` under arbitrary distribution shift. Critical for contributor-pool QA where the human writing distribution drifts across cohorts.

## Post-hoc isotonic calibration on EBM (Audit §5)

Stack `sklearn.calibration.CalibratedClassifierCV(method="isotonic", cv="prefit")` on the EBM. Expose both `p_llm_raw` (for feature explanations / interpretability) and `p_llm_calibrated` (for the decision and conformal layer). Tighter conformal sets when EBM scores drift on imbalanced or shifted test data — fewer false UNCERTAINs. References: Niculescu-Mizil & Caruana ICML 2005; Kull, Silva Filho & Flach NeurIPS 2017.

## PSI drift metric + label-shift correction (Audit §13)

Add Population Stability Index alongside KL divergence in monitoring. Trigger: `PSI > 0.2` = significant drift. Pair with Lipton et al. 2018 "Detecting and Correcting for Label Shift with Black Box Predictors" — closed-form class-prior-shift correction using only the existing classifier. Especially relevant when contributor-pool LLM/human ratio drifts across projects.

## Contributor-graph stylometric collusion (Audit §11 second half)

Authorship-verification framing rather than pure similarity. PAN shared task literature (Bevendorff et al. on cross-domain AV; Tyo et al. 2022 "On the State of the Art in Authorship Attribution and Authorship Verification"); Boenninghoff et al. 2019 Siamese-network AV. Two contributors whose stylometric embeddings cluster suspiciously closely → likely shared LLM workflow. Reuses the `cairn/index/` module built in Phase 8.

## Secondary external benchmarks (Audit §12 secondary)

Layered on top of the Phase 9 RAID workflow:
- HC3 (Guo et al. 2023) — easy baseline.
- M4GT-Bench — multilingual.
- ASAP 2.0 essays — FPR sanity check on student writing (target FPR ≤ 0.8% achievable per public results).

## Ground-truth behavioural fixtures (Audit §14)

`tests/fixtures/ground_truth_50.jsonl` with a regression test asserting minimum behaviour at controlled FPR — replaces tautological constants-checks like `test_high_p_llm_gives_red` that just re-test `_p_llm_to_label(0.82) == "RED"` against the configured threshold. Infrastructure already exists via `run_eval`.

## ONNX export for sidecar portability (Audit §6 future-proof)

Export the EBM via `interpret-ml`'s experimental ONNX support so the inference artifact does not depend on Python at all. Lets the Tauri sidecar move to Rust eventually. Long-tail follow-on to the Phase 7 skops migration.

## Bootstrap CI from EBM bagged ensemble (Audit §3 alternative)

Phase 5 derives the CI from the conformal threshold (the principled choice when conformal is calibrated). An alternative — bootstrap CI on the EBM's internal bagged GAM ensemble — gives a model-internal uncertainty signal even when no conformal artifact is loaded. ~10 LOC. File only if the conformal-derived CI proves too coarse in production.

---

# Appendix A: Audit source

This is the source audit that motivated the plan. Preserved verbatim (with markdown-link artifacts cleaned up — the original used `[name.py](http://name.py)` patterns that don't resolve) for citations, rationale, and concrete code suggestions. Each phase in the body of the plan tracks back to one or more sections here.

## Audit-to-phase mapping

| Audit § | Topic | Mapped to |
|---------|-------|-----------|
| 1(a) | Class-conditional Mondrian conformal | Phase 2 |
| 1(b) | Router-class stratification (Mondrian by router) | Phase 3 |
| 1(c) | Adaptive Conformal Inference (drift) | Deferred backlog |
| 1(d) | Empty conformal set is the strongest signal | Phase 1.2 |
| 2(a) | `<detector>_missing` indicators | Phase 4.1 |
| 2(b) | 0.5 imputation → column means stored on artifact | Phase 4.3 |
| 2(c) | Schema-mismatch warning → `RuntimeError` | Phase 4.4 |
| 3 | Hardcoded ±0.10 CI → conformal-derived / wide | Phase 5.1 |
| 4 | Hold-out by LLM family | Phase 6.1 |
| 5 | Post-hoc isotonic calibration on EBM | Deferred backlog |
| 6 | Pickle → `skops` + provenance metadata | Phase 7 |
| 7 | Override coverage-guarantee accounting + MIXED threshold from config | Phase 1.3, 1.4 |
| 8 | Neyman-Pearson thresholds | Deferred backlog |
| 9 | `<detector>_confidence` features | Phase 4.2 |
| 10 | Monotonic constraints in EBM | Phase 5.2 |
| 11 cross-sim | Retrieval-based `cross_similarity` (Krishna et al. 2023) | Phase 8 |
| 11 contrib | Stylometric collusion authorship verification | Deferred backlog |
| 12 RAID | RAID @ TPR/FPR≤1% in CI | Phase 9 |
| 12 per-source | per-source breakdown in eval report | Phase 6.2, 6.3 |
| 12 secondary | HC3 / M4GT-Bench / ASAP 2.0 | Deferred backlog |
| 13 PSI | Population Stability Index drift metric | Deferred backlog |
| 13 label-shift | Lipton et al. 2018 label-shift correction | Deferred backlog |
| 14 | Property-based ground-truth fixtures | Deferred backlog |
| Priority 1–7 | Sprint priority ordering | Phases 1–9 (waves 1–5) |
| *(out of audit)* | Semantic kinematics detector — user-originated post-audit research suggestion: embedding-trajectory macro-feature (variance/acceleration/jerk of sentence-pair cosine distances) that survives paraphrase and burstiness-prompt attacks | **Phase 10** |

## Audit author resolution notes (added during plan composition)

- **Binoculars verification (closing question of the audit):** `cairn/detectors/contrastive_lm.py` was inspected during plan composition. Confirmed: it is a real two-model Binoculars implementation (gpt2 vs gpt2-medium, mean cross-entropy ratio in bits/token, z-score against a fitted human baseline). Not a perplexity proxy. The audit author's verification ask is closed.
- **`beet → cairn` rename note (audit preface):** flagged for the integration phase. The dual `_cairn_feature_names` / `_beet_feature_names` artifact-loading guard remains correct; the README and `pyproject.toml` package-name catch-up belongs in the integration audit pass.
- **Profile config naming gotcha:** the recommended profiles are dot-prefixed: `configs/.profile_master_minus_arxiv.yaml` (prose) and `configs/.profile_prompts_only.yaml` (prompts). All audit code-snippet references to file paths (`fusion.py`, etc.) were intended literally, not as hidden-file paths.

## Audit text (verbatim, with markdown-link artifacts cleaned)

OK — code is in. Real progress since the README pass: this is no longer a midpoint heuristic, this is honest split-conformal. Several of my prior recommendations are already done (the assembler has `binoculars_p_llm`, length-bucket indicators, `log_word_count`). I'll skip those and focus on what's still open, with code references and citations.

Side note before the audit: there's an active rename in flight (`beet` → `cairn`). The artifact loading code in `EBMFusion.__init__` checks both `_cairn_feature_names` and `_beet_feature_names`, which is right; just make sure the README and `pyproject.toml` package name catch up before you ship a release, otherwise `pip install beet` and `import cairn` is a nasty footgun for users.

---

### A.1 Conformal: the math is correct, but it's not Mondrian

`fusion/conformal.py` implements vanilla split-conformal correctly — `nonconformity = 1 - p(true_class)`, finite-sample quantile `ceil((n+1)(1-α))/n`, `np.quantile(..., method="higher")` for the conservative interpolation. Good.

The remaining gap is that this is **marginal** coverage. Two issues fall out of that:

**(a) Class-conditional miscalibration.** Detector score distributions are asymmetric: a high `p_llm` from your fingerprint detector is more reliable than a low `p_llm` (low scores are mostly "I didn't see the pattern" — uninformative). Marginal CP averages over that asymmetry. The fix is **class-conditional / Mondrian conformal** — calibrate separately within each label, take the per-class quantile. Vovk (2003), and Romano, Sesia & Candès "Classification with Valid and Adaptive Coverage" (NeurIPS 2020) for the modern framing (APS/RAPS). Concrete asks:

```python
def calibrate(self, scores, labels):
    ...
    # NEW: per-class quantiles
    self._threshold_per_class = {}
    for cls in (0, 1):
        mask = labels == cls
        if mask.sum() < 10:
            raise ValueError(f"too few class-{cls} calibration points")
        nc_cls = (1.0 - scores[mask]) if cls == 1 else scores[mask]
        q = math.ceil((mask.sum() + 1) * (1.0 - self._alpha)) / mask.sum()
        self._threshold_per_class[cls] = float(np.quantile(nc_cls, min(q, 1.0), method="higher"))
```

This gives you class-conditional coverage at level `1-α` for both human and LLM submissions independently. For your QA setting where the FPR ceiling matters more than overall coverage, this is the more defensible guarantee.

**(b) Stratification by router class.** The score distribution for `domain="prompt"` vs `domain="prose"` vs `domain="mixed"` is very different (different detector ensembles fire). A single global threshold under-covers the rare router class. **Mondrian by router output** — pass the router class into `calibrate()` and `predict_set()`, store one threshold per class, fall back to global only if a class has too few cal points. Same Vovk reference.

**(c) Drift over time.** Adaptive Conformal Inference (Gibbs & Candès 2021, "Adaptive Conformal Inference Under Distribution Shift", NeurIPS) gives you online α-adjustment: after each labeled datapoint, update `α_t = α_{t-1} + γ(α − err_t)`. Trivial to add as a wrapper around the existing class. Critical for the contributor-pool QA setting because the human writing distribution drifts across cohorts.

**(d) The empty-set case is silently wrong in the decision engine.** Your conformal docstring honestly notes that `predict_set` can return `[]` when α is high. But in `decision.py` the abstention check is `len(fusion.conformal_set) >= 2`. If `conformal_set` is `[]`, that check fails → falls through to `_p_llm_to_label(fusion.p_llm)`. An empty conformal set is the **strongest** signal you can produce ("neither hypothesis is conformal under this α") and you're treating it as the **weakest** (use the bare threshold). Fix:

```python
if self._abstain and fusion.conformal_set is not None:
    if len(fusion.conformal_set) == 0:
        label = "UNCERTAIN"
        abstention_reason = "conformal_empty"
        reason = "Conformal set is empty — both hypotheses violate α-coverage; the input is out-of-distribution relative to calibration."
    elif len(fusion.conformal_set) >= 2:
        ...
```

### A.2 EBM fusion: missingness semantics are conflating three different things

`FeatureAssembler.assemble` returns `_NAN` for missing detectors. Then in `_ebm_fuse`:

```python
X = np.where(np.isnan(X), 0.5, X)
```

This is the single biggest accuracy lever I see. Three problems:

**(a) "Missing" can mean three different things, and you're collapsing all three.** A detector returning NaN can be: SKIP (intentional, e.g., batch-only), errored (the failure was logged in `Determination.detector_errors`), or absent from the artifact's feature list. The EBM should be able to distinguish them — at minimum, add a binary `<detector>_missing` indicator alongside each `<detector>_p_llm`. This is the Chen & Guestrin (XGBoost, KDD 2016) recommendation: tree-based learners handle missingness much better with explicit indicators than mean imputation. EBM is tree-based; same logic applies.

**(b) 0.5 is not a defensible imputation value.** It's "max entropy" only if you assume a uniform prior over labels, which is wrong if your dataset has class imbalance. The right defaults are either:
- Column means from the training set (store them in the EBM artifact and re-use at inference).
- The empirical marginal label frequency from the calibration set (e.g., 0.43 if 43% of your labeled data was LLM).
- For the SKIP-on-batch detectors (`cross_similarity`, `contributor_graph`), drop the feature column entirely from single-text analyze and have a separate EBM for batch.

**(c) The schema-mismatch warning needs to be a hard error.** `extra_in_artifact` (the artifact wants features the assembler doesn't produce) currently logs a warning, but those features will get NaN→0.5 imputed silently at inference, so the model is using the marginal mean for what should be informative features. That's a deployment bug waiting to happen. Promote to `RuntimeError`:

```python
if extra_in_artifact:
    raise RuntimeError(
        f"EBM artifact requires features the current assembler cannot produce: "
        f"{sorted(extra_in_artifact)}. Either retrain the EBM or update the assembler."
    )
```

### A.3 The hardcoded ±0.10 CI is now misleading

```python
ci = (max(0.0, p_llm - 0.10), min(1.0, p_llm + 0.10))
```

This appears in both `_ebm_fuse` and indirectly in `_p_llm_to_labels`. With the conformal wrapper in place, you have a real source of uncertainty — use it.

The principled options:

- **Bootstrap CI on the EBM's bagged ensemble.** EBM internally trains an ensemble of GAMs; you can extract per-bag predictions and compute an empirical 90% interval. ~10 lines of code.
- **Conformal-derived CI.** When the conformal wrapper is calibrated, derive the CI from the threshold: `ci = (max(0, p_llm - q_hat), min(1, p_llm + q_hat))`. Crude but honest.
- **Width should signal calibration status.** When `fusion_mode == "naive"` (no trained EBM), the CI should be `(0, 1)` or at least something visibly wide. Currently a naive fusion still gets a tight ±spread/2 CI even when detectors agree on a wrong answer (agreement ≠ correctness — every detector ensemble paper since Niculescu-Mizil & Caruana 2005 makes this point).

Right now an honest-and-uncalibrated naive fusion looks indistinguishable on the wire from a calibrated EBM fusion. The README has the disclaimer banner; the data structure should match.

### A.4 Train/test split doesn't hold out by LLM family — known generalization trap

`scripts/train_fusion.py` does `train_test_split(..., stratify=y, ...)` — stratifying by binary label only. The detector literature has settled on this being the wrong split for honest accuracy reporting:

- Pu et al. (2022) "On the Zero-Shot Generalization of Machine-Generated Text Detectors" — detectors that score 95%+ on in-family hold-outs collapse to 60–70% on held-out generators. RAID (Dugan et al. 2024) reproduces this systematically.
- The fix is **leave-one-LLM-out CV.** Train on samples from {GPT-4, Claude, Llama-3, Gemini}, test on Mistral. Rotate. Report worst-case accuracy.

Concrete diff:

```python
ap.add_argument("--holdout-source", default=None,
                help="LLM source/generator to hold out entirely from training (zero-shot eval).")
...
if args.holdout_source:
    train = [s for s in samples if s.source != args.holdout_source]
    test = [s for s in samples if s.source == args.holdout_source]
else:
    # current behavior
    train, test = train_test_split(...)
```

If `EvalSample` already carries `source` (your `run_eval` shows it does), this is a one-day change.

Pair this with a **per-LLM-family breakdown** in `eval_report_to_dict`. You're presumably already collecting it; surface it in the `monitoring` tab. The single-number "AUROC" is the headline that hides the failure mode.

### A.5 EBM's `predict_proba` isn't post-hoc calibrated

EBM is well-calibrated by construction relative to logistic, but on imbalanced or shifted test data it still benefits from a calibration wrapper. Niculescu-Mizil & Caruana (ICML 2005) and Kull, Silva Filho & Flach (NeurIPS 2017, "Beyond temperature scaling") are the references. Concrete:

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(estimator=ebm, method="isotonic", cv="prefit")
calibrated.fit(X_calibration, y_calibration)
```

Yes, this stacks a calibrator on top of an interpretable model and you lose some of the EBM's "raw" interpretability. The fix is to keep both: report `p_llm_raw` from the EBM (for explanations) and `p_llm_calibrated` for the decision. Your `feature_contributions` stay tied to raw; the threshold is applied to calibrated.

Why this matters for your specific use case: the conformal wrapper's coverage guarantee is **conditional on score exchangeability**. If your EBM is mis-calibrated, the conformal wrapper still gives valid coverage (CP is calibration-free), but the **prediction set sizes will be wider than necessary** — you'll abstain more than you should. Calibrating the underlying scores → tighter sets → fewer false UNCERTAINs → more useful decisions.

### A.6 Pickle is a footgun for shipping

```python
def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
```

Pickle deserialization is RCE-by-design. For a Tauri-shipped desktop app where the model artifact may live in user-writable paths, this is a real concern (model substitution attack). Options ranked by effort:

- **Cheapest:** HMAC the pickle with a key embedded in the binary. Verify on load. Not a full mitigation but raises the bar.
- **Right answer:** `skops` (https://skops.readthedocs.io/) — an sklearn-aware safe serialization format. Untrusted-load whitelist-only. Designed for exactly this case.
- **Future-proof:** ONNX export of the EBM (`interpret-ml` has experimental support) so the inference artifact doesn't depend on Python at all. Lets you move the sidecar to Rust eventually.

Also embed training metadata in the artifact: dataset hash, training timestamp, label balance, validation AUROC. Currently `_cairn_feature_names` is the only metadata — when the model misbehaves in production, no provenance.

### A.7 The CRITICAL-preamble override silently breaks the conformal coverage guarantee

`decision.py` lines ~16–22:

```python
if result.layer_id == "preamble" and result.signals.get("severity") == "CRITICAL":
    return Determination(label="RED", ...)
```

This fires before the abstention/conformal check. Conformal coverage is `P(true_label ∈ prediction_set) ≥ 1−α` over the calibration distribution. The moment you override the prediction set with a single-detector heuristic, that probability statement no longer holds for the override path. The README's "calibrated probability" UX claim leaks here.

Two principled fixes:

- **Treat preamble CRITICAL as a feature, not an override.** It already is in the EBM (`preamble_severity` and `preamble_p_llm`). Trust the trained fusion to learn that CRITICAL preamble → very high p_llm. Drop the override. This makes the conformal guarantee honest end-to-end.
- **Keep the override but mark the determination.** Add `coverage_guarantee_applies: bool` to `Determination`. Set False when an override fires. The UI can then show "RED (override — calibrated probability not applicable)" vs "RED (P=0.91, conformal set: {LLM})". Honesty preserved.

The same applies to the MIXED override at `mixed_probability >= 0.60`. The 0.60 threshold is also hardcoded — should be in `decision.mixed_threshold`.

### A.8 Neyman-Pearson framing for the thresholds

Your decision thresholds are configurable but currently set per-profile to round numbers (0.75 / 0.50 / 0.25). For a QA system where false accusation cost ≫ missed AI cost, the principled choice is **Neyman-Pearson classification** (Tong, JMLR 2013): set the threshold to the value that maximizes TPR subject to a **hard FPR ceiling** on the calibration set.

Implementation: a one-shot calibration script that takes a labeled set + an FPR ceiling and outputs the right threshold:

```python
def np_threshold(scores, labels, fpr_ceiling=0.01):
    """Find the lowest threshold t such that FPR(t) <= fpr_ceiling."""
    human_scores = scores[labels == 0]
    # Quantile of human scores that gives the desired FPR.
    return float(np.quantile(human_scores, 1.0 - fpr_ceiling))
```

For BEET-style settings, FPR ≤ 1% is the It's-AI / MGTD top-of-leaderboard regime; ASAP 2.0 student essays show 0.8% FPR is achievable. Make this a calibration step alongside `calibrate_conformal.py` rather than a config knob. The MGTD benchmark referenced earlier publicly tracks detectors at this operating point.

Pair with **Chow's rule (1970) for abstention**: abstain when the expected misclassification cost exceeds the abstention cost. Maps cleanly onto your conformal-set-non-singleton check, but with explicit cost semantics.

### A.9 Detector confidence isn't a feature in the EBM

Looking at the assembler: every detector contributes `<detector>_p_llm` and signal-specific features, but **no detector exports its `confidence` to the EBM**. Confidence is used in naive fusion (as a weight) but discarded under EBM mode.

This is a missed signal. Mixture-of-experts literature (Jacobs et al. 1991; Shazeer et al. 2017 for the modern revival) — gating by per-expert confidence is consistently better than fixed-weight averaging. Concrete: add `<detector>_confidence` for every detector to the assembler. Let the EBM learn pairwise interactions between confidence and `p_llm`. With `interactions=15` budget, those interactions will get picked up automatically.

Single-line addition per detector in the assembler:

```python
def conf(layer_id):
    return by_id[layer_id].confidence if layer_id in by_id else _NAN
...
"preamble_confidence": conf("preamble"),
"surprisal_confidence": conf("surprisal_dynamics"),
# ... etc
```

### A.10 Monotonic constraints in the EBM

`interpret-ml` supports `monotone_constraints`. For interpretability and to prevent perverse learned relationships on small datasets, enforce:

- `preamble_severity ↑ ⇒ p_llm ↑` (always)
- `fingerprint_hits_per_1000 ↑ ⇒ p_llm ↑` (always)
- `nssi_n_signals_active ↑ ⇒ p_llm ↑` (always)

For features that are inherently bidirectional (`surprisal_diversity`, `binoculars_ratio`), leave unconstrained.

This is a free win for stakeholder defensibility — you can write in the design doc that "the model is monotonically increasing in every fingerprint-style signal," which closes off "the EBM learned that more fingerprints → less LLM" failure modes that real EBMs occasionally produce on small data.

Reference: Gupta et al. 2016 "Monotonic Calibrated Interpolated Look-Up Tables" for the underlying theory; EBM's monotonic constraints implement it for tree-based shape functions.

### A.11 Cross-similarity & contributor_graph: the literature is more specific than the README implies

These are still listed as batch-only. Two pointers that pin down the implementation:

**Cross-similarity should be retrieval, not pairwise.** Krishna et al. (2023) "Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense" (NeurIPS) is the canonical method: maintain a vector index of suspected machine outputs (sentence-transformer embeddings), query at inference, flag near-neighbors. Pairwise within a batch misses inter-batch and historical contamination — exactly the case in QA where one fraudulent contributor recycles outputs across batches. Use FAISS or LanceDB; sub-millisecond after build. Persist the index in the SQLite store you already have.

**Contributor graph should be authorship verification, not just similarity.** PAN shared task literature is the right framing — Bevendorff et al. on cross-domain authorship verification, and Tyo et al. (2022) "On the State of the Art in Authorship Attribution and Authorship Verification" for a recent benchmark. The signal you actually want at the contributor level is **stylometric collusion**: two contributors whose embeddings cluster suspiciously closely are likely sharing an LLM workflow. Boenninghoff et al. (2019) on Siamese networks for AV is the practical method.

For both, the architectural change is small: add a `cairn/index/` module that wraps the sentence-transformer + ANN index, persists alongside the SQLite history, and is queried by the two batch-only detectors.

### A.12 Evaluation: get RAID into CI

You have `run_eval` with metrics + per-tier breakdown — good. Two upgrades:

- **External benchmark in CI.** RAID (Dugan et al. ACL 2024, https://raid-bench.xyz) is the most adversarially robust public detector benchmark, includes paraphrase attacks. A single-threshold TPR-at-FPR-1% number against RAID gives you the headline metric you currently can't claim. ~6h of work to pull a stable subset, run once per release tag in `.github/workflows`.
- **Per-LLM-family breakdown surfaced in monitoring.** Your `EvalSample` has `source` and `attack_name`. Add slicing in `eval_report_to_dict` so the UI shows `auroc_by_source` and `auroc_by_attack_category`. The per-family number is what reveals OOD failure modes the headline AUROC hides.

Secondary benchmarks worth pinning in the test plan: HC3 (Guo et al. 2023) for the easy baseline, M4GT-Bench for multilingual, ASAP 2.0 essays for FPR sanity check on student writing.

### A.13 Drift monitor: KL is fine, PSI is more standard

The README says drift uses KL divergence. Two adjustments backed by industry MLOps literature:

- **Population Stability Index** is the de facto standard in production drift monitoring (originated in credit risk; widely adopted in the MLOps community). More stable than KL when bins are sparse, has well-known thresholds (PSI > 0.2 → significant drift). Add as a parallel metric.
- **Lipton et al. 2018 "Detecting and Correcting for Label Shift with Black Box Predictors"** gives you something neither KL nor PSI gives: a closed-form correction for class-prior shift using only the existing classifier. If your contributor pool's LLM/human ratio drifts, you can importance-weight the calibration without retraining. Especially relevant for your QA setting where the base rate genuinely shifts across projects.

### A.14 Tests that aren't really testing what their names say

Quick read of `tests/test_decision.py`:

```python
def test_high_p_llm_gives_red(engine):
    fusion = FusionResult(p_llm=0.82, ...)
    assert engine.decide(fusion, layer_results=[]).label == "RED"
```

This tests `_p_llm_to_label(0.82) == "RED"`, which is true by construction (config: `red_threshold=0.75`). It's a constants-check, not a test of the system's behavior. Useful as a regression guard, but the README's note ("regression tests encode the heuristic thresholds, not ground truth") applies here too.

The honest fix is **property-based tests against ground-truth fixtures**: ship a small JSONL of ~50 hand-curated submissions with known labels, assert that the pipeline gets ≥X% right at FPR ≤ Y. That's the kind of test that fails meaningfully when someone breaks the fusion. The infrastructure is already there via `run_eval`.

## Priority ranking for sprint planning

If I were Dallas and had two weeks:

1. **Mondrian conformal + class-conditional thresholds.** Highest-leverage statistical fix; ~50 LOC change to `conformal.py` plus router-class plumbing through `_maybe_conformal`. Closes the biggest gap between the "calibrated" UX claim and the actual guarantee. → Phases 2 and 3.
2. **Empty-set and override coverage-guarantee accounting in `decision.py`.** Honesty fix; small code, big trust win. → Phase 1.
3. **Add `<detector>_missing` indicators and `<detector>_confidence` to the assembler; retrain.** Likely worth 2–4 AUROC points on RAID-class benchmarks. → Phase 4 (4.1, 4.2, 4.5).
4. **Hold-out by LLM family in `train_fusion.py`.** Free realism check on every retrain. → Phase 6.
5. **Wire RAID into CI as a single-number TPR@FPR≤1%.** First external benchmark you can cite without a caveat. → Phase 9.
6. **`skops` for model serialization.** Security debt that grows the longer it's there. → Phase 7.
7. **Retrieval-based `cross_similarity`** (Krishna et al. 2023). Biggest upgrade to the batch-only detector tier; pairs with the index you'd build for #5. → Phase 8.

Items 1–5 turn the README's honest "what's not really calibrated yet" disclaimers into honest measured numbers, in roughly that order of impact.

---

# Highest-risk corrections (carried forward)

- **Phase 3 synthetic router test:** synthetic data must include both labels in each router class. A test that expects `threshold_for(1, "prose")` to differ when there are no LLM prose samples will fail for the wrong reason.
- **Phase 9 workflow profile:** must be `configs/.profile_master_minus_arxiv.yaml` (calibrated, dot-prefixed) — NOT `configs/default.yaml` (heuristic).
- **Phase 8 retrieval tests:** must be behind `pytest.importorskip("sentence_transformers")` + `pytest.importorskip("faiss")` so embedding models are NOT silently downloaded during normal CI runs.
- **Privacy/history interaction:** purging the privacy vault does not necessarily purge SQLite history text if history retention remains enabled. File at integration time; not part of the sprint.

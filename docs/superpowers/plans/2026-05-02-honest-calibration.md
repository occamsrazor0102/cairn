# Calibration Honesty Implementation Plan

> **⚠ SUPERSEDED — see [`2026-05-02-honest-calibration-unified.md`](2026-05-02-honest-calibration-unified.md)** for the canonical merged plan (dispatch structure + concrete TDD code, ten phases plus integration). This document is preserved for snippet provenance only.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CAIRN's "calibrated probability" UX claim honest end-to-end: replace marginal split-conformal with class- and router-stratified Mondrian conformal, fix the empty-set / override holes in the decision engine, give the EBM honest missingness and confidence features, hold out by LLM family at training time, wire RAID into CI, replace pickle with skops + metadata, and replace pairwise word-shingle cross-similarity with a retrieval index.

**Architecture:** Each phase produces a ship-ready milestone — pytest passes, the conformal/EBM artifacts are forward-compatible, and the README's "calibrated" banner gets one more measured number behind it. Phases A-B (week 1) are pure code changes — no retraining required. Phase C (mid week 1) bumps the EBM artifact schema and requires one retrain pass. Phase D-E (end week 1, week 2) extend training and CI. Phases F-G (week 2) are infrastructure: secure serialization and retrieval-based cross-submission detection.

**Tech Stack:** Python 3.11+, `interpret-ml` EBM, `numpy`, `scikit-learn` (for `CalibratedClassifierCV`, optional), `skops` (Phase F), `sentence-transformers` + FAISS (Phase G, optional behind `cairn[retrieval]` extra), GitHub Actions (Phase E). Existing `pytest` suite (~340 tests).

**Stop-here checkpoints:**
- After **Phase B**: shippable as "honest empty-set + override accounting" — no retrain, ~50 LOC.
- After **Phase C**: shippable as "missingness-aware EBM" — requires one retrain.
- After **Phase D**: shippable as "leave-one-LLM-out reporting." Items A-D close the audit's #1-4.
- After **Phase E**: first external benchmark in CI.
- After **Phase G**: full audit priority list 1-7.

**Out of scope (future plans):** Audit items #8 (Neyman-Pearson framing), #9 absorbed into Phase C, #10 absorbed into Phase C, #11's contributor-graph half (Phase G covers cross-similarity only), #12's secondary HC3/M4GT/ASAP benchmarks, #13 (PSI drift), #14 (property-based decision tests).

---

## File Structure

**New files:**
- `cairn/fusion/aci.py` — Adaptive Conformal Inference wrapper (Phase A.6)
- `tests/test_fusion_conformal_mondrian.py` — Mondrian conformal tests (Phase A)
- `tests/test_decision_coverage.py` — empty-set + override coverage tests (Phase B)
- `tests/test_ebm_missingness.py` — `_missing` / `_confidence` feature tests (Phase C)
- `tests/test_train_fusion_holdout.py` — leave-one-LLM-out CV test (Phase D)
- `scripts/run_raid_eval.py` — RAID subset eval runner (Phase E)
- `tests/fixtures/raid_mini.jsonl` — 50-row RAID smoke fixture (Phase E)
- `.github/workflows/raid-bench.yml` — CI workflow (Phase E)
- `cairn/fusion/serialization.py` — skops save/load + HMAC + metadata (Phase F)
- `cairn/index/__init__.py` — package marker (Phase G)
- `cairn/index/retrieval.py` — sentence-transformer + FAISS wrapper (Phase G)
- `cairn/index/store.py` — SQLite-backed persistent index (Phase G)
- `tests/test_index_retrieval.py` — retrieval/store tests (Phase G)

**Modified files:**
- `cairn/fusion/conformal.py` — Mondrian per-class + per-router-class thresholds (Phase A)
- `cairn/fusion/ebm.py` — assembler `_missing`/`_confidence`, EBM column-mean imputation, hard-error on schema mismatch, conformal-derived CI, monotone constraints, domain → conformal plumb (Phases A, C)
- `cairn/fusion/training.py` — column means stored on artifact, monotone constraints, training metadata, skops save (Phases C, F)
- `cairn/contracts.py` — add `coverage_guarantee_applies` to `Determination`/`FusionResult` (Phase B)
- `cairn/decision.py` — empty-set abstention, coverage flag on overrides, MIXED threshold from config (Phase B)
- `cairn/pipeline.py` — pass `RouterDecision.domain` into fusion → conformal, retrieval index wiring (Phases A, B, G)
- `scripts/calibrate_conformal.py` — labels + domain pass-through, Mondrian dispatch (Phase A)
- `scripts/train_fusion.py` — `--holdout-source` flag, skops save with metadata (Phases D, F)
- `cairn/evaluation/runner.py` — per-source breakdown in `EvalReport.per_source` (Phase D)
- `cairn/evaluation/metrics.py` — already has `tpr_at_fpr`; add `tpr_at_fpr_ceiling` helper for Phase E
- `cairn/detectors/cross_similarity.py` — query retrieval index instead of pairwise Jaccard (Phase G)
- `pyproject.toml` — add `cairn[retrieval]` extra (Phase G), add `skops` to fusion extras (Phase F)
- `configs/default.yaml` — confirm `decision.mixed.min_mixed_probability` is read (audit's hardcode complaint; the config knob already exists, but `decision.py:28` ignored it)

---

## Phase A — Mondrian Conformal (audit #1)

Vanilla split-conformal gives marginal coverage; the audit asks for class-conditional + router-stratified ("Mondrian") coverage so per-class FPR ceilings hold even when class proportions are imbalanced. We add two parallel threshold maps: `_threshold_per_class` (always populated when calibrated) and `_threshold_per_router_class` (populated only when at least 50 calibration points fall in each router bucket; otherwise that bucket falls back to the global per-class threshold). The artifact schema bumps to `binary-mondrian-conformal-v2`; the loader stays backward-compatible with `v1` artifacts.

### Task A1: Add per-class thresholds to ConformalWrapper

**Files:**
- Modify: `cairn/fusion/conformal.py:36-131`
- Test: `tests/test_fusion_conformal_mondrian.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fusion_conformal_mondrian.py
import math
import numpy as np
import pytest

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
    assert all(0.0 <= t <= 1.0 for t in w.threshold_per_class.values())


def test_per_class_calibration_rejects_class_with_too_few_points():
    w = ConformalWrapper(alpha=0.1)
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.beta(2, 8, 200), rng.beta(8, 2, 5)])
    labels = np.concatenate([np.zeros(200, dtype=int), np.ones(5, dtype=int)])
    with pytest.raises(ValueError, match="too few class-1 calibration points"):
        w.calibrate(scores, labels)
```

Run: `pytest tests/test_fusion_conformal_mondrian.py::test_calibrate_populates_per_class_thresholds -xvs`
Expected: FAIL with `AttributeError: 'ConformalWrapper' object has no attribute 'threshold_per_class'`

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_fusion_conformal_mondrian.py -xvs`
Expected: FAIL with the AttributeError above.

- [ ] **Step 3: Implement per-class thresholds in `ConformalWrapper.calibrate`**

Replace `__init__` and `calibrate` in `cairn/fusion/conformal.py`, and add a private `_quantile_for` helper plus the per-class property:

```python
class ConformalWrapper:
    """Honest split-conformal for the binary LLM/human decision (Mondrian-capable)."""

    _MIN_PER_CLASS = 10
    _MIN_PER_ROUTER_CLASS = 50

    def __init__(self, alpha: float = 0.05):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._alpha = alpha
        self._threshold: float | None = None
        self._threshold_per_class: dict[int, float] | None = None
        self._threshold_per_router: dict[str, dict[int, float]] | None = None
        self._n_calibration: int = 0
        self._n_per_class: dict[int, int] | None = None
        self._n_per_router: dict[str, int] | None = None

    @property
    def threshold_per_class(self) -> dict[int, float] | None:
        return self._threshold_per_class

    @property
    def threshold_per_router(self) -> dict[str, dict[int, float]] | None:
        return self._threshold_per_router

    def _quantile_for(self, nonconformity: np.ndarray) -> float:
        n = len(nonconformity)
        q_level = math.ceil((n + 1) * (1.0 - self._alpha)) / n
        q_level = min(q_level, 1.0)
        return float(np.quantile(nonconformity, q_level, method="higher"))

    def calibrate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        router_classes: np.ndarray | None = None,
    ) -> None:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if scores.shape != labels.shape:
            raise ValueError("scores and labels must have the same shape")
        if scores.ndim != 1:
            raise ValueError("scores and labels must be 1-D arrays")
        n = len(scores)
        if n < self._MIN_PER_CLASS:
            raise ValueError(
                f"too few calibration points ({n}); split-conformal needs at "
                f"least ~{self._MIN_PER_CLASS}"
            )
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
        if router_classes is not None:
            router_classes = np.asarray(router_classes)
            if router_classes.shape != labels.shape:
                raise ValueError("router_classes must have the same shape as labels")
            per_router: dict[str, dict[int, float]] = {}
            n_per_router: dict[str, int] = {}
            for rc in np.unique(router_classes):
                rc_mask = router_classes == rc
                rc_total = int(rc_mask.sum())
                if rc_total < self._MIN_PER_ROUTER_CLASS:
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

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_fusion_conformal_mondrian.py -xvs`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/conformal.py tests/test_fusion_conformal_mondrian.py
git commit -m "feat(conformal): per-class Mondrian thresholds with too-few-points guard"
```

### Task A2: Use per-class thresholds in `predict_set`

**Files:**
- Modify: `cairn/fusion/conformal.py:84-108`
- Test: `tests/test_fusion_conformal_mondrian.py` (extend)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_predict_set_uses_per_class_thresholds():
    w = ConformalWrapper(alpha=0.1)
    scores, labels = _mk_scores()
    w.calibrate(scores, labels)
    assert w.threshold_per_class[0] != pytest.approx(w.threshold_per_class[1], abs=1e-6)
    assert w.predict_set(0.99) == ["LLM"]
    assert w.predict_set(0.01) == ["human"]


def test_predict_set_with_router_class_falls_back_when_unknown():
    w = ConformalWrapper(alpha=0.1)
    scores, labels = _mk_scores()
    w.calibrate(scores, labels)
    s = w.predict_set(0.5, router_class="prose")
    assert isinstance(s, list)
    assert set(s).issubset({"human", "LLM"})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_fusion_conformal_mondrian.py -xvs`
Expected: the new tests fail (predict_set doesn't accept router_class).

- [ ] **Step 3: Update `predict_set`**

```python
    def predict_set(
        self, p_llm: float, router_class: str | None = None
    ) -> list[str]:
        p_llm = max(0.0, min(1.0, float(p_llm)))
        if self._threshold is None:
            return list(_LABELS)
        per_class: dict[int, float] | None = None
        if router_class is not None and self._threshold_per_router is not None:
            per_class = self._threshold_per_router.get(str(router_class))
        if per_class is None:
            per_class = self._threshold_per_class
        out: list[str] = []
        nc = {0: p_llm, 1: 1.0 - p_llm}
        idx_to_label = {0: "human", 1: "LLM"}
        if per_class is not None:
            for cls in (0, 1):
                if nc[cls] <= per_class[cls]:
                    out.append(idx_to_label[cls])
            return out
        nc_named = {"human": p_llm, "LLM": 1.0 - p_llm}
        for cls in _LABELS:
            if nc_named[cls] <= self._threshold:
                out.append(cls)
        return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_fusion_conformal_mondrian.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/conformal.py tests/test_fusion_conformal_mondrian.py
git commit -m "feat(conformal): predict_set consumes per-class + per-router-class thresholds"
```

### Task A3: Persist Mondrian thresholds in save/load with v2 schema

**Files:**
- Modify: `cairn/fusion/conformal.py:110-131`
- Test: `tests/test_fusion_conformal_mondrian.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_save_and_load_round_trips_mondrian(tmp_path):
    w = ConformalWrapper(alpha=0.1)
    scores, labels = _mk_scores()
    rc = np.array(["prose"] * 200 + ["prompt"] * 200)
    w.calibrate(scores, labels, router_classes=rc)
    p = tmp_path / "cal.json"
    w.save(p)
    w2 = ConformalWrapper(alpha=0.5)
    w2.load(p)
    assert w2.threshold_per_class is not None
    assert set(w2.threshold_per_class.keys()) == {0, 1}
    assert w2.predict_set(0.5, router_class="prose") == w.predict_set(0.5, router_class="prose")


def test_load_v1_artifact_still_works(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(
        '{"alpha": 0.05, "threshold": 0.42, "n_calibration": 100, '
        '"schema": "binary-split-conformal-v1"}'
    )
    w = ConformalWrapper()
    w.load(p)
    assert w.calibrated
    assert w.threshold_per_class is None
    s = w.predict_set(0.5)
    assert isinstance(s, list)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_fusion_conformal_mondrian.py::test_save_and_load_round_trips_mondrian -xvs`
Expected: FAIL.

- [ ] **Step 3: Implement v2 save/load**

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
        if self._threshold_per_router is not None:
            payload["threshold_per_router"] = {
                rc: {str(k): v for k, v in inner.items()}
                for rc, inner in self._threshold_per_router.items()
            }
        if self._n_per_router is not None:
            payload["n_per_router"] = dict(self._n_per_router)
        Path(path).write_text(json.dumps(payload))

    def load(self, path: Path) -> None:
        data = json.loads(Path(path).read_text())
        schema = data.get("schema")
        if schema not in (None, self._SCHEMA_V1, self._SCHEMA_V2):
            raise ValueError(
                f"ConformalWrapper.load: unknown schema {schema!r}; "
                "artifact was produced by a different version"
            )
        self._alpha = float(data["alpha"])
        self._threshold = float(data["threshold"])
        self._n_calibration = int(data.get("n_calibration", 0))
        tpc = data.get("threshold_per_class")
        self._threshold_per_class = (
            {int(k): float(v) for k, v in tpc.items()} if tpc else None
        )
        npc = data.get("n_per_class")
        self._n_per_class = (
            {int(k): int(v) for k, v in npc.items()} if npc else None
        )
        tpr = data.get("threshold_per_router")
        self._threshold_per_router = (
            {rc: {int(k): float(v) for k, v in inner.items()} for rc, inner in tpr.items()}
            if tpr else None
        )
        self._n_per_router = data.get("n_per_router") or None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_fusion_conformal_mondrian.py -xvs`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/conformal.py tests/test_fusion_conformal_mondrian.py
git commit -m "feat(conformal): persist Mondrian thresholds in v2 schema, keep v1 load"
```

### Task A4: Plumb router class from pipeline → fusion → conformal

**Files:**
- Modify: `cairn/fusion/ebm.py:134-191`
- Modify: `cairn/pipeline.py` (call site of `fusion.fuse`)
- Test: `tests/test_fusion.py` (extend)

- [ ] **Step 1: Locate the fusion call site**

Run: `Grep --pattern "self._fusion.fuse|fusion\\.fuse" --path cairn/pipeline.py -n`
Expected: a call with `(layer_results, word_count=…, domain=…)`. Confirm before patching.

- [ ] **Step 2: Write the failing test**

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
        [LayerResult(
            layer_id="preamble", domain="universal", raw_score=0.9, p_llm=0.9,
            confidence=0.8, signals={}, determination="RED", attacker_tiers=[],
            compute_cost="trivial", min_text_length=0,
        )],
        word_count=200,
        domain="prose",
    )
    assert captured["router_class"] == "prose"
    assert fr.conformal_set == ["LLM"]
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_fusion.py::test_ebm_fuse_passes_domain_to_conformal -xvs`
Expected: FAIL.

- [ ] **Step 4: Update `_maybe_conformal` and both fusion paths**

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

In `fuse()`, pass `domain` through to `_naive_fuse` and `_ebm_fuse`. Update `_naive_fuse(self, results)` → `_naive_fuse(self, results, domain="prose")`. Replace each `self._maybe_conformal(p_llm)` call with `self._maybe_conformal(p_llm, router_class=domain)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_fusion.py -xvs`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add cairn/fusion/ebm.py tests/test_fusion.py
git commit -m "feat(fusion): plumb router class to conformal predict_set"
```

### Task A5: Update `calibrate_conformal.py` to capture router class + labels

**Files:**
- Modify: `scripts/calibrate_conformal.py:30-60`
- Test: `tests/test_calibrate_conformal_cli.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import json
import subprocess
import sys
from pathlib import Path
import pytest


def _write_dataset(path: Path):
    rows = [
        {"id": f"h{i}", "text": "Human writing sample text. " * 30, "label": 0, "source": "hc3"}
        for i in range(20)
    ] + [
        {"id": f"l{i}", "text": "LLM output sample. " * 30, "label": 1, "source": "openai"}
        for i in range(20)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_calibrate_cli_writes_v2_artifact(tmp_path):
    pytest.importorskip("interpret")
    ds = tmp_path / "cal.jsonl"
    _write_dataset(ds)
    out = tmp_path / "cal.json"
    cfg = Path("configs/default.yaml")
    res = subprocess.run(
        [sys.executable, "scripts/calibrate_conformal.py",
         "--dataset", str(ds), "--config", str(cfg),
         "--alpha", "0.2", "--out", str(out)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["schema"] == "binary-mondrian-conformal-v2"
    assert "threshold_per_class" in payload
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_calibrate_conformal_cli.py -xvs`
Expected: FAIL.

- [ ] **Step 3: Update the script body after `samples = load_dataset(...)`**

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
        np.asarray(scores),
        np.asarray(labels),
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

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_calibrate_conformal_cli.py -xvs`
Expected: PASS (or skip if `interpret` is missing).

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_conformal.py tests/test_calibrate_conformal_cli.py
git commit -m "feat(scripts): calibrate_conformal captures router class for Mondrian"
```

### Task A6: Adaptive Conformal Inference wrapper

**Files:**
- Create: `cairn/fusion/aci.py`
- Test: `tests/test_fusion_conformal_mondrian.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_aci_adjusts_alpha_toward_target_after_errors():
    from cairn.fusion.aci import AdaptiveConformal
    from cairn.fusion.conformal import ConformalWrapper

    base = ConformalWrapper(alpha=0.1)
    scores, labels = _mk_scores()
    base.calibrate(scores, labels)
    aci = AdaptiveConformal(base, target_alpha=0.1, gamma=0.05)

    initial_alpha = aci.alpha
    for _ in range(20):
        aci.observe(predicted_set=["human"], true_label="LLM")
    assert aci.alpha > initial_alpha

    aci2 = AdaptiveConformal(base, target_alpha=0.1, gamma=0.05)
    for _ in range(20):
        aci2.observe(predicted_set=["LLM"], true_label="LLM")
    assert aci2.alpha < 0.1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_fusion_conformal_mondrian.py::test_aci_adjusts_alpha_toward_target_after_errors -xvs`
Expected: FAIL.

- [ ] **Step 3: Create the ACI module**

```python
# cairn/fusion/aci.py
from __future__ import annotations

from cairn.fusion.conformal import ConformalWrapper


class AdaptiveConformal:
    def __init__(
        self,
        wrapped: ConformalWrapper,
        target_alpha: float,
        gamma: float = 0.05,
        alpha_min: float = 0.005,
        alpha_max: float = 0.5,
    ):
        if not (0.0 < target_alpha < 1.0):
            raise ValueError("target_alpha must be in (0, 1)")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive")
        self._wrapped = wrapped
        self._target = target_alpha
        self._gamma = gamma
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._alpha = float(wrapped.alpha)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def calibrated(self) -> bool:
        return self._wrapped.calibrated

    def predict_set(self, p_llm: float, router_class: str | None = None) -> list[str]:
        return self._wrapped.predict_set(p_llm, router_class=router_class)

    def observe(self, predicted_set: list[str], true_label: str) -> None:
        err = 0 if true_label in predicted_set else 1
        new_alpha = self._alpha + self._gamma * (self._target - err)
        self._alpha = max(self._alpha_min, min(self._alpha_max, new_alpha))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_fusion_conformal_mondrian.py -xvs`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/aci.py tests/test_fusion_conformal_mondrian.py
git commit -m "feat(conformal): adaptive conformal inference wrapper"
```

### Task A7: Run full pytest, confirm Phase A is green

- [ ] **Step 1: Run the full suite**

Run: `pytest -q`
Expected: ~340+ pass.

- [ ] **Step 2: If anything fails, fix the regression and re-run before proceeding**

Most likely surface: `tests/test_decision_abstention.py` consumes the conformal set. If it fails, route the fix through Phase B's tasks.

---

## Phase B — Decision-engine coverage accounting (audit #2 + #7)

Three holes:
1. Empty conformal set silently falls through to the bare threshold (audit #1d).
2. Preamble CRITICAL override fires before the conformal check, breaking the coverage guarantee silently.
3. MIXED override threshold is hardcoded; the config has the knob already at `decision.mixed.min_mixed_probability`.

### Task B1: Add `coverage_guarantee_applies` to FusionResult and Determination

**Files:**
- Modify: `cairn/contracts.py:24-65`
- Test: `tests/test_contracts.py` (extend)

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_contracts.py::test_determination_default_coverage_flag -xvs`
Expected: FAIL — field doesn't exist.

- [ ] **Step 3: Append `coverage_guarantee_applies: bool = True` to `FusionResult` and `Determination`**

In `cairn/contracts.py`, append the field as the **last** field of each dataclass (CLAUDE.md invariant: append, never reorder):

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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_contracts.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/contracts.py tests/test_contracts.py
git commit -m "feat(contracts): add coverage_guarantee_applies to FusionResult and Determination"
```

### Task B2: Empty conformal set is the strongest abstention signal

**Files:**
- Modify: `cairn/decision.py:51-65`
- Test: `tests/test_decision_coverage.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decision_coverage.py
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

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_decision_coverage.py -xvs`
Expected: FAIL — empty-set falls through.

- [ ] **Step 3: Replace the abstention block in `DecisionEngine.decide`**

Replace the current `abstention_reason = None` block with:

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

And update the trailing `return Determination(...)` to add `coverage_guarantee_applies=coverage_applies`. The full return becomes:

```python
        return Determination(
            label=label, p_llm=fusion.p_llm, confidence_interval=fusion.confidence_interval,
            prediction_set=fusion.prediction_set, reason=reason, top_features=list(fusion.top_contributors),
            override_applied=False, detectors_run=[r.layer_id for r in layer_results], cascade_phases=[],
            mixed_report=None, layer_results=list(layer_results),
            feature_contributions=dict(fusion.feature_contributions),
            conformal_set=list(fusion.conformal_set) if fusion.conformal_set is not None else None,
            conformal_alpha=fusion.conformal_alpha,
            fusion_mode=fusion.fusion_mode,
            abstention_reason=abstention_reason,
            coverage_guarantee_applies=coverage_applies,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_decision_coverage.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/decision.py tests/test_decision_coverage.py
git commit -m "fix(decision): empty conformal set yields UNCERTAIN with coverage flag off"
```

### Task B3: Overrides mark coverage as not applying

**Files:**
- Modify: `cairn/decision.py:14-50`
- Test: `tests/test_decision_coverage.py` (extend)

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_decision_coverage.py -xvs`
Expected: FAIL.

- [ ] **Step 3: Set `coverage_guarantee_applies=False` on both override returns**

In the preamble CRITICAL `return Determination(...)`, add `coverage_guarantee_applies=False`. Same for the MIXED `return Determination(...)`. Final blocks:

```python
            if result.layer_id == "preamble" and result.signals.get("severity") == "CRITICAL":
                return Determination(
                    label="RED", p_llm=fusion.p_llm, confidence_interval=fusion.confidence_interval,
                    prediction_set=["RED"], reason="Critical LLM preamble detected — raw assistant output present.",
                    top_features=list(fusion.top_contributors), override_applied=True,
                    detectors_run=[r.layer_id for r in layer_results], cascade_phases=[],
                    mixed_report=None, layer_results=list(layer_results),
                    feature_contributions=dict(fusion.feature_contributions),
                    conformal_set=list(fusion.conformal_set) if fusion.conformal_set is not None else None,
                    conformal_alpha=fusion.conformal_alpha,
                    fusion_mode=fusion.fusion_mode,
                    coverage_guarantee_applies=False,
                )
```

```python
        if mixed_report is not None:
            return Determination(
                label="MIXED", p_llm=fusion.p_llm, confidence_interval=fusion.confidence_interval,
                prediction_set=["MIXED"],
                reason=(
                    f"Mixed-authorship signals: probability {mixed_report['mixed_probability']:.2f} "
                    f"across {mixed_report['n_boundaries']} style boundaries. "
                    "Some segments look human, others look LLM."
                ),
                top_features=list(fusion.top_contributors), override_applied=True,
                detectors_run=[r.layer_id for r in layer_results], cascade_phases=[],
                mixed_report=mixed_report, layer_results=list(layer_results),
                feature_contributions=dict(fusion.feature_contributions),
                conformal_set=list(fusion.conformal_set) if fusion.conformal_set is not None else None,
                conformal_alpha=fusion.conformal_alpha,
                fusion_mode=fusion.fusion_mode,
                coverage_guarantee_applies=False,
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_decision_coverage.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/decision.py tests/test_decision_coverage.py
git commit -m "feat(decision): mark coverage_guarantee_applies=False on overrides"
```

### Task B4: MIXED threshold sourced from config

**Files:**
- Modify: `cairn/decision.py:3-12, 26-33`
- Test: `tests/test_decision_coverage.py` (extend)

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_decision_coverage.py -xvs`
Expected: FAIL — `decision.py:28` hardcodes `0.60`.

- [ ] **Step 3: Read the config and respect both knobs**

In `cairn/decision.py::__init__` add:

```python
        mixed = decision.get("mixed", {})
        self._mixed_enabled = bool(mixed.get("enabled", True))
        self._mixed_threshold = float(mixed.get("min_mixed_probability", 0.60))
```

Update the mixed_boundary scan in `decide`:

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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_decision_coverage.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/decision.py tests/test_decision_coverage.py
git commit -m "fix(decision): MIXED threshold and enable flag honour config"
```

### Task B5: Run full pytest, confirm Phase B is green

- [ ] **Step 1: Run the full suite**

Run: `pytest -q`
Expected: ~340+ pass.

> **STOP-HERE checkpoint after Phase B:** ship as `feat(calibration): Mondrian conformal + honest coverage accounting`. README's calibration banner can now claim per-class FPR holds at α and that overrides are surfaced as not-coverage-guaranteed.

---

## Phase C — EBM fusion: missingness, confidence, monotone, honest CI (audit #3 + #9 + #10)

Single biggest accuracy lever. Bumps EBM artifact schema (new feature columns + stored column means); requires one retrain pass. Conformal artifact from Phase A stays compatible.

### Task C1: Add per-detector `_missing` indicator features

**Files:**
- Modify: `cairn/fusion/ebm.py:14-91`
- Test: `tests/test_ebm_missingness.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ebm_missingness.py
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

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ebm_missingness.py -xvs`
Expected: FAIL.

- [ ] **Step 3: Add `missing` helper and emit `<detector>_missing` features**

In `cairn/fusion/ebm.py::FeatureAssembler.assemble`, add the helper:

```python
        def missing(layer_id: str) -> float:
            return 0.0 if layer_id in by_id else 1.0
```

Then add a `<detector>_missing` key alongside each `*_p_llm` in the `vec` dict literal. Add for: `preamble`, `fingerprint_vocab`, `prompt_structure`, `voice_spec`, `instruction_density`, `nssi`, `surprisal_dynamics`, `contrastive_lm`, `token_cohesiveness`, `perturbation`, `contrastive_gen`, `dna_gpt`, `mixed_boundary`, `cross_similarity`. Concretely, additions to the dict:

```python
            "preamble_missing": missing("preamble"),
            "fingerprint_missing": missing("fingerprint_vocab"),
            "ps_missing": missing("prompt_structure"),
            "vs_missing": missing("voice_spec"),
            "idi_missing": missing("instruction_density"),
            "nssi_missing": missing("nssi"),
            "surprisal_missing": missing("surprisal_dynamics"),
            "binoculars_missing": missing("contrastive_lm"),
            "tocsin_missing": missing("token_cohesiveness"),
            "perturbation_missing": missing("perturbation"),
            "contrastive_gen_missing": missing("contrastive_gen"),
            "dna_gpt_missing": missing("dna_gpt"),
            "mixed_boundary_missing": missing("mixed_boundary"),
            "cross_similarity_missing": missing("cross_similarity"),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ebm_missingness.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/ebm.py tests/test_ebm_missingness.py
git commit -m "feat(ebm): per-detector _missing indicator features in assembler"
```

### Task C2: Add per-detector `_confidence` features

**Files:**
- Modify: `cairn/fusion/ebm.py::FeatureAssembler.assemble`
- Test: `tests/test_ebm_missingness.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_confidence_per_detector():
    a = FeatureAssembler()
    vec = a.assemble([_lr("preamble", p_llm=0.9, confidence=0.85, severity="HIGH")],
                     word_count=200, domain="prose")
    assert vec["preamble_confidence"] == pytest.approx(0.85, abs=1e-6)
    assert math.isnan(vec["nssi_confidence"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ebm_missingness.py::test_confidence_per_detector -xvs`
Expected: FAIL.

- [ ] **Step 3: Add `conf` helper and emit `<detector>_confidence`**

```python
        def conf(layer_id: str) -> float:
            return float(by_id[layer_id].confidence) if layer_id in by_id else _NAN
```

Additions to `vec`:

```python
            "preamble_confidence": conf("preamble"),
            "fingerprint_confidence": conf("fingerprint_vocab"),
            "ps_confidence": conf("prompt_structure"),
            "vs_confidence": conf("voice_spec"),
            "idi_confidence": conf("instruction_density"),
            "nssi_confidence": conf("nssi"),
            "surprisal_confidence": conf("surprisal_dynamics"),
            "binoculars_confidence": conf("contrastive_lm"),
            "tocsin_confidence": conf("token_cohesiveness"),
            "perturbation_confidence": conf("perturbation"),
            "contrastive_gen_confidence": conf("contrastive_gen"),
            "dna_gpt_confidence": conf("dna_gpt"),
            "mixed_boundary_confidence": conf("mixed_boundary"),
            "cross_similarity_confidence": conf("cross_similarity"),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ebm_missingness.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/ebm.py tests/test_ebm_missingness.py
git commit -m "feat(ebm): per-detector _confidence features for MoE-style gating"
```

### Task C3: Store column means on the EBM artifact, use them at inference

**Files:**
- Modify: `cairn/fusion/training.py:14-44`
- Modify: `cairn/fusion/ebm.py:108-167`
- Test: `tests/test_ebm_missingness.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_ebm_uses_stored_column_means_not_half():
    pytest.importorskip("interpret")
    from cairn.fusion.training import train_ebm
    rows = []
    a = FeatureAssembler()
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
    assert "binoculars_ratio" in means
    assert means["binoculars_ratio"] == pytest.approx((1.0 + 0.7) / 2, abs=1e-6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ebm_missingness.py::test_ebm_uses_stored_column_means_not_half -xvs`
Expected: FAIL.

- [ ] **Step 3: Compute and store column means in `train_ebm`**

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
        feature_names=feature_names,
        max_rounds=300,
        interactions=15,
        random_state=42,
    )
    model.fit(X_arr, y_arr)
    model._cairn_feature_names = feature_names
    model._cairn_column_means = col_means
    return model
```

- [ ] **Step 4: Update `EBMFusion._ebm_fuse` to use stored means**

In `cairn/fusion/ebm.py::EBMFusion.__init__`, after the schema-warning block:

```python
        self._column_means: dict[str, float] = {}
        if model is not None:
            cm = getattr(model, "_cairn_column_means", None)
            if cm:
                self._column_means = dict(cm)
```

In `_ebm_fuse`, replace `X = np.where(np.isnan(X), 0.5, X)` with:

```python
        for j, name in enumerate(features):
            if np.isnan(X[0, j]):
                X[0, j] = self._column_means.get(name, 0.5)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_ebm_missingness.py -xvs`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add cairn/fusion/training.py cairn/fusion/ebm.py tests/test_ebm_missingness.py
git commit -m "feat(ebm): store and use per-column training means for imputation"
```

### Task C4: Schema-mismatch is a hard error

**Files:**
- Modify: `cairn/fusion/ebm.py:108-128`
- Test: `tests/test_ebm_missingness.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def test_ebm_raises_when_artifact_requires_unknown_features():
    from cairn.fusion.ebm import EBMFusion

    class _StubModel:
        _cairn_feature_names = ["preamble_p_llm", "imaginary_feature_xyz"]
        _cairn_column_means = {}

    with pytest.raises(RuntimeError, match="imaginary_feature_xyz"):
        EBMFusion(model=_StubModel())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ebm_missingness.py::test_ebm_raises_when_artifact_requires_unknown_features -xvs`
Expected: FAIL.

- [ ] **Step 3: Promote the warning to RuntimeError**

In `cairn/fusion/ebm.py::EBMFusion.__init__`, replace the warning with:

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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_ebm_missingness.py tests/test_ebm_schema_coverage.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/ebm.py tests/test_ebm_missingness.py
git commit -m "fix(ebm): hard-error when artifact requires features the assembler cannot produce"
```

### Task C5: Monotone constraints on guaranteed-direction features

**Files:**
- Modify: `cairn/fusion/training.py`
- Test: `tests/test_ebm_missingness.py` (extend)

- [ ] **Step 1: Write the failing test**

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
    assert constraints is not None
    feat_to_idx = {n: i for i, n in enumerate(model._cairn_feature_names)}
    for name in ("preamble_severity", "fingerprint_hits_per_1000", "nssi_n_signals_active"):
        idx = feat_to_idx[name]
        assert constraints[idx] == 1, f"expected monotone-up on {name}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ebm_missingness.py::test_train_ebm_applies_monotone_constraints -xvs`
Expected: FAIL.

- [ ] **Step 3: Wire monotone constraints into `train_ebm`**

Add at module top of `cairn/fusion/training.py`:

```python
_MONOTONE_UP_FEATURES = (
    "preamble_severity",
    "preamble_n_matches",
    "fingerprint_hits_per_1000",
    "fingerprint_bigram_hits",
    "nssi_n_signals_active",
    "nssi_formulaic_density",
    "nssi_discourse_scaffolding",
    "idi_score",
)
```

Update `train_ebm`:

```python
    monotone = [1 if name in _MONOTONE_UP_FEATURES else 0 for name in feature_names]
    try:
        model = ExplainableBoostingClassifier(
            feature_names=feature_names,
            max_rounds=300,
            interactions=15,
            random_state=42,
            monotone_constraints=monotone,
        )
    except TypeError:
        # Fallback for interpret-ml builds without monotone_constraints kwarg.
        import interpret
        import logging
        logging.getLogger("cairn.fusion").warning(
            "monotone_constraints unsupported in interpret-ml=%s; training without",
            getattr(interpret, "__version__", "unknown"),
        )
        model = ExplainableBoostingClassifier(
            feature_names=feature_names, max_rounds=300, interactions=15, random_state=42,
        )
    model.fit(X_arr, y_arr)
    model._cairn_feature_names = feature_names
    model._cairn_column_means = col_means
    model._cairn_monotone_constraints = monotone
    return model
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_ebm_missingness.py::test_train_ebm_applies_monotone_constraints -xvs`
Expected: PASS, or skip with version warning.

- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/training.py tests/test_ebm_missingness.py
git commit -m "feat(ebm): monotone-up constraints on fingerprint/preamble/nssi features"
```

### Task C6: Honest CI from conformal threshold (or wide when uncalibrated)

**Files:**
- Modify: `cairn/fusion/ebm.py:140-191`
- Test: `tests/test_fusion.py` (extend)

- [ ] **Step 1: Write the failing test**

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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_fusion.py::test_naive_fusion_with_no_conformal_returns_wide_ci -xvs`
Expected: FAIL — current naive path uses `±spread/2`.

- [ ] **Step 3: Add `_conformal_ci` and update both fusion paths**

```python
    def _conformal_ci(self, p_llm: float, router_class: str | None = None) -> tuple[float, float] | None:
        if self._conformal is None or not getattr(self._conformal, "calibrated", False):
            return None
        cls = 1 if p_llm >= 0.5 else 0
        per_router = getattr(self._conformal, "threshold_per_router", None)
        per_class = None
        if per_router and router_class is not None:
            per_class = per_router.get(str(router_class))
        if per_class is None:
            per_class = getattr(self._conformal, "threshold_per_class", None)
        if per_class is None:
            return None
        hw = float(per_class.get(cls, 0.10))
        return (max(0.0, p_llm - hw), min(1.0, p_llm + hw))
```

In `_ebm_fuse`, replace `ci = (max(0.0, p_llm - 0.10), min(1.0, p_llm + 0.10))` with:

```python
        ci = self._conformal_ci(p_llm, router_class=domain) or (
            max(0.0, p_llm - 0.10), min(1.0, p_llm + 0.10)
        )
```

In `_naive_fuse`, replace `spread = ...; hw = ...; ci = ...` with:

```python
        ci = self._conformal_ci(p_llm, router_class=domain) or (0.0, 1.0)
```

(Drop the `spread`/`hw` lines.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_fusion.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/fusion/ebm.py tests/test_fusion.py
git commit -m "feat(fusion): CI derives from conformal threshold or is wide-when-naive"
```

### Task C7: Retrain workflow + cache rebuild

The schema bumped; old EBM artifacts will hit Task C4's hard-error. We retrain on the v6 corpus.

- [ ] **Step 1: Confirm the v6 corpus path**

Run: `ls -lh data/calib2000_v6_train.jsonl data/all_features_v6.jsonl 2>&1 | head -5`
Expected: both files exist (per CLAUDE.md project state).

- [ ] **Step 2: Retrain against the cache**

Run:
```bash
python scripts/train_fusion.py \
  --dataset data/calib2000_v6_train.jsonl \
  --config configs/master_minus_arxiv.yaml \
  --features-cache data/all_features_v6.jsonl \
  --out models/calib2000/profiles/master_minus_arxiv/ebm_v7.pkl
```
Expected: `trained on N, held out M, wrote ...`. Bump filename to `ebm_v7.pkl` so v6 stays available for rollback.

- [ ] **Step 3: Re-calibrate conformal against the new EBM**

Run:
```bash
python scripts/calibrate_conformal.py \
  --dataset data/seed_conformal.jsonl \
  --config configs/master_minus_arxiv.yaml \
  --alpha 0.1 \
  --out models/calib2000/profiles/master_minus_arxiv/conformal_v7.json
```
Expected: `calibrated on N samples (alpha=0.1); per-class={0: ..., 1: ...}; per-router=[...]; wrote ...`.

- [ ] **Step 4: Update the profile config**

```yaml
# configs/master_minus_arxiv.yaml
fusion:
  model_path: models/calib2000/profiles/master_minus_arxiv/ebm_v7.pkl
  conformal_path: models/calib2000/profiles/master_minus_arxiv/conformal_v7.json
```

- [ ] **Step 5: Smoke test the retrained pipeline**

Run: `pytest tests/test_pipeline.py tests/test_pipeline_detector_calibration.py -xvs`
Expected: PASS.

- [ ] **Step 6: Commit the artifacts and config bump**

```bash
git add models/calib2000/profiles/master_minus_arxiv/ebm_v7.pkl \
        models/calib2000/profiles/master_minus_arxiv/conformal_v7.json \
        configs/master_minus_arxiv.yaml
git commit -m "chore(models): retrain EBM v7 with missingness/confidence/monotone features"
```

### Task C8: Run full pytest, confirm Phase C is green

- [ ] **Step 1: Run the full suite**

Run: `pytest -q`
Expected: ~340+ pass.

> **STOP-HERE checkpoint after Phase C:** ship as `feat(ebm): missingness-aware fusion v7`. Likely 2-4 AUROC point lift on RAID-class samples.

---

## Phase D — Hold-out by LLM family (audit #4)

`stratify=y` doesn't validate generalization across generators. Add `--holdout-source` and surface per-source metrics in the eval report.

### Task D1: `--holdout-source` CLI flag in `train_fusion.py`

**Files:**
- Modify: `scripts/train_fusion.py:30-66`
- Test: `tests/test_train_fusion_holdout.py` (create)

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_train_fusion_holdout.py -xvs`
Expected: FAIL.

- [ ] **Step 3: Add the flag and the leave-one-source-out branch**

In `scripts/train_fusion.py`, in the `argparse` block:

```python
    ap.add_argument(
        "--holdout-source",
        default=None,
        help="LLM source/generator (e.g. 'openai', 'anthropic') to hold out "
             "entirely from training. The held-out samples become the test "
             "split; everything else trains. Used for honest zero-shot eval.",
    )
```

After the featurize loop, replace the `train_test_split(...)` call with:

```python
    if args.holdout_source:
        idx_train = [
            i for i, s in enumerate(samples)
            if (s.source or "").lower() != args.holdout_source.lower()
        ]
        idx_test = [
            i for i, s in enumerate(samples)
            if (s.source or "").lower() == args.holdout_source.lower()
        ]
        if not idx_train or not idx_test:
            raise SystemExit(
                f"--holdout-source {args.holdout_source!r} produced an empty "
                f"split (train={len(idx_train)}, test={len(idx_test)})"
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

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_train_fusion_holdout.py -xvs`
Expected: PASS (or skip if `interpret` missing).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_fusion.py tests/test_train_fusion_holdout.py
git commit -m "feat(scripts): --holdout-source for leave-one-LLM-out training splits"
```

### Task D2: Per-source breakdown in `EvalReport`

**Files:**
- Modify: `cairn/evaluation/runner.py:24-110`
- Test: `tests/test_evaluation/test_runner.py` (extend)

- [ ] **Step 1: Write the failing test**

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

    samples = [
        EvalSample(id=f"a{i}", text="Anthropic sample.", label=1, source="anthropic")
        for i in range(20)
    ] + [
        EvalSample(id=f"o{i}", text="OpenAI sample.", label=1, source="openai")
        for i in range(20)
    ] + [
        EvalSample(id=f"h{i}", text="Human writing.", label=0, source="hc3")
        for i in range(20)
    ]
    rep = run_eval(_FakePipeline(), samples)
    assert "anthropic" in rep.per_source
    assert "openai" in rep.per_source
    assert "hc3" in rep.per_source
    assert rep.per_source["anthropic"]["n_samples"] == 20
    assert "auroc" in rep.per_source["anthropic"] or rep.per_source["anthropic"].get("auroc_unavailable")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_evaluation/test_runner.py::test_per_source_breakdown_in_report -xvs`
Expected: FAIL.

- [ ] **Step 3: Add `per_source` to `EvalReport` and populate it**

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

In `run_eval`, after the existing `per_attack` aggregation:

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

Pass `per_source=per_source` into the `EvalReport(...)` constructor at the bottom of `run_eval`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_evaluation/test_runner.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/evaluation/runner.py tests/test_evaluation/test_runner.py
git commit -m "feat(eval): per-source breakdown in EvalReport for cross-family analysis"
```

### Task D3: Surface `per_source` in eval-report serialization

**Files:**
- Modify: `cairn/evaluation/__init__.py` (or wherever `eval_report_to_dict` lives)
- Test: `tests/test_eval_serialization.py` (extend)

- [ ] **Step 1: Locate `eval_report_to_dict`**

Run: `Grep --pattern "def eval_report_to_dict" -n`

- [ ] **Step 2: Write the failing test**

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

- [ ] **Step 3: Run, fail, fix, run**

Add `"per_source": dict(report.per_source)` to the serializer's output dict. Re-run.

- [ ] **Step 4: Commit**

```bash
git add cairn/evaluation/ tests/test_eval_serialization.py
git commit -m "feat(eval): per_source surfaced in serialized eval report"
```

### Task D4: Run full pytest, confirm Phase D is green

- [ ] **Step 1: Run the full suite**

Run: `pytest -q`
Expected: ~340+ pass.

> **STOP-HERE checkpoint after Phase D:** ship as `feat(eval): leave-one-LLM-out training splits + per-source eval`.

---

## Phase E — RAID benchmark in CI (audit #5)

A 50-row RAID smoke fixture goes into the repo; a separate workflow runs the full RAID subset against the latest profile artifact at release-tag time.

### Task E1: TPR-at-FPR-ceiling primitive

**Files:**
- Modify: `cairn/evaluation/metrics.py`
- Test: `tests/test_evaluation/test_metrics.py` (extend)

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Implement and re-run**

Add to `cairn/evaluation/metrics.py`:

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

Run: `pytest tests/test_evaluation/test_metrics.py -xvs`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add cairn/evaluation/metrics.py tests/test_evaluation/test_metrics.py
git commit -m "feat(eval): tpr_at_fpr_ceiling primitive for Neyman-Pearson reporting"
```

### Task E2: RAID smoke fixture (50 rows)

**Files:**
- Create: `scripts/build_raid_fixture.py`
- Create: `tests/fixtures/raid_mini.jsonl` (one-shot output)

- [ ] **Step 1: Write the builder script**

```python
# scripts/build_raid_fixture.py
"""One-shot builder: pull a deterministic 50-row sample from RAID public CSV.

Run manually once; resulting JSONL is checked in. Re-run when bumping
fixture cardinality. Do NOT run from CI — the network call is offline-unfriendly.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path,
                    help="Path to a previously-downloaded RAID test CSV.")
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
                "text": text,
                "label": label,
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

- [ ] **Step 2: Run the builder once locally**

```bash
curl -L -o /tmp/raid-test.csv https://raid-bench.xyz/data/raid-test.csv
python scripts/build_raid_fixture.py --source /tmp/raid-test.csv --n-per-class 25
```
Expected: `wrote 50 rows to tests/fixtures/raid_mini.jsonl`.

If the URL is unreachable, document the manual download step in the script's docstring and ask the user to provide the CSV.

- [ ] **Step 3: Verify the fixture loads**

Run: `python -c "from cairn.evaluation.dataset import load_dataset; print(len(load_dataset('tests/fixtures/raid_mini.jsonl')))"`
Expected: `50`.

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/raid_mini.jsonl scripts/build_raid_fixture.py
git commit -m "feat(test): 50-row RAID smoke fixture and one-shot builder"
```

### Task E3: `scripts/run_raid_eval.py` runner

- [ ] **Step 1: Write the runner**

```python
# scripts/run_raid_eval.py
"""Run a CAIRN profile against a RAID-format JSONL and report TPR@FPR≤1%."""
from __future__ import annotations

import argparse
import json
import sys
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
    tpr, threshold, achieved_fpr = tpr_at_fpr_ceiling(
        scores, labels, ceiling=args.fpr_ceiling
    )
    summary = {
        "n_samples": report.n_samples,
        "fpr_ceiling": args.fpr_ceiling,
        "tpr_at_fpr_ceiling": tpr,
        "threshold": threshold,
        "achieved_fpr": achieved_fpr,
        "auroc": report.metrics.get("auroc"),
        "per_source": report.per_source,
    }
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

- [ ] **Step 2: Smoke-run against the fixture**

```bash
python scripts/run_raid_eval.py \
  --dataset tests/fixtures/raid_mini.jsonl \
  --config configs/default.yaml \
  --fpr-ceiling 0.01
```
Expected: JSON summary; non-zero TPR; AUROC > 0.5.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_raid_eval.py
git commit -m "feat(scripts): run_raid_eval.py — TPR@FPR≤1% reporter"
```

### Task E4: GitHub Actions workflow

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/raid-bench.yml
name: RAID benchmark

on:
  push:
    tags: ['v*']
  workflow_dispatch:
  pull_request:
    paths:
      - 'cairn/fusion/**'
      - 'cairn/detectors/**'
      - 'configs/master_minus_arxiv.yaml'
      - 'models/calib2000/**'
      - '.github/workflows/raid-bench.yml'

jobs:
  raid-mini:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install
        run: |
          python -m pip install --upgrade pip
          pip install -e '.[fusion]'
      - name: Run RAID-mini
        run: |
          python scripts/run_raid_eval.py \
            --dataset tests/fixtures/raid_mini.jsonl \
            --config configs/default.yaml \
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
git add .github/workflows/raid-bench.yml
git commit -m "ci: RAID-mini benchmark on PR + tag with TPR@FPR≤1% gate"
```

### Task E5: Run full pytest, confirm Phase E is green

- [ ] **Step 1: Run the full suite**

Run: `pytest -q`
Expected: ~340+ pass.

> **STOP-HERE checkpoint after Phase E:** ship as `feat(ci): RAID smoke benchmark with TPR@FPR≤1% gate`.

---

## Phase F — `skops` serialization + training metadata (audit #6)

Replace `pickle.dump`/`pickle.load` with `skops.io.save`/`skops.io.load`. Add an HMAC sidecar for tamper detection. Embed training-time provenance metadata in a `.meta.json` sidecar.

### Task F1: Add skops dep, write serialization module

**Files:**
- Modify: `pyproject.toml`
- Create: `cairn/fusion/serialization.py`
- Test: `tests/test_fusion_serialization.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import pytest


def test_save_and_load_round_trips_with_metadata(tmp_path):
    pytest.importorskip("interpret")
    pytest.importorskip("skops")
    import numpy as np
    from cairn.fusion.training import train_ebm
    from cairn.fusion.serialization import save_model, load_model

    X = [{"preamble_p_llm": 0.9, "preamble_missing": 0.0,
          "fingerprint_p_llm": 0.8, "fingerprint_missing": 0.0,
          "word_count": 200.0, "log_word_count": 5.3,
          "is_short": 0.0, "is_medium": 1.0, "is_long": 0.0,
          "domain_prompt": 0.0, "domain_prose": 1.0}
         for _ in range(40)]
    y = [1 if i >= 20 else 0 for i in range(40)]
    model = train_ebm(X, y, feature_names=list(X[0].keys()))
    out = tmp_path / "ebm.skops"
    save_model(model, out, metadata={"dataset_hash": "abc123", "n_samples": 40})
    loaded, metadata = load_model(out)
    assert metadata["dataset_hash"] == "abc123"
    assert metadata["n_samples"] == 40
    Xt = np.array([[0.9, 0.0, 0.8, 0.0, 200.0, 5.3, 0.0, 1.0, 0.0, 0.0, 1.0]])
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_fusion_serialization.py -xvs`
Expected: FAIL.

- [ ] **Step 3: Add skops to extras**

```toml
[project.optional-dependencies]
fusion = ["interpret>=0.5", "scikit-learn>=1.3", "skops>=0.10"]
```

- [ ] **Step 4: Implement `cairn/fusion/serialization.py`**

```python
# cairn/fusion/serialization.py
from __future__ import annotations

import hashlib
import hmac
import json
import os
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


def load_model(
    path: Path, *, verify_hmac: bool = True, trusted: list[str] | None = None
) -> tuple[Any, dict]:
    try:
        import skops.io as sio
    except ImportError as e:
        raise RuntimeError(
            "skops is required. Install with: pip install 'cairn[fusion]'"
        ) from e
    path = Path(path)
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

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_fusion_serialization.py -xvs`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml cairn/fusion/serialization.py tests/test_fusion_serialization.py
git commit -m "feat(fusion): skops + HMAC + metadata serialization"
```

### Task F2: Wire `serialization.save_model` into `train_fusion.py` with provenance

**Files:**
- Modify: `scripts/train_fusion.py`
- Modify: `cairn/pipeline.py` (model load path)

- [ ] **Step 1: Compute provenance and use skops in the training script**

Replace the `from cairn.fusion.training import save_model, train_ebm` line with `from cairn.fusion.training import train_ebm` and add `from cairn.fusion.serialization import save_model as save_model_skops`. Then after `model = train_ebm(...)`:

```python
    import hashlib
    from datetime import datetime, timezone
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
        "schema_version": "ebm-v7",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model_skops(model, args.out, metadata=metadata)
    print(f"trained on {len(X_train)}, held out {len(X_test)}, wrote {args.out}")
    print(f"metadata: dataset={dataset_hash}, n_train={len(X_train)}, label_balance={metadata['label_balance_train']:.3f}")
```

- [ ] **Step 2: Update the model loader in pipeline**

Find the load site:
Run: `Grep --pattern "load_model|pickle\\.load" --path cairn/ -n`

Add a small dispatcher that prefers `.skops` over pickle:

```python
def _load_fusion_model(path: Path) -> Any:
    if str(path).endswith(".skops"):
        from cairn.fusion.serialization import load_model as _load
        model, _meta = _load(path)
        return model
    import pickle
    import logging
    logging.getLogger("cairn").warning(
        "loading legacy pickle EBM at %s; retrain with skops", path
    )
    with open(path, "rb") as f:
        return pickle.load(f)
```

Wire it where the EBM is currently loaded.

- [ ] **Step 3: Run pytest**

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_fusion.py cairn/pipeline.py
git commit -m "feat(fusion): train_fusion writes skops artifact with provenance metadata"
```

### Task F3: Run full pytest, confirm Phase F is green

- [ ] **Step 1: Run the full suite**

Run: `pytest -q`
Expected: ~340+ pass.

> **STOP-HERE checkpoint after Phase F:** ship as `feat(security): skops + HMAC + provenance for EBM artifacts`.

---

## Phase G — Retrieval-based cross-similarity (audit #7)

Replace pairwise word-shingle Jaccard with a sentence-transformer + FAISS index. Persist alongside the SQLite history. Cross-similarity becomes a retrieval query against historical and in-batch outputs.

### Task G1: New `cairn/index/` module — retrieval wrapper

**Files:**
- Create: `cairn/index/__init__.py`
- Create: `cairn/index/retrieval.py`
- Modify: `pyproject.toml`
- Test: `tests/test_index_retrieval.py` (create)

- [ ] **Step 1: Write the failing test**

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
    nbr_ids = [nid for nid, _score in nbrs if nid != "c"]
    assert nbr_ids[0] == "a"


def test_retrieval_index_empty_returns_no_neighbors():
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from cairn.index.retrieval import RetrievalIndex
    idx = RetrievalIndex()
    idx.add("a", "Short text. " * 20)
    nbrs = idx.query("a", k=5)
    nbr_ids = [nid for nid, _score in nbrs if nid != "a"]
    assert nbr_ids == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_index_retrieval.py -xvs`
Expected: FAIL.

- [ ] **Step 3: Add the extras and the module**

```toml
[project.optional-dependencies]
retrieval = ["sentence-transformers>=2.2", "faiss-cpu>=1.7"]
```

```python
# cairn/index/__init__.py
from cairn.index.retrieval import RetrievalIndex

__all__ = ["RetrievalIndex"]
```

```python
# cairn/index/retrieval.py
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

- [ ] **Step 4: Install the extra and run the tests**

```bash
pip install -e '.[retrieval]'
pytest tests/test_index_retrieval.py -xvs
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/index/ tests/test_index_retrieval.py pyproject.toml
git commit -m "feat(index): RetrievalIndex (sentence-transformer + FAISS)"
```

### Task G2: Persistent index store (SQLite-backed)

**Files:**
- Create: `cairn/index/store.py`
- Test: `tests/test_index_retrieval.py` (extend)

- [ ] **Step 1: Write the failing test**

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


def test_persistent_index_failed_db_write_does_not_raise(tmp_path):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("faiss")
    from cairn.index.store import PersistentRetrievalIndex
    bad = tmp_path / "nonexistent" / "ix.db"
    idx = PersistentRetrievalIndex(db_path=bad)
    idx.add("a", "Sample. " * 20)
    nbrs = idx.query("a", k=1)
    assert isinstance(nbrs, list)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_index_retrieval.py -xvs`
Expected: FAIL.

- [ ] **Step 3: Implement the store**

```python
# cairn/index/store.py
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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_index_retrieval.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/index/store.py tests/test_index_retrieval.py
git commit -m "feat(index): SQLite-backed persistent retrieval index, best-effort writes"
```

### Task G3: Wire `cross_similarity` to the retrieval index

**Files:**
- Modify: `cairn/detectors/cross_similarity.py:42-100`
- Test: `tests/test_detectors/test_cross_similarity.py` (extend)

- [ ] **Step 1: Write the failing test**

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


def test_cross_similarity_falls_back_to_pairwise_when_no_index():
    from cairn.detectors.cross_similarity import CrossSimilarityDetector
    det = CrossSimilarityDetector()
    res = det.analyze("Some text. " * 30, {"use_retrieval_index": False})
    assert res.determination == "SKIP"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_detectors/test_cross_similarity.py -xvs`
Expected: FAIL.

- [ ] **Step 3: Update `CrossSimilarityDetector.analyze`**

Replace the body of `analyze` and add `_retrieval_analyze` + `_heuristic_p_llm`:

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
                signals={"reason": "empty_index"},
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
            raw_score=float(top_score),
            p_llm=float(p_llm),
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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_detectors/test_cross_similarity.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/detectors/cross_similarity.py tests/test_detectors/test_cross_similarity.py
git commit -m "feat(cross_similarity): retrieval-index path with pairwise fallback"
```

### Task G4: Pipeline wiring — instantiate index, inject handle, best-effort write

**Files:**
- Modify: `cairn/pipeline.py`
- Test: `tests/test_pipeline.py` (extend)

- [ ] **Step 1: Write the failing test**

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
        "detectors": {
            "cross_similarity": {
                "enabled": True,
                "use_retrieval_index": True,
                "neighbor_score_threshold": 0.85,
            },
        },
        "retrieval": {
            "enabled": True,
            "db_path": str(tmp_path / "ix.db"),
        },
    }
    from cairn.pipeline import CairnPipeline
    p = CairnPipeline(config)
    assert getattr(p, "_retrieval_index", None) is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_pipeline.py::test_pipeline_instantiates_index_when_configured -xvs`
Expected: FAIL.

- [ ] **Step 3: Add lazy index instantiation in `CairnPipeline.__init__`**

After detector registry init:

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
                    "retrieval extra not installed; cross_similarity falls back to pairwise. "
                    "Install with: pip install 'cairn[retrieval]'"
                )
            except Exception:
                logger.warning("retrieval index init failed; continuing without", exc_info=True)
```

In the analyze entrypoint where detectors are dispatched, inject the handle:

```python
        if det_id == "cross_similarity" and self._retrieval_index is not None:
            det_config = {**det_config, "_index_handle": self._retrieval_index, "use_retrieval_index": True}
```

After analyze completes, best-effort add:

```python
        if self._retrieval_index is not None:
            try:
                import hashlib
                sid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                self._retrieval_index.add(sid, text)
            except Exception:
                logger.warning("retrieval index add failed", exc_info=True)
```

(Hash-by-default keeps the privacy retention invariant in CLAUDE.md.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_pipeline.py -xvs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cairn/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): wire RetrievalIndex into cross_similarity dispatch"
```

### Task G5: Run full pytest, confirm Phase G is green

- [ ] **Step 1: Run the full suite**

Run: `pytest -q`
Expected: ~340+ pass.

> **STOP-HERE checkpoint after Phase G:** ship as `feat(detectors): retrieval-based cross-similarity (Krishna et al. 2023)`. Closes priority list 1-7.

---

## Final verification

- [ ] **Run the full suite end-to-end**: `pytest -q` (~340+ pass)
- [ ] **Run RAID smoke benchmark** against new EBM v7 + Mondrian conformal:
  ```bash
  python scripts/run_raid_eval.py \
    --dataset tests/fixtures/raid_mini.jsonl \
    --config configs/master_minus_arxiv.yaml \
    --fpr-ceiling 0.01
  ```
  Record TPR for the PR description.
- [ ] **Update `README.md`**: per-class FPR bounded; overrides marked as not-coverage-guaranteed; missingness-aware EBM; RAID smoke metric; skops artifacts.
- [ ] **Update `docs/architecture.md`**: Mondrian conformal + retrieval index sections.
- [ ] **Open the PR** with the per-source RAID delta against v6 in the description.

---

## Self-review

Spec coverage check (priority items 1-7):
- **#1 Mondrian conformal** → Phase A (A1-A6). Covers (a) class-conditional, (b) router-stratified, (c) ACI; (d) empty-set lives in Phase B.
- **#2 Decision-engine coverage accounting** → Phase B (B1-B4).
- **#3 _missing + _confidence + retrain** → Phase C (C1-C7).
- **#4 Hold-out by LLM family** → Phase D (D1-D3).
- **#5 RAID in CI** → Phase E (E1-E5).
- **#6 skops + metadata** → Phase F (F1-F2).
- **#7 Retrieval cross-similarity** → Phase G (G1-G4).

Type/signature consistency check:
- `ConformalWrapper.predict_set(p_llm, router_class=None)` consistent across A2, A4, A5, A6, C6.
- `EBMFusion.fuse(layer_results, word_count, domain)`, `_naive_fuse(results, domain)`, `_ebm_fuse(results, word_count, domain)`, `_maybe_conformal(p_llm, router_class)` consistent across A4 and C6.
- `Determination` and `FusionResult` only **append** new fields with defaults — CLAUDE.md invariant honoured.
- `EvalReport.per_source` is plain `dict` — consistent in D2, D3, E3.
- `RetrievalIndex.query(sid_or_text, k=5)` and `RetrievalIndex.add(sid, text)` consistent across G1, G2, G3.

No-placeholder check: every step contains either complete code or a concrete bash invocation with expected output. The two non-code steps (Task E2 Step 2's network download, Task C7's retrain) have explicit fallback instructions.

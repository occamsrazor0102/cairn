# TOCSIN Calibration Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale TOCSIN calibration path with a versioned, signature-guarded calibrator and build a coherent `prompts_only_tocsin_v2` calibrator/EBM/conformal candidate from fresh current-schema features.

**Architecture:** Add detector signatures and one centralized calibration guard, propagate one compute device through faithful TOCSIN, generate and validate fresh GPU feature caches behind a measured `$15` Modal cost gate, then fit and evaluate a matched calibrator, EBM, and conformal wrapper. The existing `prompts_only` profile stays untouched; the new profile is a separately versioned candidate and is promoted only by deterministic gates.

**Tech Stack:** Python 3.11/3.14, pytest, NumPy, SciPy, scikit-learn isotonic regression and metrics, InterpretML EBM, skops/HMAC serialization, PyTorch, Hugging Face Transformers, Modal H200, YAML, JSONL, Markdown/CSV reports.

## Global Constraints

- Preserve the existing dirty worktree. Do not stage or alter `.claude/homunculus/`, `docs/reports/`, unrelated `outputs/`, or the modified `cairn/privacy/__pycache__/*.pyc` files. The final conditional commit may include only the sanitized TOCSIN manifests/reports named in Task 11.
- Keep `configs/.profile_prompts_only.yaml` and `models/calib2000/profiles/prompts_only/` unchanged.
- Keep the 14 July 10 labelled rows out of detector calibration, EBM training, and conformal calibration. They are an adaptive challenge slice only.
- Preserve the approved binary target: Human is label `0`; LLM and humanized/AI-edited positives are label `1`. Do not claim that AI-edited text is equivalent to untouched model output.
- Do not promote the July-derived `0.836` threshold or any other challenge-set operating point.
- Keep `contrastive_gen` and `dna_gpt` disabled. Any active result from either API-backed layer invalidates a generated cache.
- Use `configs/.profile_full_coverage_with_specdetect.yaml` for fresh feature generation and force every enabled phase to run. Do not claim coverage from research layers that this approved profile intentionally leaves disabled (`lapd`, `cross_similarity`, and `contributor_graph`).
- Fit the TOCSIN calibrator only to the combined signature `tocsin-bart-fastdetectgpt-combined-w-v1` and raw channel `combined_w`. Treat the cohesiveness-only signature as a separate, uncalibrated fallback.
- Source all unchanged non-TOCSIN calibration entries from `models/calib2000/profiles/prompts_specdetect/detector_calibration.json` and prove their parsed JSON remains identical.
- Stop before the full Modal pass unless the 256-row pilot projects cumulative pilot-plus-full compute at or below `$15.00`.
- Use unique run IDs containing UTC time and the current Git revision; never resume the legacy `v8` checkpoint.
- Large caches and the private challenge data remain untracked under `outputs/tocsin_retrain_2026-07-10/`.
- The checkout currently lacks `models/calib2000/profiles/prompts_specdetect/ebm.skops` and `ebm.skops.hmac`. Never rebuild an approximation and call it the historical comparator. Candidate engineering may proceed, but the promotion decision must be `DO_NOT_PROMOTE` until those exact artifacts are restored from a trusted backup and pass HMAC verification.
- Even a gate-passing candidate remains a deployment-review candidate, not a production-readiness claim; a fresh group-disjoint AI-edited-versus-Human benchmark remains required.
- Follow red-green-refactor for every behavior change. Run the named focused test first and confirm the stated failure before editing production code.

---

### Task 1: Version detector-calibration artifacts and guard their application

**Files:**

- Modify: `cairn/calibration.py`
- Modify: `cairn/detectors/token_cohesiveness.py`
- Modify: `tests/test_calibration.py`
- Modify: `tests/test_detectors/test_token_cohesiveness.py`
- Create: `tests/test_calibration_guard.py`

- [ ] **Step 1: Add failing tests for TOCSIN signatures and versioned metadata**

Add these public constants to the test imports and assert them in the existing combined/fallback detector tests:

```python
TOCSIN_COMBINED_SIGNATURE = "tocsin-bart-fastdetectgpt-combined-w-v1"
TOCSIN_COHESIVENESS_SIGNATURE = "tocsin-bart-cohesiveness-u-v1"
TOCSIN_SIGNATURE_TO_RAW_CHANNEL = {
    TOCSIN_COMBINED_SIGNATURE: "combined_w",
    TOCSIN_COHESIVENESS_SIGNATURE: "cohesiveness_u",
}
```

For the base-available branch, require:

```python
assert result.signals["calibration_signature"] == TOCSIN_COMBINED_SIGNATURE
assert result.signals["calibration_raw_channel"] == "combined_w"
assert result.raw_score == pytest.approx(
    result.signals["combined_w"], abs=1e-5
)
```

For the base-unavailable branch, require:

```python
assert result.signals["calibration_signature"] == TOCSIN_COHESIVENESS_SIGNATURE
assert result.signals["calibration_raw_channel"] == "cohesiveness_u"
assert result.raw_score == pytest.approx(
    result.signals["cohesiveness_u"], abs=1e-5
)
```

Create `tests/test_calibration_guard.py` with a `LayerResult` factory and tests covering:

```python
def test_current_signature_rejects_legacy_entry_before_clipping():
    legacy = DetectorCalibrator()
    legacy.fit("token_cohesiveness", [0.0, 0.01], [0, 1])
    result = layer(raw=0.005)

    assert legacy.apply(result) is False
    assert result.calibrator_applied is False
    assert result.determination == "SKIP"
    assert result.signals["calibrator_signature_mismatch"] == {
        "result_signature": TOCSIN_COMBINED_SIGNATURE,
        "artifact_signature": None,
        "result_raw_channel": "combined_w",
        "artifact_raw_channel": None,
    }


def test_matching_entry_rejects_beyond_support_tolerance():
    result = layer(raw=0.226)

    assert fitted_versioned().apply(result) is False
    assert result.calibrator_applied is False
    assert result.determination == "SKIP"
    assert result.signals["calibrator_out_of_domain"] == {
        "raw_score": pytest.approx(0.226),
        "support_min": pytest.approx(-0.4),
        "support_max": pytest.approx(0.1),
        "tolerance": pytest.approx(0.125),
    }


def test_matching_entry_allows_score_on_tolerance_boundary():
    result = layer(raw=0.225)

    assert fitted_versioned().apply(result) is True
    assert result.calibrator_applied is True


def test_legacy_entry_still_applies_to_unsigned_legacy_result():
    legacy = DetectorCalibrator()
    legacy.fit("preamble", [0.0, 1.0], [0, 1])
    result = layer(
        raw=0.5,
        layer_id="preamble",
        signature=None,
        raw_channel=None,
    )

    assert legacy.apply(result) is True
    assert result.calibrator_applied is True


def test_preexisting_skip_is_ignored_without_mutation():
    result = layer(
        raw=999.0,
        signature="wrong",
        raw_channel="wrong",
        determination="SKIP",
    )
    before = copy.deepcopy(result)

    assert fitted_versioned().apply(result) is False
    assert result == before
```

Also require a v2 save/load round trip with this exact entry shape:

```json
{
  "X": [-0.4, -0.2, 0.0, 0.1],
  "y": [0.0, 0.0, 1.0, 1.0],
  "schema": "isotonic-thresholds-v2",
  "detector_signature": "tocsin-bart-fastdetectgpt-combined-w-v1",
  "raw_channel": "combined_w",
  "support_min": -0.4,
  "support_max": 0.1,
  "training_rows": 4,
  "class_counts": {"0": 2, "1": 2},
  "training_dataset_hash": "sha256:abc",
  "weighting_policy": {"policy": "source-length-balanced-v1"}
}
```

- [ ] **Step 2: Run the focused tests and confirm red**

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m pytest -q tests/test_detectors/test_token_cohesiveness.py -k "combined_channel or standalone_fallback"
python -m pytest -q tests/test_calibration_guard.py
```

Expected: detector assertions fail with missing signal keys, and the guard test fails to import `CalibrationMetadata` or find `DetectorCalibrator.apply`.

- [ ] **Step 3: Implement metadata-aware fitting, v1 compatibility, and one guarded apply method**

Add to `cairn/calibration.py`:

```python
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from cairn.contracts import LayerResult

CALIBRATION_SUPPORT_TOLERANCE_FRACTION = 0.25
CALIBRATION_SUPPORT_TOLERANCE_FLOOR = 1e-12


@dataclass(frozen=True)
class CalibrationMetadata:
    detector_signature: str
    raw_channel: str
    training_dataset_hash: str
    weighting_policy: dict[str, object]
```

Extend `DetectorCalibrator` with `_metadata`, `_fit_stats`, and `_serialized_entries` dictionaries and this fitting contract:

```python
def fit(
    self,
    detector_id: str,
    raw_scores: Iterable[float],
    labels: Iterable[int],
    *,
    sample_weight: Iterable[float] | None = None,
    metadata: CalibrationMetadata | None = None,
) -> None:
    x = np.asarray(list(raw_scores), dtype=float)
    y = np.asarray(list(labels), dtype=float)
    weights = (
        None
        if sample_weight is None
        else np.asarray(list(sample_weight), dtype=float)
    )
    if x.shape != y.shape:
        raise ValueError("raw_scores and labels must have the same shape")
    if weights is not None and weights.shape != x.shape:
        raise ValueError("sample_weight must match raw_scores")
    if not np.isfinite(x).all():
        raise ValueError("raw_scores must all be finite")
    if weights is not None and (
        not np.isfinite(weights).all() or (weights <= 0).any()
    ):
        raise ValueError("sample_weight must be finite and positive")
    if len(x) < 2 or len(set(y.tolist())) < 2:
        return
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(x, y, sample_weight=weights)
    self._models[detector_id] = ir
    self._metadata[detector_id] = metadata
    self._fit_stats[detector_id] = {
        "support_min": float(x.min()),
        "support_max": float(x.max()),
        "training_rows": int(len(x)),
        "class_counts": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
    }
    self._serialized_entries.pop(detector_id, None)
```

When `metadata is not None`, the v2 serializer reads `support_min`, `support_max`, `training_rows`, and `class_counts` from `_fit_stats` and stores the metadata fields. When `metadata is None`, a newly fit model serializes in the existing v1 `{X, y, schema}` shape; add a regression test for that all-detector compatibility path and never call `asdict(None)`. A v2 load reconstructs `_fit_stats` from the artifact. A v1 load keeps the parsed entry in `_serialized_entries`; saving an untouched legacy entry writes that dictionary back unchanged. This parsed-equality behavior is required for the isolated merge in Task 6.

Implement the single application path:

```python
def apply(self, result: LayerResult) -> bool:
    if result.determination == "SKIP":
        return False
    result.calibrator_applied = False
    model = self._models.get(result.layer_id)
    if model is None:
        return False
    raw = float(result.raw_score)
    if not np.isfinite(raw):
        result.signals["calibrator_nonfinite_raw"] = {"raw_score": repr(raw)}
        result.determination = "SKIP"
        return False

    entry = self._entry_metadata(result.layer_id)
    artifact_signature = entry.get("detector_signature")
    artifact_channel = entry.get("raw_channel")
    result_signature = result.signals.get("calibration_signature")
    result_channel = result.signals.get("calibration_raw_channel")
    identity_declared = any(
        value is not None
        for value in (
            artifact_signature,
            artifact_channel,
            result_signature,
            result_channel,
        )
    )
    if identity_declared and (
        artifact_signature != result_signature
        or artifact_channel != result_channel
    ):
        result.signals["calibrator_signature_mismatch"] = {
            "result_signature": result_signature,
            "artifact_signature": artifact_signature,
            "result_raw_channel": result_channel,
            "artifact_raw_channel": artifact_channel,
        }
        result.determination = "SKIP"
        return False

    support_min, support_max = self.support(result.layer_id)
    tolerance = max(
        CALIBRATION_SUPPORT_TOLERANCE_FRACTION
        * (support_max - support_min),
        CALIBRATION_SUPPORT_TOLERANCE_FLOOR,
    )
    if raw < support_min - tolerance or raw > support_max + tolerance:
        result.signals["calibrator_out_of_domain"] = {
            "raw_score": raw,
            "support_min": support_min,
            "support_max": support_max,
            "tolerance": tolerance,
        }
        result.determination = "SKIP"
        return False

    result.p_llm = float(model.predict([raw])[0])
    result.calibrator_applied = True
    return True
```

For legacy entries, `support()` returns the first and last fitted `X` knots. For v2 entries it returns the stored support. Do not alter `raw_score`, native `p_llm`, or detector evidence on rejection; only set the rejection signal and `determination="SKIP"` so both EBM and naive fusion treat the layer as missing.

Emit `calibration_signature` and `calibration_raw_channel` only for active TOCSIN results in `token_cohesiveness.py`. SKIP results remain unsigned.

- [ ] **Step 4: Run the calibration tests and confirm green**

```powershell
python -m pytest -q tests/test_calibration.py tests/test_calibration_guard.py tests/test_detectors/test_token_cohesiveness.py
```

Expected: all tests pass, including v1 compatibility and the 25% support-boundary cases.

- [ ] **Step 5: Commit the versioned guard**

```powershell
git add cairn/calibration.py cairn/detectors/token_cohesiveness.py tests/test_calibration.py tests/test_calibration_guard.py tests/test_detectors/test_token_cohesiveness.py
git commit -m "feat(calibration): version and guard detector calibrations"
```

---

### Task 2: Route every live, cached, and replay caller through the guard

**Files:**

- Modify: `cairn/pipeline.py`
- Modify: `cairn/feature_cache.py`
- Modify: `scripts/per_detector_auroc.py`
- Modify: `scripts/run_loss_filter.py`
- Modify: `scripts/_formb_profile_grid.py`
- Modify: `scripts/_formb_profile_grid_auc.py`
- Modify: `scripts/_stage1_cpu.py`
- Modify: `scripts/_stage1_cpu_clampoff.py`
- Modify: `tests/test_pipeline_detector_calibration.py`
- Create: `tests/test_calibration_application.py`

- [ ] **Step 1: Add a failing live-versus-cache parity test**

Create a parametrized test with these cases:

```python
@pytest.mark.parametrize(
    "raw,signature,determination,expected_applied,rejection_key",
    [
        (0.0, TOCSIN_COMBINED_SIGNATURE, "AMBER", True, None),
        (0.0, "wrong-signature", "AMBER", False,
         "calibrator_signature_mismatch"),
        (0.226, TOCSIN_COMBINED_SIGNATURE, "AMBER", False,
         "calibrator_out_of_domain"),
        (999.0, "wrong-signature", "SKIP", False, None),
    ],
)
def test_live_and_cached_pipeline_share_calibration_guard(
    tmp_path,
    monkeypatch,
    raw,
    signature,
    determination,
    expected_applied,
    rejection_key,
):
    calibrator = make_versioned_calibrator()
    seed = make_tocsin_result(raw, signature, determination)
    live = run_stubbed_live_phase(calibrator, seed)
    cached = run_stubbed_cached_pipeline(tmp_path, calibrator, seed)

    assert cached.p_llm == pytest.approx(live.p_llm)
    assert cached.calibrator_applied is live.calibrator_applied
    assert cached.determination == live.determination
    assert cached.signals == live.signals
    assert live.calibrator_applied is expected_applied
    if rejection_key is not None:
        assert rejection_key in live.signals
        vector = FeatureAssembler().assemble(
            [live], word_count=100, domain="prose"
        )
        assert vector["tocsin_missing"] == 1.0
```

Spy on `calibrator.apply` and require exactly one call in each path.

- [ ] **Step 2: Run the parity test and confirm red**

```powershell
python -m pytest -q tests/test_calibration_application.py
```

Expected: the apply spy records no calls because both paths still call `transform` directly.

- [ ] **Step 3: Replace every runtime/replay transform block**

Use one form everywhere:

```python
if calibrator is not None:
    calibrator.apply(result)
```

In `CairnPipeline._run_phase`, use `self._detector_calibrator.apply(result)`. In `CachedPipeline`, deserialize a fresh result and call the inner calibrator. In replay scripts, construct `LayerResult` before applying calibration; keep rejected results in detailed output but let their `SKIP` determination remove them from fusion.

Also apply the guard to any batch-produced `cross_similarity` results before fusion. Remove now-unused `math` imports after replacing direct finite checks.

- [ ] **Step 4: Prove no direct runtime transforms remain**

```powershell
python -m pytest -q tests/test_calibration_application.py tests/test_pipeline_detector_calibration.py
rg -n "calibrator\.transform|cal\.transform|_detector_calibrator\.transform" cairn scripts
```

Expected: tests pass and `rg` exits with no matches. Calls inside calibration-focused tests may still call `transform` for diagnostic assertions.

- [ ] **Step 5: Commit centralized application**

```powershell
git add cairn/pipeline.py cairn/feature_cache.py scripts/per_detector_auroc.py scripts/run_loss_filter.py scripts/_formb_profile_grid.py scripts/_formb_profile_grid_auc.py scripts/_stage1_cpu.py scripts/_stage1_cpu_clampoff.py tests/test_pipeline_detector_calibration.py tests/test_calibration_application.py
git commit -m "refactor(calibration): centralize live and cached application"
```

---

### Task 3: Move faithful TOCSIN and standalone Fast-DetectGPT to one device

**Files:**

- Modify: `cairn/benchmarks/fast_detectgpt.py`
- Modify: `cairn/detectors/fast_detectgpt.py`
- Modify: `cairn/detectors/token_cohesiveness.py`
- Modify: `tests/test_detectors/test_fast_detectgpt.py`
- Modify: `tests/test_detectors/test_token_cohesiveness.py`

- [ ] **Step 1: Add lightweight CUDA/CPU tests without model downloads**

Use fake tensor, tokenizer, BART, and torch objects. The central assertions are:

```python
def test_tocsin_moves_bart_tensors_and_fastdetectgpt_to_cuda(monkeypatch):
    tokenizer = FakeTokenizer()
    bart = FakeBart()
    monkeypatch.setattr(tc, "_try_import_torch", lambda: FakeCudaTorch)
    monkeypatch.setattr(
        tc, "_load_bart_components", lambda _name: (tokenizer, bart)
    )

    detector = tc.TokenCohesivenessDetector()

    assert detector._device == "cuda"
    assert detector._base._device == "cuda"
    detector._ensure_bart()
    assert bart.to_calls == ["cuda"]
    assert bart.eval_called is True
    assert detector._bart_score("source", "target") == -1.25
    assert [tensor.to_calls for tensor in tokenizer.tensors] == [
        ["cuda"], ["cuda"]
    ]


def test_tocsin_defaults_to_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(tc, "_try_import_torch", lambda: FakeCpuTorch)
    detector = tc.TokenCohesivenessDetector()
    assert detector._device == "cpu"
    assert detector._base._device == "cpu"


def test_tocsin_honors_explicit_device(monkeypatch):
    monkeypatch.setattr(tc, "_try_import_torch", lambda: FakeCudaTorch)
    detector = tc.TokenCohesivenessDetector(device="cpu")
    assert detector._device == "cpu"
    assert detector._base._device == "cpu"


def test_standalone_fastdetectgpt_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr(
        "cairn.benchmarks.fast_detectgpt._try_import_torch",
        lambda: (FakeCudaTorch, True),
    )
    detector = FastDetectGPTDetector()
    assert detector._adapter._device == "cuda"
```

- [ ] **Step 2: Run the device tests and confirm red**

```powershell
python -m pytest -q tests/test_detectors/test_token_cohesiveness.py tests/test_detectors/test_fast_detectgpt.py -k "cuda or cpu or explicit_device"
```

Expected: the constructor rejects `device`, or `_load_bart_components` is missing.

- [ ] **Step 3: Implement device selection and tensor movement**

Add the shared selector to `cairn/benchmarks/fast_detectgpt.py` and import it from TOCSIN:

```python
def _select_device(torch_module, requested: str | None = None) -> str:
    if requested is not None:
        return requested
    return (
        "cuda"
        if torch_module is not None and torch_module.cuda.is_available()
        else "cpu"
    )
```

Add the BART loader to `cairn/detectors/token_cohesiveness.py`:

```python
def _load_bart_components(model_name: str):
    from transformers import AutoTokenizer, BartForConditionalGeneration
    return (
        AutoTokenizer.from_pretrained(model_name),
        BartForConditionalGeneration.from_pretrained(model_name),
    )
```

In `FastDetectGPTAdapter.__init__`, precedence is explicit constructor device, `CAIRN_FAST_DETECTGPT_DEVICE`, then `_select_device(torch_module)`. Extend `FastDetectGPTDetector(scoring_model=None, device=None)` and its lazy wrapper so the detector config's optional `device` value reaches the adapter. This makes the standalone phase-2 detector use CUDA on Modal as well as TOCSIN's embedded base channel.

Change the constructor to:

```python
def __init__(
    self,
    model_name: str = _BART_MODEL,
    device: str | None = None,
) -> None:
    self._model_name = model_name
    torch = _try_import_torch()
    self._device = _select_device(torch, device)
    self._bart = None
    self._bart_tok = None
    self._base = FastDetectGPTAdapter(device=self._device)
```

Load and score on that device:

```python
self._bart_tok, self._bart = _load_bart_components(self._model_name)
self._bart = self._bart.to(self._device)
self._bart.eval()

enc = {
    key: value.to(self._device)
    for key, value in self._bart_tok(
        source, return_tensors="pt", truncation=True, max_length=512
    ).items()
}
labels = self._bart_tok(
    target, return_tensors="pt", truncation=True, max_length=512
)["input_ids"].to(self._device)
```

Pass optional `device` from `_LazyTOCSIN.analyze()` alongside the configured model name. CPU behavior and lazy optional imports must remain intact.

- [ ] **Step 4: Run detector tests and commit**

```powershell
python -m pytest -q tests/test_detectors/test_token_cohesiveness.py tests/test_detectors/test_fast_detectgpt.py
git add cairn/benchmarks/fast_detectgpt.py cairn/detectors/fast_detectgpt.py cairn/detectors/token_cohesiveness.py tests/test_detectors/test_fast_detectgpt.py tests/test_detectors/test_token_cohesiveness.py
git commit -m "feat(detectors): propagate GPU device through TOCSIN"
```

Expected: all focused detector tests pass without downloading model weights.

---

### Task 4: Build, validate, union, and slice current-schema TOCSIN caches

**Files:**

- Create: `cairn/tocsin_cache.py`
- Create: `scripts/build_tocsin_cache_inputs.py`
- Create: `scripts/validate_tocsin_cache.py`
- Create: `scripts/slice_tocsin_feature_cache.py`
- Create: `scripts/hash_artifacts.py`
- Create: `tests/test_tocsin_cache.py`

- [ ] **Step 1: Add cache-contract tests**

Define this immutable summary:

```python
@dataclass(frozen=True)
class TocsinCacheSummary:
    dataset_sha256: str
    cache_sha256: str
    expected_unique_hashes: int
    cached_unique_hashes: int
    active_rows: int
    skip_rows: int
    calibration_signature: str | None
    calibration_raw_channel: str | None
    raw_min: float | None
    raw_max: float | None
    identical_duplicate_hashes: int
    api_layer_ids: tuple[str, ...]
    layer_counts: dict[str, dict[str, int]]
    unexpected_layer_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
```

Add tests proving a valid combined-signature cache passes and each of these fails independently:

- normalized-text hash coverage mismatch;
- more than one TOCSIN result per record;
- an eligible row missing an active TOCSIN result;
- a short row not represented as `SKIP` with `skipped="too_short"`;
- retired `mean_deletion_impact` evidence;
- missing or mixed signatures;
- channel/signature disagreement;
- a nonfinite raw score;
- disagreement between `raw_score` and the named signal channel beyond `1e-5`;
- conflicting duplicate hashes;
- any `contrastive_gen` or `dna_gpt` result.
- any configured enabled phase-1/2/3 layer absent from a router-eligible cache record.

Resolve enabled layer IDs from the supplied profile's merged `detectors` map, intersected with `PHASE1_DETECTORS + PHASE2_DETECTORS + PHASE3_DETECTORS`. For each cache record, subtract that record's `router.skip_detectors` before requiring results. For each layer, report `present`, `active`, `skip`, `router_skipped`, and `unexpected_absent` counts. Always fail when `unexpected_absent > 0`. Fail when an enabled layer is never present anywhere only when `require_all_layers_seen=True`; use that strict mode for the multi-domain union cache, not for individual slices where every row may legitimately router-skip a layer. Add tests for both modes. `contributor_graph` is batch-only and `cross_similarity` is phase 4; neither is expected from per-text cache generation. Record disabled layers separately so the report cannot imply they ran. This preserves the historical full-coverage profile's router semantics and keeps the restored comparator on its training-time feature-presence contract.

Representative failure test:

```python
def test_rejects_mixed_current_signatures(tmp_path):
    texts = [
        " ".join(["combined"] * 80),
        " ".join(["fallback"] * 80),
    ]
    dataset = write_dataset(tmp_path / "data.jsonl", texts)
    cache = write_cache(
        tmp_path / "features.jsonl",
        [
            cache_record(texts[0], TOCSIN_COMBINED_SIGNATURE),
            cache_record(texts[1], TOCSIN_COHESIVENESS_SIGNATURE),
        ],
    )

    with pytest.raises(ValueError, match="mixed TOCSIN signatures"):
        validate_tocsin_cache(dataset, cache)
```

- [ ] **Step 2: Run the cache tests and confirm red**

```powershell
python -m pytest -q tests/test_tocsin_cache.py
```

Expected: collection fails because `cairn.tocsin_cache` does not exist.

- [ ] **Step 3: Implement the pure cache APIs**

Expose `validate_tocsin_cache(dataset_path, cache_path, *, expected_signature=None, expected_layer_ids=(), require_all_layers_seen=False) -> TocsinCacheSummary`. Read JSONL directly so duplicate hashes remain visible before indexing; compare the cache keys with `sha256(normalize_text(text))`; enforce every invariant in Step 1 with a `ValueError` that names the path, hash, layer, and violation. Do not use `load_cache()` for validation because it overwrites duplicate keys.

Expose `build_union_dataset(dataset_paths, output_path) -> dict[str, object]`. Preserve the first complete JSON object for each normalized-text hash in deterministic dataset/row order. Its returned manifest records each input's path, full SHA-256, total rows, unique rows, duplicate rows, eligible rows, plus the union's total/unique/eligible counts and full SHA-256.

Expose `extract_cache_for_dataset(source_cache, dataset_path, output_path) -> int`. Index the already-validated source cache by hash, write records exactly once in dataset hash order, and raise with the first absent hash instead of producing a partial slice.

Add `hash_artifacts(paths: Sequence[Path]) -> dict[str, object]` to `cairn/tocsin_cache.py`. It returns a sorted mapping from normalized path string to `{sha256, bytes}` and rejects absent or non-file paths. `scripts/hash_artifacts.py` accepts one or more paths plus required `--out`, writes indented JSON atomically, and is covered by a test that changes one byte and observes a changed digest.

Implement all four script CLIs as thin argument-parsing wrappers. `validate_tocsin_cache.py` requires `--config`, accepts `--require-all-layers-seen`, resolves enabled phase-1/2/3 layers from that merged profile, and writes `summary.to_dict()` as indented JSON only after successful validation.

- [ ] **Step 4: Confirm green and commit**

```powershell
python -m pytest -q tests/test_tocsin_cache.py
git add cairn/tocsin_cache.py scripts/build_tocsin_cache_inputs.py scripts/validate_tocsin_cache.py scripts/slice_tocsin_feature_cache.py scripts/hash_artifacts.py tests/test_tocsin_cache.py
git commit -m "feat(cache): validate current TOCSIN features"
```

Expected: all cache tests pass.

---

### Task 5: Add measured Modal timing, unique run IDs, and the hard cost gate

**Files:**

- Modify: `scripts/modal_cache_features.py`
- Create: `tests/test_modal_cache_features.py`

- [ ] **Step 1: Add failing tests for pilot selection and cost projection**

Load `scripts/modal_cache_features.py` with a stubbed `modal` module, following the import pattern in `tests/test_modal_run_hc3_benchmark.py`. Test these pure helpers:

```python
def validate_run_id(run_id: str) -> str:
    if run_id in {"v8", "default"} or len(run_id) < 12:
        raise ValueError("run_id must be a unique descriptive identifier")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]+", run_id) is None:
        raise ValueError("run_id contains unsafe path characters")
    return run_id


def test_pilot_is_deterministic_and_stratified(modal_cache_module):
    rows = []
    for index in range(300):
        rows.append({"text": " ".join([f"long{index}"] * 80)})
        rows.append({"text": f"short {index}"})
    payload = "".join(
        json.dumps(row) + "\n" for row in rows
    ).encode("utf-8")

    first = modal_cache_module.select_pilot_rows(
        payload, n_rows=256, seed=20260710
    )
    second = modal_cache_module.select_pilot_rows(
        payload, n_rows=256, seed=20260710
    )
    selected = [json.loads(line) for line in first.decode().splitlines()]

    assert first == second
    assert len(selected) == 256
    assert sum(len(row["text"].split()) >= 60 for row in selected) == 128
```

Cost tests use this exact projection:

```python
def project_modal_cost(
    pilot_manifest: Mapping[str, object],
    *,
    total_rows: int,
    eligible_rows: int,
    compute_usd_per_second: Decimal,
    cap_usd: Decimal = Decimal("15.00"),
) -> dict[str, object]:
    n_eligible = int(pilot_manifest["n_eligible"])
    n_ineligible = int(pilot_manifest["n_ineligible"])
    eligible_seconds_per_row = (
        Decimal(str(pilot_manifest["eligible_scoring_seconds"]))
        / Decimal(n_eligible)
    )
    ineligible_seconds_per_row = (
        Decimal(str(pilot_manifest["ineligible_scoring_seconds"]))
        / Decimal(n_ineligible)
    )
    projected_full_seconds = (
        Decimal(str(pilot_manifest["warmup_seconds"]))
        + Decimal(eligible_rows) * eligible_seconds_per_row
        + Decimal(total_rows - eligible_rows) * ineligible_seconds_per_row
    )
    projected_cumulative_usd = (
        Decimal(str(pilot_manifest["worker_elapsed_seconds"]))
        + projected_full_seconds
    ) * compute_usd_per_second
    return {
        "projected_full_seconds": float(projected_full_seconds),
        "projected_cumulative_usd": format(
            projected_cumulative_usd, "f"
        ),
        "gate": (
            "PASS"
            if projected_cumulative_usd <= cap_usd
            else "STOP_COST_CAP"
        ),
    }
```

Require `PASS` at exactly `$15.00` and `STOP_COST_CAP` above it. Require the projection to include both the measured pilot and projected full run.

- [ ] **Step 2: Run the Modal helper tests and confirm red**

```powershell
python -m pytest -q tests/test_modal_cache_features.py
```

Expected: the helper attributes are absent.

- [ ] **Step 3: Make GPU and host-resource accounting explicit**

Set the function resources to:

```python
@app.function(
    image=image,
    gpu="H200",
    cpu=4.0,
    memory=49152,
    timeout=21600,
    secrets=[modal.Secret.from_dotenv()],
    volumes={"/root/.cache/huggingface": hf_cache},
)
```

The July 10 accounting rate is `$0.00141996/sec`: H200 `$0.001261/sec`, four physical cores `$0.0000524/sec`, and 48 GiB host memory `$0.00010656/sec`, based on [Modal's current resource pricing](https://modal.com/pricing). Keep the rate a required/recorded CLI value so a later run can substitute the then-current price.

Remove the default `run_id="v8"` from the remote and local entrypoints. Require a validated run ID. Add local arguments:

```text
--pilot-rows 256
--manifest-path PATH
--gate-manifest-path PATH
--prior-manifest-path PATH
--cost-cap-usd 15
--compute-usd-per-second 0.00141996
```

`--pilot-rows` selects exactly 128 eligible and 128 ineligible rows with seed `20260710`. The remote worker warms the pipeline once on an eligible row, then records per-row elapsed time into eligible and ineligible accumulators.

Return and persist these manifest keys:

```text
run_id,source_dataset_sha256,scored_payload_sha256,config_path,
resolved_config_sha256,git_revision,n_total,n_eligible,
n_ineligible,warmup_seconds,eligible_scoring_seconds,
ineligible_scoring_seconds,worker_elapsed_seconds,
eligible_rows_per_second,total_rows_per_second,gpu_type,remote_path,
python_version,torch_version,transformers_version,cuda_version,
pilot_seed,detector_seed,pyproject_sha256,package_versions
```

The local entrypoint hashes the complete union bytes before pilot selection and passes that value separately from the selected payload. Thus a pilot has the union hash in `source_dataset_sha256` and a different 256-row `scored_payload_sha256`; a full run has identical source/payload hashes. A pilot always writes its manifest. If the projected cumulative cost exceeds the cap, write `gate="STOP_COST_CAP"` and exit nonzero after the manifest exists. A full run with `--gate-manifest-path` must verify `gate="PASS"` plus identical `source_dataset_sha256`, resolved-config SHA-256, and Git revision before invoking Modal. Persist the measured eligible/ineligible seconds-per-row and warmup seconds as `cost_model` and propagate that model into later manifests. Every completed run records `run_compute_usd = worker_elapsed_seconds * compute_usd_per_second`; a run with `--prior-manifest-path` uses the inherited cost model to project its own rows, adds the prior manifest's `cumulative_compute_usd`, refuses to start when the projection would cross the cap, and writes the new cumulative value after completion.

- [ ] **Step 4: Confirm green and commit**

```powershell
python -m pytest -q tests/test_modal_cache_features.py
git add scripts/modal_cache_features.py tests/test_modal_cache_features.py
git commit -m "feat(modal): gate TOCSIN cache refresh by cost"
```

Expected: all pure helper tests pass without opening a Modal connection.

---

### Task 6: Add deterministic source/length weighting and isolated calibration merge

**Files:**

- Create: `cairn/calibration_training.py`
- Modify: `scripts/calibrate_detectors.py`
- Create: `tests/test_calibration_training.py`
- Create: `tests/test_calibrate_detectors_cli.py`

- [ ] **Step 1: Test the exact weighting policy**

Implement tests against this interface:

```python
@dataclass(frozen=True)
class SourceLengthWeights:
    weights: tuple[float, ...]
    manifest: dict[str, object]


def compute_source_length_weights(
    samples: Sequence[EvalSample],
) -> SourceLengthWeights:
    rows = [
        {
            "index": index,
            "label": int(sample.label),
            "source": sample.source or sample.source_id or "<missing>",
            "bin": _length_bin(len(normalize_text(sample.text).split())),
        }
        for index, sample in enumerate(samples)
    ]
    by_stratum: dict[tuple[str, int], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for row in rows:
        by_stratum[(str(row["bin"]), int(row["label"]))].append(row)

    weights = [0.0] * len(samples)
    strata: list[dict[str, object]] = []
    for bin_name in ("60-100", "100-150", "150-250", "250-inf"):
        n_human = len(by_stratum[(bin_name, 0)])
        n_positive = len(by_stratum[(bin_name, 1)])
        if not n_human or not n_positive:
            raise ValueError(f"length bin {bin_name} lacks both classes")
        target = float(min(n_human, n_positive))

        for label in (0, 1):
            members = by_stratum[(bin_name, label)]
            source_counts = Counter(str(row["source"]) for row in members)
            initial = [
                1.0 / source_counts[str(row["source"])]
                for row in members
            ]
            initial_median = median(initial)
            cap = 10.0 * initial_median
            clipped = [min(value, cap) for value in initial]
            scale = target / sum(clipped)
            for row, value in zip(members, clipped):
                weights[int(row["index"])] = value * scale

            stratum_weights = [
                weights[int(row["index"])] for row in members
            ]
            source_totals = {
                source: sum(
                    weights[int(row["index"])]
                    for row in members
                    if str(row["source"]) == source
                )
                for source in sorted(source_counts)
            }
            effective_sample_size = (
                sum(stratum_weights) ** 2
                / sum(value * value for value in stratum_weights)
            )
            strata.append({
                "length_bin": bin_name,
                "label": label,
                "rows": len(members),
                "target_weight": target,
                "actual_weight": sum(stratum_weights),
                "median_inverse_source_weight": initial_median,
                "clip_cap": cap,
                "effective_sample_size": effective_sample_size,
                "source_rows": dict(sorted(source_counts.items())),
                "source_total_weights": source_totals,
            })

    return SourceLengthWeights(
        weights=tuple(weights),
        manifest={
            "policy": "source-length-balanced-v1",
            "bins": ["60-100", "100-150", "150-250", "250-inf"],
            "clip_multiple": 10.0,
            "strata": strata,
        },
    )
```

The function imports `Counter` and `defaultdict` from `collections`, `median` from `statistics`, and the canonical `normalize_text` helper from `cairn.pipeline`. It uses this bin helper:

```python
LENGTH_BINS = ((60, 100), (100, 150), (150, 250), (250, None))


def _source(sample: EvalSample) -> str:
    return sample.source or sample.source_id or "<missing>"


def _length_bin(word_count: int) -> str:
    for lower, upper in LENGTH_BINS:
        if word_count >= lower and (upper is None or word_count < upper):
            return f"{lower}-{upper if upper is not None else 'inf'}"
    raise ValueError(
        f"TOCSIN weighting received ineligible row: {word_count} words"
    )
```

For each `(length_bin, label)` stratum:

1. Set each class's target total to `min(n_human, n_positive)` for that bin.
2. Give each row initial weight `1 / source_count` within its stratum.
3. Clip each row to `10 * median(initial_weights)`.
4. Renormalize the clipped weights to the stratum target.
5. Record row counts, target/actual weight, median, cap, effective sample size, per-source rows, and per-source total weights.

Test the full train corpus inventory separately from the pure synthetic unit test:

| Word bin | Human rows | Positive rows | Target weight per class |
|---|---:|---:|---:|
| 60–99 | 1,968 | 442 | 442 |
| 100–149 | 1,583 | 320 | 320 |
| 150–249 | 912 | 1,644 | 912 |
| 250+ | 53 | 1,562 | 53 |
| Total | 4,516 | 3,968 | 1,727 |

- [ ] **Step 2: Add failing CLI/merge tests**

The CLI contract is:

```text
--only-detector token_cohesiveness
--merge-from PATH
--detector-signature tocsin-bart-fastdetectgpt-combined-w-v1
--raw-channel combined_w
--source-length-weighting
--manifest-out PATH
```

Test that:

- `--only-detector` requires `--merge-from`;
- `--merge-from` is invalid without `--only-detector`;
- only exact-signature, exact-channel, active, finite rows enter the fit;
- the fit aborts unless both accepted classes contain at least 500 rows;
- a target entry may be added or replaced;
- every non-target parsed JSON entry is identical before and after;
- the merge source file remains byte-for-byte unchanged;
- the manifest stores full dataset/cache hashes, accepted/rejected counts, class counts, support, and the complete weighting manifest.

Use a single-purpose public function for direct unit testing:

```python
def fit_isolated_detector_calibration(
    observations: Iterable[
        tuple[LayerResult, EvalSample]
    ],
    *,
    detector_id: str,
    merge_from: Path,
    detector_signature: str,
    raw_channel: str,
    training_dataset_hash: str,
    min_class_rows: int = 500,
) -> tuple[DetectorCalibrator, dict[str, object]]:
    accepted = _select_matching_observations(
        observations,
        detector_id,
        detector_signature,
        raw_channel,
    )
    weighted = compute_source_length_weights(
        [sample for _, sample in accepted]
    )
    class_counts = Counter(sample.label for _, sample in accepted)
    if min(class_counts.get(0, 0), class_counts.get(1, 0)) < min_class_rows:
        raise ValueError(
            f"accepted class counts {dict(class_counts)} are below "
            f"minimum {min_class_rows}"
        )
    calibrator = _load_merge_base(merge_from)
    calibrator.fit(
        detector_id,
        [result.raw_score for result, _ in accepted],
        [sample.label for _, sample in accepted],
        sample_weight=weighted.weights,
        metadata=CalibrationMetadata(
            detector_signature=detector_signature,
            raw_channel=raw_channel,
            training_dataset_hash=training_dataset_hash,
            weighting_policy=weighted.manifest,
        ),
    )
    return calibrator, _build_training_manifest(accepted, weighted)
```

Preserve the current no-flag all-detector mode unchanged: when `args.only_detector is None`, reject `--merge-from`, signature/channel, source-length-weighting, and manifest-only options, then execute the existing unweighted loop for every observed detector. When `--only-detector` is present, require `--merge-from`, `--detector-signature`, `--raw-channel`, `--source-length-weighting`, and `--manifest-out`, and call only `fit_isolated_detector_calibration` with the default minimum of 500. Unit tests may pass `min_class_rows=2`; the CLI does not expose an override. Add a regression test proving the legacy no-flag mode still fits two synthetic detector IDs.

- [ ] **Step 3: Run both suites and confirm red**

```powershell
python -m pytest -q tests/test_calibration_training.py tests/test_calibrate_detectors_cli.py
```

Expected: imports fail because the weighting module and new CLI interface do not exist.

- [ ] **Step 4: Implement collection, weighted fit, and merge assertions**

When `--features-cache` is supplied, use the cache only to obtain raw detector results and router word counts. Do not let a configured calibrator pre-transform results during collection. Select TOCSIN observations before any calibration application, and verify `rd.word_count == len(normalize_text(sample.text).split())` for the weighting bin.

After save, parse the base and output JSON and assert:

```python
base_non_target = {
    key: value for key, value in base_payload.items()
    if key != args.only_detector
}
output_non_target = {
    key: value for key, value in output_payload.items()
    if key != args.only_detector
}
if output_non_target != base_non_target:
    raise RuntimeError("isolated calibration changed non-target entries")
```

Write the training manifest only after that assertion passes.

- [ ] **Step 5: Confirm green and commit**

```powershell
python -m pytest -q tests/test_calibration_training.py tests/test_calibrate_detectors_cli.py tests/test_calibration_guard.py
git add cairn/calibration_training.py scripts/calibrate_detectors.py tests/test_calibration_training.py tests/test_calibrate_detectors_cli.py
git commit -m "feat(training): add weighted isolated detector calibration"
```

Expected: all focused tests pass.

---

### Task 7: Support full-corpus EBM training and exact contribution reconstruction

**Files:**

- Modify: `scripts/train_fusion.py`
- Modify: `scripts/calibrate_conformal.py`
- Modify: `cairn/fusion/ebm.py`
- Modify: `tests/test_train_fusion_holdout.py`
- Modify: `tests/test_fusion.py`
- Create: `tests/test_calibrate_conformal_cli.py`

- [ ] **Step 1: Add failing `--train-all` tests**

Add `--train-all` as a boolean flag mutually exclusive with `--holdout-source`. The split logic must be:

```python
if args.train_all:
    X_train, y_train = X, y
    X_test, y_test = [], []
elif args.holdout_source:
    X_train, X_test, y_train, y_test = source_holdout_split(
        X, y, samples, args.holdout_source
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=42,
    )
```

The subprocess test must assert `trained on 30, held out 0` for a 30-row fixture and assert an argument-parser failure when both flags are supplied.

Require the `.meta.json` sidecar to record:

```text
dataset_path,dataset_sha256,features_cache_path,features_cache_sha256,
n_train,n_test,label_balance_train,label_balance_test,trained_at,
interactions,max_rounds,random_state,python_version,numpy_version,
sklearn_version,interpret_version,skops_version,feature_names,
dropped_features,schema_version
```

Use full 64-character SHA-256 values. Set `label_balance_test` to `null` when `n_test == 0`.

Add a regression test for the training/runtime length contract:

```python
def test_training_assembler_uses_profile_word_count_clamp():
    config = {"fusion": {"word_count_clamp": 200}}
    assembler = build_training_assembler(config, drop_features=set())
    vector = assembler.assemble([], word_count=600, domain="prompt")

    assert vector["word_count"] == 200.0
    assert vector["log_word_count"] == pytest.approx(math.log1p(200))
    assert vector["is_medium"] == 1.0
    assert vector["is_long"] == 0.0
```

The EBM metadata must record `word_count_clamp: 200` for the candidate.

Add optional `--manifest-out PATH` to `scripts/calibrate_conformal.py`. After the wrapper is saved, write an atomic JSON manifest containing the full dataset and feature-cache SHA-256 values, alpha, calibration row/class/router counts, fitted class/router thresholds, Python/NumPy/scikit-learn versions, and UTC timestamp. A subprocess test must assert those hashes and counts against a synthetic cached fixture.

- [ ] **Step 2: Add a failing interaction-reconstruction test**

The existing explanation code zips all term scores against only base feature names, losing interactions. Change the test model so local explanation returns one interaction name and require:

```python
result = fusion.fuse(layers, word_count=100, domain="prose")
assert "preamble_p_llm & nssi_p_llm" in result.feature_contributions

intercept = float(np.ravel(model.intercept_)[0])
logit = intercept + sum(result.feature_contributions.values())
reconstructed = 1.0 / (1.0 + np.exp(-logit))
assert reconstructed == pytest.approx(result.p_llm, abs=1e-9)
```

- [ ] **Step 3: Run focused tests and confirm red**

```powershell
python -m pytest -q tests/test_train_fusion_holdout.py tests/test_fusion.py tests/test_calibrate_conformal_cli.py -k "train_all or reconstruction or interaction or manifest"
```

Expected: the CLI rejects `--train-all`, and the interaction term is absent from contributions.

- [ ] **Step 4: Implement the training flag, metadata, and complete term attribution**

Load the fully resolved config in `train_fusion.py` and construct the training assembler through:

```python
def build_training_assembler(
    config: dict,
    *,
    drop_features: set[str],
) -> FeatureAssembler:
    clamp = (config.get("fusion") or {}).get("word_count_clamp")
    return FeatureAssembler(
        drop_features=drop_features,
        word_count_clamp=int(clamp) if clamp is not None else None,
    )
```

Use the same resolved config both to construct the pipeline and to construct this assembler. This prevents training on raw 600-word length features while runtime clamps them to 200.

Use the EBM explanation's own names:

```python
local_data = self._model.explain_local(X).data(0)
names = list(local_data.get("names") or features)
scores = list(local_data["scores"])
if len(names) != len(scores):
    raise RuntimeError(
        f"EBM explanation returned {len(names)} names "
        f"for {len(scores)} scores"
    )
contribs = {
    str(name): float(score)
    for name, score in zip(names, scores)
}
top = sorted(
    contribs.items(), key=lambda item: abs(item[1]), reverse=True
)[:5]
```

Keep the existing warning/empty-contribution fallback for model implementations that do not support local explanations; the promotion runner will fail candidate integrity if contributions are unavailable.

- [ ] **Step 5: Confirm green and commit**

```powershell
python -m pytest -q tests/test_train_fusion_holdout.py tests/test_fusion.py tests/test_fusion_serialization.py tests/test_calibrate_conformal_cli.py
git add scripts/train_fusion.py scripts/calibrate_conformal.py cairn/fusion/ebm.py tests/test_train_fusion_holdout.py tests/test_fusion.py tests/test_calibrate_conformal_cli.py
git commit -m "feat(fusion): support matched full-corpus TOCSIN training"
```

Expected: all focused training/fusion tests pass.

---

### Task 8: Run the 256-row pilot, enforce the cap, and generate fresh caches

**Files produced under untracked local output:**

- `outputs/tocsin_retrain_2026-07-10/inputs/all_corpora_union.jsonl`
- `outputs/tocsin_retrain_2026-07-10/pilot/pilot-256.features.jsonl`
- `outputs/tocsin_retrain_2026-07-10/caches/*.features.jsonl`
- `outputs/tocsin_retrain_2026-07-10/manifests/*.json`

- [ ] **Step 1: Build the union corpus and record immutable input provenance**

Run from `C:\Users\occam\My Projects\cairn`:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$Root = "outputs/tocsin_retrain_2026-07-10"
$Datasets = @(
  "data/prompts_v2_train.jsonl",
  "data/prompts_only_calib.jsonl",
  "data/prompts_v5_test.jsonl",
  "data/_humanized_eval.jsonl",
  "data/_hard_human_clean.jsonl",
  "data/ood_m4_test.jsonl",
  "data/ood_raid_test.jsonl",
  "data/ood_mage_test.jsonl"
)
New-Item -ItemType Directory -Force `
  "$Root/inputs", "$Root/pilot", "$Root/caches", "$Root/manifests" `
  | Out-Null

$ProductionFiles = @(
  "configs/.profile_prompts_only.yaml"
) + @(
  Get-ChildItem "models/calib2000/profiles/prompts_only" -File `
    | ForEach-Object { Resolve-Path -Relative $_.FullName }
)
python scripts/hash_artifacts.py `
  --out "$Root/manifests/production-before.json" `
  $ProductionFiles
if ($LASTEXITCODE -ne 0) { throw "Production hash baseline failed" }

python scripts/build_tocsin_cache_inputs.py `
  --union-out "$Root/inputs/all_corpora_union.jsonl" `
  --manifest-out "$Root/manifests/input-corpora.json" `
  --datasets $Datasets
if ($LASTEXITCODE -ne 0) { throw "Union-input build failed" }
```

Expected union inventory: `31,192` unique normalized texts and `11,926` TOCSIN-eligible texts. If the hashes or counts differ, inspect the manifest and stop; do not substitute the expected number into the file.

- [ ] **Step 2: Run the deterministic 256-row pilot**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$Revision = (git rev-parse --short=12 HEAD).Trim()
$RunBase = "tocsin-v2-$Stamp-$Revision"
Set-Content -Path "$Root/manifests/run-base.txt" -Value $RunBase

modal run scripts/modal_cache_features.py `
  --dataset-path "$Root/inputs/all_corpora_union.jsonl" `
  --out-path "$Root/pilot/pilot-256.features.jsonl" `
  --config-path "configs/.profile_full_coverage_with_specdetect.yaml" `
  --run-id "$RunBase-pilot" `
  --pilot-rows 256 `
  --manifest-path "$Root/manifests/modal-pilot.json" `
  --cost-cap-usd 15 `
  --compute-usd-per-second 0.00141996
if ($LASTEXITCODE -ne 0) {
  $Pilot = Get-Content "$Root/manifests/modal-pilot.json" -Raw `
    | ConvertFrom-Json
  if ($Pilot.gate -eq "STOP_COST_CAP") {
    throw "STOP_COST_CAP: projected cumulative spend is $($Pilot.projected_cumulative_usd)"
  }
  throw "Modal pilot failed before producing a valid cost decision"
}
```

Expected: exactly 256 rows, 128 eligible and 128 ineligible; the manifest contains measured timing, GPU type, input hash, config, revision, and cost projection.

- [ ] **Step 3: Enforce the cost decision before any full run**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$Gate = Get-Content "$Root/manifests/modal-pilot.json" -Raw `
  | ConvertFrom-Json
if ($Gate.gate -ne "PASS") {
  throw "STOP_COST_CAP: projected cumulative spend is $($Gate.projected_cumulative_usd)"
}
if ([decimal]$Gate.projected_cumulative_usd -gt [decimal]15.00) {
  throw "Cost manifest is internally inconsistent"
}
```

If this step throws, Task 8 ends here. Report the measured projection and do not launch the full Modal function.

- [ ] **Step 4: Generate one deduplicated full cache**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$RunBase = (Get-Content "$Root/manifests/run-base.txt" -Raw).Trim()
modal run scripts/modal_cache_features.py `
  --dataset-path "$Root/inputs/all_corpora_union.jsonl" `
  --out-path "$Root/caches/all_corpora_union.features.jsonl" `
  --config-path "configs/.profile_full_coverage_with_specdetect.yaml" `
  --run-id "$RunBase-full" `
  --manifest-path "$Root/manifests/modal-full.json" `
  --gate-manifest-path "$Root/manifests/modal-pilot.json" `
  --prior-manifest-path "$Root/manifests/modal-pilot.json" `
  --cost-cap-usd 15 `
  --compute-usd-per-second 0.00141996
if ($LASTEXITCODE -ne 0) { throw "Full Modal cache run failed" }

python scripts/validate_tocsin_cache.py `
  --dataset "$Root/inputs/all_corpora_union.jsonl" `
  --cache "$Root/caches/all_corpora_union.features.jsonl" `
  --config configs/.profile_full_coverage_with_specdetect.yaml `
  --expected-signature "tocsin-bart-fastdetectgpt-combined-w-v1" `
  --require-all-layers-seen `
  --out-manifest "$Root/manifests/all_corpora_union.cache.json"
if ($LASTEXITCODE -ne 0) { throw "Union cache validation failed" }
```

The validator must report no API-backed layers, no retired schema, exactly one active signature, and complete normalized-text hash coverage.

- [ ] **Step 5: Slice and validate all eight consumer caches**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$Pairs = @(
  @("data/prompts_v2_train.jsonl", "prompts_v2_train"),
  @("data/prompts_only_calib.jsonl", "prompts_only_calib"),
  @("data/prompts_v5_test.jsonl", "prompts_v5_test"),
  @("data/_humanized_eval.jsonl", "_humanized_eval"),
  @("data/_hard_human_clean.jsonl", "_hard_human_clean"),
  @("data/ood_m4_test.jsonl", "ood_m4_test"),
  @("data/ood_raid_test.jsonl", "ood_raid_test"),
  @("data/ood_mage_test.jsonl", "ood_mage_test")
)

foreach ($Pair in $Pairs) {
  $Dataset = $Pair[0]
  $Stem = $Pair[1]
  $Cache = "$Root/caches/$Stem.features.jsonl"
  $Manifest = "$Root/manifests/$Stem.cache.json"

  python scripts/slice_tocsin_feature_cache.py `
    --source-cache "$Root/caches/all_corpora_union.features.jsonl" `
    --dataset $Dataset `
    --out $Cache
  if ($LASTEXITCODE -ne 0) { throw "Cache slice failed: $Stem" }

  python scripts/validate_tocsin_cache.py `
    --dataset $Dataset `
    --cache $Cache `
    --config configs/.profile_full_coverage_with_specdetect.yaml `
    --expected-signature "tocsin-bart-fastdetectgpt-combined-w-v1" `
    --out-manifest $Manifest
  if ($LASTEXITCODE -ne 0) { throw "Cache validation failed: $Stem" }
}
```

Expected slice inventories:

| Cache | Unique | Active TOCSIN | TOCSIN SKIP |
|---|---:|---:|---:|
| `prompts_v2_train` | 20,732 | 8,484 | 12,248 |
| `prompts_only_calib` | 3,660 | 1,026 | 2,634 |
| `prompts_v5_test` | 6,098 | 1,822 | 4,276 |
| `_humanized_eval` | 502 | 440 | 62 |
| `_hard_human_clean` | 296 | 296 | 0 |
| `ood_m4_test` | 200 | 160 | 40 |
| `ood_raid_test` | 200 | 197 | 3 |
| `ood_mage_test` | 200 | 135 | 65 |

Any mismatch is a stop condition. Do not edit a cache to force these counts.

---

### Task 9: Fit the matched calibrator, EBM, conformal wrapper, and candidate profile

**Files produced locally:**

- Create: `configs/.profile_prompts_only_tocsin_v2.yaml`
- Create: `models/calib2000/profiles/prompts_only_tocsin_v2/detector_calibration.json`
- Create: `models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops`
- Create: `models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops.hmac`
- Create: `models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops.meta.json`
- Create: `models/calib2000/profiles/prompts_only_tocsin_v2/conformal.json`
- Create: `tests/test_tocsin_candidate_profile.py`

`scripts/run_tocsin_retrain_benchmark.py` uses `action="append"` for repeated `--baseline-config`, `nargs=2` for each dataset/cache pair, and a `--preflight-only` flag that validates candidate/comparator artifacts plus one cached smoke row without computing metrics. Preflight always writes `promotion_decision.json` and `PROMOTION_REPORT.md`; invalid artifacts exit `1` and lock the decision to `DO_NOT_PROMOTE`. Add `--diagnostic-on-invalid-comparator`: it may be used only after a failed preflight, excludes invalid comparators from metric comparisons, evaluates the candidate and every valid comparator from caches, writes the complete diagnostic report set, keeps the invalid-comparator failure in the decision, and exits `1`. When the challenge pair is omitted in this mode, write `july10_per_task.csv` with its header only and record `challenge_not_run_reason="invalid_comparator_avoided_additional_compute"` in the JSON/Markdown reports.

- [ ] **Step 1: Fit only the TOCSIN calibration entry**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$Candidate = "models/calib2000/profiles/prompts_only_tocsin_v2"
New-Item -ItemType Directory -Force $Candidate | Out-Null

python scripts/calibrate_detectors.py `
  --dataset data/prompts_v2_train.jsonl `
  --config configs/.profile_full_coverage_with_specdetect.yaml `
  --features-cache "$Root/caches/prompts_v2_train.features.jsonl" `
  --only-detector token_cohesiveness `
  --merge-from models/calib2000/profiles/prompts_specdetect/detector_calibration.json `
  --detector-signature "tocsin-bart-fastdetectgpt-combined-w-v1" `
  --raw-channel combined_w `
  --source-length-weighting `
  --manifest-out "$Root/manifests/tocsin-calibration.json" `
  --out "$Candidate/detector_calibration.json"
if ($LASTEXITCODE -ne 0) { throw "TOCSIN calibration fit failed" }
```

Expected: `8,484` accepted rows (`4,516` Human and `3,968` positive), more than 500 per class, and a v2 TOCSIN entry. Every non-TOCSIN entry must be parsed-identical to the base artifact.

- [ ] **Step 2: Create the candidate config**

Write exactly:

```yaml
_profile: profile_prompts_only_tocsin_v2
_extends: .profile_full_coverage_with_specdetect

fusion:
  model_path: models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops
  word_count_clamp: 200
  conformal_path: models/calib2000/profiles/prompts_only_tocsin_v2/conformal.json
  detector_calibration_path: models/calib2000/profiles/prompts_only_tocsin_v2/detector_calibration.json
```

Do not add API detector overrides; the parent profile already leaves them disabled.

- [ ] **Step 3: Train the EBM on the entire training corpus**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$Candidate = "models/calib2000/profiles/prompts_only_tocsin_v2"
python scripts/train_fusion.py `
  --dataset data/prompts_v2_train.jsonl `
  --config configs/.profile_prompts_only_tocsin_v2.yaml `
  --features-cache "$Root/caches/prompts_v2_train.features.jsonl" `
  --train-all `
  --interactions 15 `
  --drop-features tocsin_mean_deletion_impact `
  --out "$Candidate/ebm.skops"
if ($LASTEXITCODE -ne 0) { throw "Candidate EBM training failed" }
```

Expected: `trained on 20732, held out 0`. The artifact feature names contain `tocsin_p_llm`, `tocsin_cohesiveness_u`, `tocsin_combined_w`, and `tocsin_missing`, and omit `tocsin_mean_deletion_impact`.

- [ ] **Step 4: Calibrate conformal prediction only on the disjoint corpus**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$Candidate = "models/calib2000/profiles/prompts_only_tocsin_v2"
python scripts/calibrate_conformal.py `
  --dataset data/prompts_only_calib.jsonl `
  --config configs/.profile_prompts_only_tocsin_v2.yaml `
  --features-cache "$Root/caches/prompts_only_calib.features.jsonl" `
  --alpha 0.05 `
  --out "$Candidate/conformal.json" `
  --manifest-out "$Root/manifests/conformal-training.json"
if ($LASTEXITCODE -ne 0) { throw "Candidate conformal calibration failed" }
```

Expected: `calibrated on 3660 samples (alpha=0.05)` with class thresholds and qualifying router thresholds.

- [ ] **Step 5: Add and run the candidate integrity test**

Create a test that loads the normal profile and asserts:

```python
PROFILE = Path("configs/.profile_prompts_only_tocsin_v2.yaml")
pytestmark = pytest.mark.skipif(
    not PROFILE.exists(),
    reason="local TOCSIN v2 candidate has not been generated",
)


def test_candidate_profile_loads_coherent_artifacts():
    pipeline = CairnPipeline.from_config_file(PROFILE)
    assert pipeline._calibration_status == "calibrated"
    assert pipeline._detector_calibrator.has("token_cohesiveness")
    assert pipeline._fusion._model is not None
    assert pipeline._fusion._conformal.calibrated
    assert pipeline._fusion._assembler._word_count_clamp == 200

    names = set(pipeline._fusion._model._cairn_feature_names)
    assert {
        "tocsin_p_llm",
        "tocsin_cohesiveness_u",
        "tocsin_combined_w",
        "tocsin_missing",
    } <= names
    assert "tocsin_mean_deletion_impact" not in names

    entry = json.loads(
        Path(
            "models/calib2000/profiles/prompts_only_tocsin_v2/"
            "detector_calibration.json"
        ).read_text()
    )["token_cohesiveness"]
    assert entry["detector_signature"] == TOCSIN_COMBINED_SIGNATURE
    assert entry["raw_channel"] == "combined_w"
```

Run:

```powershell
python -m pytest -q tests/test_tocsin_candidate_profile.py
```

Expected: pass, including skops HMAC verification through the normal loader.

- [ ] **Step 6: Hash the coherent trio without committing it yet**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$Candidate = "models/calib2000/profiles/prompts_only_tocsin_v2"
python scripts/hash_artifacts.py `
  --out "$Root/manifests/candidate-integrity.json" `
  "$Candidate/detector_calibration.json" `
  "$Candidate/ebm.skops" `
  "$Candidate/ebm.skops.hmac" `
  "$Candidate/ebm.skops.meta.json" `
  "$Candidate/conformal.json" `
  "configs/.profile_prompts_only_tocsin_v2.yaml"
```

Keep the candidate config and model directory uncommitted until Task 11 determines whether the performance gates pass.

---

### Task 10: Implement deterministic detector, profile, conformal, and promotion evaluation

**Files:**

- Create: `cairn/evaluation/tocsin_retrain.py`
- Create: `scripts/run_tocsin_retrain_benchmark.py`
- Create: `scripts/prepare_july10_challenge.py`
- Create: `tests/test_tocsin_retrain_evaluation.py`
- Create: `tests/test_prepare_july10_challenge.py`
- Create: `tests/test_tocsin_candidate_profile.py`

- [ ] **Step 1: Add pure metric and gate tests**

Expose:

```python
def load_evaluation_slice(path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    seen_hashes: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "text" not in row or "label" not in row:
                raise ValueError(
                    f"{path}:{line_number}: text and label are required"
                )
            digest = text_hash(normalize_text(str(row["text"])))
            label = int(row["label"])
            sample_id = f"{path.stem}:{digest}"
            original_id = row.get("id")
            original_source_id = row.get("source_id")
            source_id_parts = []
            if original_source_id is not None:
                source_id_parts.append(str(original_source_id))
            if original_id is not None:
                source_id_parts.append(f"id={original_id}")
            evaluation_source_id = (
                "|".join(source_id_parts) if source_id_parts else None
            )
            if digest in seen_hashes:
                if seen_hashes[digest] != label:
                    raise ValueError(f"{path}: cross-label duplicate {digest}")
                raise ValueError(f"{path}: duplicate normalized text {digest}")
            seen_hashes[digest] = label
            samples.append(EvalSample(
                id=sample_id,
                text=str(row["text"]),
                label=label,
                tier=row.get("tier"),
                source=row.get("source"),
                attack_name=row.get("attack_name"),
                attack_category=row.get("attack_category"),
                source_id=evaluation_source_id,
            ))
    return samples


def reliability_curve(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int,
) -> list[dict[str, float | int | None]]:
    truth = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int | None]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (
            (scores >= lower) & (scores <= upper)
            if index == n_bins - 1
            else (scores >= lower) & (scores < upper)
        )
        rows.append({
            "lower": float(lower),
            "upper": float(upper),
            "count": int(mask.sum()),
            "mean_probability": (
                float(scores[mask].mean()) if mask.any() else None
            ),
            "positive_rate": (
                float(truth[mask].mean()) if mask.any() else None
            ),
        })
    return rows


def probability_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, object]:
    scores = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=int)
    return {
        "n": int(len(truth)),
        "n_unique": int(len(np.unique(scores))),
        "finite": bool(np.isfinite(scores).all()),
        "in_unit_interval": bool(
            ((scores >= 0.0) & (scores <= 1.0)).all()
        ),
        "brier": brier(truth, scores),
        "log_loss": float(sklearn_log_loss(truth, scores)),
        "auroc": auroc(truth, scores),
        "ece": ece(truth, scores),
        "human_mean": float(scores[truth == 0].mean()),
        "positive_mean": float(scores[truth == 1].mean()),
        "endpoint_fraction": float(
            np.mean((scores == 0.0) | (scores == 1.0))
        ),
        "reliability_curve": reliability_curve(truth, scores, n_bins=10),
    }


def detector_comparison(
    labels: Sequence[int],
    raw_scores: Sequence[float],
    stale_probabilities: Sequence[float],
    candidate_probabilities: Sequence[float],
    candidate_applied: Sequence[bool],
    *,
    support: tuple[float, float],
    tolerance: float,
    detector_skip_rows: int,
) -> dict[str, object]:
    truth = np.asarray(labels, dtype=int)
    raw = np.asarray(raw_scores, dtype=float)
    stale = np.asarray(stale_probabilities, dtype=float)
    candidate = np.asarray(candidate_probabilities, dtype=float)
    applied = np.asarray(candidate_applied, dtype=bool)
    lower, upper = support
    in_support = (raw >= lower - tolerance) & (raw <= upper + tolerance)
    if not np.array_equal(in_support, applied):
        raise ValueError("candidate apply mask disagrees with support guard")
    common = applied
    prevalence = float(truth[common].mean())
    constant = np.full(int(common.sum()), prevalence, dtype=float)
    return {
        "eligible_rows": int(len(truth)),
        "detector_skip_rows": int(detector_skip_rows),
        "in_support_rows": int(common.sum()),
        "support_coverage": float(common.mean()),
        "support": [float(lower), float(upper)],
        "tolerance": float(tolerance),
        "raw": {
            "min": float(raw.min()),
            "max": float(raw.max()),
            "auroc": auroc(truth, raw),
            "human_mean": float(raw[truth == 0].mean()),
            "positive_mean": float(raw[truth == 1].mean()),
        },
        "candidate": probability_metrics(truth[common], candidate[common]),
        "legacy_unsafe_clip_diagnostic": probability_metrics(
            truth[common], stale[common]
        ),
        "constant_prevalence": probability_metrics(
            truth[common], constant
        ),
    }
```

Use this loader for every locked evaluation slice. It leaves source JSONL files untouched, always derives the evaluation ID from path stem plus full normalized-text hash, and retains any original `id`/`source_id` in the loaded sample's `source_id` metadata. Add fixtures proving the same file loads to the same hash-derived IDs twice, repeated source IDs attached to different texts remain distinct evaluation rows, same-label duplicate text fails, and cross-label duplicate text fails. This is required because 400 of 502 `_humanized_eval.jsonl` rows and all 296 `_hard_human_clean.jsonl` rows lack an `id`, while `prompts_v5_test.jsonl` contains 30 source-ID collisions and `_humanized_eval.jsonl` contains four.

`profile_metrics()` additionally returns balanced accuracy at `0.5`, Human FPR/specificity, TPR at 5%/10% FPR, conformal truth coverage, singleton rate, selective accuracy, and the maximum EBM reconstruction error.

Compute the two-sided 95% binomial count interval at the configured target:

```python
def binomial_target_interval(n: int, target: float) -> tuple[int, int]:
    lower, upper = scipy.stats.binom.interval(0.95, n, target)
    return int(lower), int(upper)
```

`decide_promotion(report)` returns exactly:

```python
{
    "status": "PASS" if not failures else "DO_NOT_PROMOTE",
    "failures": failures,
}
```

Test every gate at its boundary and immediately beyond it. The July challenge block must not be read by `decide_promotion()`.

- [ ] **Step 2: Encode the detector-level gates**

On eligible `prompts_v5_test` rows:

- exactly `1,822` current combined-signature rows are available and every one lies within fitted support plus tolerance (`support_coverage == 1.0`);
- every candidate probability is finite and in `[0, 1]`;
- `n_unique > 3`;
- candidate Brier and log loss are strictly lower than the stale-calibrator diagnostic;
- candidate Brier is strictly lower than constant-prevalence Brier;
- endpoint fraction is below `0.95`;
- active/SKIP counts, class means, AUROC, ECE, support coverage, and a ten-bin reliability curve are always reported.

Compute candidate, stale, and constant-prevalence probability metrics on the same `candidate_applied` mask. The 100% support-coverage gate prevents the candidate from improving Brier/log loss by abstaining on difficult eligible rows. Report the current uncalibrated `combined_w` range, class means, and AUROC in the separate `raw` block; do not compute Brier or log loss on that unbounded raw channel.

Generate stale diagnostic scores only through this isolated helper and never pass them to fusion:

```python
def legacy_unsafe_transform_for_diagnostic(
    calibrator: DetectorCalibrator,
    detector_id: str,
    raw_score: float,
) -> float:
    return calibrator.transform(detector_id, raw_score)  # AUDITED_UNSAFE_STALE_DIAGNOSTIC
```

Label the metric block `legacy_unsafe_clip_diagnostic`. A test must prove the helper is used only to reproduce the stale clipped mapping and is never called by `CairnPipeline`, `CachedPipeline`, or profile fusion.

- [ ] **Step 3: Encode the end-to-end gates against each valid comparator**

For both `prompts_only` and `prompts_specdetect`:

```python
candidate_prompt["brier"] <= baseline_prompt["brier"]
candidate_prompt["log_loss"] <= baseline_prompt["log_loss"]
candidate_prompt["balanced_accuracy"] >= (
    baseline_prompt["balanced_accuracy"] - 0.01
)
candidate_prompt["auroc"] >= baseline_prompt["auroc"] - 0.01
candidate_prompt["human_fpr"] <= baseline_prompt["human_fpr"]
```

For the combined humanized-positive versus clean-hard-Human slice:

```python
def build_humanized_hard_slice(
    humanized_samples: Sequence[EvalSample],
    hard_human_samples: Sequence[EvalSample],
) -> list[EvalSample]:
    positives = [sample for sample in humanized_samples if sample.label == 1]
    negatives = [sample for sample in hard_human_samples if sample.label == 0]
    if len(positives) != 102 or len(negatives) != 296:
        raise ValueError(
            "expected 102 humanized positives and 296 clean hard Humans"
        )
    positive_hashes = {
        text_hash(normalize_text(sample.text)) for sample in positives
    }
    negative_hashes = {
        text_hash(normalize_text(sample.text)) for sample in negatives
    }
    if len(positive_hashes) != len(positives):
        raise ValueError("humanized-positive slice contains duplicate text")
    if len(negative_hashes) != len(negatives):
        raise ValueError("clean hard-Human slice contains duplicate text")
    conflicts = positive_hashes & negative_hashes
    if conflicts:
        raise ValueError("humanized/hard-Human slice has cross-label hashes")
    return positives + negatives
```

Do not concatenate all 502 rows from `_humanized_eval.jsonl`: that file also contains 400 label-0 controls, including 73 normalized-text overlaps with `_hard_human_clean.jsonl`. Test the `102/296` filtered composition and cross-label conflict rejection.

```python
for metric in ("auroc", "tpr_at_fpr_05", "tpr_at_fpr_10"):
    candidate_humanized[metric] >= baseline_humanized[metric] - 0.02
```

For each of `ood_m4_test`, `ood_raid_test`, and `ood_mage_test`:

```python
candidate_ood[corpus]["auroc"] >= baseline_ood[corpus]["auroc"] - 0.02
```

An OOD miss remains a failure. A compensating improvement can be documented for later manual review but cannot automatically change `DO_NOT_PROMOTE` to `PASS`.

Candidate conformal truth-coverage count must fall inside `binomial_target_interval(n, 0.95)`. Report singleton/selective accuracy regardless of pass/fail.

Every evaluated row must have `fusion_mode="ebm"`, a finite `p_llm`, and EBM reconstruction error at or below `1e-9`.

The adaptive July block reports Brier score, balanced accuracy, Human specificity, conformal truth coverage, and per-task current-versus-candidate changes. These values are serialized and displayed but never enter automatic promotion logic.

- [ ] **Step 4: Hard-fail missing or fallback comparators**

Before reading evaluation rows, require each profile's model, HMAC, conformal, and detector-calibration files. Instantiate the profile and run one cached smoke row. Reject it unless `fusion_mode="ebm"` and `calibration_status` starts with `calibrated`.

For the current checkout, the expected preflight finding is:

```text
INVALID_COMPARATOR prompts_specdetect:
missing models/calib2000/profiles/prompts_specdetect/ebm.skops
missing models/calib2000/profiles/prompts_specdetect/ebm.skops.hmac
```

The runner writes `promotion_decision.json` with `DO_NOT_PROMOTE`, includes these failures, and exits `1`. It must not benchmark `prompts_specdetect` through naive fusion and must not label a reconstructed EBM as historical.

- [ ] **Step 5: Test private challenge preparation without leaking it into training**

`scripts/prepare_july10_challenge.py` accepts:

```text
--rows outputs/cairn_blinded_rerun_2026-07-10/source_rows.json
--truth outputs/cairn_blinded_rerun_2026-07-10/ground_truth_analysis.json
--out outputs/tocsin_retrain_2026-07-10/challenge/july10_labelled.jsonl
```

The test requires exactly 14 rows, task numbers 41–54, 9 positive and 5 Human labels, and excludes task 55. It also asserts none of the 14 normalized hashes occur in `prompts_v2_train.jsonl` or `prompts_only_calib.jsonl`.

- [ ] **Step 6: Run evaluation tests and confirm red, then green**

Red:

```powershell
python -m pytest -q tests/test_tocsin_retrain_evaluation.py tests/test_prepare_july10_challenge.py
```

Expected: imports fail because the evaluation module and scripts are absent.

After implementing the exact interfaces and gates above:

```powershell
python -m pytest -q tests/test_tocsin_retrain_evaluation.py tests/test_prepare_july10_challenge.py
```

Expected: all synthetic gate, boundary, comparator-preflight, and leakage tests pass.

- [ ] **Step 7: Commit the evaluation engine**

```powershell
git add cairn/evaluation/tocsin_retrain.py scripts/run_tocsin_retrain_benchmark.py scripts/prepare_july10_challenge.py tests/test_tocsin_retrain_evaluation.py tests/test_prepare_july10_challenge.py tests/test_tocsin_candidate_profile.py
git commit -m "feat(evaluation): add TOCSIN retrain promotion report"
```

---

### Task 11: Run the locked evaluation and make the conditional artifact decision

**Generated report paths after a valid comparator preflight:**

```text
outputs/tocsin_retrain_2026-07-10/reports/detector_calibration.json
outputs/tocsin_retrain_2026-07-10/reports/benchmarks.json
outputs/tocsin_retrain_2026-07-10/reports/benchmarks_long.csv
outputs/tocsin_retrain_2026-07-10/reports/july10_per_task.csv
outputs/tocsin_retrain_2026-07-10/reports/promotion_decision.json
outputs/tocsin_retrain_2026-07-10/reports/PROMOTION_REPORT.md
```

`benchmarks_long.csv` columns are exactly:

```text
profile,slice,id,label,p_llm,predicted_label,conformal_set,
fusion_mode,reconstruction_error
```

- [ ] **Step 1: Preflight every candidate and comparator artifact before more compute**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
New-Item -ItemType Directory -Force "$Root/reports" | Out-Null
python scripts/run_tocsin_retrain_benchmark.py `
  --preflight-only `
  --candidate-config configs/.profile_prompts_only_tocsin_v2.yaml `
  --baseline-config configs/.profile_prompts_only.yaml `
  --baseline-config configs/.profile_prompts_specdetect.yaml `
  --prompt-test data/prompts_v5_test.jsonl "$Root/caches/prompts_v5_test.features.jsonl" `
  --out-dir "$Root/reports"
$PreflightExit = $LASTEXITCODE
if ($PreflightExit -eq 1) {
  $Blocked = Get-Content "$Root/reports/promotion_decision.json" -Raw `
    | ConvertFrom-Json
  if ($Blocked.status -ne "DO_NOT_PROMOTE") {
    throw "Invalid comparator did not lock promotion"
  }
  Set-Content "$Root/manifests/comparator-blocked.txt" "true"
} elseif ($PreflightExit -eq 0) {
  Set-Content "$Root/manifests/comparator-blocked.txt" "false"
} else {
  throw "Unexpected promotion preflight exit code $PreflightExit"
}
```

The current checkout is expected to record `true` because the exact `prompts_specdetect` EBM/HMAC pair is absent. This permanently locks the run to `DO_NOT_PROMOTE`, but cached candidate-versus-valid-baseline diagnostics still continue. Do not spend more Modal compute on the adaptive challenge while this flag is true.

- [ ] **Step 2: Prepare and cache the adaptive challenge slice separately**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$RunBase = (Get-Content "$Root/manifests/run-base.txt" -Raw).Trim()
$ComparatorBlocked = (
  Get-Content "$Root/manifests/comparator-blocked.txt" -Raw
).Trim() -eq "true"
if ($ComparatorBlocked) {
  Write-Output "Skipping adaptive challenge compute: promotion is already blocked"
} else {
  New-Item -ItemType Directory -Force `
    "$Root/challenge", "$Root/reports" | Out-Null

  python scripts/prepare_july10_challenge.py `
    --rows outputs/cairn_blinded_rerun_2026-07-10/source_rows.json `
    --truth outputs/cairn_blinded_rerun_2026-07-10/ground_truth_analysis.json `
    --out "$Root/challenge/july10_labelled.jsonl"
  if ($LASTEXITCODE -ne 0) { throw "Challenge preparation failed" }

  $ChallengeRun = "$RunBase-july10-challenge"
  modal run scripts/modal_cache_features.py `
    --dataset-path "$Root/challenge/july10_labelled.jsonl" `
    --out-path "$Root/challenge/july10_labelled.features.jsonl" `
    --config-path configs/.profile_full_coverage_with_specdetect.yaml `
    --run-id $ChallengeRun `
    --manifest-path "$Root/manifests/modal-july10-challenge.json" `
    --prior-manifest-path "$Root/manifests/modal-full.json" `
    --cost-cap-usd 15 `
    --compute-usd-per-second 0.00141996
  if ($LASTEXITCODE -ne 0) { throw "Challenge cache generation failed" }

  python scripts/validate_tocsin_cache.py `
    --dataset "$Root/challenge/july10_labelled.jsonl" `
    --cache "$Root/challenge/july10_labelled.features.jsonl" `
    --config configs/.profile_full_coverage_with_specdetect.yaml `
    --expected-signature "tocsin-bart-fastdetectgpt-combined-w-v1" `
    --out-manifest "$Root/manifests/july10-challenge.cache.json"
  if ($LASTEXITCODE -ne 0) { throw "Challenge cache validation failed" }
}
```

The challenge cache contributes to actual cumulative cost. Before launching it, update the cost manifest with the full-run measured cost and assert projected/actual remaining headroom keeps total spend at or below `$15.00`.

- [ ] **Step 3: Run the benchmark with all locked slices**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$ComparatorBlocked = (
  Get-Content "$Root/manifests/comparator-blocked.txt" -Raw
).Trim() -eq "true"
$BenchmarkArgs = @(
  "scripts/run_tocsin_retrain_benchmark.py",
  "--candidate-config", "configs/.profile_prompts_only_tocsin_v2.yaml",
  "--baseline-config", "configs/.profile_prompts_only.yaml",
  "--baseline-config", "configs/.profile_prompts_specdetect.yaml",
  "--prompt-test", "data/prompts_v5_test.jsonl", "$Root/caches/prompts_v5_test.features.jsonl",
  "--humanized", "data/_humanized_eval.jsonl", "$Root/caches/_humanized_eval.features.jsonl",
  "--hard-human", "data/_hard_human_clean.jsonl", "$Root/caches/_hard_human_clean.features.jsonl",
  "--ood-m4", "data/ood_m4_test.jsonl", "$Root/caches/ood_m4_test.features.jsonl",
  "--ood-raid", "data/ood_raid_test.jsonl", "$Root/caches/ood_raid_test.features.jsonl",
  "--ood-mage", "data/ood_mage_test.jsonl", "$Root/caches/ood_mage_test.features.jsonl",
  "--stale-calibration", "models/calib2000/profiles/prompts_only/detector_calibration.json",
  "--out-dir", "$Root/reports"
)
if ($ComparatorBlocked) {
  $BenchmarkArgs += "--diagnostic-on-invalid-comparator"
} else {
  $BenchmarkArgs += @(
    "--challenge",
    "$Root/challenge/july10_labelled.jsonl",
    "$Root/challenge/july10_labelled.features.jsonl"
  )
}
& python @BenchmarkArgs
$BenchmarkExit = $LASTEXITCODE
if ($BenchmarkExit -notin @(0, 1, 2)) {
  throw "Benchmark exited with unexpected code $BenchmarkExit"
}
```

Exit codes are `0=PASS`, `2=valid evaluation with failed performance gates`, and `1=invalid or missing artifact/comparator`. In the current checkout, `1` is expected until the exact `prompts_specdetect` EBM/HMAC pair is restored.

- [ ] **Step 4: Verify report completeness and privacy**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$RequiredReports = @(
  "detector_calibration.json",
  "benchmarks.json",
  "benchmarks_long.csv",
  "july10_per_task.csv",
  "promotion_decision.json",
  "PROMOTION_REPORT.md"
)
foreach ($Name in $RequiredReports) {
  if (-not (Test-Path "$Root/reports/$Name")) {
    throw "Missing report: $Name"
  }
}

rg -n '"text"|source_rows|ground_truth_analysis' "$Root/reports"
if ($LASTEXITCODE -eq 0) {
  throw "Report contains private raw-text or source-path material"
}
```

The Markdown report leads with `PASS` or `DO_NOT_PROMOTE`, lists every gate and comparator validity result, and clearly labels July results as adaptive/secondary.

- [ ] **Step 5: Apply the conditional artifact policy**

Read the decision:

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$ProductionFiles = @(
  "configs/.profile_prompts_only.yaml"
) + @(
  Get-ChildItem "models/calib2000/profiles/prompts_only" -File `
    | ForEach-Object { Resolve-Path -Relative $_.FullName }
)
python scripts/hash_artifacts.py `
  --out "$Root/manifests/production-after.json" `
  $ProductionFiles
if ($LASTEXITCODE -ne 0) { throw "Production final hashing failed" }
$Before = Get-Content "$Root/manifests/production-before.json" -Raw `
  | ConvertFrom-Json
$After = Get-Content "$Root/manifests/production-after.json" -Raw `
  | ConvertFrom-Json
if (($Before | ConvertTo-Json -Depth 10 -Compress) -ne `
    ($After | ConvertTo-Json -Depth 10 -Compress)) {
  throw "Production prompt profile or artifact hash changed"
}

$Decision = Get-Content "$Root/reports/promotion_decision.json" -Raw `
  | ConvertFrom-Json
```

If `$Decision.status -eq "PASS"`, add explicit `.gitignore` allow rules for only the six candidate model/config files, stage them with the non-private manifests/reports, and commit:

```gitignore
!/models/calib2000/profiles/prompts_only_tocsin_v2/
/models/calib2000/profiles/prompts_only_tocsin_v2/*
!/models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops
!/models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops.hmac
!/models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops.meta.json
!/models/calib2000/profiles/prompts_only_tocsin_v2/conformal.json
!/models/calib2000/profiles/prompts_only_tocsin_v2/detector_calibration.json
```

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
git add .gitignore configs/.profile_prompts_only_tocsin_v2.yaml `
  models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops `
  models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops.hmac `
  models/calib2000/profiles/prompts_only_tocsin_v2/ebm.skops.meta.json `
  models/calib2000/profiles/prompts_only_tocsin_v2/conformal.json `
  models/calib2000/profiles/prompts_only_tocsin_v2/detector_calibration.json
git add -f "$Root/manifests" "$Root/reports"
git commit -m "feat(profile): add validated prompts-only TOCSIN v2 candidate"
```

If the status is `DO_NOT_PROMOTE`, do not stage the candidate config or model directory. Force-add only sanitized manifests/reports and commit:

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
git add -f "$Root/manifests" "$Root/reports"
git commit -m "docs(benchmark): record rejected TOCSIN v2 candidate"
```

Never stage `$Root/caches`, `$Root/inputs`, `$Root/pilot`, or `$Root/challenge`.

---

### Task 12: Run final repository verification and audit scope

- [ ] **Step 1: Run focused integration tests**

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m pytest -q `
  tests/test_calibration.py `
  tests/test_calibration_guard.py `
  tests/test_calibration_application.py `
  tests/test_pipeline_detector_calibration.py `
  tests/test_detectors/test_token_cohesiveness.py `
  tests/test_tocsin_cache.py `
  tests/test_modal_cache_features.py `
  tests/test_calibration_training.py `
  tests/test_calibrate_detectors_cli.py `
  tests/test_train_fusion_holdout.py `
  tests/test_calibrate_conformal_cli.py `
  tests/test_fusion.py `
  tests/test_tocsin_retrain_evaluation.py `
  tests/test_prepare_july10_challenge.py `
  tests/test_tocsin_candidate_profile.py
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the entire suite**

```powershell
python -m pytest -q
```

Expected: all repository tests pass. If an unrelated pre-existing failure appears, record the exact command/output and keep it separate from TOCSIN changes; do not weaken the new tests.

- [ ] **Step 3: Audit direct calibration calls, API state, artifact scope, and production immutability**

```powershell
$Root = "outputs/tocsin_retrain_2026-07-10"
$DirectTransforms = @(
  rg -n "calibrator\.transform|cal\.transform|_detector_calibrator\.transform" cairn scripts
)
$UnexpectedTransforms = @(
  $DirectTransforms | Where-Object {
    $_ -notmatch "cairn[\\/]evaluation[\\/]tocsin_retrain.py:.*AUDITED_UNSAFE_STALE_DIAGNOSTIC"
  }
)
if ($UnexpectedTransforms.Count -ne 0) {
  $UnexpectedTransforms | Write-Output
  throw "Unexpected direct calibrator transform call"
}
if ($DirectTransforms.Count -ne 1) {
  throw "Expected exactly one isolated stale-diagnostic transform"
}
rg -n -C 2 "contrastive_gen:|dna_gpt:" configs/.profile_full_coverage.yaml configs/default.yaml
git diff --name-only 8eab8c4..HEAD
git status --short
git diff --exit-code 8eab8c4 -- configs/.profile_prompts_only.yaml models/calib2000/profiles/prompts_only

$ExpectedProduction = Get-Content `
  "$Root/manifests/production-after.json" -Raw | ConvertFrom-Json
foreach ($Entry in $ExpectedProduction.PSObject.Properties) {
  $ActualHash = (
    Get-FileHash -Algorithm SHA256 -LiteralPath $Entry.Name
  ).Hash.ToLowerInvariant()
  if ($ActualHash -ne $Entry.Value.sha256) {
    throw "Production artifact changed after final staging: $($Entry.Name)"
  }
}
```

Expected:

- exactly one direct transform exists, on the marked stale-diagnostic helper line, and no runtime/replay transform exists;
- API-backed detectors remain disabled;
- only planned code/tests/docs plus a conditional sanitized artifact/report commit appear;
- existing production prompt profile and artifacts have no diff;
- unrelated dirty paths are still unstaged.

- [ ] **Step 4: Record the final handoff**

Report:

- the candidate integrity status;
- actual Modal elapsed time and spend accounting versus `$15.00`;
- fresh cache counts and hashes;
- detector-level calibration deltas;
- end-to-end gate results;
- whether the missing historical `prompts_specdetect` artifact forced `DO_NOT_PROMOTE`;
- exact commits produced;
- the unchanged production profile path.

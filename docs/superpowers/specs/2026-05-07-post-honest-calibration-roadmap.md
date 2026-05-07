# CAIRN — Post-Honest-Calibration Roadmap

**Status:** Draft, ready for review
**Date:** 2026-05-07
**Scope:** ~10 weeks of post-honest-calibration work across three independent tracks
**Supersedes:** Nothing. Sits alongside [`2026-05-02-honest-calibration-unified.md`](../plans/2026-05-02-honest-calibration-unified.md); does not block on it but assumes its assembler-`_missing` / `_confidence` features land before A2/A3 retrain.

## Goal

Make CAIRN's calibrated-probability claim externally verifiable and the
system deployable past loopback. Today CAIRN ships calib2000 calibration
artifacts at the `calibrated-router-conditional` tier
(`master_minus_arxiv` for prose, `prompts_only` for prompts) with
honest in-domain numbers (AUROC ~0.94 HC3, ~0.68 RAID, ~0.57 MAGE,
~0.89 short-form). What it does not ship: head-to-head numbers against
published baselines, an OOD story for RAID/MAGE that closes the
in-vs-out gap, parity on OpenAI hidden-reasoning text (~6 AUROC behind
Gemini), a multilingual probe, auth/rate-limit on the HTTP transports,
a persistent drift window, profile auto-routing, an audit chain wired
into `analyze`, and a closed-loop recalibration runner. This roadmap
sequences those.

The user-stated priority is **harden first, then build**: scientific
proof and operational hardening over net-new product surfaces.

## Non-goals

- SaaS / hosted offering, browser extension, multi-tenant accounts,
  marketing site, cloud SDK. Deferred to a future "distribution"
  roadmap once accuracy and hardening land.
- Multilingual product. A4 below is a beachhead probe, gated as
  experimental in `/health`.
- Replacing the unified honest-calibration plan. That plan owns
  Mondrian conformal, empty-set decision, missingness features, skops
  serialization. This roadmap layers on top.

## Architecture overview

```
                Honest calibration (in flight)
                ──────────────────────────────
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   Track A —           Track B —           Track C —
   Honest accuracy     Operational         Closed-loop
   & external proof    hardening           improvement

   A1 competitor       B1 auth +           C1 feedback_log →
      adapter shelf       rate-limit          calibration export
   A2 OOD attack       B2 persistent       C2 monthly retrain
      pack               DriftMonitor         runner
   A3 hidden-          B3 profile          C3 per-detector
      reasoning           auto-routing         AUROC regression
      sub-detector    B4 audit-chain          gate
   A4 multilingual        on every
      seed corpus         analyze
   A5 public
      benchmark
      dashboard

   ────────────────────────────────────────────────────
                            │
                            ▼
            Recurring ship cadence:
            monthly retrain + benchmark refresh
```

The three tracks are independent. Track A produces external proof,
Track B closes deployability gaps, Track C closes the data-back-to-model
loop. Track A's A2/A3 retrain steps depend on the unified honest-calibration
plan reaching its Phase 4 stop-here checkpoint
(`feat(ebm): missingness-aware fusion v7`). Tracks B and C have no such
dependency.

---

## Track A — Honest accuracy & external proof (~4 weeks)

### A1 — Competitor adapter shelf

**Goal:** Land the third module described in
[`2026-04-14-evaluation-adversarial-benchmarks-design.md`](2026-04-14-evaluation-adversarial-benchmarks-design.md)
that was scoped but never landed: head-to-head benchmarking of CAIRN
against open-source and commercial detectors.

**Files to create:**

```
cairn/benchmarks/__init__.py
cairn/benchmarks/base.py             # CompetitorAdapter Protocol
cairn/benchmarks/binoculars.py       # local, OSS
cairn/benchmarks/fast_detectgpt.py   # local, OSS
cairn/benchmarks/gptzero.py          # commercial, env-keyed
cairn/benchmarks/originality.py      # commercial, env-keyed
scripts/run_competitor_benchmark.py
tests/test_benchmarks/__init__.py
tests/test_benchmarks/test_adapters.py
tests/test_benchmarks/fixtures/      # snapshotted API responses
```

**Reuse:**

- `cairn/evaluation/runner.py:54` — `run_eval` already returns an
  `EvalReport` keyed off the adapter callable. The competitor adapter
  Protocol mirrors `Detector` shape so the same runner consumes it.
- `cairn/evaluation/dataset.py` — `EvalSample` JSONL is the input
  contract; no new format.
- `is_available()` gate pattern from
  `cairn/pipeline.py::detector_availability` — adapters report missing
  deps without crashing the run.

**CLI surface:**

```bash
cairn benchmark data/ood_raid_test.jsonl \
    --adapters binoculars,fast_detectgpt \
    --out docs/benchmarks/competitor/raid_competitor.json
```

**Privacy:** Commercial adapters (GPTZero, Originality) send the full
text to a third party. Document this in the adapter docstring, gate
behind explicit opt-in flag (`--allow-commercial`) and a
`CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1` env var. Default is
local-OSS-only.

**Test gate:** Two open-source adapters land with passing structural
tests; commercial adapters smoke-test against snapshotted fixtures
(real API calls only when `CAIRN_BENCHMARK_LIVE=1`).

### A2 — OOD attack pack: close the RAID/MAGE gap

**Goal:** Move RAID AUROC from 0.6841 → ≥ 0.73 and MAGE from
0.5667 → ≥ 0.60 on `master_minus_arxiv`, in-domain regression
≤ -0.005.

**Approach:**

1. Mine RAID's per-attack confusion matrix from
   `scripts/run_raid_benchmark.py`. Identify the three transforms with
   the largest negative AUROC delta vs `raid_base`.
2. Generate a matched adversarial training slice using
   `cairn/adversarial/generator.py` driven by an **expanded transform
   set** in `cairn/adversarial/transforms.py`. Existing transforms
   stay: `paraphrase`, `synonym_swap`, `typo`, `casualize`,
   `mix_human`. **New transforms to add as an A2 acceptance gate:**

   - `translation_pivot` — round-trip translation through one of
     {es, fr, de, zh, ja} via a local MT model (NLLB-200-distilled-600M
     is the smallest competent option). Two-hop pivot only; skip
     three-hop to bound semantic drift.
   - `grammar_normalize` — push informal human text through a small
     LM ("fix grammar, preserve meaning") to *raise* its surface
     polish toward LLM register. Targets the dominant false-positive
     case: a polished human draft.
   - `style_transfer` — formal↔casual / academic↔conversational
     rewrite via the same small LM with a style prompt. Catches the
     "I asked the LLM to write like me" attack.
   - `burstiness_injection` — synthetic perplexity-burstiness
     manipulation: split the text into windows, perturb the
     lowest-entropy windows by injecting a higher-entropy phrase
     drawn from a human-prose pool. Targets the burstiness statistic
     that `surprisal_dynamics` and `contrastive_lm` consume.
   - `emoji_noise` — interleave low-frequency emoji at sentence
     boundaries; defuses the "humanize-by-emoji" trick.
   - `summarization_attack` — replace text with its LLM summary and
     re-expand to original length via LLM, simulating the
     "compress-then-expand" round-trip pattern.
   - `humanizer_tools` — opt-in integration with commercial
     humanizers (StealthGPT, Undetectable.ai, BypassGPT). Gated by
     `CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1` (same gate as A1
     commercial adapters), budget-capped, snapshot-tested; live runs
     only on explicit opt-in.

   Sized to ~15k rows total, balanced human/LLM, **provider-stratified
   across calib2000 v6 providers AND transform-stratified** so no
   single transform dominates the augmentation.
3. Append to `data/calib2000_v6_train.jsonl` as a new source tag
   (`adversarial_raid_v1`).
4. Retrain `master_minus_arxiv` via the existing pipeline:
   `scripts/calibrate_detectors.py` → `scripts/train_fusion.py` →
   `scripts/calibrate_conformal.py` (router-conditional).
5. Validate with `scripts/run_raid_benchmark.py` and the OOD MAGE
   slice.

**Files to create / modify:**

```
cairn/adversarial/transforms.py       (modify: add 7 new transforms —
                                       translation_pivot,
                                       grammar_normalize, style_transfer,
                                       burstiness_injection, emoji_noise,
                                       summarization_attack, humanizer_tools)
tests/test_adversarial/test_transforms.py  (extend: per-transform
                                            structural test — transform
                                            produces text of comparable
                                            length, does not crash on
                                            edge inputs, preserves
                                            label semantics for the
                                            label-preserving transforms)
scripts/build_raid_aware_corpus.py    (new)
data/adversarial_raid_v1.jsonl        (generated, ~15k rows)
docs/benchmarks/calib2000/raid_aware_v1/   (new — per-attack eval reports)
docs/benchmarks/attack_coverage.md    (new — published before/after AUROC
                                       per attack; regenerated by C3's
                                       per_attack_auroc.py)
models/calib2000/profiles/master_minus_arxiv_robust_v1/  (new robust
                                       profile; ships alongside the
                                       existing master_minus_arxiv)
```

**Reuse:**

- `cairn/adversarial/generator.py`, `cairn/adversarial/transforms.py`
- `cairn/evaluation/robustness.py::run_robustness_eval` — already
  computes per-attack delta from baseline; track gains/regressions
  here.
- `scripts/regenerate_llm_outputs.py` patterns for provider-stratified
  generation.

**Test gates:**

- **Aggregate metrics:** RAID ≥ 0.73, MAGE ≥ 0.60, in-domain
  HC3-prose ≥ 0.94, `verb_low` not regressed > 0.01.
- **Per-attack metrics published:** `docs/benchmarks/attack_coverage.md`
  shows before/after AUROC for *every* transform in the expanded
  set (existing 5 + new 7 = 12). No transform's after-AUROC may
  regress > 0.02 versus the pre-A2 baseline. **This is a release
  blocker, enforced in CI by C3.**
- **Robust profile produced:** A new
  `models/calib2000/profiles/master_minus_arxiv_robust_v1/` ships
  alongside the existing `master_minus_arxiv`. The unaugmented
  baseline stays as the OOD reference; the new profile becomes the
  recommended production default once the gate passes. Future
  attack-aware variants of `prompts_only` and other profiles follow
  the same `_robust_v<N>` naming convention.
- **No silent baseline updates.** A profile that fails the
  per-attack gate cannot ship by raising the baseline; the baseline
  CSV is updated only when the regression is intentional and
  justified in the commit message (see C3).

### A3 — Hidden-reasoning sub-detector

**Goal:** Close the OpenAI hidden-reasoning gap. README's
matched-comparison eval shows OpenAI Responses-API reasoning text at
AUROC 0.9358 under `calibrated_full` vs Gemini at 0.9961 — ~6 AUROC
points stealthier than other frontier providers.

**Approach:** Add a Phase-2 detector targeting the surface artefacts
hidden reasoning produces: uniform register, low surprisal-variance
across reasoning-segment boundaries, characteristic transition
phrasing ("Let me think...", "Let's break this down...", "Looking at
this..."), and a high ratio of intent-stating clauses to action
clauses.

**Files to create / modify:**

```
cairn/detectors/reasoning_artifact.py        (new)
tests/test_detectors/test_reasoning_artifact.py (new)
cairn/cascade.py                              (modify: add to PHASE2_DETECTORS at line 10)
configs/default.yaml                          (modify: detectors.reasoning_artifact)
README.md                                     (modify: phase table at lines 47-50)
```

**Reuse:**

- `cairn/contracts.py::Detector` Protocol; `LayerResult.spans` for
  high-confidence trigger highlighting (see existing
  `voice_spec.py` / `prompt_structure.py` for the span pattern).
- `cairn/detectors/_utils.py` for shared text utilities.
- Test fixture: provider-tagged samples from
  `data/calib2000_v6_train.jsonl` filtered to
  `source LIKE 'openai_%_reasoning'` + the matched-eval q1500-1999
  pairs from `data/sweep_*.jsonl`.

**Test gate:** Per-provider AUROC delta on `calibrated_full`'s
matched-comparison eval narrows from ~6 points to ≤ 3 points
(OpenAI ≥ 0.96 with the new detector wired in). Provider parity
asserted via `EvalReport.per_source` already implemented in
`cairn/evaluation/runner.py`.

**Risk:** Detector overfits to current OpenAI surface artefacts and
collapses if OpenAI changes its reasoning UX. Mitigation: cross-provider
parity check is the gate; if Gemini's AUROC drops by > 0.02 the
detector is over-fit and the threshold is rolled back.

### A4 — Multilingual seed corpus

**Goal:** Establish an honest baseline for multilingual prose. M4's
best AUROC across the calib2000 grid is 0.6391 (`general_full`);
that's barely above random for a calibrated system.

**Approach:** Pull a ~5k-sample slice from M4 covering es/fr/de/zh/ja/ar
(the same language pool M4 uses), train a `profile_multilingual`,
report the numbers. Frame it as a probe, not a product.

**Files to create / modify:**

```
scripts/download_m4_multilingual.py          (new)
configs/.profile_multilingual.yaml           (new, dot-prefixed per the
                                              honest-calibration plan
                                              Phase-0 note)
docs/benchmarks/calib2000/profile_multilingual/  (new)
README.md                                     (modify: limitations block,
                                              add multilingual caveat)
cairn/api.py                                  (modify: /health adds
                                              `profile_status` field
                                              with "experimental" tag
                                              when this profile is loaded)
```

**Reuse:** `scripts/download_ood_samples.py` patterns;
`scripts/run_profile_grid.py` to fold the new profile into the grid.

**Test gate:** Profile lands; M4 AUROC ≥ 0.75 on its own slice. If we
can't hit that, the honest answer is to ship the experimental tag and
a caveat rather than misleading numbers.

### A5 — Public benchmark dashboard

**Goal:** A single comparison page that combines the calib2000
profile grid + the new competitor numbers from A1, regenerated on
every push.

**Files to create / modify:**

```
scripts/build_benchmark_dashboard.py         (new)
docs/benchmarks/dashboard.md                 (generated)
.github/workflows/raid.yml                   (modify: add a build-dashboard
                                              step that runs the script
                                              and commits the generated
                                              dashboard.md if changed)
README.md                                    (modify: link to
                                              `docs/benchmarks/dashboard.md`
                                              from the Documentation
                                              list)
```

**Reuse:**

- `docs/benchmarks/calib2000/profile_grid_summary.csv`
- `docs/benchmarks/calib2000/loo_summary.csv`
- The competitor `EvalReport` JSONs from A1.

**Test gate:** Dashboard regenerates in CI without manual
intervention; rendered `dashboard.md` shows ≥ 5 profiles × ≥ 3 test
sets × CAIRN + ≥ 2 competitors.

---

## Track B — Operational hardening (~3 weeks)

### B1 — Auth + rate-limit on `cairn serve` and `cairn gui`

**Goal:** Close README's "no auth or rate limiting on the HTTP/FastAPI
transports" gap. Default-off so the loopback-friendly local
experience does not regress.

**Files to modify:**

```
cairn/api.py                          (add bearer-token + rate-limit
                                       middleware; extend `_redact` if
                                       needed — note `auth` is already
                                       in the sensitive-token list at
                                       line 53)
cairn/gui/server.py                   (matching guard in `_make_handler`)
configs/default.yaml                  (new `auth:` and `rate_limit:`
                                       sections, both disabled)
docs/deployment.md                    (production-deploy guide for
                                       enabling token + reverse proxy)
tests/test_http_api.py                (extend with token-required and
                                       token-rejected cases)
tests/test_gui_server.py              (matching cases)
```

**Design:**

- Bearer token via `CAIRN_API_TOKEN` env var (single token) or
  `auth.tokens` config list (multi-token). Constant-time comparison.
- Rate limiter: per-token + per-IP sliding window, in-memory by
  default, sqlite when `CAIRN_RATE_LIMIT_STORE=sqlite` (mirrors the
  history-store pattern).
- Both default to disabled. When disabled, `/health` reports
  `auth: "disabled"` honestly; when enabled, reports
  `auth: "bearer-token"` and the rate-limit window.

**Privacy:** No token material logged. The existing `_redact` already
scrubs any key matching `auth` per `cairn/api.py:51-77`.

**Test gate:** Token-required mode rejects unauthenticated requests
with 401; rate-limit returns 429 after the configured threshold;
default-off mode preserves current behaviour exactly.

### B2 — Persistent DriftMonitor

**Goal:** Close the gap CLAUDE.md flags: "DriftMonitor observations
are in-memory. Baselines rebuild from the persistent history store on
demand, but the in-flight observation window is lost on sidecar
restart."

**Files to create / modify:**

```
cairn/monitoring/store.py              (new — sqlite-backed
                                        observation store)
cairn/monitoring/drift.py              (modify: line 74's
                                        `self._observations: list[dict]`
                                        becomes a thin shim over
                                        `DriftStore`; window resets
                                        and check_drift() read from
                                        sqlite)
tests/test_monitoring.py               (extend: assert observations
                                        survive a `DriftMonitor`
                                        re-instantiation)
```

**Reuse:** `cairn/history.py` is the model for sqlite-backed
best-effort writes (single connection per call, WAL, daemon thread
fallback for non-critical writes). The drift store mirrors this
exactly.

**Invariant:** A failed drift-store write must never fail
`pipeline.analyze`. Same discipline as `history.py` and
`provenance/chain.py` writes.

**Test gate:** `tests/test_monitoring.py` adds a case that records 50
observations, instantiates a fresh `DriftMonitor` against the same
store path, and asserts `_observations` is recovered before
`check_drift` runs.

### B3 — Profile auto-routing

**Goal:** CAIRN's README acknowledges that
`RuntimeContext.switch_profile()` (`cairn/runtime.py:43`) swaps
manually but the pipeline does not auto-pick. Build the auto-pick
layer; keep manual selection as the default.

**Files to create / modify:**

```
cairn/auto_routing.py                 (new — `recommend_profile(
                                       router_decision, task_metadata,
                                       available_profiles) -> str`)
cairn/sidecar.py                      (modify: register
                                       `recommend_profile` method body,
                                       dispatch entry per CLAUDE.md's
                                       "Adding an RPC method — 4 touch
                                       points")
cairn/gui/server.py                   (modify: HTTP route in
                                       `_make_handler`)
src-tauri/src/commands.rs             (modify: Tauri command)
src-tauri/src/main.rs                 (modify: `invoke_handler`
                                       registration)
cairn/gui/static/index.html           (modify: add `cairnApi` entry
                                       in BOTH branches — Tauri
                                       `invoke` and HTTP `fetch`)
configs/default.yaml                  (new `runtime.auto_route: false`
                                       — opt-in)
tests/test_auto_routing.py            (new)
tests/test_rpc_sidecar.py             (extend)
tests/test_http_api.py                (extend)
```

**Logic:**

- `RouterDecision.domain == "prompt"` → `prompts_only`
- `RouterDecision.domain == "prose"` and word_count ≥ 50 →
  `master_minus_arxiv`
- `task_metadata.language` set and not English →
  `profile_multilingual` (once A4 lands; until then, fall through to
  `master_minus_arxiv` with a `caveat: "non-english"` field on the
  recommendation)

**Reuse:** `cairn/router.py:52::TextRouter` already classifies
prompt/prose/mixed; only the profile mapping is new.

**Test gate:** RPC returns sensible profile for ≥ 5 fixture inputs
covering prompt/prose/short/long/multilingual.

### B4 — Audit-chain on every analyze

**Goal:** `cairn/provenance/chain.py` exists and `tests/test_provenance_chain.py`
already covers it, but the chain is not invoked from
`pipeline.analyze`. Wire it.

**Files to modify:**

```
cairn/pipeline.py                     (modify: `CairnPipeline.analyze`
                                       at class line 126 — best-effort
                                       append to the audit chain after
                                       fusion + decision)
cairn/console.py                      (modify: add `cairn audit verify`
                                       subcommand)
tests/test_provenance_chain.py        (extend: end-to-end inclusion
                                       through the pipeline)
docs/architecture.md                  (modify: document the audit-chain
                                       record shape)
```

**Record shape:**

```json
{
  "submission_id": "...",
  "text_hash": "sha256:...",
  "profile": "master_minus_arxiv",
  "p_llm": 0.84,
  "determination": "AMBER",
  "calibration_status": "calibrated-router-conditional",
  "model_artifact_hash": "sha256:..."
}
```

The text itself is never written to the audit log — only the hash.
This holds even when `gui.history.retain_text=true`.

**Invariant:** Failed chain-append must not fail analyze. Same
best-effort discipline as history and drift.

**Test gate:** A 1k-call synthetic load produces a chain whose
`cairn audit verify` returns `valid=True` and prev-hash links across
all 1k entries.

---

## Track C — Closed-loop improvement (~3 weeks)

### C1 — `feedback_log` → calibration export

**Goal:** Existing `cairn/history.py:139::record_feedback` writes
confirmed labels to the `feedback_log` table. Nothing reads them back
into a training format.

**Files to create / modify:**

```
scripts/export_feedback_corpus.py     (new)
tests/test_feedback_export.py         (new)
docs/calibration.md                   (modify: add a "feedback corpus"
                                       section)
```

**Output:** `data/feedback_<YYYY-MM>.jsonl` matching the `EvalSample`
schema in `cairn/evaluation/dataset.py`.

**Privacy:** Honour `gui.history.retain_text=false`. When
`retain_text` was off at write time, the row has only a hash, so the
exporter writes a stub entry with `text_hash` and `confirmed_label`
that is unusable for training but recorded for audit/diagnosis.
Document this clearly: enabling text retention is a privacy choice the
operator must make.

**Test gate:** Round-trip — a synthetic feedback row exported, then
`cairn.evaluation.dataset.load_dataset` accepts the output without
schema errors.

### C2 — Monthly retrain runner

**Goal:** A single command that runs the recalibration cadence
described in `beet2_implementation_roadmap.md` Phase 7 + the existing
calibration scripts, with a hard promotion gate.

**Files to create:**

```
scripts/run_monthly_retrain.py
docs/recalibration_protocol.md
```

**Pipeline:**

1. Run C1 to export this month's feedback to
   `data/feedback_<YYYY-MM>.jsonl`.
2. Concatenate to `data/calib2000_v6_train.jsonl` (idempotent — track
   already-merged rows by submission_id).
3. Run `scripts/calibrate_detectors.py` →
   `scripts/train_fusion.py` →
   `scripts/calibrate_conformal.py` for `master_minus_arxiv`.
4. Run the full benchmark grid: in-domain HC3, RAID, MAGE, M4,
   `verb_low`, plus prompts heldout.
5. **Promotion gate:** AUROC delta ≥ 0 on every held-out OOD slice.
   Any regression aborts promotion. Honest reporting: write the
   regression to `docs/benchmarks/calib2000/retrain_failures/` so the
   failure is visible.
6. On promotion: write to
   `models/calib2000/profiles/master_minus_arxiv/` and move the
   previous artifact to a date-stamped sibling
   (`master_minus_arxiv_2026-04-07/`) for rollback.

**Reuse:** All four calibration scripts already exist
(`scripts/calibrate_detectors.py`, `scripts/train_fusion.py`,
`scripts/calibrate_conformal.py`, `scripts/run_benchmark.py`).

**Test gate:** Synthetic regression injection — feed an obviously-bad
augmented corpus, verify the promotion gate aborts and emits a
failure report. Rollback path tested by manually triggering and
confirming the previous artifact is restored bit-identical.

### C3 — Per-detector AND per-attack AUROC regression gate

**Goal:** `scripts/per_detector_auroc.py` already runs ad-hoc; nothing
fails CI when a detector's AUROC silently regresses. Once A2 lands,
the same discipline must extend to **per-attack** metrics so that a
regression on (say) `translation_pivot` blocks release the same way a
regression on `contrastive_lm` does. Both axes are first-class
release blockers.

**Files to modify / create:**

```
.github/workflows/raid.yml            (modify: add two CI steps —
                                       per-detector AUROC and
                                       per-attack AUROC; either step
                                       fails the build on a > 0.02
                                       drop vs its baseline CSV)
scripts/per_attack_auroc.py           (new — companion to
                                       per_detector_auroc.py; computes
                                       per-attack AUROC across the
                                       expanded transform set, mirrors
                                       the per_detector_auroc.py
                                       output schema for tooling
                                       reuse)
docs/benchmarks/calib2000/per_detector_auroc.csv  (committed
                                                   baseline; updated
                                                   only on intentional
                                                   regressions with
                                                   commit-message
                                                   justification)
docs/benchmarks/calib2000/per_attack_auroc.csv    (new — committed
                                                   baseline, lands
                                                   when A2 ships)
```

**Reuse:** Same TPR-gate pattern as the existing RAID workflow in
`.github/workflows/raid.yml`.

**Test gates:**

- A synthetic injected per-detector regression (force one detector to
  return a flat `p_llm=0.5`) fails the workflow. A no-op change passes.
- A synthetic injected per-attack regression (drop `paraphrase` AUROC
  by 0.05 via a corpus mutation) fails the workflow. A no-op change
  passes.
- **No silent baseline overrides.** Intentional regressions are
  unblocked only by an explicit baseline-CSV update in the same PR
  whose commit message documents the rationale. CI rejects PRs where
  the baseline CSV changes without an accompanying justification line
  matching `Baseline-Update: <reason>` in the commit body.

---

## Dependency graph

```text
honest-calibration-unified  ──►  Phase 4 stop (missingness fusion v7)
                                          │
                                          ▼
                          ┌─── A2 OOD attack pack
                          ├─── A3 hidden-reasoning detector
                          │
   A1 competitors ────────┴──► A5 public benchmark dashboard

   B1 ── auth + rate-limit ─────────────┐
   B2 ── persistent DriftMonitor        │ Track B: independent of
   B3 ── profile auto-routing           │ Tracks A and C
   B4 ── audit-chain ───────────────────┘

   C1 feedback export ── C2 monthly retrain ── C3 regression gate
                                  │                       │
                                  │                       └─► per-attack
                                  │                           gate consumes
                                  │                           A2's per_attack
                                  │                           baseline CSV
                                  │
                                  └─► consumes A1+A2 numbers as
                                      promotion-gate inputs
```

**Recommended dispatch waves:**

- Wave 1 (parallel): A1, B1, B2, C1.
- Wave 2 (after honest-calibration Phase 4): A2, A3, B3.
- Wave 3 (after Wave 2): A4, A5, B4, C2.
- Wave 4: C3.

---

## Risk register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPTZero / Originality APIs churn between adapter write and run | Medium | Adapter breaks under live key | Snapshot expected response shapes in `tests/test_benchmarks/fixtures/`; live run gated by `CAIRN_BENCHMARK_LIVE=1` |
| Adversarial-augmented corpus harms in-domain | Medium | Regression on HC3 prose | C2's promotion gate aborts on any in-domain delta < -0.005 |
| Hidden-reasoning detector over-fits to current OpenAI behaviour | Medium | Detector collapses on UX change | Cross-provider parity gate; Gemini AUROC must not drop > 0.02 when the new detector is wired in |
| Auth scope creep into multi-tenant accounts | Medium | Roadmap derails into product work | Resist; bearer + rate-limit only. Multi-tenant lives in the future "distribution" roadmap |
| Multilingual scope creep | Low | Spend exceeds the probe budget | A4 ships with an explicit "experimental" tag in `/health`; a full multilingual product is not on this roadmap |
| Retrain runner promotes silently bad artifact | Low | Production calibration regresses | Promotion gate is hard; failures are reported, not buried |
| Persistent DriftMonitor write contention | Low | Drift writes block analyze | Best-effort discipline; failed write logs to stderr and continues |
| Commercial humanizer APIs (StealthGPT etc.) churn or sunset | Medium | A2's `humanizer_tools` transform breaks; budget overrun if pricing changes | Same gating pattern as A1's commercial adapters: snapshot-test against fixtures, gate live runs on `CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1`, budget cap per-run. If a humanizer API sunsets, the transform falls back to skip with a recorded reason; the eval continues without it. |
| Translation-pivot or summarization-attack transforms drift label semantics | Medium | Augmented corpus has noisy labels; trained profile regresses | Per-transform structural test in `tests/test_adversarial/test_transforms.py` asserts label preservation via embedding-cosine threshold; transforms below threshold are excluded from the augmentation. |

---

## Success criteria

| Track | Gate |
|-------|------|
| A1 | Competitor `EvalReport` JSON committed under `docs/benchmarks/competitor/` for ≥ 2 OSS baselines (Binoculars, Fast-DetectGPT) on RAID and HC3 |
| A2 | RAID AUROC ≥ 0.73, MAGE ≥ 0.60, HC3 in-domain ≥ 0.94, `verb_low` not regressed > 0.01; per-attack before/after table published in `docs/benchmarks/attack_coverage.md` for all 12 transforms with no transform regressing > 0.02; `master_minus_arxiv_robust_v1` profile ships alongside the unaugmented baseline |
| A3 | OpenAI matched-comparison AUROC within 3 points of Gemini on `calibrated_full` (≥ 0.96 absolute) without > 0.02 drop on Gemini |
| A4 | `profile_multilingual` lands; M4 AUROC ≥ 0.75 on its own slice OR ships with explicit experimental tag and honest caveat |
| A5 | Dashboard regenerates in CI on every push to master; linked from `README.md` |
| B1 | Bearer-token + rate-limit covered by tests in `tests/test_http_api.py` and `tests/test_gui_server.py`; `/health` reports `auth` field honestly |
| B2 | DriftMonitor observations survive sidecar restart (asserted in `tests/test_monitoring.py`) |
| B3 | `recommend_profile` RPC returns sensible profile across ≥ 5 fixture inputs (prompt / prose / short / long / multilingual) |
| B4 | `cairn audit verify` returns `valid=True` after a synthetic 1k-call run; `tests/test_provenance_chain.py` extended for end-to-end inclusion |
| C1 | `scripts/export_feedback_corpus.py` round-trips through `EvalSample` schema; honours `retain_text=false` |
| C2 | Monthly runner promotes only on AUROC-delta ≥ 0 across RAID/MAGE/M4/`verb_low`; rollback path tested |
| C3 | `.github/workflows/raid.yml` fails on a synthetic injected per-detector regression > 0.02 AND on a per-attack regression > 0.02; baseline CSV updates require a `Baseline-Update: <reason>` line in the commit body |

---

## What this roadmap does NOT promise

- A fixed calendar date. Effort estimates above are CAIRN-developer-time
  approximations; actual ship dates depend on the honest-calibration
  unified plan landing first and on the size of the calib2000 retrain
  loop on available hardware (Modal GPU offload available for feature
  caching per `scripts/modal_cache_features.py`).
- Beating every commercial detector on every benchmark. Track A1
  produces honest numbers, not curated wins. If a competitor beats
  CAIRN on a slice, the dashboard reports that.
- A multilingual product. A4 is a probe, full stop.
- A SaaS or hosted offering. That work belongs in a future roadmap.

---

## References

- [`docs/superpowers/plans/2026-05-02-honest-calibration-unified.md`](../plans/2026-05-02-honest-calibration-unified.md) — the in-flight plan this layers on
- [`docs/superpowers/specs/2026-04-14-evaluation-adversarial-benchmarks-design.md`](2026-04-14-evaluation-adversarial-benchmarks-design.md) — original competitor-benchmark spec (A1 finishes the deferred third module)
- [`docs/superpowers/specs/2026-04-23-drift-monitor-wiring-design.md`](2026-04-23-drift-monitor-wiring-design.md) — the drift wiring B2 hardens
- [`docs/benchmarks/README.md`](../../benchmarks/README.md) — current calib2000 numbers (the baseline this roadmap improves on)
- [`beet2_implementation_roadmap.md`](../../../beet2_implementation_roadmap.md) — the original 2026-04-19 roadmap (Phase 7 recalibration protocol becomes C2)

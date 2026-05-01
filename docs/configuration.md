# CAIRN — Configuration Guide

All pipeline behavior is driven by YAML config files in `configs/`.
This doc covers the profile system, the config schema, and how to
tune thresholds, detectors, and the GUI's history / drift store.

## Profile layout

Profiles divide into two groups: **heuristic** (no calibration
artifacts; per-detector tables drive `p_llm`) and **calibrated**
(trained EBM + isotonic + split-conformal artifacts override
heuristics).

### Heuristic profiles

| Profile | Intent | Key choices |
|---------|--------|-------------|
| `default` | Balanced baseline | all Tier-1+2 detectors on, tier-3 off, history+drift enabled |
| `screening` | Fast triage | only Tier-1 detectors; tier-2/3 disabled |
| `strict` | High-recall | more aggressive thresholds; tier-3 enabled if credentials present |
| `no-api` | Offline | tier-3 detectors forcibly off (no Anthropic/OpenAI calls) |
| `production` | Containerised | paths rooted at `/app` and `/data`, logging + retention dialed up |

### Calibrated profiles (production)

These set `fusion.model_path`, `fusion.conformal_path`, and
`fusion.detector_calibration_path` to trained artifacts under
`models/calib2000/`. See [`docs/calibration.md`](calibration.md) for
how the corpus and artifacts are produced.

| Profile | Trained on | Best for |
|---------|------------|----------|
| `calibrated_calib2000` | Full v6 master corpus (~94K rows, 6 providers, all source tags) | General-purpose default |
| `general_hc3` | HC3 only (~20K rows) | Pure prose, in-domain — peak in-distribution AUROC (0.94) |
| `general_full` | HC3 + OOD (raid/m4/mage) | HC3 plus OOD coverage |
| `master_minus_arxiv` | v6 minus arxiv abstracts (~70K rows) | **Cross-domain prose, RAID/MAGE, short-form. Recommended.** |
| `drop_top3_drags` | v6 minus arxiv + reddit + magpie (~63K rows) | Most aggressive OOD gain; sacrifices prompt-detection |
| `prompts_only` | All prompt corpora (~24K rows) | Prompt detection (human-written vs LLM-generated prompts) |
| `chat_prompts` | Conversational prompt subsets | Chat-style prompts |
| `medical` | HC3 medicine + MedQA prompts | Medical domain |
| `finance` | HC3 finance only | Finance domain |
| `stem` | arxiv + GSM8K + HumanEval | Math / code / scientific prose |
| `stem_with_prompts` | STEM + math/code/reasoning prompts | STEM responses + STEM prompts |
| `short_form` | concise + verb_low cells | Short-form responses |
| `everything` | Full v6 (reference baseline) | Strictly worse than `master_minus_arxiv` on OOD; kept for comparison |
| `general_plus_*` | HC3+OOD plus stem / short / medical | Adjustable mixes |

### Picking a profile

```
short, prose, in-domain HC3   → general_hc3
RAID / MAGE / cross-genre     → master_minus_arxiv
human-vs-LLM prompts          → prompts_only
domain-specific corpus        → medical / finance / stem
unsure / mixed inputs         → calibrated_calib2000 (full v6)
no labeled data, just heuristics → screening / default
```

Every profile except `default` uses `_extends: default` so they only
need to override what differs.

## Profile inheritance

```yaml
_profile: screening
_extends: default

detectors:
  surprisal_dynamics:
    enabled: false
  contrastive_lm:
    enabled: false
```

`cairn.config.load_config` deep-merges the child into the parent:
scalars overwrite, dicts merge recursively, lists replace. Keys
starting with `_` (like `_profile`, `_extends`) are metadata only and
don't survive the merge into the active config dict.

## Required top-level keys

`load_config` enforces three keys must be present after merging:

```yaml
detectors: {...}
cascade: {...}
decision: {...}
```

Missing any of these raises `ConfigError`.

## Detectors block

```yaml
detectors:
  preamble:
    enabled: true
    weight: 1.0
  fingerprint_vocab:
    enabled: true
    weight: 0.8
  contrastive_gen:
    enabled: false
    weight: 0.95
    provider: "anthropic"
    model: "claude-sonnet-4-6"
    n_baselines: 3
  ...
```

- `enabled` — gate the detector on/off at load time.
- `weight` — multiplier applied in the naive-fusion fallback; ignored
  when a trained EBM fusion model is loaded.
- Detector-specific keys (`provider`, `model`, `n_baselines`, `hf_model`,
  etc.) flow through to `detector.analyze(text, config)` as the second
  argument — each detector documents what it reads.

## Cascade block

```yaml
cascade:
  phase1_short_circuit_high: 0.85  # above this, stop after Phase 1
  phase1_short_circuit_low: 0.10   # below this, stop after Phase 1
  phase2_short_circuit_high: 0.80
  phase2_short_circuit_low: 0.15
  phase3_always_run: false         # force Phase 3 regardless
```

Raise the `_high` / lower the `_low` thresholds to short-circuit more
aggressively (faster, fewer Tier-2/3 calls, less accurate on
ambiguous inputs). `phase3_always_run: true` is useful for evaluation
runs where you want every detector's raw output regardless of
confidence.

## Decision block

```yaml
decision:
  red_threshold: 0.75
  amber_threshold: 0.50
  yellow_threshold: 0.25
  abstention:
    enabled: true
    max_prediction_set_size: 3
```

- Thresholds are inclusive lower bounds — fused `p_llm >=
  red_threshold` → RED, etc. Anything below `yellow_threshold` → GREEN.
- Abstention: if the conformal prediction set covers `>=
  max_prediction_set_size` severity bands, the engine returns
  UNCERTAIN regardless of the point estimate. Disable by setting
  `enabled: false`.

## GUI block (optional)

```yaml
gui:
  history:
    enabled: true
    db_path: data/cairn_history.sqlite3
    retain_text: true           # store full text in the SQLite submission log
  drift:
    enabled: true
    store_path: data/drift_alerts
    window_size: 1000           # observations per drift-check flush
    kl_threshold: 0.20          # per-feature drift trigger
    ece_threshold: 0.15         # calibration-drift trigger
```

- History: `retain_text: false` stores only the sha256 hash, suitable
  for privacy-sensitive deployments where the raw text should stay out
  of the index.
- Drift: the three tuning keys can also live at the config top level
  under `drift_monitoring` (the historical location); top-level wins
  if both are set.

## Router block

```yaml
router:
  minimum_words:
    prompt: 30
    prose: 150
    mixed: 300
```

Below these word counts the router returns `insufficient` and
downstream phases operate on the minimum-viable detector set.

## Privacy block

```yaml
privacy:
  raw_text_retention_days: 90
  access_logging: true
  contributor_anonymization_days: 365
  external_api:
    mode: "restricted"            # or "open", "disabled"
    allowed_providers: ["anthropic"]
    log_all_api_calls: true
```

Consumed by `cairn.privacy.retention` (purge CLI) and
`cairn.privacy.store.PrivacyStore` (access log).

## Runtime profile switching

The GUI's Settings tab includes a profile dropdown. Clicking **Switch**
calls `RuntimeContext.switch_profile(name)`, which:

1. Resolves `configs/<name>.yaml` (name is validated against
   `^[A-Za-z0-9][A-Za-z0-9_-]*$` to block path traversal).
2. Calls `load_config` — if the file is missing or malformed, the old
   pipeline stays in place and the UI surfaces `ERR_BAD_PROFILE`.
3. Constructs a fresh `CairnPipeline` outside the lock.
4. Atomically swaps the pipeline, profile name, and config under the
   lock.

A failed switch leaves the old context untouched.

## Adding a new profile

```bash
cat > configs/my-profile.yaml <<EOF
_profile: my-profile
_extends: default

detectors:
  preamble:
    weight: 1.2       # boost the preamble weight
  perturbation:
    enabled: true     # enable a Tier-3 detector

decision:
  red_threshold: 0.70  # be more willing to flag RED
EOF

cairn gui --profile my-profile
```

The profile appears automatically in the Settings-tab dropdown via
`cairn.config.list_profiles`, which scans `configs/*.yaml`.

## Pattern files

Some detectors load additional pattern lists from
`configs/patterns/*.yaml`:

- `preamble_patterns.yaml` — LLM-preamble regexes grouped by severity
- `fingerprint_words.yaml` — LLM-biased vocabulary and bigrams
- `constraint_frames.yaml` — constraint-framing phrases for
  `prompt_structure`

Edit these files to tune detection without touching Python. Changes
take effect on next pipeline construction.

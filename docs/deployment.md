# CAIRN — Deployment Guide

This guide covers the three runtimes: embedded web GUI, REST API
(containerised), and the stdio sidecar used by the Tauri desktop shell.

## Installation

```bash
pip install -e .
```

Installs the `beet` and `cairn-sidecar` console scripts.

## Embedded web GUI

No external dependencies beyond the base install:

```bash
cairn gui --port 8877 --profile screening
```

Serves a single-file SPA at `http://127.0.0.1:8877/` with tabs for
Analyze, Batch, History, Monitoring, Evaluation, and Settings. The
server is single-threaded and suitable for local/desktop use; for
multi-user deployments, front with a reverse proxy or use the REST API.

History persists to `data/cairn_history.sqlite3` by default; override
via the `gui.history.db_path` field in the active config.

## REST API (containerised)

Install API extras locally, or use the provided Dockerfile.

```bash
pip install -e ".[api]"
cairn serve --host 0.0.0.0 --port 8000 --profile production
```

Docker:

```bash
docker compose up --build
```

Endpoints:

- `GET  /health`   — liveness + model info
- `GET  /config`   — current (redacted) config
- `POST /analyze`  — `{text, task_metadata?, submission_id?}`
- `POST /batch`    — `{items: [{id, text}], cross_similarity?: bool}`

The web GUI (`cairn gui`) exposes a larger surface — history, batch,
profile switching, monitoring, evaluation — via its own routes; see
`cairn/gui/server.py` for the complete list.

## Desktop shell (Tauri)

The Tauri app spawns `cairn.sidecar` as a subprocess and talks line-
delimited JSON-RPC over stdio. Dev-mode override:

```bash
export CAIRN_SIDECAR_CMD="python -m cairn.sidecar --profile default"
cd src-tauri && cargo tauri dev
```

See `docs/building.md` for the full build, bundling, and PyInstaller
sidecar pipeline.

## Production configuration

`configs/production.yaml` points at `/app/models/` for trained artifacts
and `/data/` for monitoring + privacy vault. Mount these directories
into the container:

```yaml
volumes:
  - ./models:/app/models:ro
  - beet-data:/data
```

## Single-file analysis (programmatic)

The pre-refactor `cairn analyze` / `beet eval` / `beet batch` /
`beet robustness` / `beet fairness` / `beet privacy purge` /
`beet audit-validate` subcommands are currently unavailable on the
restored CLI. Use the Python module surface directly:

```python
from cairn.config import load_config
from cairn.pipeline import CairnPipeline
from cairn.report import build_json_report

pipeline = CairnPipeline.from_config_file("configs/screening.yaml")
report = build_json_report(pipeline.analyze("submission text..."))
```

For evaluation against a labeled JSONL dataset, the Evaluation tab in
the web GUI wraps the same `cairn.evaluation.runner.run_eval` with a
200-sample default cap and 1000-sample hard cap. For larger runs, call
`run_eval` directly in a script:

```python
from cairn.evaluation.dataset import load_dataset
from cairn.evaluation.runner import run_eval, eval_report_to_dict

samples = load_dataset("data/dataset.jsonl")
result = eval_report_to_dict(run_eval(pipeline, samples))
```

## Model artifacts

See `scripts/` for training:

- `scripts/calibrate_detectors.py` — fit per-detector isotonic calibration
- `scripts/train_fusion.py` — train the EBM fusion model
- `scripts/calibrate_conformal.py` — calibrate conformal threshold

Outputs land in `models/` and are referenced by `fusion.model_path` etc.
in the active config. Execution requires a labeled JSONL dataset (see
`scripts/build_seed_dataset.py` — Phase 7).

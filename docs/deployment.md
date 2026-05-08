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

- `GET  /health`   — liveness + model info + `calibration_status` (one of
  `heuristic`, `fusion-only`, `calibrated`, `calibrated-router-conditional`);
  see [`docs/calibration.md`](calibration.md) for what each tier means and
  which profiles produce it.
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

## Production hardening

CAIRN's HTTP transports (`cairn serve` and `cairn gui`) bind to
loopback by default and ship with **no auth and no rate limiting**.
When you expose either past loopback, enable both. Both surfaces share
the same configuration shape under `auth:` and `rate_limit:` —
documented in `configs/default.yaml`.

### Bearer-token auth

Two equivalent sources for tokens:

1. **Env var** (one token, easiest in containers):

   ```bash
   export CAIRN_API_TOKEN="$(openssl rand -hex 32)"
   cairn serve --host 0.0.0.0 --port 8000
   ```

2. **Config file** (multi-token, useful for rolling rotation):

   ```yaml
   auth:
     enabled: true
     tokens:
       - "01J9G1...primary..."
       - "01J9G2...rotation-window..."
   ```

When enabled, every endpoint *except* `/health` requires
`Authorization: Bearer <token>`. Tokens are compared via
`hmac.compare_digest` (constant-time) and never logged. The
`/config` endpoint redacts any key matching the existing
`auth | secret | api_key | token | password | credential` pattern.

The `/health` endpoint is intentionally left open so liveness probes
keep working without a token. It honestly reports the auth state:

```json
{
  "status": "ok",
  "auth": "bearer-token",
  "rate_limit": {
    "store": "memory",
    "window_seconds": 60,
    "max_per_window": 60
  }
}
```

Both fields read `"disabled"` when the corresponding feature is off,
so an operator can verify the live deploy.

### Rate limiting

```yaml
rate_limit:
  enabled: true
  window_seconds: 60
  max_per_window: 60
  store: memory                    # or "sqlite"
  hash_keys: false                 # set true to keep raw IPs off disk
```

The limiter buckets per token (when authed; SHA-256 prefix of the
header) or per client IP (otherwise). When the cap is exceeded the
endpoint returns:

```json
HTTP/1.1 429 Too Many Requests
Content-Type: application/json

{ "error": "rate limit exceeded", "code": "ERR_RATE_LIMIT",
  "window_seconds": 60, "max_per_window": 60 }
```

Choose `store: memory` for single-process deploys; `store: sqlite`
when you need the limit to survive a worker restart or be shared
across `cairn serve` workers behind a reverse proxy. Set
`CAIRN_RATE_LIMIT_STORE=sqlite` to override at runtime without
editing the config.

### Reverse-proxy example (nginx)

CAIRN is loopback-only by default. The recommended non-loopback
deployment fronts it with a TLS-terminating reverse proxy:

```nginx
server {
    listen 443 ssl http2;
    server_name cairn.example.com;

    ssl_certificate     /etc/letsencrypt/live/cairn.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/cairn.example.com/privkey.pem;

    # CAIRN already does bearer auth + rate limiting; the proxy adds
    # TLS, request-size cap, and a higher-level connection limit.
    client_max_body_size 1m;
    limit_req_zone $binary_remote_addr zone=cairn_burst:10m rate=10r/s;

    location / {
        limit_req zone=cairn_burst burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        # Bearer header passes through unchanged.
    }
}
```

Note: CAIRN's rate limiter buckets on the direct peer IP. When
fronting with a proxy, deploy the same auth token to all upstream
clients and rely on the proxy's `limit_req` plus CAIRN's per-token
bucket — or run CAIRN with `auth.enabled: true` so each upstream
client is identified by its token rather than the proxy's IP.

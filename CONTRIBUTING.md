# Contributing

Thanks for looking at CAIRN. This doc covers the development loop,
code conventions, and how to add detectors.

## Dev setup

```bash
git clone https://github.com/occamsrazor0102/LLM-Detector-Reboot.git
cd LLM-Detector-Reboot
pip install -e ".[dev]"
pytest
```

Optional extras:

- `".[tier2]"` — transformers + sentence-transformers (Phase 2 detectors)
- `".[tier3]"` — anthropic + openai (Phase 3 LLM-backed detectors)
- `".[fusion]"` — interpret + scikit-learn + scipy (trained fusion model)
- `".[batch]"` — networkx + scikit-learn (contributor_graph, cross_similarity)
- `".[api]"` — fastapi + uvicorn (`cairn serve`)
- `".[full]"` — everything

## Running the pieces

```bash
cairn gui --profile screening            # web UI, default port 8877
cairn sidecar --profile default          # stdio JSON-RPC (for Tauri)
cairn serve --profile production         # FastAPI, needs [api]
python -m cairn.sidecar --help           # same as `cairn sidecar`
```

For the Tauri desktop shell, see `docs/building.md`.

## Code conventions

- Python 3.11+; type hints on public surfaces.
- Vanilla stdlib preferred at module boundaries (`http.server`,
  `sqlite3`, `json`). External deps are OK for algorithms
  (`interpret`, `scikit-learn`, `transformers`).
- Dataclasses for contracts; `Protocol` for structural typing
  (see `cairn/contracts.py::Detector`).
- Pipeline side effects go through the runtime context
  (`cairn/runtime.py::RuntimeContext`) so `switch_profile` stays atomic.
- History / drift writes are best-effort — a failed write should log
  and continue, never fail the underlying analyze call.

## Tests

`pytest` against ~35 test files across:

- `tests/test_contracts.py` — dataclass shapes
- `tests/test_detectors/` — one file per detector
- `tests/test_router.py`, `test_cascade.py`, `test_fusion.py`,
  `test_decision.py`, `test_calibration.py`
- `tests/test_pipeline.py` — end-to-end pipeline runs
- `tests/test_http_api.py`, `tests/test_rpc_sidecar.py` — live
  transport tests against the stdlib handler and the Sidecar class
- `tests/test_history.py`, `tests/test_history_stats.py`,
  `tests/test_drift_wiring.py` — persistence + monitoring
- `tests/test_evaluation/` — dataset, runner, metrics, ablation
- `tests/test_privacy.py`, `tests/test_privacy_retention.py`,
  `tests/test_provenance_chain.py` — privacy & audit

Every new feature or fix should ship with tests. Prefer tests against
the public contract (pipeline.analyze, the RPC methods, Determination
shape) over tests against internal helpers.

## Adding a detector

1. Create `cairn/detectors/your_detector.py` implementing the protocol
   from `cairn/contracts.py`:

   ```python
   from cairn.contracts import LayerResult

   class YourDetector:
       id = "your_detector"
       domain = "prose"            # or "prompt" or "universal"
       compute_cost = "cheap"       # or "trivial" / "moderate" / "expensive"

       def analyze(self, text: str, config: dict) -> LayerResult:
           ...
           return LayerResult(
               layer_id=self.id, domain=self.domain,
               raw_score=...,
               p_llm=...,
               confidence=...,
               signals={...},
               determination=...,
               attacker_tiers=["A0", "A1"],
               compute_cost=self.compute_cost,
               min_text_length=30,
               spans=[],   # populate if you can point at the text
           )

       def calibrate(self, labeled_data: list) -> None:
           pass

   DETECTOR = YourDetector()
   ```

2. Register it in `cairn/pipeline.py` (detector registry) and a cascade
   phase in `cairn/cascade.py::PHASE{1,2,3,4}_DETECTORS`.

3. Add a default entry in `configs/default.yaml::detectors`.

4. Add `tests/test_detectors/test_your_detector.py` with at least one
   positive and one negative case.

5. If your detector has positional evidence (regex matches, token
   offsets), populate `spans` — the UI will pick it up automatically.
   Add a new `.span-<kind>` class in `cairn/gui/static/index.html` if
   the existing palette doesn't cover your detector's output.

## Adding an RPC method

Both the HTTP server (`cairn/gui/server.py`) and the sidecar
(`cairn/sidecar.py`) route to the same shared functions. A new method
needs four touch points:

1. Method body on `Sidecar` class (or a helper it calls).
2. Dispatch entry in `Sidecar.handle`.
3. Matching HTTP route in `_make_handler`.
4. Tauri command in `src-tauri/src/commands.rs` + handler registration
   in `src-tauri/src/main.rs::invoke_handler`.

Add a test per transport (`tests/test_rpc_sidecar.py`,
`tests/test_http_api.py`). If you need to touch the frontend, extend
`beetApi` in both branches (tauri + http) in
`cairn/gui/static/index.html`.

## Writing design specs

Non-trivial features should land a spec in
`docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` before the code.
See that directory for worked examples (six GUI phases plus the drift
wiring enhancement).

## Commit style

Short subject line, bullet body with details, and co-author attribution
if applicable. Conventional-commit prefixes (`feat:`, `fix:`, `docs:`,
`chore:`) are used throughout the history but not mandatory.

## Licensing

CAIRN is licensed under the
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
(see `LICENSE`). By contributing you agree your work is made
available under the same license.

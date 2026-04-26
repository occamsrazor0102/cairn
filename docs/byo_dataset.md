# Calibrating BEET on your own dataset

The calibration scripts (`scripts/calibrate_detectors.py`,
`scripts/train_fusion.py`, `scripts/calibrate_conformal.py`) are
corpus-agnostic. They all take `--dataset path/to/file.jsonl` and
read the **EvalSample JSONL** schema below — `download_hc3.py` is
just one possible producer.

This guide shows two ways to plug in your own labels.

## Schema: EvalSample JSONL

One JSON object per line. Required fields:

| Field   | Type | Notes |
|---------|------|-------|
| `id`    | str  | Unique within the dataset. |
| `text`  | str  | The submission to score. |
| `label` | int  | `1` = LLM-generated, `0` = human-written. |

Optional fields (passed through to evaluation reports):

| Field             | Type | Notes |
|-------------------|------|-------|
| `tier`            | str  | Adversary/source tag. `"human"`, `"A0"`, `"A1"`, `"A2"`, `"A3"`. |
| `source`          | str  | Where the sample came from (e.g. `"hc3_finance"`). |
| `attack_name`     | str  | If the sample was generated/transformed by a known attack. |
| `attack_category` | str  | Coarse attack class (paraphrase, mix, etc.). |
| `source_id`       | str  | Upstream id (e.g. the original corpus row). |

Example line:

```json
{"id": "mydata_000001", "text": "Sample text here.", "label": 1, "tier": "A0", "source": "mycorpus_v1"}
```

## Path 1 — You already have JSONL

Point the calibration scripts at it directly:

```bash
python scripts/calibrate_detectors.py \
  --dataset data/my_corpus.jsonl \
  --config  configs/default.yaml \
  --out     models/calibrators/my_corpus.json

python scripts/train_fusion.py \
  --dataset data/my_corpus.jsonl \
  --config  configs/default.yaml \
  --out     models/fusion/my_corpus_ebm.pkl

python scripts/calibrate_conformal.py \
  --dataset data/my_corpus_calib.jsonl \
  --config  configs/default.yaml \
  --alpha   0.05 \
  --out     models/conformal/my_corpus.json
```

Use a separate held-out file for the conformal calibration step —
otherwise the coverage guarantee is invalid (you'd be calibrating on
the training distribution).

## Path 2 — You have CSV/TSV

Convert via `scripts/import_dataset.py`:

```bash
python scripts/import_dataset.py \
  --in        data/raw/my_corpus.csv \
  --out       data/my_corpus.jsonl \
  --text-col  body \
  --label-col is_llm \
  --source    mycorpus_v1
```

`--label-col` accepts `0`/`1`, `human`/`llm`, or
`human`/`ai`/`gpt`/`chatgpt`/`machine`/`synthetic` (case-insensitive).
TSV inputs are detected by the `.tsv` / `.tab` extension.

Cap class sizes for balanced subsets with `--max-per-class N`.

## Path 3 — You have folders of `.txt` files

Use `beet.evaluation.dataset.build_dataset` programmatically:

```python
from beet.evaluation.dataset import build_dataset, save_dataset

samples = build_dataset([
  {"path": "data/raw/human/",   "label": 0, "tier": "human", "source": "in_house"},
  {"path": "data/raw/chatgpt/", "label": 1, "tier": "A0",    "source": "in_house"},
])
save_dataset(samples, "data/in_house.jsonl")
```

## Wiring the artifacts into a profile

Once calibration artifacts exist, point a config profile at them:

```yaml
# configs/calibrated.yaml
fusion:
  model_path:     models/fusion/my_corpus_ebm.pkl
  conformal_path: models/conformal/my_corpus.json
detectors:
  calibration_path: models/calibrators/my_corpus.json
```

Then `BeetPipeline.from_config_file("configs/calibrated.yaml")`
auto-loads them and `/health` reports
`calibration_status: "calibrated"` (UI banner suppressed).

## Splits & leakage

Best practice: split by *source document*, not by row, before any
of the three calibration steps. Then:

- ~70% → `train_fusion.py`
- ~15% → `calibrate_conformal.py` (must be separate)
- ~15% → held-out test for `scripts/run_benchmark.py`

If your labels come from one model family (e.g. only ChatGPT
outputs), expect the conformal coverage guarantee to weaken on
samples from a different family — re-calibrate when you broaden
attacker scope.

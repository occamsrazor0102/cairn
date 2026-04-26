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

## Path 4 — Regenerate the LLM side of HC3 with a current model

HC3's `chatgpt_answers` are early-2023 gpt-3.5 outputs. Calibrating
against them teaches "ChatGPT-2023" patterns, not modern-LLM
patterns. `scripts/regenerate_llm_outputs.py` keeps HC3's real human
answers and replaces the LLM side with fresh generations from a
current model (Anthropic OR OpenAI):

```bash
# Dry-run first — 10 questions, ~$0.03 on Haiku.
export ANTHROPIC_API_KEY=...
python scripts/regenerate_llm_outputs.py \
  --subset finance \
  --max-questions 10 \
  --provider anthropic \
  --model claude-haiku-4-5-20251001 \
  --out data/hc3_finance_regen_haiku.jsonl

# Production — ~1000 questions (~$1.50 on Haiku).
python scripts/regenerate_llm_outputs.py \
  --subset all \
  --max-questions 1000 \
  --provider anthropic \
  --model claude-haiku-4-5-20251001 \
  --out data/hc3_all_regen_haiku.jsonl
```

OpenAI works the same way:

```bash
export OPENAI_API_KEY=...
python scripts/regenerate_llm_outputs.py \
  --subset all --max-questions 1000 \
  --provider openai --model gpt-4o-mini \
  --out data/hc3_all_regen_gpt4o.jsonl
```

The prompt is intentionally minimal — `system="You are a helpful
assistant. Answer the user's question accurately and helpfully."` +
the HC3 question verbatim. No "be concise" / "be human" / "match
this style" instructions, since those would teach the detector to
spot prompt-engineering artifacts instead of native model output.

The script is **resumable** — it appends to `--out` and skips
questions already generated. A crash, rate-limit, or Ctrl-C doesn't
waste credits.

### Multi-provider runs

For cross-family robustness, point each provider at a *different
slice* of HC3 questions via `--start-question`. Each question is
processed by exactly one model — no human-side duplication, no ID
collisions:

```bash
# Anthropic handles questions 0-499
python scripts/regenerate_llm_outputs.py \
  --subset all --start-question 0 --max-questions 500 \
  --provider anthropic --model claude-sonnet-4-6 \
  --out data/regen_anthropic.jsonl

# OpenAI handles questions 500-999
python scripts/regenerate_llm_outputs.py \
  --subset all --start-question 500 --max-questions 500 \
  --provider openai --model gpt-4o-mini \
  --out data/regen_openai.jsonl

# Concatenate, split, calibrate as usual
cat data/regen_anthropic.jsonl data/regen_openai.jsonl > data/regen_mixed.jsonl
python scripts/split_dataset.py --in data/regen_mixed.jsonl --out-dir data --prefix regen_mixed
```

The detector learns "LLM-ness" rather than "Claude-ness" or
"GPT-ness". Scale up to N providers by extending the slice math.

### Anthropic notes

Default is Haiku 4.5 (cheapest, ~$0.0015/question). Pass
`--model claude-sonnet-4-6` for stronger output (~$0.008/question)
or `--model claude-opus-4-7` (~$0.025/question). For Sonnet/Opus the
script automatically passes `thinking={"type": "disabled"}` so we
get clean answer text without spending tokens on hidden reasoning.

### Using a saved OpenAI prompt template (Responses API)

If you've built a Prompt template in
[platform.openai.com → Prompts](https://platform.openai.com/playground/prompts),
the script can call it via the Responses API instead of raw
`chat.completions`:

```bash
# Template uses `input` for the question (most common)
python scripts/regenerate_llm_outputs.py \
  --provider openai \
  --openai-prompt-id pmpt_abc123... \
  --openai-prompt-version 1 \
  --max-questions 10 \
  --out data/hc3_finance_regen_via_template.jsonl

# Template uses {{question}} variable substitution
python scripts/regenerate_llm_outputs.py \
  --provider openai \
  --openai-prompt-id pmpt_abc123... \
  --openai-template-variable question \
  --max-questions 10 \
  --out data/hc3_finance_regen_via_template.jsonl
```

When `--openai-prompt-id` is set, `--model` is ignored (the template
specifies its own model). To check which input mode your template
uses, open it in the Prompts playground — variables show up as a
side panel.

**Calibration warning when using a custom template:** anything in
the template beyond a minimal "answer the question" instruction
(style guides, length caps, persona constraints, output formats)
will bake into the LLM samples. The detector then learns to spot
*your prompting choices* on top of native LLM patterns. This is
fine if every deployment will use that same template; less useful
as a general LLM detector.

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

# `raid_mini.jsonl` — synthetic placeholder

This file is a **SYNTHETIC PLACEHOLDER for CI smoke testing only**. Every row
carries `source: "synthetic_placeholder"` so it cannot be confused with real
RAID data downstream.

CI green on this fixture is **not** a meaningful TPR@FPR<=0.01 measurement.
The 25 human-flavored and 25 LLM-flavored templates were hand-authored to
exercise the cascade end-to-end (every row is >= 50 words so detectors
actually fire), but they are not drawn from the licensed RAID corpus and
the score distribution they produce will not match real RAID.

## To replace with real RAID data

Real RAID validation requires running

```bash
python scripts/build_raid_fixture.py --source raid-test.csv --n-per-class 25
```

against the licensed RAID test set (https://raid-bench.xyz/). Verify the
RAID licence permits redistribution before committing the resulting
`raid_mini.jsonl`; if not, commit only a SHA256 manifest and have CI fetch
on the fly.

## To regenerate the placeholder

If this file gets corrupted or you want to refresh the synthetic content:

```bash
python scripts/gen_raid_synthetic_placeholder.py
```

The generator is deterministic (seed=42); re-running produces a
byte-identical file.

## Schema

Each line is a UTF-8 JSON object with these fields, matching what
`scripts/build_raid_fixture.py` would emit so that
`scripts/run_raid_benchmark.py` can consume either source:

| field         | type           | example                           |
|---------------|----------------|-----------------------------------|
| `id`          | string         | `"raid_synth_human_007"`          |
| `text`        | string (>=50w) | `"I remember the summer my..."`   |
| `label`       | int (0 or 1)   | `0` (human) or `1` (LLM)          |
| `source`      | string         | `"synthetic_placeholder"`         |
| `attack_name` | string \| null | always `null` in the placeholder  |
| `tier`        | string         | always `"raid"`                   |

50 rows total, 25 per class, deterministically shuffled.

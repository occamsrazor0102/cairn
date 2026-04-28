"""Convert a CSV / TSV of (text, label) into BEET's EvalSample JSONL.

Use this when you want to calibrate on your own labeled corpus rather
than HC3 (or any other built-in source). The downstream calibration
scripts (`scripts/calibrate_detectors.py`, `scripts/train_fusion.py`,
`scripts/calibrate_conformal.py`) all accept the JSONL produced here
via their `--dataset` flag.

Label column accepts 0/1 integers, "human"/"llm" strings, or
"chatgpt"/"gpt"/"ai" (treated as LLM).

Usage:
    python scripts/import_dataset.py \\
        --in   data/raw/my_corpus.csv \\
        --out  data/my_corpus.jsonl \\
        --text-col body \\
        --label-col is_llm \\
        --source mycorpus_v1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cairn.evaluation.dataset import EvalSample, save_dataset


_LLM_LABELS = {"1", "llm", "ai", "gpt", "chatgpt", "machine", "synthetic", "true"}
_HUMAN_LABELS = {"0", "human", "person", "real", "false"}


def _normalise_label(raw: str) -> int:
    v = raw.strip().lower()
    if v in _LLM_LABELS:
        return 1
    if v in _HUMAN_LABELS:
        return 0
    raise ValueError(
        f"unrecognised label {raw!r}; expected 0/1 or human/llm "
        f"(also accepts: {sorted(_LLM_LABELS | _HUMAN_LABELS)})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="src", required=True, type=Path,
                    help="input CSV or TSV; delimiter inferred from extension")
    ap.add_argument("--out", required=True, type=Path,
                    help="output JSONL (overwritten)")
    ap.add_argument("--text-col", default="text",
                    help="column name holding the text (default: 'text')")
    ap.add_argument("--label-col", default="label",
                    help="column name holding the label (default: 'label')")
    ap.add_argument("--source", default=None,
                    help="optional source tag stored on every sample")
    ap.add_argument("--id-prefix", default="sample",
                    help="prefix for generated row IDs (default: 'sample')")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="optional cap per class to keep balanced subsets manageable")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"input not found: {args.src}")

    delimiter = "\t" if args.src.suffix.lower() in {".tsv", ".tab"} else ","
    samples: list[EvalSample] = []
    counts = {0: 0, 1: 0}
    skipped = 0

    with open(args.src, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise SystemExit(f"{args.src} has no header row")
        for col in (args.text_col, args.label_col):
            if col not in reader.fieldnames:
                raise SystemExit(
                    f"column {col!r} not in input header {reader.fieldnames!r}"
                )
        for i, row in enumerate(reader):
            text = (row.get(args.text_col) or "").strip()
            raw_label = row.get(args.label_col) or ""
            if not text:
                skipped += 1
                continue
            try:
                label = _normalise_label(raw_label)
            except ValueError as e:
                raise SystemExit(f"row {i + 1}: {e}")
            if args.max_per_class is not None and counts[label] >= args.max_per_class:
                continue
            samples.append(EvalSample(
                id=f"{args.id_prefix}_{i:06d}",
                text=text,
                label=label,
                tier="human" if label == 0 else "A0",
                source=args.source,
            ))
            counts[label] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(samples, args.out)
    print(
        f"wrote {len(samples)} samples to {args.out} "
        f"(human={counts[0]}, llm={counts[1]}, skipped_blank={skipped})"
    )


if __name__ == "__main__":
    main()

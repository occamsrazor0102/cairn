"""Prepare the private July 10 adaptive challenge as a labelled JSONL slice."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

from cairn.evaluation.dataset import EvalSample, save_dataset
from cairn.feature_cache import text_hash
from cairn.normalizer import normalize_text


DEFAULT_TRAINING_PATHS = (
    Path("data/prompts_v2_train.jsonl"),
    Path("data/prompts_only_calib.jsonl"),
)
_TASK_RE = re.compile(r"^\s*(\d+)")


def _task_id(value: object) -> int:
    if isinstance(value, int):
        return value
    match = _TASK_RE.match(str(value))
    if match is None:
        raise ValueError(f"cannot parse task number {value!r}")
    return int(match.group(1))


def _training_hashes(paths: Sequence[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "text" not in row:
                    raise ValueError(f"{path}:{line_number}: text is required")
                hashes.add(text_hash(normalize_text(str(row["text"]))))
    return hashes


def prepare_challenge(
    rows_path: Path,
    truth_path: Path,
    out_path: Path,
    *,
    training_paths: Sequence[Path] = DEFAULT_TRAINING_PATHS,
) -> list[EvalSample]:
    rows_payload = json.loads(Path(rows_path).read_text(encoding="utf-8"))
    truth_payload = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    source_rows = rows_payload.get("rows")
    origins = truth_payload.get("actual_origins")
    if not isinstance(source_rows, list) or not isinstance(origins, list):
        raise ValueError("challenge sources require rows and actual_origins lists")

    texts_by_task: dict[int, str] = {}
    for row in source_rows:
        if not isinstance(row, dict):
            raise ValueError("challenge source row must be an object")
        task = _task_id(row.get("task_no"))
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"task {task}: non-empty text is required")
        if task in texts_by_task:
            raise ValueError(f"duplicate challenge task {task}")
        texts_by_task[task] = text

    labels_by_task: dict[int, int] = {}
    for row in origins:
        if not isinstance(row, dict):
            raise ValueError("actual-origin row must be an object")
        task = _task_id(row.get("task_id", row.get("task_no")))
        label = int(row.get("binary_label"))
        if label not in (0, 1):
            raise ValueError(f"task {task}: binary_label must be 0 or 1")
        if task in labels_by_task:
            raise ValueError(f"duplicate actual-origin task {task}")
        labels_by_task[task] = label

    locked_ids = list(range(41, 55))
    if any(task not in texts_by_task for task in locked_ids):
        raise ValueError("challenge source must contain every task 41-54")
    if any(task not in labels_by_task for task in locked_ids):
        raise ValueError("challenge truth must contain every task 41-54")
    labels = [labels_by_task[task] for task in locked_ids]
    if labels.count(1) != 9 or labels.count(0) != 5:
        raise ValueError("challenge must contain exactly 9 positive and 5 Human labels")

    samples = [
        EvalSample(
            id=f"july10_{task}",
            text=texts_by_task[task],
            label=labels_by_task[task],
            tier="adaptive",
            source="july10_challenge",
            source_id=str(task),
        )
        for task in locked_ids
    ]
    challenge_hashes = {
        text_hash(normalize_text(sample.text)) for sample in samples
    }
    if len(challenge_hashes) != 14:
        raise ValueError("challenge contains duplicate normalized text")
    overlaps = challenge_hashes & _training_hashes(training_paths)
    if overlaps:
        raise ValueError(
            f"challenge overlaps training data at {len(overlaps)} normalized hashes"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(samples, out_path)
    return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_challenge(args.rows, args.truth, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import json
from pathlib import Path

import pytest

from cairn.feature_cache import text_hash
from cairn.normalizer import normalize_text
from scripts.prepare_july10_challenge import prepare_challenge


NAMED_TRAINING_PATHS = (
    Path("data/prompts_v2_train.jsonl"),
    Path("data/prompts_only_calib.jsonl"),
)
ACTUAL_ROWS = Path("outputs/cairn_blinded_rerun_2026-07-10/source_rows.json")
ACTUAL_TRUTH = Path(
    "outputs/cairn_blinded_rerun_2026-07-10/ground_truth_analysis.json"
)
_requires_named_training = pytest.mark.skipif(
    not all(path.is_file() for path in NAMED_TRAINING_PATHS),
    reason="git-ignored training corpora under data/ are absent from this checkout",
)


def _challenge_sources(tmp_path: Path) -> tuple[Path, Path]:
    rows = {
        "source": "private.xlsx",
        "sheet": "Sheet1",
        "headers": ["Task No", "Text"],
        "rows": [
            {
                "source_row": task_id - 39,
                "task_no": f"{task_id} - blinded",
                "text": f"Private challenge text number {task_id}",
            }
            for task_id in range(41, 56)
        ],
    }
    labels = [1] * 9 + [0] * 5 + [1]
    truth = {
        "analysis_metadata": {
            "evaluated_task_ids": list(range(41, 55)),
            "excluded_task_ids": [55],
        },
        "actual_origins": [
            {
                "task_id": task_id,
                "task_no": f"{task_id} - blinded",
                "actual_origin": "AI-Edited" if label else "Human",
                "binary_label": label,
            }
            for task_id, label in zip(range(41, 56), labels)
        ],
    }
    rows_path = tmp_path / "source_rows.json"
    truth_path = tmp_path / "ground_truth_analysis.json"
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    return rows_path, truth_path


@_requires_named_training
def test_prepare_challenge_emits_exact_locked_slice_and_excludes_training(tmp_path):
    rows_path, truth_path = _challenge_sources(tmp_path)
    out = tmp_path / "july10_labelled.jsonl"

    prepared = prepare_challenge(
        rows_path,
        truth_path,
        out,
        training_paths=NAMED_TRAINING_PATHS,
    )

    assert len(prepared) == 14
    assert [int(sample.source_id) for sample in prepared] == list(range(41, 55))
    assert sum(sample.label == 1 for sample in prepared) == 9
    assert sum(sample.label == 0 for sample in prepared) == 5
    assert all(sample.source_id != "55" for sample in prepared)
    output_rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["source_id"] for row in output_rows] == [str(i) for i in range(41, 55)]

    training_hashes = {
        text_hash(normalize_text(row["text"]))
        for path in NAMED_TRAINING_PATHS
        for row in (json.loads(line) for line in path.read_text().splitlines())
    }
    assert not {
        text_hash(normalize_text(sample.text)) for sample in prepared
    } & training_hashes


@_requires_named_training
def test_prepare_challenge_rejects_normalized_training_overlap(tmp_path):
    rows_path, truth_path = _challenge_sources(tmp_path)
    named_training_row = json.loads(
        NAMED_TRAINING_PATHS[0].read_text(encoding="utf-8").splitlines()[0]
    )
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    rows["rows"][0]["text"] = f"  {named_training_row['text']}  "
    rows_path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="overlaps training data"):
        prepare_challenge(
            rows_path,
            truth_path,
            tmp_path / "out.jsonl",
            training_paths=NAMED_TRAINING_PATHS,
        )


def test_prepare_challenge_rejects_wrong_locked_composition(tmp_path):
    rows_path, truth_path = _challenge_sources(tmp_path)
    truth = json.loads(truth_path.read_text())
    truth["actual_origins"][9]["binary_label"] = 1
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    with pytest.raises(ValueError, match="9 positive and 5 Human"):
        prepare_challenge(
            rows_path,
            truth_path,
            tmp_path / "out.jsonl",
            training_paths=NAMED_TRAINING_PATHS,
        )


@pytest.mark.skipif(
    not (ACTUAL_ROWS.is_file() and ACTUAL_TRUTH.is_file()),
    reason="private July 10 source/truth artifacts are not present in this checkout",
)
def test_actual_private_challenge_has_locked_14_row_composition(tmp_path):
    prepared = prepare_challenge(
        ACTUAL_ROWS,
        ACTUAL_TRUTH,
        tmp_path / "actual-july10.jsonl",
        training_paths=NAMED_TRAINING_PATHS,
    )

    assert [int(sample.source_id) for sample in prepared] == list(range(41, 55))
    assert sum(sample.label == 1 for sample in prepared) == 9
    assert sum(sample.label == 0 for sample in prepared) == 5

"""Export confirmed-label rows from `feedback_log` as EvalSample JSONL.

Bridges CAIRN's reviewer feedback loop into the training pipeline. The GUI
writes confirmed labels to `feedback_log` via
`cairn/history.py::HistoryStore.record_feedback`; this script joins those
rows back to the original `submissions` and emits a JSONL ready for
`cairn.evaluation.dataset.load_dataset` (and therefore for
`scripts/calibrate_detectors.py` / `scripts/train_fusion.py`).

Privacy invariant
-----------------
The default `gui.history.retain_text=false` privacy posture means
`submissions.text` may be NULL. We honour that:

- Rows where the original text is unavailable are exported with
  ``text: null`` so they are recorded for audit / diagnosis but ARE
  NOT usable for retraining (the calibration scripts require text).
- A stderr warning is emitted whenever any text-null row is written
  so the operator notices the privacy gap.
- The ``--include-text-only`` flag filters those rows out entirely
  for callers wiring directly into a retraining job.

To enable feedback as training data, set ``gui.history.retain_text:
true`` in the config BEFORE the feedback was recorded. Toggling it
later does not retroactively recover dropped text.

Output schema
-------------
One JSON object per line, EvalSample-compatible::

    {
      "id": "<submission_id>",
      "text": "<original text or null>",
      "label": 0 | 1,                  # 0=human, 1=LLM
      "tier": "feedback",
      "attack_name": null,
      "attack_category": null,
      "metadata": {
        "source": "feedback_log",
        "recorded_at": "<feedback timestamp ISO>",
        "profile": "<profile name>",
        "reviewer_notes": "<notes>"
      }
    }

`load_dataset` ignores unknown keys, so the ``metadata`` field is
preserved for audit but does not break round-trip; the canonical
EvalSample fields all round-trip cleanly.

Usage
-----

::

    python scripts/export_feedback_corpus.py \\
        --history-path ~/.cairn/history.sqlite3 \\
        --out data/feedback_2026-05.jsonl

    # Filter to one calendar month and one profile:
    python scripts/export_feedback_corpus.py \\
        --history-path ~/.cairn/history.sqlite3 \\
        --out data/feedback_2026-05.jsonl \\
        --since 2026-05-01 \\
        --until 2026-06-01 \\
        --profile master_minus_arxiv

    # Drop privacy-redacted rows (training-ready output):
    python scripts/export_feedback_corpus.py \\
        --history-path ~/.cairn/history.sqlite3 \\
        --out data/feedback_2026-05.jsonl \\
        --include-text-only
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable


def _row_iter(
    db_path: Path,
    *,
    since: str | None,
    until: str | None,
    profile: str | None,
) -> Iterable[sqlite3.Row]:
    """Join feedback_log to submissions; yield rows respecting filters.

    The filter compares against `feedback_log.recorded_at` (when the
    label was confirmed), not the submission's recorded_at — the
    feedback timestamp is the meaningful "when did we learn this"
    signal for monthly retrain windows.
    """
    where: list[str] = []
    params: list[Any] = []
    if since:
        where.append("f.recorded_at >= ?")
        params.append(since)
    if until:
        where.append("f.recorded_at < ?")
        params.append(until)
    if profile:
        where.append("s.profile = ?")
        params.append(profile)
    clause = ("WHERE " + " AND ".join(where)) if where else ""

    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            f"""
            SELECT
              s.submission_id  AS submission_id,
              s.text           AS text,
              s.text_hash      AS text_hash,
              s.profile        AS profile,
              f.confirmed_label AS confirmed_label,
              f.reviewer_notes  AS reviewer_notes,
              f.recorded_at     AS recorded_at
            FROM feedback_log f
            JOIN submissions s ON f.submission_id = s.submission_id
            {clause}
            ORDER BY f.recorded_at ASC
            """,
            tuple(params),
        )
        yield from cur.fetchall()
    finally:
        conn.close()


def _to_evalsample_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a joined feedback+submission row to EvalSample-shaped dict."""
    text = row["text"]  # may be None when retain_text was off
    return {
        "id": row["submission_id"],
        "text": text,  # None -> JSON null, honoured by load_dataset
        "label": int(row["confirmed_label"]),
        "tier": "feedback",
        "attack_name": None,
        "attack_category": None,
        "metadata": {
            "source": "feedback_log",
            "recorded_at": row["recorded_at"],
            "profile": row["profile"],
            "reviewer_notes": row["reviewer_notes"],
        },
    }


def export_feedback(
    *,
    history_path: Path,
    out_path: Path,
    since: str | None = None,
    until: str | None = None,
    profile: str | None = None,
    include_text_only: bool = False,
    stderr=None,
    stdout=None,
) -> dict[str, Any]:
    """Run the export. Returns a stats dict.

    ``stderr`` / ``stdout`` are injectable for tests; default to the
    real streams when None.
    """
    if stderr is None:
        stderr = sys.stderr
    if stdout is None:
        stdout = sys.stdout

    history_path = Path(history_path)
    out_path = Path(out_path)
    if not history_path.exists():
        raise SystemExit(f"history db not found: {history_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_label_human = 0
    n_label_llm = 0
    n_text_null = 0
    n_dropped_text_null = 0
    earliest: str | None = None
    latest: str | None = None

    with open(out_path, "w", encoding="utf-8") as fout:
        for row in _row_iter(
            history_path, since=since, until=until, profile=profile
        ):
            obj = _to_evalsample_dict(row)
            text_is_null = obj["text"] is None

            if text_is_null and include_text_only:
                n_dropped_text_null += 1
                continue

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_total += 1
            if text_is_null:
                n_text_null += 1
            label = obj["label"]
            if label == 0:
                n_label_human += 1
            elif label == 1:
                n_label_llm += 1

            ts = row["recorded_at"]
            if ts:
                if earliest is None or ts < earliest:
                    earliest = ts
                if latest is None or ts > latest:
                    latest = ts

    if n_text_null > 0:
        # Privacy invariant: surface text-null rows so operators know
        # the gap. These rows are not usable for retraining.
        print(
            f"WARNING: {n_text_null} rows have text=null because "
            f"retain_text was disabled at write time. These rows "
            f"cannot be used for retraining.",
            file=stderr,
        )

    print(f"exported {n_total} rows to {out_path}", file=stdout)
    print(
        f"  by label: human={n_label_human} llm={n_label_llm}",
        file=stdout,
    )
    print(f"  text-null rows: {n_text_null}", file=stdout)
    if include_text_only and n_dropped_text_null:
        print(
            f"  dropped (text-null, --include-text-only): "
            f"{n_dropped_text_null}",
            file=stdout,
        )
    if earliest and latest:
        print(f"  date range: {earliest} -> {latest}", file=stdout)
    elif n_total == 0:
        print("  (no feedback rows matched the filters)", file=stdout)

    return {
        "total": n_total,
        "human": n_label_human,
        "llm": n_label_llm,
        "text_null": n_text_null,
        "dropped_text_null": n_dropped_text_null,
        "earliest": earliest,
        "latest": latest,
        "out_path": str(out_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--history-path",
        required=True,
        type=Path,
        help="Path to the CAIRN history sqlite database "
        "(typically ~/.cairn/history.sqlite3).",
    )
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output JSONL path "
        "(convention: data/feedback_YYYY-MM.jsonl).",
    )
    ap.add_argument(
        "--since",
        default=None,
        help="Filter to feedback recorded_at >= this ISO date "
        "(inclusive). Example: 2026-05-01.",
    )
    ap.add_argument(
        "--until",
        default=None,
        help="Filter to feedback recorded_at < this ISO date "
        "(exclusive). Example: 2026-06-01.",
    )
    ap.add_argument(
        "--profile",
        default=None,
        help="Filter to submissions analysed under this profile "
        "(e.g. master_minus_arxiv).",
    )
    ap.add_argument(
        "--include-text-only",
        action="store_true",
        help="Drop rows whose original text is null (retain_text was "
        "off at write time). Use this when feeding the output "
        "directly into retraining.",
    )
    args = ap.parse_args(argv)

    export_feedback(
        history_path=args.history_path,
        out_path=args.out,
        since=args.since,
        until=args.until,
        profile=args.profile,
        include_text_only=args.include_text_only,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

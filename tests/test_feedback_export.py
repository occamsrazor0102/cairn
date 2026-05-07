"""Tests for `scripts/export_feedback_corpus.py`.

Covers the round-trip through `cairn.evaluation.dataset.load_dataset`,
the privacy invariant for text-null rows, the date / profile filters,
and the empty-store edge case.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

# Ensure the script under test is importable as a module.
_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import export_feedback_corpus  # noqa: E402  (path-augmented)

from cairn.history import HistoryStore  # noqa: E402
from cairn.evaluation.dataset import load_dataset  # noqa: E402


def _report(det="AMBER", p=0.62, sid=""):
    return {
        "submission_id": sid,
        "timestamp": "2026-04-22T12:00:00Z",
        "determination": det,
        "p_llm": p,
        "confidence_interval": [p - 0.1, p + 0.1],
        "prediction_set": [det],
        "reason": "test",
        "top_features": [],
        "detectors_run": ["preamble"],
        "cascade_phases": [1],
        "mixed_report": None,
        "layer_results": [],
        "feature_contributions": {},
        "override_applied": False,
    }


def _stamp_recorded_at(db_path: Path, submission_id: str, iso_ts: str) -> None:
    """Force `feedback_log.recorded_at` for a row so date filters are deterministic."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "UPDATE feedback_log SET recorded_at=? WHERE submission_id=?",
            (iso_ts, submission_id),
        )
        conn.commit()
    finally:
        conn.close()


def _make_store(tmp_path: Path, *, retain_text: bool = True) -> HistoryStore:
    return HistoryStore(tmp_path / "history.sqlite3", retain_text=retain_text)


# --------------------------------------------------------------------------
# 1. Round-trip through EvalSample schema
# --------------------------------------------------------------------------

def test_export_round_trips_through_evalsample_schema(tmp_path):
    store = _make_store(tmp_path, retain_text=True)
    sid = store.record(
        _report(sid="rt-1"),
        source="analyze",
        text="A synthetic body of LLM-looking prose for round-trip testing.",
        profile="master_minus_arxiv",
    )
    store.record_feedback(sid, confirmed_label=1, reviewer_notes="clear LLM")

    out_path = tmp_path / "feedback.jsonl"
    stats = export_feedback_corpus.export_feedback(
        history_path=tmp_path / "history.sqlite3",
        out_path=out_path,
        stderr=io.StringIO(),
        stdout=io.StringIO(),
    )

    assert stats["total"] == 1
    assert stats["llm"] == 1
    assert stats["human"] == 0
    assert stats["text_null"] == 0

    samples = load_dataset(out_path)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.id == "rt-1"
    assert sample.label == 1
    assert sample.tier == "feedback"
    assert sample.text == (
        "A synthetic body of LLM-looking prose for round-trip testing."
    )

    # Sanity-check the metadata sidecar dict made it into the JSONL.
    raw = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert raw["metadata"]["source"] == "feedback_log"
    assert raw["metadata"]["profile"] == "master_minus_arxiv"
    assert raw["metadata"]["reviewer_notes"] == "clear LLM"
    assert raw["metadata"]["recorded_at"]


# --------------------------------------------------------------------------
# 2. Privacy invariant: text=null + stderr warning
# --------------------------------------------------------------------------

def test_export_handles_text_null_rows(tmp_path):
    store = _make_store(tmp_path, retain_text=False)  # privacy default
    sid = store.record(
        _report(sid="redacted-1"),
        source="analyze",
        text="this text will not be persisted",
        profile="master_minus_arxiv",
    )
    store.record_feedback(sid, confirmed_label=0)

    out_path = tmp_path / "feedback.jsonl"
    stderr_buf = io.StringIO()
    stats = export_feedback_corpus.export_feedback(
        history_path=tmp_path / "history.sqlite3",
        out_path=out_path,
        stderr=stderr_buf,
        stdout=io.StringIO(),
    )

    assert stats["total"] == 1
    assert stats["text_null"] == 1
    assert stats["dropped_text_null"] == 0

    raw = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert raw["text"] is None
    assert raw["id"] == "redacted-1"
    assert raw["label"] == 0

    err = stderr_buf.getvalue()
    assert "WARNING" in err
    assert "text=null" in err
    assert "retain_text was disabled" in err
    assert "cannot be used for retraining" in err


# --------------------------------------------------------------------------
# 3. --since / --until filter
# --------------------------------------------------------------------------

def test_since_until_filter(tmp_path):
    store = _make_store(tmp_path, retain_text=True)
    db_path = tmp_path / "history.sqlite3"

    timestamps = {
        "early": "2026-04-01T00:00:00Z",
        "middle": "2026-05-01T00:00:00Z",
        "late": "2026-06-01T00:00:00Z",
    }
    for tag, iso in timestamps.items():
        sid = store.record(
            _report(sid=f"sid-{tag}"),
            source="analyze",
            text=f"text for {tag}",
            profile="master_minus_arxiv",
        )
        store.record_feedback(sid, confirmed_label=1)
        _stamp_recorded_at(db_path, sid, iso)

    out_path = tmp_path / "feedback.jsonl"
    stats = export_feedback_corpus.export_feedback(
        history_path=db_path,
        out_path=out_path,
        since="2026-05-01",
        stderr=io.StringIO(),
        stdout=io.StringIO(),
    )

    assert stats["total"] == 2  # middle + late
    samples = load_dataset(out_path)
    ids = {s.id for s in samples}
    assert ids == {"sid-middle", "sid-late"}

    # Bounded window: middle only.
    out_path2 = tmp_path / "feedback_bounded.jsonl"
    stats2 = export_feedback_corpus.export_feedback(
        history_path=db_path,
        out_path=out_path2,
        since="2026-05-01",
        until="2026-06-01",
        stderr=io.StringIO(),
        stdout=io.StringIO(),
    )
    assert stats2["total"] == 1
    samples2 = load_dataset(out_path2)
    assert {s.id for s in samples2} == {"sid-middle"}


# --------------------------------------------------------------------------
# 4. Profile filter
# --------------------------------------------------------------------------

def test_profile_filter(tmp_path):
    store = _make_store(tmp_path, retain_text=True)

    sid_a = store.record(
        _report(sid="prof-a"),
        source="analyze",
        text="alpha body",
        profile="master_minus_arxiv",
    )
    store.record_feedback(sid_a, confirmed_label=1)

    sid_b = store.record(
        _report(sid="prof-b"),
        source="analyze",
        text="beta body",
        profile="prompts_only",
    )
    store.record_feedback(sid_b, confirmed_label=0)

    out_path = tmp_path / "feedback.jsonl"
    stats = export_feedback_corpus.export_feedback(
        history_path=tmp_path / "history.sqlite3",
        out_path=out_path,
        profile="master_minus_arxiv",
        stderr=io.StringIO(),
        stdout=io.StringIO(),
    )

    assert stats["total"] == 1
    samples = load_dataset(out_path)
    assert len(samples) == 1
    assert samples[0].id == "prof-a"


# --------------------------------------------------------------------------
# 5. --include-text-only drops null-text rows
# --------------------------------------------------------------------------

def test_include_text_only_flag(tmp_path):
    # Mix retained + unretained submissions in the same DB by writing
    # against the same file via two HistoryStore instances with
    # different retain_text. The DB rows then differ in whether
    # `text` is NULL.
    db_path = tmp_path / "history.sqlite3"

    keep = HistoryStore(db_path, retain_text=True)
    sid_kept = keep.record(
        _report(sid="kept-1"),
        source="analyze",
        text="this text persists",
        profile="master_minus_arxiv",
    )
    keep.record_feedback(sid_kept, confirmed_label=1)

    drop = HistoryStore(db_path, retain_text=False)
    sid_dropped = drop.record(
        _report(sid="dropped-1"),
        source="analyze",
        text="this text is null in db",
        profile="master_minus_arxiv",
    )
    drop.record_feedback(sid_dropped, confirmed_label=0)

    # Without the flag — both rows present, one with text=null.
    out_default = tmp_path / "feedback_default.jsonl"
    stats_default = export_feedback_corpus.export_feedback(
        history_path=db_path,
        out_path=out_default,
        stderr=io.StringIO(),
        stdout=io.StringIO(),
    )
    assert stats_default["total"] == 2
    assert stats_default["text_null"] == 1

    # With the flag — null row dropped entirely.
    out_filtered = tmp_path / "feedback_filtered.jsonl"
    stats_filtered = export_feedback_corpus.export_feedback(
        history_path=db_path,
        out_path=out_filtered,
        include_text_only=True,
        stderr=io.StringIO(),
        stdout=io.StringIO(),
    )
    assert stats_filtered["total"] == 1
    assert stats_filtered["text_null"] == 0
    assert stats_filtered["dropped_text_null"] == 1
    samples = load_dataset(out_filtered)
    assert {s.id for s in samples} == {"kept-1"}


# --------------------------------------------------------------------------
# 6. Empty feedback log -> empty output, exit 0
# --------------------------------------------------------------------------

def test_empty_feedback_log_produces_empty_output(tmp_path):
    # No feedback rows at all (we don't even need a submission).
    HistoryStore(tmp_path / "history.sqlite3", retain_text=True)

    out_path = tmp_path / "feedback.jsonl"
    stats = export_feedback_corpus.export_feedback(
        history_path=tmp_path / "history.sqlite3",
        out_path=out_path,
        stderr=io.StringIO(),
        stdout=io.StringIO(),
    )

    assert stats["total"] == 0
    assert stats["text_null"] == 0
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == ""

    # And via the CLI `main` entry point, exit 0 on empty.
    rc = export_feedback_corpus.main([
        "--history-path", str(tmp_path / "history.sqlite3"),
        "--out", str(tmp_path / "via_main.jsonl"),
    ])
    assert rc == 0
    assert (tmp_path / "via_main.jsonl").read_text(encoding="utf-8") == ""


# --------------------------------------------------------------------------
# 7. Bonus: missing history db raises cleanly
# --------------------------------------------------------------------------

def test_missing_history_db_raises(tmp_path):
    with pytest.raises(SystemExit):
        export_feedback_corpus.export_feedback(
            history_path=tmp_path / "does-not-exist.sqlite3",
            out_path=tmp_path / "out.jsonl",
            stderr=io.StringIO(),
            stdout=io.StringIO(),
        )

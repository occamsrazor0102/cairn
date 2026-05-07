"""SQLite-backed drift observation store.

Mirrors `cairn/history.py`: single connection per call, WAL mode,
best-effort writes that never fail the pipeline. The store persists
per-observation feature rows so the in-flight DriftMonitor window
survives sidecar restart.

Schema is one row per (observation, feature_name) pair. An observation
is reconstructed by grouping rows by observation_id; observations with
no numeric features still get a single sentinel row keyed off
feature_name='__obs__' so they can be retrieved.

Writes are best-effort: a failed DB write must not fail the pipeline RPC.
Callers should catch and log to stderr.
"""
from __future__ import annotations

import datetime
import random
import sqlite3
import sys
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS drift_observations (
  observation_id TEXT PRIMARY KEY,
  recorded_at    TEXT NOT NULL,
  feature_name   TEXT NOT NULL,
  feature_value  REAL NOT NULL,
  p_llm          REAL,
  determination  TEXT,
  reviewer_label INTEGER,
  profile        TEXT
);
CREATE INDEX IF NOT EXISTS idx_drift_recorded_at ON drift_observations(recorded_at);
CREATE INDEX IF NOT EXISTS idx_drift_feature ON drift_observations(feature_name);
"""

# Sentinel feature_name used when an observation has no numeric features but
# we still need a row in the table to make `recent(N)` see it. The dict-shape
# observation it reconstructs to has `features: {}`.
_NO_FEATURES_KEY = "__obs__"


def _now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
        + "Z"
    )


def _mint_observation_id() -> str:
    """Short, lexicographically-time-ordered id, mirroring history.mint_submission_id."""
    ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    rand = random.randint(0, 0xFFFFFFFF)
    return f"obs_{ms:011x}_{rand:08x}"


def _is_nan(x: float) -> bool:
    # NaN != NaN; avoid pulling math just for this.
    return x != x


def _to_float_or_none(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(v) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class DriftStore:
    """Thin SQLite wrapper for drift observations.

    Single connection per call, short-lived. Mirrors `HistoryStore` so the
    same operational discipline (WAL, best-effort) applies.
    """

    def __init__(self, db_path: Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialize()

    def _initialize(self) -> None:
        with self._init_lock, self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def record(self, observation: dict) -> None:
        """Insert one observation. Best-effort: catches sqlite errors.

        `observation` shape mirrors DriftMonitor's in-memory dict:
            {"p_llm": float, "determination": str,
             "features": {name: float, ...},
             "confirmed": int | None,
             "timestamp": str (iso),
             "profile": str | None}

        Each numeric feature becomes its own row sharing a base
        observation_id. If the observation has no numeric features, a
        single sentinel row is written so it remains retrievable.
        """
        recorded_at = str(observation.get("timestamp") or _now_iso())
        p_llm = observation.get("p_llm")
        determination = observation.get("determination")
        confirmed = observation.get("confirmed")
        profile = observation.get("profile")
        features = observation.get("features") or {}

        base_id = str(observation.get("observation_id") or _mint_observation_id())
        numeric_features = [
            (name, float(val))
            for name, val in features.items()
            if isinstance(val, (int, float)) and not _is_nan(float(val))
        ]
        rows: list[tuple] = []
        if numeric_features:
            for idx, (name, value) in enumerate(numeric_features):
                row_id = f"{base_id}::{idx:04x}"
                rows.append((
                    row_id, recorded_at, str(name), float(value),
                    _to_float_or_none(p_llm),
                    None if determination is None else str(determination),
                    _to_int_or_none(confirmed),
                    None if profile is None else str(profile),
                ))
        else:
            rows.append((
                base_id, recorded_at, _NO_FEATURES_KEY, 0.0,
                _to_float_or_none(p_llm),
                None if determination is None else str(determination),
                _to_int_or_none(confirmed),
                None if profile is None else str(profile),
            ))

        try:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO drift_observations (
                      observation_id, recorded_at, feature_name, feature_value,
                      p_llm, determination, reviewer_label, profile
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    rows,
                )
        except sqlite3.Error as e:
            # Best-effort discipline: a failed drift-store write must never
            # fail the pipeline. Surface to stderr so a full disk /
            # permission error is visible, then continue.
            sys.stderr.write(f"[drift_store] record failed: {e}\n")

    def recent(self, window_size: int) -> list[dict]:
        """Return the most-recent `window_size` observations, regrouped.

        Output ordering: newest first (DESC by recorded_at, then by base
        observation_id for ties). Each element is the observation dict
        shape DriftMonitor expects.
        """
        if window_size <= 0:
            return []
        try:
            with self._connect() as conn:
                # Group by base id (everything before "::") so a multi-feature
                # observation counts as one toward the window.
                base_rows = conn.execute(
                    """
                    SELECT
                      CASE
                        WHEN instr(observation_id, '::') > 0
                        THEN substr(observation_id, 1, instr(observation_id, '::') - 1)
                        ELSE observation_id
                      END AS base_id,
                      MAX(recorded_at) AS recorded_at
                    FROM drift_observations
                    GROUP BY base_id
                    ORDER BY recorded_at DESC, base_id DESC
                    LIMIT ?
                    """,
                    (int(window_size),),
                ).fetchall()
                if not base_rows:
                    return []
                base_ids = [r["base_id"] for r in base_rows]
                # Chunked retrieval: a literal OR/IN clause with 1000+ ids
                # blows past SQLite's expression-tree depth (default 1000).
                # Pull rows in fixed-size chunks and concatenate. Each chunk
                # uses a single IN list so the planner can use the PK index.
                feature_rows = []
                # 200 ids per chunk -> 200 IN literals; comfortably under
                # the SQLITE_LIMIT_EXPR_DEPTH default of 1000.
                CHUNK = 200
                for start in range(0, len(base_ids), CHUNK):
                    chunk_ids = base_ids[start:start + CHUNK]
                    in_marks = ",".join("?" * len(chunk_ids))
                    like_clauses = " OR ".join(
                        ["observation_id LIKE ?"] * len(chunk_ids)
                    )
                    like_params = [f"{bid}::%" for bid in chunk_ids]
                    rows = conn.execute(
                        f"""
                        SELECT observation_id, recorded_at, feature_name, feature_value,
                               p_llm, determination, reviewer_label, profile
                        FROM drift_observations
                        WHERE observation_id IN ({in_marks})
                           OR {like_clauses}
                        """,
                        (*chunk_ids, *like_params),
                    ).fetchall()
                    feature_rows.extend(rows)
        except sqlite3.Error as e:
            sys.stderr.write(f"[drift_store] recent failed: {e}\n")
            return []

        grouped: dict[str, dict] = {}
        for row in feature_rows:
            obs_id = row["observation_id"]
            base_id = obs_id.split("::", 1)[0]
            entry = grouped.setdefault(base_id, {
                "observation_id": base_id,
                "p_llm": row["p_llm"],
                "determination": row["determination"],
                "features": {},
                "confirmed": row["reviewer_label"],
                "profile": row["profile"],
                "timestamp": row["recorded_at"],
            })
            # All sibling rows for one observation share scalar values; the
            # latest recorded_at wins on ties (rows that were re-recorded).
            if row["recorded_at"] > entry["timestamp"]:
                entry["timestamp"] = row["recorded_at"]
                entry["p_llm"] = row["p_llm"]
                entry["determination"] = row["determination"]
                entry["confirmed"] = row["reviewer_label"]
                entry["profile"] = row["profile"]
            if row["feature_name"] != _NO_FEATURES_KEY:
                entry["features"][row["feature_name"]] = float(row["feature_value"])

        # Newest first, in the same order as base_rows.
        return [grouped[bid] for bid in base_ids if bid in grouped]

    def purge_older_than(self, cutoff_iso: str) -> int:
        """Delete observations recorded before `cutoff_iso`. Returns count."""
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM drift_observations WHERE recorded_at < ?",
                    (str(cutoff_iso),),
                )
                return int(cur.rowcount or 0)
        except sqlite3.Error as e:
            sys.stderr.write(f"[drift_store] purge failed: {e}\n")
            return 0


__all__ = ["DriftStore", "SCHEMA"]

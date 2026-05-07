"""Direct unit tests for the sqlite-backed drift observation store."""
from __future__ import annotations

import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from cairn.monitoring.store import DriftStore


def _iso_at(year: int, month: int, day: int, *, hour: int = 0, minute: int = 0,
            second: int = 0, micro: int = 0) -> str:
    return (
        datetime.datetime(year, month, day, hour, minute, second, micro,
                          tzinfo=datetime.timezone.utc)
        .replace(tzinfo=None)
        .isoformat()
        + "Z"
    )


def _obs(i: int, ts: str | None = None) -> dict:
    """Build a synthetic observation with deterministic feature values."""
    return {
        "p_llm": 0.40 + (i % 50) * 0.01,
        "determination": "AMBER" if (i % 2 == 0) else "GREEN",
        "features": {"x": 0.10 + (i % 10) * 0.01, "y": 0.50 + (i % 7) * 0.02},
        "confirmed": (i % 2),
        "timestamp": ts or _iso_at(2026, 1, 1, second=i % 60, micro=i),
    }


def test_record_and_retrieve(tmp_path):
    """Record N observations; retrieve the same N via recent(N)."""
    store = DriftStore(tmp_path / "drift.sqlite3")
    n = 25
    for i in range(n):
        store.record(_obs(i))
    out = store.recent(n)
    assert len(out) == n
    # Newest first by timestamp; our observations have monotonically
    # increasing microseconds within the same second.
    timestamps = [o["timestamp"] for o in out]
    assert timestamps == sorted(timestamps, reverse=True), (
        "recent() must return newest-first"
    )
    # Each observation has both features hydrated.
    for o in out:
        assert set(o["features"].keys()) == {"x", "y"}
        assert isinstance(o["p_llm"], float)
        assert o["determination"] in {"AMBER", "GREEN"}


def test_window_size_trim(tmp_path):
    """Record 100 observations, recent(50) returns the last 50."""
    store = DriftStore(tmp_path / "drift.sqlite3")
    for i in range(100):
        store.record(_obs(i))
    out = store.recent(50)
    assert len(out) == 50
    timestamps = [o["timestamp"] for o in out]
    assert timestamps == sorted(timestamps, reverse=True)
    # And recent(0) returns []; recent(>N) returns all.
    assert store.recent(0) == []
    assert len(store.recent(500)) == 100


def test_purge_older_than(tmp_path):
    """Insert observations across 3 days; purge older than yesterday."""
    store = DriftStore(tmp_path / "drift.sqlite3")
    day_old = [_obs(i, ts=_iso_at(2026, 5, 5, second=i)) for i in range(10)]
    yesterday = [_obs(i + 100, ts=_iso_at(2026, 5, 6, second=i)) for i in range(7)]
    today = [_obs(i + 200, ts=_iso_at(2026, 5, 7, second=i)) for i in range(5)]
    for o in day_old + yesterday + today:
        store.record(o)
    cutoff = _iso_at(2026, 5, 6)  # everything before 2026-05-06 00:00:00Z
    deleted = store.purge_older_than(cutoff)
    # Each observation produces 2 feature rows, so 10 obs => 20 rows deleted.
    assert deleted == len(day_old) * 2
    remaining = store.recent(1000)
    assert len(remaining) == len(yesterday) + len(today)
    for o in remaining:
        assert o["timestamp"] >= cutoff


def test_concurrent_writes_do_not_corrupt(tmp_path):
    """10 threads each write 50 observations; the store stays internally consistent.

    Best-effort writes can drop under contention — assert no corruption
    via a clean recent() round-trip and a permissive count bound.
    """
    store = DriftStore(tmp_path / "drift.sqlite3")
    n_threads = 10
    per_thread = 50

    def _worker(start: int) -> int:
        local_count = 0
        for i in range(per_thread):
            store.record(_obs(start * per_thread + i))
            local_count += 1
        return local_count

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(_worker, t) for t in range(n_threads)]
        completed = sum(f.result() for f in as_completed(futures))
    assert completed == n_threads * per_thread

    out = store.recent(n_threads * per_thread + 50)
    # Best-effort: writes may drop under contention, but the rows that did
    # land must hydrate cleanly — no partial / null-feature corruption.
    assert 0 < len(out) <= n_threads * per_thread
    for o in out:
        assert set(o["features"].keys()) == {"x", "y"}
        assert isinstance(o["p_llm"], float)
        assert o["determination"] in {"AMBER", "GREEN"}

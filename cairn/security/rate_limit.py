"""Sliding-window rate limiter for CAIRN's HTTP transports.

Default-disabled, opt-in per the deployment guide. Two backends are
shipped:

* :class:`InMemoryRateLimiter` — a per-key :class:`collections.deque`
  of timestamps. Cheap, no I/O, lost on restart. Suitable for single-
  process deployments and the embedded GUI.
* :class:`SqliteRateLimiter` — a sqlite-backed counterpart that
  survives restarts and (with WAL) is safe to share across the
  ``cairn serve`` worker pool. Mirrors the connection-per-call pattern
  in :mod:`cairn.history`.

The factory :func:`make_rate_limiter` picks the backend from the
configured ``store`` field. ``CAIRN_RATE_LIMIT_STORE`` overrides the
config when set.

Privacy note
------------
Rate-limit keys carry caller identity. The HTTP/GUI middleware uses
``ip:<client_ip>`` for unauthenticated callers and
``token:<sha256-prefix>`` for token-authenticated callers. The IP form
is PII; operators who want to keep PII out of the limiter can set
``rate_limit.hash_keys: true``, which causes the limiter to hash any
key it receives via SHA-256 before storage. The hash is one-way, so
queries by raw IP still hit the same bucket on the same process but
cannot be reversed off-host.

Schema (sqlite backend)
-----------------------
``events(key TEXT NOT NULL, ts REAL NOT NULL)`` with index
``idx_rate_limit_events_key`` on ``key``. ``ts`` is a Unix epoch
float (``time.time()`` result), e.g. ``1746662400.123``; not an ISO
string. The history store uses ISO-8601 because it surfaces dates in
queries; here we only need numeric arithmetic, so a float is correct.
"""

from __future__ import annotations

import abc
import hashlib
import os
import sqlite3
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_RATE_LIMIT_DB_PATH = Path("data/cairn_rate_limit.sqlite3")


@dataclass(frozen=True)
class RateLimitConfig:
    """Resolved rate-limit configuration.

    Attributes
    ----------
    enabled:
        Master switch. ``False`` short-circuits the middleware before
        any limiter call.
    window_seconds:
        Sliding window in seconds. Defaults to 60.
    max_per_window:
        Maximum events allowed per key inside the window. Defaults to
        60 (i.e. 1 req/s sustained).
    store:
        Either ``"memory"`` or ``"sqlite"``.
    hash_keys:
        When ``True`` the limiter hashes raw keys via SHA-256 before
        using them, so PII (IPs) is not stored in plaintext. Defaults
        to ``False`` to keep debug output legible.
    """

    enabled: bool = False
    window_seconds: int = 60
    max_per_window: int = 60
    store: Literal["memory", "sqlite"] = "memory"
    hash_keys: bool = False


def rate_limit_config_from_dict(cfg: dict | None) -> RateLimitConfig:
    """Read the ``rate_limit`` section out of the config dict.

    Falls back to the dataclass defaults — i.e. disabled, in-memory.
    The ``CAIRN_RATE_LIMIT_STORE`` env var, when set to ``"memory"`` or
    ``"sqlite"``, overrides ``store``.
    """
    section = (cfg or {}).get("rate_limit") or {}
    enabled = bool(section.get("enabled", False))
    window_seconds = int(section.get("window_seconds", 60))
    max_per_window = int(section.get("max_per_window", 60))
    store_value = str(section.get("store", "memory")).lower()
    env_store = os.environ.get("CAIRN_RATE_LIMIT_STORE", "").strip().lower()
    if env_store in ("memory", "sqlite"):
        store_value = env_store
    if store_value not in ("memory", "sqlite"):
        store_value = "memory"
    hash_keys = bool(section.get("hash_keys", False))
    return RateLimitConfig(
        enabled=enabled,
        window_seconds=max(1, window_seconds),
        max_per_window=max(1, max_per_window),
        store=store_value,  # type: ignore[arg-type]
        hash_keys=hash_keys,
    )


def _maybe_hash(key: str, *, hash_keys: bool) -> str:
    """Return the storage form of ``key`` honouring ``hash_keys``.

    The hash is SHA-256 hex digest. We do not truncate so that
    collisions cannot widen accidentally.
    """
    if not hash_keys:
        return key
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class RateLimiter(abc.ABC):
    """Abstract base class for the limiter backends.

    Implementations must record an event under ``key`` and return
    ``True`` iff the *post-record* count is at or below the configured
    ``max_per_window``. Implementations are responsible for trimming
    older events out of the window.
    """

    def __init__(self, cfg: RateLimitConfig):
        self._cfg = cfg

    @property
    def config(self) -> RateLimitConfig:
        return self._cfg

    @abc.abstractmethod
    def check(self, key: str) -> bool:
        """Record an event under ``key``; return ``True`` if under cap."""


class InMemoryRateLimiter(RateLimiter):
    """Process-local sliding-window limiter.

    Each key gets its own ``deque[float]`` of timestamps. On every
    :meth:`check` we pop expired timestamps off the left and append a
    new one on the right; the deque length after that is the post-record
    count.
    """

    def __init__(self, cfg: RateLimitConfig):
        super().__init__(cfg)
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> bool:
        if not self._cfg.enabled:
            return True
        store_key = _maybe_hash(key, hash_keys=self._cfg.hash_keys)
        now = time.time()
        cutoff = now - self._cfg.window_seconds
        with self._lock:
            bucket = self._buckets.get(store_key)
            if bucket is None:
                bucket = deque()
                self._buckets[store_key] = bucket
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            bucket.append(now)
            return len(bucket) <= self._cfg.max_per_window

    def _bucket_size(self, key: str) -> int:
        """Test-only helper: report current bucket length."""
        store_key = _maybe_hash(key, hash_keys=self._cfg.hash_keys)
        with self._lock:
            bucket = self._buckets.get(store_key)
            return len(bucket) if bucket is not None else 0


class SqliteRateLimiter(RateLimiter):
    """SQLite-backed sliding-window limiter.

    Mirrors :mod:`cairn.history` discipline: connection-per-call,
    short-lived, WAL journal. Schema is created on first use.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS events (
        key TEXT NOT NULL,
        ts  REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_rate_limit_events_key ON events(key);
    """

    def __init__(self, cfg: RateLimitConfig, db_path: Path):
        super().__init__(cfg)
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = threading.Lock()
        self._initialize()

    def _initialize(self) -> None:
        with self._init_lock, self._connect() as conn:
            conn.executescript(self.SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def check(self, key: str) -> bool:
        if not self._cfg.enabled:
            return True
        store_key = _maybe_hash(key, hash_keys=self._cfg.hash_keys)
        now = time.time()
        cutoff = now - self._cfg.window_seconds
        try:
            with self._connect() as conn:
                # BEGIN IMMEDIATE acquires the writer lock up front so the
                # DELETE / INSERT / SELECT-COUNT execute as a single
                # transaction. Without this, two concurrent workers can
                # each DELETE expired rows, INSERT, then both SELECT a
                # count under the cap — letting bursts slip past the
                # configured limit. WAL alone serialises commits but does
                # NOT serialise this read-after-write within a connection.
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "DELETE FROM events WHERE key=? AND ts<?",
                    (store_key, cutoff),
                )
                conn.execute(
                    "INSERT INTO events (key, ts) VALUES (?, ?)",
                    (store_key, now),
                )
                count = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE key=? AND ts>=?",
                    (store_key, cutoff),
                ).fetchone()[0]
                conn.execute("COMMIT")
        except sqlite3.Error as e:
            # Storage failure: fail-CLOSED for a security gate. An
            # operator who enabled rate-limiting should see a 429-storm
            # over a sqlite outage rather than have the limiter silently
            # turn off. Log loudly so the failure is visible.
            sys.stderr.write(
                f"[rate_limit] sqlite check failed ({e!r}); "
                f"failing closed (denying request)\n"
            )
            return False
        return int(count) <= self._cfg.max_per_window


def make_rate_limiter(
    cfg: RateLimitConfig,
    db_path: Path | None = None,
) -> RateLimiter:
    """Factory: pick the backend from ``cfg.store``.

    Parameters
    ----------
    cfg:
        The resolved :class:`RateLimitConfig`.
    db_path:
        Path to the sqlite store. Ignored when ``cfg.store == "memory"``.
        Falls back to :data:`DEFAULT_RATE_LIMIT_DB_PATH` when ``cfg.store
        == "sqlite"`` and no path is supplied.
    """
    if cfg.store == "sqlite":
        return SqliteRateLimiter(cfg, db_path or DEFAULT_RATE_LIMIT_DB_PATH)
    return InMemoryRateLimiter(cfg)


def rate_limit_status_for_health(cfg: RateLimitConfig) -> dict | str:
    """Return the dict reported under ``/health::rate_limit``.

    When disabled returns the literal string ``"disabled"`` so the
    health check stays small. When enabled returns the active window
    + cap so an operator can verify the deployed value without
    reading config.
    """
    if not cfg.enabled:
        return "disabled"
    return {
        "store": cfg.store,
        "window_seconds": cfg.window_seconds,
        "max_per_window": cfg.max_per_window,
    }

"""Unit tests for ``cairn.security.rate_limit``.

Covers:

* ``rate_limit_config_from_dict`` defaults + env-var override.
* ``InMemoryRateLimiter`` under-cap, at-cap, over-cap, window expiry.
* ``SqliteRateLimiter`` same behaviours plus persistence across
  instances (the property the in-memory store cannot offer).
* ``make_rate_limiter`` factory dispatch.
"""

from __future__ import annotations

import time

import pytest

from cairn.security.rate_limit import (
    DEFAULT_RATE_LIMIT_DB_PATH,  # noqa: F401  re-exported for stability
    InMemoryRateLimiter,
    RateLimitConfig,
    SqliteRateLimiter,
    _maybe_hash,
    make_rate_limiter,
    rate_limit_config_from_dict,
    rate_limit_status_for_health,
)


# ---------------------------------------------------------------------------
# rate_limit_config_from_dict
# ---------------------------------------------------------------------------


def test_config_defaults_disabled(monkeypatch):
    monkeypatch.delenv("CAIRN_RATE_LIMIT_STORE", raising=False)
    cfg = rate_limit_config_from_dict({})
    assert cfg.enabled is False
    assert cfg.window_seconds == 60
    assert cfg.max_per_window == 60
    assert cfg.store == "memory"
    assert cfg.hash_keys is False


def test_config_reads_yaml_keys(monkeypatch):
    monkeypatch.delenv("CAIRN_RATE_LIMIT_STORE", raising=False)
    cfg = rate_limit_config_from_dict({
        "rate_limit": {
            "enabled": True,
            "window_seconds": 30,
            "max_per_window": 10,
            "store": "sqlite",
            "hash_keys": True,
        }
    })
    assert cfg.enabled is True
    assert cfg.window_seconds == 30
    assert cfg.max_per_window == 10
    assert cfg.store == "sqlite"
    assert cfg.hash_keys is True


def test_config_env_var_overrides_store(monkeypatch):
    """`CAIRN_RATE_LIMIT_STORE` overrides the YAML value at runtime."""
    monkeypatch.setenv("CAIRN_RATE_LIMIT_STORE", "sqlite")
    cfg = rate_limit_config_from_dict({"rate_limit": {"store": "memory"}})
    assert cfg.store == "sqlite"


def test_config_env_var_unknown_value_falls_back(monkeypatch):
    monkeypatch.setenv("CAIRN_RATE_LIMIT_STORE", "redis")  # not supported
    cfg = rate_limit_config_from_dict({"rate_limit": {"store": "memory"}})
    assert cfg.store == "memory"


def test_config_clamps_silly_window_and_cap(monkeypatch):
    monkeypatch.delenv("CAIRN_RATE_LIMIT_STORE", raising=False)
    cfg = rate_limit_config_from_dict({
        "rate_limit": {"window_seconds": 0, "max_per_window": -3}
    })
    assert cfg.window_seconds == 1
    assert cfg.max_per_window == 1


# ---------------------------------------------------------------------------
# _maybe_hash
# ---------------------------------------------------------------------------


def test_maybe_hash_off_passes_through():
    assert _maybe_hash("ip:1.2.3.4", hash_keys=False) == "ip:1.2.3.4"


def test_maybe_hash_on_returns_sha256_hex():
    out = _maybe_hash("ip:1.2.3.4", hash_keys=True)
    assert out != "ip:1.2.3.4"
    assert len(out) == 64
    int(out, 16)  # parses as hex


# ---------------------------------------------------------------------------
# InMemoryRateLimiter
# ---------------------------------------------------------------------------


def _enabled_cfg(**overrides) -> RateLimitConfig:
    base = {
        "enabled": True,
        "window_seconds": 60,
        "max_per_window": 3,
        "store": "memory",
        "hash_keys": False,
    }
    base.update(overrides)
    return RateLimitConfig(**base)


def test_memory_under_limit_returns_true():
    rl = InMemoryRateLimiter(_enabled_cfg(max_per_window=3))
    assert rl.check("ip:1.2.3.4") is True
    assert rl.check("ip:1.2.3.4") is True


def test_memory_at_limit_still_true():
    rl = InMemoryRateLimiter(_enabled_cfg(max_per_window=3))
    for _ in range(3):
        assert rl.check("ip:1.2.3.4") is True


def test_memory_over_limit_returns_false():
    rl = InMemoryRateLimiter(_enabled_cfg(max_per_window=2))
    assert rl.check("k") is True
    assert rl.check("k") is True
    assert rl.check("k") is False
    assert rl.check("k") is False


def test_memory_separate_keys_separate_buckets():
    rl = InMemoryRateLimiter(_enabled_cfg(max_per_window=1))
    assert rl.check("ip:a") is True
    assert rl.check("ip:b") is True  # different key, fresh bucket
    assert rl.check("ip:a") is False  # over cap on `a`


def test_memory_window_expiry_drops_old_events(monkeypatch):
    """Wind the clock forward past the window; old events must drop."""
    cfg = _enabled_cfg(max_per_window=2, window_seconds=10)
    rl = InMemoryRateLimiter(cfg)

    fake_now = [1_000.0]

    def fake_time():
        return fake_now[0]

    monkeypatch.setattr(time, "time", fake_time)

    assert rl.check("k") is True
    assert rl.check("k") is True
    assert rl.check("k") is False  # at cap

    fake_now[0] += 11.0  # past the 10 s window
    assert rl.check("k") is True  # bucket reset by window expiry


def test_memory_disabled_short_circuits_true():
    cfg = RateLimitConfig(enabled=False)
    rl = InMemoryRateLimiter(cfg)
    for _ in range(100):
        assert rl.check("anything") is True


def test_memory_hash_keys_changes_storage_form():
    """When hash_keys=True the limiter stores hashed keys, not raw IPs."""
    cfg = _enabled_cfg(hash_keys=True, max_per_window=1)
    rl = InMemoryRateLimiter(cfg)
    assert rl.check("ip:1.2.3.4") is True
    # Internal bucket size is reachable via the hashed key.
    assert rl._bucket_size("ip:1.2.3.4") == 1


# ---------------------------------------------------------------------------
# SqliteRateLimiter
# ---------------------------------------------------------------------------


def test_sqlite_basic_under_then_over_limit(tmp_path):
    cfg = _enabled_cfg(store="sqlite", max_per_window=2)
    rl = SqliteRateLimiter(cfg, tmp_path / "rl.sqlite3")
    assert rl.check("ip:1.2.3.4") is True
    assert rl.check("ip:1.2.3.4") is True
    assert rl.check("ip:1.2.3.4") is False


def test_sqlite_persists_across_instances(tmp_path):
    """Second instance against same file sees the first instance's events."""
    db = tmp_path / "rl.sqlite3"
    cfg = _enabled_cfg(store="sqlite", max_per_window=2)

    rl_a = SqliteRateLimiter(cfg, db)
    assert rl_a.check("ip:9.9.9.9") is True
    assert rl_a.check("ip:9.9.9.9") is True

    rl_b = SqliteRateLimiter(cfg, db)
    assert rl_b.check("ip:9.9.9.9") is False  # carries over the existing 2


def test_sqlite_separate_keys_separate_buckets(tmp_path):
    cfg = _enabled_cfg(store="sqlite", max_per_window=1)
    rl = SqliteRateLimiter(cfg, tmp_path / "rl.sqlite3")
    assert rl.check("ip:a") is True
    assert rl.check("ip:b") is True
    assert rl.check("ip:a") is False


def test_sqlite_window_expiry(tmp_path, monkeypatch):
    cfg = _enabled_cfg(store="sqlite", max_per_window=1, window_seconds=5)
    rl = SqliteRateLimiter(cfg, tmp_path / "rl.sqlite3")

    fake_now = [10.0]
    monkeypatch.setattr(time, "time", lambda: fake_now[0])

    assert rl.check("k") is True
    assert rl.check("k") is False  # at cap
    fake_now[0] += 6.0
    assert rl.check("k") is True  # window has rolled over


def test_sqlite_disabled_short_circuits(tmp_path):
    cfg = RateLimitConfig(enabled=False, store="sqlite")
    rl = SqliteRateLimiter(cfg, tmp_path / "rl.sqlite3")
    for _ in range(50):
        assert rl.check("k") is True


# ---------------------------------------------------------------------------
# make_rate_limiter factory
# ---------------------------------------------------------------------------


def test_factory_returns_memory_for_memory_store():
    cfg = _enabled_cfg(store="memory")
    rl = make_rate_limiter(cfg)
    assert isinstance(rl, InMemoryRateLimiter)


def test_factory_returns_sqlite_for_sqlite_store(tmp_path):
    cfg = _enabled_cfg(store="sqlite")
    rl = make_rate_limiter(cfg, db_path=tmp_path / "rl.sqlite3")
    assert isinstance(rl, SqliteRateLimiter)


def test_factory_falls_back_to_default_db_path_when_unspecified(tmp_path, monkeypatch):
    """When the caller doesn't supply a db_path the factory still works.

    We point DEFAULT_RATE_LIMIT_DB_PATH at a tmp dir so the test can't
    pollute the working tree.
    """
    cfg = _enabled_cfg(store="sqlite")
    monkeypatch.setattr(
        "cairn.security.rate_limit.DEFAULT_RATE_LIMIT_DB_PATH",
        tmp_path / "rl.sqlite3",
    )
    rl = make_rate_limiter(cfg)
    assert isinstance(rl, SqliteRateLimiter)


# ---------------------------------------------------------------------------
# rate_limit_status_for_health
# ---------------------------------------------------------------------------


def test_status_disabled_string():
    cfg = RateLimitConfig(enabled=False)
    assert rate_limit_status_for_health(cfg) == "disabled"


def test_status_enabled_dict():
    cfg = _enabled_cfg(window_seconds=30, max_per_window=10, store="sqlite")
    out = rate_limit_status_for_health(cfg)
    assert isinstance(out, dict)
    assert out["window_seconds"] == 30
    assert out["max_per_window"] == 10
    assert out["store"] == "sqlite"

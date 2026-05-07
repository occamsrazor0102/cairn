"""Unit tests for ``cairn.security.auth``.

The bearer-token gate is opt-in; default-off behaviour must be
preserved across every check. These tests pin down the merge logic
between the env var and the config list, the constant-time match,
and the status string surfaced to /health.
"""

from __future__ import annotations

import pytest

from cairn.security.auth import (
    AuthConfig,
    _extract_bearer,
    auth_config_from_dict,
    auth_status_for_health,
    check_bearer,
)


# ---------------------------------------------------------------------------
# auth_config_from_dict — env / config / merge precedence
# ---------------------------------------------------------------------------


def test_config_disabled_when_no_tokens(monkeypatch):
    """No env var, empty/missing config -> disabled."""
    monkeypatch.delenv("CAIRN_API_TOKEN", raising=False)
    cfg = auth_config_from_dict({})
    assert cfg.enabled is False
    assert cfg.tokens == ()


def test_config_disabled_when_none_passed(monkeypatch):
    """Defensive: `None` config behaves as `{}`."""
    monkeypatch.delenv("CAIRN_API_TOKEN", raising=False)
    cfg = auth_config_from_dict(None)
    assert cfg.enabled is False


def test_config_enabled_via_env_var_only(monkeypatch):
    """Bare env var is enough to enable auth."""
    monkeypatch.setenv("CAIRN_API_TOKEN", "env-token-aaaa")
    cfg = auth_config_from_dict({})
    assert cfg.enabled is True
    assert "env-token-aaaa" in cfg.tokens


def test_config_enabled_via_config_only(monkeypatch):
    """Config tokens alone enable auth."""
    monkeypatch.delenv("CAIRN_API_TOKEN", raising=False)
    cfg = auth_config_from_dict({"auth": {"tokens": ["cfg-token-bbbb"]}})
    assert cfg.enabled is True
    assert cfg.tokens == ("cfg-token-bbbb",)


def test_config_merges_env_and_config_in_order(monkeypatch):
    """Env-var token lands first, then config tokens; duplicates collapse."""
    monkeypatch.setenv("CAIRN_API_TOKEN", "env-token-aaaa")
    cfg = auth_config_from_dict({
        "auth": {"tokens": ["cfg-token-bbbb", "env-token-aaaa"]}
    })
    assert cfg.enabled is True
    # env first, dedup keeps only one copy of env-token, config-only token last
    assert cfg.tokens == ("env-token-aaaa", "cfg-token-bbbb")


def test_config_drops_blank_env_var(monkeypatch):
    """Empty env var must not enable auth (common deploy mistake)."""
    monkeypatch.setenv("CAIRN_API_TOKEN", "   ")
    cfg = auth_config_from_dict({})
    assert cfg.enabled is False
    assert cfg.tokens == ()


def test_config_drops_blank_and_non_string_config_tokens(monkeypatch):
    """Garbage in the config list is dropped silently."""
    monkeypatch.delenv("CAIRN_API_TOKEN", raising=False)
    cfg = auth_config_from_dict({
        "auth": {"tokens": ["good-token", "", "  ", None, 42, "another-good"]}
    })
    assert cfg.enabled is True
    assert cfg.tokens == ("good-token", "another-good")


def test_config_explicit_enabled_false_overrides_token_presence(monkeypatch):
    """`auth.enabled: false` keeps the gate open even if tokens are listed."""
    monkeypatch.delenv("CAIRN_API_TOKEN", raising=False)
    cfg = auth_config_from_dict({
        "auth": {"enabled": False, "tokens": ["a-token"]}
    })
    assert cfg.enabled is False


def test_config_custom_header_name(monkeypatch):
    """Header override carried through."""
    monkeypatch.delenv("CAIRN_API_TOKEN", raising=False)
    cfg = auth_config_from_dict({
        "auth": {
            "tokens": ["t"],
            "header_name": "X-Cairn-Token",
        }
    })
    assert cfg.header_name == "X-Cairn-Token"


# ---------------------------------------------------------------------------
# _extract_bearer — header parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("header,expected", [
    ("Bearer abc", "abc"),
    ("bearer abc", "abc"),  # case-insensitive scheme
    ("BEARER abc", "abc"),
    ("Bearer   abc  ", "abc"),  # whitespace tolerated around token
    ("Bearer abc def", "abc def"),  # whole rest is the token
])
def test_extract_bearer_happy(header, expected):
    assert _extract_bearer(header) == expected


@pytest.mark.parametrize("header", [
    None,
    "",
    "   ",
    "Basic abc",
    "Token abc",
    "Bearer",
    "Bearer ",  # empty token
])
def test_extract_bearer_rejects(header):
    assert _extract_bearer(header) is None


# ---------------------------------------------------------------------------
# check_bearer — disabled / open vs enabled / closed
# ---------------------------------------------------------------------------


def test_check_bearer_disabled_always_true():
    """Disabled config -> every call passes regardless of header."""
    cfg = AuthConfig(enabled=False, tokens=())
    assert check_bearer(None, cfg) is True
    assert check_bearer("", cfg) is True
    assert check_bearer("garbage", cfg) is True
    assert check_bearer("Bearer whatever", cfg) is True


def test_check_bearer_enabled_missing_header_false():
    cfg = AuthConfig(enabled=True, tokens=("good-token",))
    assert check_bearer(None, cfg) is False
    assert check_bearer("", cfg) is False


def test_check_bearer_enabled_wrong_token_false():
    cfg = AuthConfig(enabled=True, tokens=("good-token",))
    assert check_bearer("Bearer wrong", cfg) is False


def test_check_bearer_enabled_correct_token_true():
    cfg = AuthConfig(enabled=True, tokens=("good-token",))
    assert check_bearer("Bearer good-token", cfg) is True


def test_check_bearer_enabled_multi_token_first_match():
    cfg = AuthConfig(enabled=True, tokens=("primary", "rotation"))
    assert check_bearer("Bearer primary", cfg) is True
    assert check_bearer("Bearer rotation", cfg) is True
    assert check_bearer("Bearer neither", cfg) is False


def test_check_bearer_rejects_wrong_scheme():
    cfg = AuthConfig(enabled=True, tokens=("good-token",))
    assert check_bearer("Basic good-token", cfg) is False
    assert check_bearer("Token good-token", cfg) is False


def test_check_bearer_rejects_token_substring():
    """A prefix must NOT be accepted; constant-time compare requires equal len."""
    cfg = AuthConfig(enabled=True, tokens=("aaaaaaaaaaaaaaaaaaaa",))
    assert check_bearer("Bearer aaaaaaaaaa", cfg) is False  # half length
    assert check_bearer("Bearer aaaaaaaaaaaaaaaaaaaaaaaa", cfg) is False  # extra


# ---------------------------------------------------------------------------
# auth_status_for_health — health surface
# ---------------------------------------------------------------------------


def test_status_disabled():
    cfg = AuthConfig(enabled=False, tokens=())
    assert auth_status_for_health(cfg) == "disabled"


def test_status_enabled():
    cfg = AuthConfig(enabled=True, tokens=("a-token",))
    assert auth_status_for_health(cfg) == "bearer-token"

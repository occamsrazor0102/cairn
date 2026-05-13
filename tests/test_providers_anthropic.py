"""Unit tests for cairn/providers/anthropic.py.

No live API calls -- the test patches ``anthropic.Anthropic`` with a
fake client to verify the closure forwards the prompt, honors model and
max_tokens overrides, and raises clearly when the API key is missing.
"""
from __future__ import annotations

import sys
import types

import pytest


def _install_fake_anthropic(monkeypatch, *, recorded: list):
    """Inject a fake `anthropic` module into sys.modules with a client
    that records every messages.create() call into ``recorded``."""

    class _Block:
        def __init__(self, text: str):
            self.text = text

    class _Msg:
        def __init__(self, text: str):
            self.content = [_Block(text)]

    class _Messages:
        def __init__(self):
            pass

        def create(self, *, model, max_tokens, messages):
            recorded.append({"model": model, "max_tokens": max_tokens, "messages": messages})
            return _Msg(f"REPLY[{messages[0]['content'][:40]}]")

    class _FakeClient:
        def __init__(self, *, api_key=None):
            self.api_key = api_key
            self.messages = _Messages()

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)


def test_provider_callable_returns_string_and_forwards_prompt(monkeypatch):
    recorded: list = []
    _install_fake_anthropic(monkeypatch, recorded=recorded)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-xyz")
    monkeypatch.delenv("CAIRN_ADVERSARIAL_MODEL", raising=False)

    from cairn.providers.anthropic import provider_callable, DEFAULT_MODEL
    call = provider_callable()

    out = call("paraphrase this prose, please")
    assert isinstance(out, str)
    assert out.startswith("REPLY[")
    assert len(recorded) == 1
    assert recorded[0]["model"] == DEFAULT_MODEL
    assert recorded[0]["messages"] == [
        {"role": "user", "content": "paraphrase this prose, please"}
    ]


def test_provider_callable_honors_explicit_model_and_max_tokens(monkeypatch):
    recorded: list = []
    _install_fake_anthropic(monkeypatch, recorded=recorded)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-xyz")

    from cairn.providers.anthropic import provider_callable
    call = provider_callable(model="claude-opus-4-7", max_tokens=256)
    call("hi")

    assert recorded[0]["model"] == "claude-opus-4-7"
    assert recorded[0]["max_tokens"] == 256


def test_provider_callable_env_var_overrides_default(monkeypatch):
    recorded: list = []
    _install_fake_anthropic(monkeypatch, recorded=recorded)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-xyz")
    monkeypatch.setenv("CAIRN_ADVERSARIAL_MODEL", "claude-sonnet-4-6")

    from cairn.providers.anthropic import provider_callable
    call = provider_callable()
    call("hi")
    assert recorded[0]["model"] == "claude-sonnet-4-6"


def test_provider_callable_raises_when_api_key_missing(monkeypatch):
    recorded: list = []
    _install_fake_anthropic(monkeypatch, recorded=recorded)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from cairn.providers.anthropic import provider_callable
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        provider_callable()

"""Anthropic provider callable for adversarial transforms.

``provider_callable(...)`` returns a thin closure that satisfies the
``Callable[[str], str]`` contract the prompt-category transforms in
``cairn/adversarial/transforms.py`` expect: take a prompt, return the
model's text reply.

Imports of the ``anthropic`` package are lazy so the base install stays
free of the SDK; callers that don't opt into prompt-category attacks
never trigger the import path.
"""
from __future__ import annotations

import os
from typing import Callable

# Matches the default model used by cairn/task_description.py:125 --
# claude-haiku-4-5 is the project's go-to cheap-task model.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024


def provider_callable(
    *,
    model: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    api_key: str | None = None,
) -> Callable[[str], str]:
    """Build a one-shot ``str -> str`` Anthropic caller.

    The SDK client is constructed once and captured in the closure, so
    repeated calls reuse the same HTTPS connection pool.

    The model defaults to ``DEFAULT_MODEL``; pass ``model=`` to override,
    or set ``CAIRN_ADVERSARIAL_MODEL`` in the environment as a deployment
    knob without code changes.
    """
    try:
        import anthropic  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed; pip install cairn[tier3]"
        ) from e

    resolved_model = (
        model or os.environ.get("CAIRN_ADVERSARIAL_MODEL") or DEFAULT_MODEL
    )
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set; export it or pass api_key="
        )
    client = anthropic.Anthropic(api_key=resolved_key)

    def _call(prompt: str) -> str:
        msg = client.messages.create(
            model=resolved_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns a list of content blocks; the prompt-category
        # transforms expect a flat string. Use the first text block.
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                return text
        return ""

    return _call

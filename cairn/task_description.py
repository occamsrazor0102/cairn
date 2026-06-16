"""Auto-derive a short task description from submission text.

Background. ``contrastive_gen`` (and other prompt-aware detectors) need a
short description of what the submission is *trying to do* — distinct
from the submission itself. Historically the caller had to supply it via
``task_metadata={"task_description": "..."}``. This module makes that
optional by deriving one automatically.

Two modes are exposed; the privacy posture differs:

- ``"local"`` — pure heuristic extraction (last interrogative sentence,
  first sentence, or a leading slice). No data leaves the host. This is
  the default once auto mode is enabled and preserves the
  ``contrastive_gen`` privacy invariant: the provider only ever sees the
  derived task description, never the submission.

- ``"llm"`` — sends the submission to a provider (e.g. Anthropic Haiku)
  with a one-sentence-summary prompt. **This sends the submission text
  to the provider** — the call lives outside the contrastive_gen privacy
  guard on purpose, and callers must opt in. The pipeline records a
  warning on every Determination produced under this mode so it's
  visible at the verdict surface, not buried in a config.
"""
from __future__ import annotations

import os
import re

# Cap derived descriptions at this many characters. The provider prompt
# in contrastive_gen treats the description as a brief; a long one
# defeats the privacy constraint and bloats the API call.
_MAX_LEN = 240


def _truncate(s: str, n: int = _MAX_LEN) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _last_question(text: str) -> str | None:
    sentences = re.findall(r"[^.!?]*\?", text)
    if not sentences:
        return None
    return sentences[-1].strip()


def _first_sentence(text: str) -> str | None:
    m = re.search(r"^([^.!?]+[.!?])", text.strip())
    return m.group(1).strip() if m else None


_LOCAL_PREFIX = "Task: "


def derive_local(text: str) -> str:
    """Pure-Python, zero-API-call task description.

    Order of preference: last interrogative sentence, first declarative
    sentence, leading slice. Always truncated to ``_MAX_LEN``.

    The result is prefixed with ``"Task: "`` so it isn't a verbatim
    substring of the submission. ``contrastive_gen`` rejects task
    descriptions that appear inside the submission (caller-misuse
    guard); auto-derivation legitimately extracts from submission text,
    so the prefix is the minimum perturbation that lets the description
    pass the guard while preserving meaning.
    """
    text = text.strip()
    if not text:
        return ""
    q = _last_question(text)
    if q:
        return _LOCAL_PREFIX + _truncate(q, _MAX_LEN - len(_LOCAL_PREFIX))
    s = _first_sentence(text)
    if s:
        return _LOCAL_PREFIX + _truncate(s, _MAX_LEN - len(_LOCAL_PREFIX))
    # No sentence boundary found — return a leading slice with the
    # ellipsis marker preserved so downstream consumers can see this
    # is a fragment, not a complete brief.
    return _LOCAL_PREFIX + _truncate(text, _MAX_LEN - len(_LOCAL_PREFIX))


def _resolve_api_key(config: dict, provider: str) -> str | None:
    key = config.get("api_key") or config.get(f"{provider}_api_key")
    if key:
        return key
    env_name = config.get("api_key_env") or f"{provider.upper()}_API_KEY"
    return os.environ.get(env_name) or None


def derive_llm(
    text: str,
    *,
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 80,
) -> tuple[str, str | None]:
    """LLM-summarised task description. Returns ``(description, error)``.

    The provider sees the full submission text — that's the privacy
    cost callers opt in to when they pick this mode. Errors do not
    raise; ``("", reason)`` is returned and the pipeline falls back to
    local derivation.
    """
    text = text.strip()
    if not text:
        return "", "empty_text"
    prompt = (
        "In one short sentence (under 30 words), describe the TASK that "
        "the following text is responding to or attempting. Output only "
        "the sentence, no preamble.\n\n"
        f"{text}"
    )
    if provider == "anthropic":
        try:
            import anthropic  # type: ignore
        except ImportError:
            return "", "anthropic package not installed"
        try:
            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            msg = client.messages.create(
                model=model or "claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return _truncate(msg.content[0].text), None
        except Exception as e:
            return "", f"{type(e).__name__}: {str(e)[:160]}"
    if provider == "openai":
        try:
            import openai  # type: ignore
        except ImportError:
            return "", "openai package not installed"
        try:
            client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
            resp = client.chat.completions.create(
                model=model or "gpt-4o-mini",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return _truncate(resp.choices[0].message.content or ""), None
        except Exception as e:
            return "", f"{type(e).__name__}: {str(e)[:160]}"
    return "", f"unknown_provider: {provider}"


def auto_derive(text: str, config: dict) -> tuple[str, str, str | None]:
    """Dispatch on ``config['auto_mode']``.

    Returns ``(description, mode_used, warning)``. ``mode_used`` may be
    ``"off"``, ``"local"``, ``"llm"``, or ``"llm_fallback_local"`` when
    LLM derivation failed and we fell back. ``warning`` is a human-
    readable string the pipeline appends to ``Determination.warnings``
    so the verdict surface reflects whether submission text was sent
    to a provider.
    """
    cfg = config or {}
    mode = (cfg.get("auto_mode") or "off").lower()
    if mode in ("off", "false", "no", "none"):
        return "", "off", None
    if mode == "local":
        return derive_local(text), "local", None
    if mode == "llm":
        provider = cfg.get("provider", "anthropic")
        api_key = _resolve_api_key(cfg, provider)
        desc, err = derive_llm(
            text,
            provider=provider,
            model=cfg.get("model"),
            api_key=api_key,
        )
        if desc:
            warning = (
                f"task_description auto-derived via LLM ({provider}); "
                "submission text was sent to the provider"
            )
            return desc, "llm", warning
        local = derive_local(text)
        return (
            local,
            "llm_fallback_local",
            f"task_description LLM derivation failed ({err}); used local fallback",
        )
    return "", "off", f"unknown task_description.auto_mode: {mode!r}"

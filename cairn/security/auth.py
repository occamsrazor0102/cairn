"""Bearer-token authentication for CAIRN's HTTP transports.

CAIRN's `cairn serve` (FastAPI) and `cairn gui` (stdlib HTTP) bind to
loopback by default. When operators expose either transport beyond a
single host, they need an authentication gate. This module is that gate.

Design notes:
    * **Default-disabled.** When no token sources are present the
      :class:`AuthConfig` carries ``enabled=False`` and
      :func:`check_bearer` returns ``True`` for every request. The
      loopback-friendly local experience must not regress.
    * **Token sources.** Two sources are merged: the
      ``CAIRN_API_TOKEN`` env var (single token, convenient for
      container deploys) and the ``auth.tokens`` config list
      (multi-token, useful for rolling key rotation). Empty / blank
      strings are dropped at parse time so an empty env var does not
      accidentally enable auth.
    * **Constant-time check.** :func:`hmac.compare_digest` is used so
      timing side-channels cannot reveal a valid token prefix.
    * **No token material logged.** Callers must not include the
      provided header in log lines on rejection. The existing
      ``cairn.api._redact`` already scrubs ``auth.*`` keys when the
      config is rendered for ``/config``.

The header name defaults to ``Authorization`` and the bearer scheme is
``Bearer <token>``. Anything else (missing scheme, wrong scheme,
empty header) is treated as unauthenticated.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AuthConfig:
    """Resolved authentication configuration.

    Frozen so the middleware can hold a reference without worrying about
    later mutation.

    Attributes
    ----------
    enabled:
        ``True`` when at least one token has been configured. The
        middleware checks this before doing anything else.
    tokens:
        Tuple of accepted token strings. Compared against the bearer
        token via :func:`hmac.compare_digest`.
    header_name:
        Header to read the bearer credential from. Defaults to
        ``Authorization``; rarely overridden.
    """

    enabled: bool = False
    tokens: tuple[str, ...] = field(default_factory=tuple)
    header_name: str = "Authorization"


def auth_config_from_dict(cfg: dict | None) -> AuthConfig:
    """Build an :class:`AuthConfig` from a config dict + environment.

    Parameters
    ----------
    cfg:
        The full CAIRN config dict. Reads ``cfg["auth"]`` if present;
        treats absence as "no config-side tokens".

    Notes
    -----
    Token sources are merged in this order:

    1. ``CAIRN_API_TOKEN`` — a single token, common for env-var-driven
       deployments. Blank/whitespace-only values are ignored.
    2. ``auth.tokens`` — a list under the config dict. Non-string and
       blank entries are dropped.

    Duplicate tokens are de-duplicated while preserving first-seen
    order. ``enabled`` is set to ``True`` iff the merged token list is
    non-empty. The header name defaults to ``Authorization`` and can
    be overridden via ``auth.header_name``.
    """
    auth_section = (cfg or {}).get("auth") or {}

    tokens: list[str] = []

    env_token = os.environ.get("CAIRN_API_TOKEN", "").strip()
    if env_token:
        tokens.append(env_token)

    cfg_tokens = auth_section.get("tokens") or []
    if isinstance(cfg_tokens, (list, tuple)):
        for tok in cfg_tokens:
            if isinstance(tok, str) and tok.strip():
                tokens.append(tok.strip())

    seen: set[str] = set()
    deduped: list[str] = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            deduped.append(tok)

    explicit_enabled = auth_section.get("enabled")
    enabled = bool(deduped) if explicit_enabled is None else bool(explicit_enabled) and bool(deduped)

    header_name = str(auth_section.get("header_name") or "Authorization")

    return AuthConfig(
        enabled=enabled,
        tokens=tuple(deduped),
        header_name=header_name,
    )


def _extract_bearer(provided_header: str | None) -> str | None:
    """Pull the credential out of a ``Bearer <token>`` header.

    Returns ``None`` for missing / empty / wrongly-schemed headers.
    Whitespace is stripped from both sides of the credential so a
    trailing newline (common when a token is loaded from a file via
    ``$(cat token.txt)``) does not silently cause a 401.
    """
    if not provided_header:
        return None
    parts = provided_header.strip().split(None, 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def check_bearer(provided_header: str | None, cfg: AuthConfig) -> bool:
    """Return ``True`` when the caller is authorized.

    Parameters
    ----------
    provided_header:
        Raw header value as received from the transport, e.g.
        ``"Bearer abc123"``. ``None`` when the header was absent.
    cfg:
        The resolved :class:`AuthConfig`.

    Behaviour
    ---------
    * ``cfg.enabled`` is ``False`` -> always ``True`` (open).
    * ``cfg.enabled`` is ``True`` and the header is missing / empty /
      not a bearer scheme -> ``False``.
    * Otherwise compare the provided token against each configured
      token using :func:`hmac.compare_digest` and return ``True`` on
      the first match.
    """
    if not cfg.enabled:
        return True

    token = _extract_bearer(provided_header)
    if token is None:
        return False

    token_bytes = token.encode("utf-8")
    for valid in cfg.tokens:
        valid_bytes = valid.encode("utf-8")
        if len(token_bytes) != len(valid_bytes):
            continue
        if hmac.compare_digest(token_bytes, valid_bytes):
            return True
    return False


def auth_status_for_health(cfg: AuthConfig) -> str:
    """Return the string reported under ``/health::auth``.

    This is the only place the rest of the system needs to know about
    the auth status: ``"disabled"`` when the gate is open,
    ``"bearer-token"`` when a token is required.
    """
    return "bearer-token" if cfg.enabled else "disabled"

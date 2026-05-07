"""URL policy guards for outbound HTTP from CAIRN detectors.

This module exists to defend against SSRF when an openai-compat detector
(``cairn/detectors/dna_gpt.py``, ``cairn/detectors/contrastive_gen.py``)
accepts a ``base_url`` from config. Without validation, an attacker (or
poisoned config) could redirect traffic to internal services such as
AWS instance metadata (``169.254.169.254``) or localhost, exfiltrating
the API key passed alongside the request.

The single public surface is :func:`validate_public_https_base_url`,
which enforces that the URL uses HTTPS and resolves to a publicly
routable IP address. It is intentionally narrow -- pure validation
logic, no I/O beyond DNS resolution.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


def validate_public_https_base_url(url: str) -> str:
    """Return ``url`` unchanged if it's https + resolves to a public IP.

    Raises :class:`ValueError` otherwise. The error message names the
    specific rejection reason (scheme, missing hostname, resolution
    failure, or IP category) to aid debugging without leaking secrets.

    Parameters
    ----------
    url:
        The candidate ``base_url`` from config. Typically an
        openai-compat provider endpoint, e.g. ``https://api.openai.com/v1``.

    Returns
    -------
    str
        The URL, returned unchanged on success so this function can be
        used inline in an assignment.

    Raises
    ------
    ValueError
        If the URL fails any of the checks below.

    Known limitations (best-effort, not a complete SSRF defense)
    ------------------------------------------------------------
    * **DNS rebinding.** :func:`socket.gethostbyname` resolves the
      hostname once at validation time. The HTTP client used by the
      detector resolves the same hostname AGAIN at connect time. An
      adversary controlling the DNS for the hostname can rotate the A
      record between the two resolutions — `api.evil.example` resolves
      to a public IP at validate time and to ``169.254.169.254`` at
      connect time. Closing this hole requires either short-circuiting
      the connect-time resolution (substituting the validated IP into
      the URL host) or pinning resolution at the HTTP-client layer; both
      are out of scope here. Treat this guard as "stops accidental
      misconfiguration and naive SSRF," not "stops a determined
      adversary controlling DNS."
    * **IPv4-mapped IPv6 / dual-stack.** Only the first A record is
      checked (``gethostbyname`` is IPv4-only). A hostname that resolves
      to a public IPv4 plus a private IPv6 will pass validation but the
      HTTP client may pick the IPv6 address depending on platform. For
      strict SSRF, switch to :func:`socket.getaddrinfo` and validate
      every returned address.
    * **Single-A lookup.** Multi-record DNS (round-robin) is not
      enumerated here; we trust the first answer. A poisoned authoritative
      response could mix categories.
    """
    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise ValueError(
            f"base_url must use https scheme, got {parsed.scheme!r}"
        )

    if not parsed.hostname:
        raise ValueError("base_url must include a hostname")

    try:
        resolved = socket.gethostbyname(parsed.hostname)
    except socket.gaierror as e:
        raise ValueError(
            f"base_url hostname could not be resolved: {parsed.hostname!r} ({e})"
        ) from e

    ip = ipaddress.ip_address(resolved)
    if ip.is_loopback:
        raise ValueError(
            f"base_url resolved to loopback address {resolved!r}"
        )
    if ip.is_link_local:
        raise ValueError(
            f"base_url resolved to link-local address {resolved!r}"
        )
    if ip.is_private:
        raise ValueError(
            f"base_url resolved to private address {resolved!r}"
        )
    if ip.is_reserved:
        raise ValueError(
            f"base_url resolved to reserved address {resolved!r}"
        )

    return url

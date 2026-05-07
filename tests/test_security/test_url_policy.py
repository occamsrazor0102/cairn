"""Unit tests for cairn.security.url_policy.

Network access is mocked throughout via unittest.mock.patch so the
suite never touches DNS or the public internet.
"""
from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from cairn.security.url_policy import validate_public_https_base_url


def test_valid_public_https_url_returns_unchanged() -> None:
    """A public https URL with a globally-routable IP passes through."""
    url = "https://api.example.com/v1"
    with patch("cairn.security.url_policy.socket.gethostbyname", return_value="93.184.216.34"):
        result = validate_public_https_base_url(url)
    assert result == url


def test_rejects_http_scheme() -> None:
    """Plain http is rejected; the error names the scheme for debuggability."""
    with pytest.raises(ValueError) as excinfo:
        validate_public_https_base_url("http://example.com/v1")
    msg = str(excinfo.value)
    assert "http" in msg
    assert "scheme" in msg


def test_rejects_localhost_hostname() -> None:
    """`localhost` resolves to 127.0.0.1; reject as loopback."""
    with patch("cairn.security.url_policy.socket.gethostbyname", return_value="127.0.0.1"):
        with pytest.raises(ValueError) as excinfo:
            validate_public_https_base_url("https://localhost/v1")
    assert "loopback" in str(excinfo.value)


def test_rejects_loopback_ip_literal() -> None:
    """A literal 127.0.0.1 in the URL must also be rejected."""
    with patch("cairn.security.url_policy.socket.gethostbyname", return_value="127.0.0.1"):
        with pytest.raises(ValueError) as excinfo:
            validate_public_https_base_url("https://127.0.0.1/v1")
    assert "loopback" in str(excinfo.value)


def test_rejects_link_local_aws_metadata() -> None:
    """169.254.169.254 is the AWS instance metadata endpoint -- the explicit attack we defend against."""
    with patch("cairn.security.url_policy.socket.gethostbyname", return_value="169.254.169.254"):
        with pytest.raises(ValueError) as excinfo:
            validate_public_https_base_url("https://169.254.169.254/v1")
    assert "link-local" in str(excinfo.value)


def test_rejects_private_10_dot_range() -> None:
    """RFC 1918 10.0.0.0/8 must be rejected."""
    with patch("cairn.security.url_policy.socket.gethostbyname", return_value="10.0.0.1"):
        with pytest.raises(ValueError) as excinfo:
            validate_public_https_base_url("https://10.0.0.1/v1")
    assert "private" in str(excinfo.value)


def test_rejects_private_192_dot_168_range() -> None:
    """RFC 1918 192.168.0.0/16 must be rejected."""
    with patch("cairn.security.url_policy.socket.gethostbyname", return_value="192.168.1.1"):
        with pytest.raises(ValueError) as excinfo:
            validate_public_https_base_url("https://192.168.1.1/v1")
    assert "private" in str(excinfo.value)


def test_rejects_private_172_dot_16_range() -> None:
    """RFC 1918 172.16.0.0/12 must be rejected."""
    with patch("cairn.security.url_policy.socket.gethostbyname", return_value="172.16.0.1"):
        with pytest.raises(ValueError) as excinfo:
            validate_public_https_base_url("https://172.16.0.1/v1")
    assert "private" in str(excinfo.value)


def test_rejects_missing_hostname() -> None:
    """`https://` parses with an empty hostname; reject before DNS."""
    with pytest.raises(ValueError) as excinfo:
        validate_public_https_base_url("https://")
    assert "hostname" in str(excinfo.value)


def test_rejects_unresolvable_hostname() -> None:
    """A DNS failure is surfaced as ValueError, not a bare gaierror."""
    with patch(
        "cairn.security.url_policy.socket.gethostbyname",
        side_effect=socket.gaierror("Name or service not known"),
    ):
        with pytest.raises(ValueError) as excinfo:
            validate_public_https_base_url("https://nope.invalid/v1")
    msg = str(excinfo.value)
    assert "resolve" in msg or "resolved" in msg

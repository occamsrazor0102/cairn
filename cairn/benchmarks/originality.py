"""Originality.ai adapter — SKELETON.

Commercial detector by Originality.ai (https://originality.ai). API
docs: ``https://api.originality.ai/api/v1/scan/ai`` (POST, JSON body,
header ``X-OAI-API-KEY``).

This adapter is a SKELETON: ``score()`` raises :class:`NotImplementedError`.
The HTTP integration is deferred until we have a live API key under
change-control plus snapshot fixtures.

**Privacy concern — read before wiring this up.** Calling ``score()``
sends the *full submission text* to a third-party server. The
commercial-adapter shelf is gated behind both the
``--allow-commercial`` CLI flag and the
``CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1`` environment variable to make
this trade-off explicit.

The adapter reports ``is_available()=False`` whenever
``ORIGINALITY_API_KEY`` is missing so default benchmark runs route
around it without manual configuration.
"""
from __future__ import annotations

import os

from cairn.benchmarks.base import CompetitorAdapter

ORIGINALITY_API_URL = "https://api.originality.ai/api/v1/scan/ai"


class OriginalityAdapter:
    """Skeleton adapter for the Originality.ai commercial detection API."""

    id: str = "originality"
    domain: str = "universal"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._api_key = api_key or os.environ.get("ORIGINALITY_API_KEY")
        self._base_url = base_url or os.environ.get(
            "ORIGINALITY_BASE_URL", ORIGINALITY_API_URL
        )

    def available_dependencies(self) -> dict[str, bool]:
        return {
            "ORIGINALITY_API_KEY": bool(self._api_key),
        }

    def is_available(self) -> bool:
        return bool(self._api_key)

    def score(self, text: str) -> float:  # pragma: no cover — skeleton
        # TODO(commercial-integration): replace this stub with an HTTP
        # POST that:
        #   1. Sends ``{"content": text, "title": "", "aiModelVersion": "1"}``
        #      to self._base_url with header ``X-OAI-API-KEY``.
        #   2. Parses the response: Originality returns
        #      ``{"score": {"ai": float, "original": float}, ...}``.
        #   3. Maps ``score.ai`` → p_llm directly.
        #   4. Surfaces network/auth/quota failures via raise; the
        #      runner records the exception in ``failed_samples``.
        # Snapshot fixtures live in tests/test_benchmarks/fixtures/.
        raise NotImplementedError(
            "OriginalityAdapter.score is a skeleton; live integration "
            "is deferred until snapshot fixtures and a privacy-review "
            "checkpoint are in place. See module docstring."
        )


ADAPTER = OriginalityAdapter()


def _verify_protocol() -> None:
    """Sanity-check the adapter satisfies the structural Protocol."""
    assert isinstance(ADAPTER, CompetitorAdapter)


_verify_protocol()

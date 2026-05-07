"""GPTZero adapter — SKELETON.

Commercial detector by GPTZero (https://gptzero.me). API docs:
``https://api.gptzero.me/v2/predict/text`` (POST, JSON body, header
``x-api-key``).

This adapter is a SKELETON: ``score()`` raises :class:`NotImplementedError`.
The HTTP integration is deliberately deferred until we have:

* A live API key under change-control.
* Snapshot fixtures captured under ``tests/test_benchmarks/fixtures/``
  so live runs can be smoke-tested offline.
* A concrete privacy review checkpoint (see below).

**Privacy concern — read before wiring this up.** Calling ``score()``
sends the *full submission text* to a third-party server. CAIRN's
default privacy stance keeps text on-device or behind ``retain_text=false``
(hashes only). The commercial-adapter shelf is gated behind both the
``--allow-commercial`` CLI flag *and* the
``CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1`` environment variable to make
this trade-off explicit. Operators must opt in deliberately.

The adapter reports ``is_available()=False`` whenever
``GPTZERO_API_KEY`` is missing so default benchmark runs route around
it without manual configuration.
"""
from __future__ import annotations

import os

from cairn.benchmarks.base import CompetitorAdapter

GPTZERO_API_URL = "https://api.gptzero.me/v2/predict/text"


class GPTZeroAdapter:
    """Skeleton adapter for the GPTZero commercial detection API."""

    id: str = "gptzero"
    domain: str = "universal"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self._api_key = api_key or os.environ.get("GPTZERO_API_KEY")
        self._base_url = base_url or os.environ.get(
            "GPTZERO_BASE_URL", GPTZERO_API_URL
        )

    def available_dependencies(self) -> dict[str, bool]:
        return {
            "GPTZERO_API_KEY": bool(self._api_key),
        }

    def is_available(self) -> bool:
        return bool(self._api_key)

    def score(self, text: str) -> float:  # pragma: no cover — skeleton
        # TODO(commercial-integration): once a live API key is under
        # change-control, replace this stub with an HTTP POST that:
        #   1. Sends ``text`` to self._base_url with ``x-api-key`` header.
        #   2. Parses the response: GPTZero returns
        #      ``{"documents": [{"completely_generated_prob": float, ...}]}``.
        #   3. Maps ``completely_generated_prob`` → p_llm directly.
        #   4. Surfaces network/auth failures via the
        #      ``run_eval.failed_samples`` mechanism (raise on failure;
        #      the runner already records the exception).
        # Snapshot fixtures live in tests/test_benchmarks/fixtures/.
        raise NotImplementedError(
            "GPTZeroAdapter.score is a skeleton; live integration is "
            "deferred until snapshot fixtures and a privacy-review "
            "checkpoint are in place. See module docstring."
        )


ADAPTER = GPTZeroAdapter()


def _verify_protocol() -> None:
    """Sanity-check the adapter satisfies the structural Protocol."""
    assert isinstance(ADAPTER, CompetitorAdapter)


_verify_protocol()

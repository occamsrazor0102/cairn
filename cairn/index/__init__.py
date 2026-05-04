"""Retrieval index module - imports kept lazy for base installs.

Use explicit paths:
    from cairn.index.retrieval import RetrievalIndex
    from cairn.index.store import PersistentRetrievalIndex
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cairn.index.retrieval import RetrievalIndex
    from cairn.index.store import PersistentRetrievalIndex

__all__: list[str] = []

"""Tests for cairn.detectors registry exception handling."""
import sys
from unittest.mock import patch
import importlib

import pytest

import cairn.detectors as registry


@pytest.fixture(autouse=True)
def _restore_registry_state():
    """Snapshot registry state before each test and rebuild it after, so a
    test's forced re-discovery does not leave the rest of the suite with an
    empty / partial registry."""
    saved_registry = dict(registry._registry)
    saved_missing = dict(registry._missing)
    saved_discovered = registry._discovered
    try:
        yield
    finally:
        registry._registry = saved_registry
        registry._missing = saved_missing
        registry._discovered = saved_discovered


def test_constructor_failure_lands_in_missing():
    """Regression: a detector whose import raises a non-ImportError must be
    recorded in get_missing_detectors() rather than silently disappearing."""
    # Force re-discovery
    registry._discovered = False
    registry._registry = {}
    registry._missing = {}

    real_import = importlib.import_module

    def import_with_failure(name, *args, **kwargs):
        if name == "cairn.detectors.preamble":
            raise RuntimeError("simulated constructor failure")
        return real_import(name, *args, **kwargs)

    with patch("cairn.detectors.importlib.import_module", side_effect=import_with_failure):
        registry._discover()

    missing = registry.get_missing_detectors()
    assert "preamble" in missing
    assert "RuntimeError" in missing["preamble"]
    assert "simulated" in missing["preamble"]


def test_attribute_error_during_register_lands_in_missing():
    """Regression: an AttributeError after import (e.g., d.id missing) must
    also land in _missing, not silently swallow."""
    registry._discovered = False
    registry._registry = {}
    registry._missing = {}

    class BrokenDetector:
        # Missing .id attribute
        domain = "universal"

    fake_module = type(sys)("cairn.detectors.broken_test")
    fake_module.DETECTOR = BrokenDetector()

    real_import = importlib.import_module

    def import_with_broken(name, *args, **kwargs):
        if name == "cairn.detectors.broken_test":
            return fake_module
        return real_import(name, *args, **kwargs)

    # Inject the broken module into the package by patching iter_modules too.
    # _discover() calls pkgutil.iter_modules(__path__, __name__ + "."), so the
    # yielded ModuleInfo.name is the fully-qualified dotted name (which is what
    # the frozen importer also produces) — mirror that here.
    fake_iter = [type("M", (), {"name": "cairn.detectors.broken_test"})()]
    real_iter = registry.pkgutil.iter_modules

    def iter_with_broken(*args, **kwargs):
        for m in real_iter(*args, **kwargs):
            yield m
        for m in fake_iter:
            yield m

    with patch("cairn.detectors.pkgutil.iter_modules", side_effect=iter_with_broken), \
         patch("cairn.detectors.importlib.import_module", side_effect=import_with_broken):
        registry._discover()

    missing = registry.get_missing_detectors()
    assert "broken_test" in missing
    assert "AttributeError" in missing["broken_test"]


def test_keyboard_interrupt_propagates():
    """Regression: KeyboardInterrupt must NOT be swallowed by the broad
    Exception handler. Operators need to be able to abort discovery."""
    registry._discovered = False
    registry._registry = {}
    registry._missing = {}

    real_import = importlib.import_module

    def import_with_kbi(name, *args, **kwargs):
        if name == "cairn.detectors.preamble":
            raise KeyboardInterrupt()
        return real_import(name, *args, **kwargs)

    with patch("cairn.detectors.importlib.import_module", side_effect=import_with_kbi):
        try:
            registry._discover()
            assert False, "KeyboardInterrupt should have propagated"
        except KeyboardInterrupt:
            pass  # expected

"""
Auto-discovery registry for CAIRN detectors.
Each module in this package should expose a DETECTOR module-level attribute
implementing the Detector protocol.
"""
import importlib
import logging
import pkgutil
from cairn.contracts import Detector

_logger = logging.getLogger("cairn.detectors")

_registry: dict[str, Detector] = {}
_missing: dict[str, str] = {}
_discovered = False


def _discover() -> None:
    global _discovered
    # Enumerate submodules via the package's own __path__/__name__ rather
    # than a raw filesystem directory string. pkgutil.iter_modules over a
    # literal path only sees real files on disk, which finds nothing in a
    # PyInstaller frozen build where modules live in the embedded archive;
    # the frozen importer participates only when given the package __path__.
    for module_info in pkgutil.iter_modules(__path__, __name__ + "."):
        short_name = module_info.name.rsplit(".", 1)[-1]
        if short_name == "__init__" or short_name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(module_info.name)
            if hasattr(mod, "DETECTOR"):
                d = mod.DETECTOR
                _registry[d.id] = d
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            _missing[short_name] = f"{type(e).__name__}: {e}"
            # Preserve the full traceback so deep import-chain failures
            # (e.g., a detector whose optional dep fails partway through
            # its own __init__) can be diagnosed from logs. The single-
            # line `_missing` string is enough for /health UI display;
            # the traceback is what an operator needs when triaging.
            _logger.debug(
                "detector discovery failed for %s",
                short_name,
                exc_info=True,
            )
    _discovered = True


def get_detector(detector_id: str) -> Detector | None:
    if not _discovered:
        _discover()
    return _registry.get(detector_id)


def get_all_detectors() -> dict[str, Detector]:
    if not _discovered:
        _discover()
    return dict(_registry)


def get_missing_detectors() -> dict[str, str]:
    if not _discovered:
        _discover()
    return dict(_missing)

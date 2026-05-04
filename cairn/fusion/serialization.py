from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


_HMAC_KEY_ENV = "CAIRN_MODEL_HMAC_KEY"
_LEGACY_PICKLE_ENV = "CAIRN_ALLOW_LEGACY_PICKLE"


def _hmac_key() -> bytes:
    """Return the HMAC key for model integrity verification.

    Reads from env var CAIRN_MODEL_HMAC_KEY. The default fallback
    "cairn-default-model-hmac-key" is published in source — HMAC with this
    key protects only against passive corruption (filesystem bit-rot,
    accidental edits), NOT active tampering. For tamper-evident artifacts,
    set CAIRN_MODEL_HMAC_KEY to a private secret in the environment.
    """
    return os.environ.get(_HMAC_KEY_ENV, "cairn-default-model-hmac-key").encode()


def _digest(payload: bytes) -> str:
    return hmac.new(_hmac_key(), payload, hashlib.sha256).hexdigest()


def save_model(model: Any, path: Path, *, metadata: dict | None = None) -> None:
    """Save a fitted model with HMAC integrity verification + provenance metadata.

    Writes three files at the target path:
      * `<path>` — the skops-serialized model (zip-format container)
      * `<path>.hmac` — HMAC-SHA256 of the skops payload, hex-encoded ASCII
      * `<path>.meta.json` — optional metadata dict, present only when metadata is supplied

    Expected metadata keys (informational, not enforced):
      dataset_path, dataset_hash, n_train, n_test,
      label_balance_train, label_balance_test, trained_at,
      interactions, max_rounds, schema_version

    HMAC key is read from env CAIRN_MODEL_HMAC_KEY; defaults to a published
    fallback that protects only against passive corruption (see _hmac_key).
    """
    try:
        import skops.io as sio
    except ImportError as e:
        raise RuntimeError(
            "skops is required. Install with: pip install 'cairn[fusion]'"
        ) from e
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sio.dump(model, path)
    payload = path.read_bytes()
    sidecar = path.with_suffix(path.suffix + ".hmac")
    sidecar.write_text(_digest(payload), encoding="ascii")
    # Warn once per save if the default key is in use
    if not os.environ.get(_HMAC_KEY_ENV):
        logger.warning(
            "save_model: HMAC computed with default public key (%s unset); "
            "artifact is tamper-evident only against passive corruption, not adversaries.",
            _HMAC_KEY_ENV,
        )
    if metadata is not None:
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _looks_like_skops(path: Path) -> bool:
    if path.suffix == ".skops":
        return path.exists() and path.stat().st_size > 0
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def load_model(
    path: Path, *, verify_hmac: bool = True, trusted: list[str] | None = None
) -> tuple[Any, dict]:
    path = Path(path)
    if not _looks_like_skops(path):
        if not os.environ.get(_LEGACY_PICKLE_ENV):
            raise RuntimeError(
                f"Refusing to load legacy pickle artifact {path}. "
                f"Pickle deserialization is unsafe — retrain with skops, OR "
                f"set {_LEGACY_PICKLE_ENV}=1 to opt in during migration."
            )
        warnings.warn(
            f"Loading legacy pickle artifact {path}; retrain with skops",
            DeprecationWarning, stacklevel=2,
        )
        with open(path, "rb") as f:
            return pickle.load(f), {}
    try:
        import skops.io as sio
    except ImportError as e:
        raise RuntimeError(
            "skops is required. Install with: pip install 'cairn[fusion]'"
        ) from e
    payload = path.read_bytes()
    if verify_hmac:
        sidecar = path.with_suffix(path.suffix + ".hmac")
        if not sidecar.exists():
            raise ValueError(f"HMAC sidecar missing for {path}")
        expected = sidecar.read_text(encoding="ascii").strip()
        actual = _digest(payload)
        if not hmac.compare_digest(expected, actual):
            raise ValueError(f"HMAC mismatch for {path} - artifact tampered")
    untrusted = sio.get_untrusted_types(file=path)
    if trusted is None:
        if untrusted:
            logger.warning(
                "skops auto-trust: accepting %d untrusted types from %s: %s. "
                "Supply trusted= explicitly when loading artifacts from untrusted sources.",
                len(untrusted), path, sorted(untrusted),
            )
        trusted = list(untrusted)
    model = sio.load(path, trusted=trusted)
    metadata: dict = {}
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read metadata for %s: %s", path, exc)
            metadata = {}
    return model, metadata

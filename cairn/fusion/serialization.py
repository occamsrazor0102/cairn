from __future__ import annotations

import hashlib
import hmac
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any


_HMAC_KEY_ENV = "CAIRN_MODEL_HMAC_KEY"


def _hmac_key() -> bytes:
    return os.environ.get(_HMAC_KEY_ENV, "cairn-default-model-hmac-key").encode()


def _digest(payload: bytes) -> str:
    return hmac.new(_hmac_key(), payload, hashlib.sha256).hexdigest()


def save_model(model: Any, path: Path, *, metadata: dict | None = None) -> None:
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
    sidecar.write_text(_digest(payload))
    if metadata is not None:
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def _looks_like_skops(path: Path) -> bool:
    if path.suffix == ".skops":
        return True
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
        expected = sidecar.read_text().strip()
        actual = _digest(payload)
        if not hmac.compare_digest(expected, actual):
            raise ValueError(f"HMAC mismatch for {path} - artifact tampered")
    untrusted = sio.get_untrusted_types(file=path)
    if trusted is None:
        trusted = list(untrusted)
    model = sio.load(path, trusted=trusted)
    metadata: dict = {}
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    return model, metadata

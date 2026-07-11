"""Validate a current-schema TOCSIN cache and write its manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

from cairn.cascade import PHASE1_DETECTORS, PHASE2_DETECTORS, PHASE3_DETECTORS
from cairn.config import load_config
from cairn.tocsin_cache import validate_tocsin_cache, write_json_atomic


def _cache_layer_inventory(
    config: dict[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    configured = config.get("detectors", {})
    if not isinstance(configured, dict):
        return (), ()
    phase_layer_ids = PHASE1_DETECTORS + PHASE2_DETECTORS + PHASE3_DETECTORS
    enabled: list[str] = []
    disabled: list[str] = []
    for layer_id in phase_layer_ids:
        settings = configured.get(layer_id)
        if isinstance(settings, dict) and settings.get("enabled") is True:
            enabled.append(layer_id)
        elif isinstance(settings, dict) and settings.get("enabled") is False:
            disabled.append(layer_id)
    return tuple(enabled), tuple(disabled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-signature")
    parser.add_argument("--require-all-layers-seen", action="store_true")
    parser.add_argument("--out-manifest", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    enabled_layer_ids, disabled_layer_ids = _cache_layer_inventory(config)
    summary = validate_tocsin_cache(
        args.dataset,
        args.cache,
        expected_signature=args.expected_signature,
        expected_layer_ids=enabled_layer_ids,
        require_all_layers_seen=args.require_all_layers_seen,
    )
    manifest = summary.to_dict()
    manifest["disabled_layer_ids"] = disabled_layer_ids
    write_json_atomic(args.out_manifest, manifest)


if __name__ == "__main__":
    main()

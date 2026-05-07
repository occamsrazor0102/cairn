"""Run head-to-head benchmark of competitor adapters against an EvalSample dataset.

Default suite is local-OSS-only. Commercial adapters require *both*
``--allow-commercial`` AND ``CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1`` so
they are never enabled accidentally — calling them sends full text to
a third-party.

Usage::

    python scripts/run_competitor_benchmark.py \\
        --dataset data/ood_raid_test.jsonl \\
        --adapters binoculars,fast_detectgpt \\
        --out docs/benchmarks/competitor/raid_competitor.json

Output is a JSON file with one ``EvalReport``-shaped block per adapter
(serialised via ``cairn.evaluation.runner.eval_report_to_dict``).
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path

from cairn.benchmarks.base import to_pipeline, wrap_as_detector
from cairn.evaluation.dataset import load_dataset
from cairn.evaluation.runner import eval_report_to_dict, run_eval

log = logging.getLogger("cairn.scripts.run_competitor_benchmark")


# Map CLI adapter id → (module path, exposed name, is_commercial).
_ADAPTER_REGISTRY: dict[str, tuple[str, str, bool]] = {
    "binoculars": ("cairn.benchmarks.binoculars", "ADAPTER", False),
    "fast_detectgpt": ("cairn.benchmarks.fast_detectgpt", "ADAPTER", False),
    "gptzero": ("cairn.benchmarks.gptzero", "ADAPTER", True),
    "originality": ("cairn.benchmarks.originality", "ADAPTER", True),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run competitor adapters against a CAIRN EvalSample dataset.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to a JSONL EvalSample dataset.",
    )
    parser.add_argument(
        "--adapters",
        required=True,
        type=str,
        help=(
            "Comma-separated adapter ids. Available: "
            + ", ".join(sorted(_ADAPTER_REGISTRY)) + "."
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Path to write the combined results JSON.",
    )
    parser.add_argument(
        "--allow-commercial",
        action="store_true",
        default=False,
        help=(
            "Permit commercial adapters. Also requires "
            "CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1 in the environment."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally cap the number of samples evaluated.",
    )
    parser.add_argument(
        "--include-predictions",
        action="store_true",
        default=False,
        help=(
            "Include per-sample predictions in the output (default off "
            "to keep the JSON small)."
        ),
    )
    return parser.parse_args(argv)


def _resolve_adapter(adapter_id: str):
    if adapter_id not in _ADAPTER_REGISTRY:
        raise SystemExit(
            f"Unknown adapter '{adapter_id}'. Available: "
            + ", ".join(sorted(_ADAPTER_REGISTRY))
        )
    module_path, attr, _ = _ADAPTER_REGISTRY[adapter_id]
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _commercial_allowed(args: argparse.Namespace) -> bool:
    return bool(args.allow_commercial) and (
        os.environ.get("CAIRN_ALLOW_COMMERCIAL_BENCHMARK") == "1"
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)

    requested = [a.strip() for a in args.adapters.split(",") if a.strip()]
    if not requested:
        raise SystemExit("--adapters must list at least one adapter id.")

    # Reject commercial adapters unless both gates are set.
    selected_commercial: list[str] = [
        a for a in requested if _ADAPTER_REGISTRY.get(a, ("", "", False))[2]
    ]
    if selected_commercial and not _commercial_allowed(args):
        raise SystemExit(
            "Commercial adapters requested ({0}) but the gate is closed. "
            "Pass --allow-commercial AND set "
            "CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1 to opt in.".format(
                ", ".join(selected_commercial)
            )
        )

    log.info("loading dataset: %s", args.dataset)
    samples = load_dataset(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    log.info("loaded %d samples", len(samples))

    results: dict[str, dict] = {
        "_meta": {
            "dataset": str(args.dataset),
            "n_samples": len(samples),
            "adapters_requested": requested,
        },
    }

    for adapter_id in requested:
        log.info("running adapter: %s", adapter_id)
        adapter = _resolve_adapter(adapter_id)
        if not adapter.is_available():
            deps = adapter.available_dependencies()
            log.warning(
                "adapter %s is unavailable (deps=%s); skipping",
                adapter_id,
                deps,
            )
            results[adapter_id] = {
                "skipped": True,
                "reason": "adapter_unavailable",
                "dependencies": deps,
            }
            continue
        pipeline = to_pipeline(wrap_as_detector(adapter))
        report = run_eval(pipeline, samples)
        results[adapter_id] = eval_report_to_dict(
            report, include_predictions=args.include_predictions
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("wrote results to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

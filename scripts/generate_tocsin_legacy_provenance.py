"""Generate sanitized retrospective TOCSIN cache and cost manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence

from cairn.evaluation.tocsin_legacy_provenance import (
    generate_retrospective_manifests,
)


LOCKED_PILOT_SHA256 = "f752f280e5de0d4c580201cfaf4281203296b7a9be07e92b0d3af7ad948aaf55"
LOCKED_FULL_SHA256 = "d0159e1dedd6c24371272e9ea38701658e2d29e91499caa81b7c791a9110905f"


def _git_revision(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if len(revision) != 40:
        raise ValueError(f"git rev-parse returned malformed revision {revision!r}")
    return revision


def _render(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_pair(rendered: Mapping[Path, str]) -> None:
    temporary: dict[Path, Path] = {}
    try:
        for destination, content in rendered.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
            )
            path = Path(name)
            with open(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
            temporary[destination] = path
        for destination, path in temporary.items():
            path.replace(destination)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--git-revision")
    parser.add_argument("--expected-active-rows", type=int, default=11926)
    parser.add_argument("--expected-pilot-rows", type=int, default=256)
    parser.add_argument(
        "--expected-pilot-sha256", default=LOCKED_PILOT_SHA256
    )
    parser.add_argument("--expected-full-sha256", default=LOCKED_FULL_SHA256)
    parser.add_argument("--original-cost-cap-usd", default="15.00")
    parser.add_argument("--provenance-out", type=Path, required=True)
    parser.add_argument("--cost-out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    revision = args.git_revision or _git_revision(repo_root)
    provenance, cost = generate_retrospective_manifests(
        repo_root=repo_root,
        run_root=args.run_root,
        dataset_root=args.dataset_root,
        config_path=args.config,
        git_revision=revision,
        expected_active_rows=args.expected_active_rows,
        expected_pilot_rows=args.expected_pilot_rows,
        expected_pilot_sha256=args.expected_pilot_sha256,
        expected_full_sha256=args.expected_full_sha256,
        original_cost_cap_usd=Decimal(args.original_cost_cap_usd),
    )
    _write_pair(
        {
            args.provenance_out: _render(provenance),
            args.cost_out: _render(cost),
        }
    )
    print(
        json.dumps(
            {
                "provenance_path": args.provenance_out.as_posix(),
                "provenance_sha256": hashlib.sha256(
                    args.provenance_out.read_bytes()
                ).hexdigest(),
                "cost_path": args.cost_out.as_posix(),
                "cost_sha256": hashlib.sha256(args.cost_out.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

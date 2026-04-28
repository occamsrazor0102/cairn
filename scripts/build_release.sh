#!/usr/bin/env bash
# Build a release bundle of the CAIRN Tauri desktop shell on macOS / Linux.
#
# Steps:
#   1. Build the Python sidecar binary via scripts/build_sidecar.py
#      (renames to cairn-sidecar-<target-triple> under src-tauri/binaries/).
#   2. Run `cargo tauri build` to produce an unsigned .dmg / .app / .deb /
#      .AppImage under src-tauri/target/release/bundle/.
#
# Prereqs:
#   - Python 3.11+ with the `dist` extra installed:
#       pip install -e ".[dist]"
#   - Rust toolchain + Tauri CLI:
#       cargo install tauri-cli --version "^2"
#
# Usage:
#   ./scripts/build_release.sh                # full build
#   ./scripts/build_release.sh --no-smoke     # skip sidecar smoke test
#   ./scripts/build_release.sh --sidecar-only # build only the sidecar binary

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

sidecar_args=()
sidecar_only=0
for arg in "$@"; do
    case "$arg" in
        --no-smoke)     sidecar_args+=("--no-smoke") ;;
        --sidecar-only) sidecar_only=1 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

echo "==> building sidecar binary"
python scripts/build_sidecar.py "${sidecar_args[@]}"

if [[ "$sidecar_only" -eq 1 ]]; then
    echo "==> sidecar-only build complete"
    exit 0
fi

echo "==> running cargo tauri build"
(
    cd src-tauri
    cargo tauri build
)

echo "==> done. Artifacts under: $REPO_ROOT/src-tauri/target/release/bundle"

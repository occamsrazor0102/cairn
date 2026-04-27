#!/usr/bin/env pwsh
# Build a release bundle of the BEET Tauri desktop shell on Windows.
#
# Steps:
#   1. Build the Python sidecar binary via scripts/build_sidecar.py
#      (renames to beet-sidecar-<target-triple>.exe under src-tauri/binaries/).
#   2. Run `cargo tauri build` to produce an unsigned .msi / .exe installer
#      under src-tauri/target/release/bundle/.
#
# Prereqs:
#   - Python 3.11+ with the `dist` extra installed:
#       pip install -e ".[dist]"
#   - Rust toolchain + Tauri CLI:
#       cargo install tauri-cli --version "^2"
#
# Usage:
#   ./scripts/build_release.ps1                  # full build
#   ./scripts/build_release.ps1 -SkipSmoke       # skip sidecar smoke test
#   ./scripts/build_release.ps1 -SidecarOnly     # build only the sidecar binary

param(
    [switch]$SkipSmoke,
    [switch]$SidecarOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

Push-Location $RepoRoot
try {
    $sidecarArgs = @()
    if ($SkipSmoke) { $sidecarArgs += "--no-smoke" }

    Write-Host "==> building sidecar binary" -ForegroundColor Cyan
    python scripts/build_sidecar.py @sidecarArgs
    if ($LASTEXITCODE -ne 0) { throw "build_sidecar.py failed (exit $LASTEXITCODE)" }

    if ($SidecarOnly) {
        Write-Host "==> sidecar-only build complete" -ForegroundColor Green
        return
    }

    Write-Host "==> running cargo tauri build" -ForegroundColor Cyan
    Push-Location (Join-Path $RepoRoot "src-tauri")
    try {
        cargo tauri build
        if ($LASTEXITCODE -ne 0) { throw "cargo tauri build failed (exit $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }

    $bundleDir = Join-Path $RepoRoot "src-tauri/target/release/bundle"
    Write-Host "==> done. Artifacts under: $bundleDir" -ForegroundColor Green
} finally {
    Pop-Location
}

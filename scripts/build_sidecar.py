"""Build the Python sidecar binary for Tauri bundling.

Runs PyInstaller against `beet/sidecar.py` and renames the output to the
target-triple suffix Tauri 2 expects when resolving an `externalBin`
declared as `binaries/cairn-sidecar` in `tauri.conf.json`.

Usage:
    python scripts/build_sidecar.py            # build for the host triple
    python scripts/build_sidecar.py --no-smoke # skip post-build RPC smoke
    python scripts/build_sidecar.py --target x86_64-pc-windows-msvc  # override

Exits non-zero if PyInstaller is missing, the build fails, or the smoke
test does not get a `{"status": "ok"}` response back from `health`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIDECAR_SRC = REPO_ROOT / "cairn" / "sidecar.py"
DIST_DIR = REPO_ROOT / "src-tauri" / "binaries"
SPEC_FILE = REPO_ROOT / "cairn-sidecar.spec"
BUILD_DIR = REPO_ROOT / "build"


def detect_target_triple() -> str:
    """Return the Rust host target triple via `rustc -vV`.

    Tauri 2 resolves `externalBin: ["binaries/cairn-sidecar"]` to a file
    named `cairn-sidecar-<target-triple>(.exe)` at bundle time.
    """
    try:
        out = subprocess.check_output(
            ["rustc", "-vV"], text=True, stderr=subprocess.STDOUT
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise SystemExit(
            "rustc not available — cannot infer target triple. "
            "Install Rust (https://rustup.rs) or pass --target explicitly."
        ) from e
    for line in out.splitlines():
        if line.startswith("host:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"could not parse `host:` from rustc -vV output:\n{out}")


def run_pyinstaller(*, target: str, clean: bool) -> Path:
    """Invoke PyInstaller and return the path of the produced binary."""
    if not SIDECAR_SRC.exists():
        raise SystemExit(f"sidecar source not found: {SIDECAR_SRC}")
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        "cairn-sidecar",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(REPO_ROOT),
        "--noconfirm",
        # Pull in lazy/optional submodules we only import inside functions.
        "--collect-submodules",
        "cairn",
        str(SIDECAR_SRC),
    ]
    if clean:
        cmd.insert(3, "--clean")

    print(f"[build_sidecar] running PyInstaller for target={target}", flush=True)
    subprocess.check_call(cmd, cwd=REPO_ROOT)

    is_windows = sys.platform.startswith("win") or "windows" in target
    raw_name = "cairn-sidecar.exe" if is_windows else "cairn-sidecar"
    raw_path = DIST_DIR / raw_name
    if not raw_path.exists():
        raise SystemExit(f"PyInstaller output missing: {raw_path}")

    final_name = (
        f"cairn-sidecar-{target}.exe" if is_windows else f"cairn-sidecar-{target}"
    )
    final_path = DIST_DIR / final_name
    if final_path.exists():
        final_path.unlink()
    raw_path.rename(final_path)
    print(f"[build_sidecar] wrote {final_path}", flush=True)
    return final_path


def smoke_test(binary: Path, timeout: float = 30.0) -> None:
    """Pipe one `health` JSON-RPC call to the binary; assert ok."""
    print(f"[build_sidecar] smoke test: {binary} health", flush=True)
    proc = subprocess.Popen(
        [str(binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        request = json.dumps({"id": "smoke", "method": "health"}) + "\n"
        out, err = proc.communicate(request, timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise SystemExit("smoke test timed out waiting for sidecar response")

    response = None
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("id") == "smoke":
            response = obj
            break

    if response is None:
        raise SystemExit(
            "smoke test: no response with id='smoke' on stdout.\n"
            f"stdout:\n{out}\nstderr:\n{err}"
        )
    result = (response.get("result") or {})
    if result.get("status") != "ok":
        raise SystemExit(
            f"smoke test: unexpected response: {response}\nstderr:\n{err}"
        )
    print(
        f"[build_sidecar] smoke ok: calibration={result.get('calibration_status')}, "
        f"protocol={result.get('protocol_version')}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        help="Rust target triple (default: rustc host)",
        default=None,
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip PyInstaller --clean (faster incremental builds)",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="Skip the post-build sidecar RPC smoke test",
    )
    args = parser.parse_args()

    target = args.target or detect_target_triple()
    binary = run_pyinstaller(target=target, clean=not args.no_clean)

    if not args.no_smoke:
        smoke_test(binary)

    for path in (SPEC_FILE, BUILD_DIR):
        if path.exists():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except OSError:
                pass

    print(f"[build_sidecar] done: {binary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

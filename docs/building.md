# Building CAIRN

Three shells are supported and share one frontend (`cairn/gui/static/index.html`):

| Shell | Backend | Frontend transport | Dist format |
|-------|---------|-------------------|-------------|
| `cairn gui` | local HTTP server (`cairn/gui/server.py`) | `fetch()` | Python package |
| `cairn serve` | FastAPI (requires `cairn[api]` extras) | `fetch()` | Python package |
| Tauri desktop | Python sidecar subprocess over stdio JSON-RPC | `invoke()` | `.dmg` / `.msi` / `.AppImage` |

The `index.html` auto-detects the host (`window.__TAURI_INTERNALS__`) and routes calls through `cairnApi`. The same file works in all three.

---

## Prerequisites

- **Python** 3.11+ (project already requires this)
- **Rust** (stable toolchain) — https://rustup.rs
- **Tauri CLI** — `cargo install tauri-cli --version "^2.0"`
- **Node.js** 18+ — Tauri requires it even for vanilla-HTML frontends
- **PyInstaller** — install via the `dist` extra: `pip install -e ".[dist]"` (only needed for bundled builds, not dev)
- Platform-native build toolchains:
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Windows: Microsoft C++ Build Tools + WebView2 runtime (pre-installed on Win11)
  - Linux: `webkit2gtk-4.1`, `libayatana-appindicator3-dev`, `librsvg2-dev`, `build-essential`

Verify:
```bash
rustc --version
cargo tauri --version
node --version
python --version
```

---

## Repo layout (Tauri-relevant)

```
.
├── cairn/                   # Python package
│   ├── gui/
│   │   ├── server.py        # HTTP shell
│   │   └── static/index.html # single-file SPA, dual-mode (fetch + invoke)
│   └── sidecar.py           # JSON-RPC loop over stdio (Tauri backend)
├── src-tauri/
│   ├── Cargo.toml
│   ├── tauri.conf.json      # frontendDist: ../cairn/gui/static
│   ├── build.rs
│   ├── src/
│   │   ├── main.rs          # Tauri app entry + sidecar lifecycle
│   │   ├── commands.rs      # #[tauri::command] handlers
│   │   └── sidecar.rs       # spawn/manage Python sidecar subprocess
│   ├── icons/               # populated via `cargo tauri icon`
│   └── binaries/            # PyInstaller output lives here for bundling
└── docs/building.md         # this file
```

---

## Development loop (no bundling)

You want the fastest iteration cycle. Run the Python sidecar from source, launch Tauri in dev mode, and skip PyInstaller entirely.

**1. One-time icon setup** (Tauri won't build without them):
```bash
cd src-tauri
cargo tauri icon path/to/any-square-png.png
cd ..
```
This generates `src-tauri/icons/*.{png,icns,ico}` in all required sizes.

**2. Point the Rust side at your local Python sidecar**:

macOS / Linux:
```bash
export CAIRN_SIDECAR_CMD="python -m cairn.sidecar --profile default"
```

Windows (PowerShell):
```powershell
$env:CAIRN_SIDECAR_CMD = "python -m cairn.sidecar --profile default"
```

The Rust sidecar manager checks this env var first; if set, it uses it verbatim instead of looking for a bundled binary. (Whitespace-split — don't put paths with spaces in the command.)

**3. Launch**:
```bash
cd src-tauri
cargo tauri dev
```

First run compiles the Rust dependencies (~2–5 min). Subsequent runs are incremental. The window opens with the CAIRN frontend; paste text, analyze, submit feedback, all backed by your local Python.

**Debugging tips:**
- Sidecar stderr (Python warnings, logs) is inherited into the Tauri process stderr. Watch the terminal.
- Add `--log-level DEBUG` to `CAIRN_SIDECAR_CMD` for verbose logs.
- Frontend DevTools: right-click in the window (dev builds only) → Inspect Element.
- Rust logs: `RUST_LOG=cairn=debug cargo tauri dev`.

---

## Production build (bundled sidecar + Tauri binary)

Builds are **per-platform** — you build on macOS to get `.dmg`, on Windows to get `.msi`, on Linux to get `.AppImage`. Cross-compilation is painful; use CI runners or separate machines.

**Recommended path — wrapper scripts:**

```bash
# macOS / Linux
./scripts/build_release.sh

# Windows (PowerShell)
./scripts/build_release.ps1
```

Each wrapper does two things:

1. Calls `python scripts/build_sidecar.py`, which runs PyInstaller, renames the output to `src-tauri/binaries/cairn-sidecar-<rust-target-triple>(.exe)` (the form Tauri 2 expects when resolving the `externalBin` declaration), and pipes a `health` JSON-RPC call into the binary as a smoke test.
2. Runs `cargo tauri build` from `src-tauri/`, producing the platform installer under `src-tauri/target/release/bundle/`.

Useful flags: `--no-smoke` / `-SkipSmoke` skips the sidecar smoke; `--sidecar-only` / `-SidecarOnly` stops after step 1.

**Output paths:**
- macOS: `src-tauri/target/release/bundle/dmg/CAIRN_2.0.0_aarch64.dmg` (or `x86_64`)
- Windows: `src-tauri/target/release/bundle/msi/CAIRN_2.0.0_x64_en-US.msi`
- Linux: `src-tauri/target/release/bundle/appimage/cairn_2.0.0_amd64.AppImage`

**Smoke test the bundle:**
1. Launch the installed app.
2. Paste text, click Analyze. Verdict should render within ~1–2 seconds.
3. Submit Human/LLM feedback. Check that `data/reviewer_feedback.jsonl` gets written — the path is relative to the app's working directory, which Tauri sets per-OS.
4. Close the window. Verify the Python subprocess exits (no orphaned `cairn-sidecar` process in Task Manager / `ps`).

**Manual fallback** (drive PyInstaller and Tauri yourself):

```bash
pip install -e ".[dist]"
python scripts/build_sidecar.py        # writes src-tauri/binaries/cairn-sidecar-<triple>(.exe)
unset CAIRN_SIDECAR_CMD                 # or Remove-Item Env:CAIRN_SIDECAR_CMD on PowerShell
cd src-tauri && cargo tauri build
```

---

## Unsigned-build caveats

Shipping without code signing is fine for internal use. End users will see:

- **macOS**: *"CAIRN can't be opened because it is from an unidentified developer."* Right-click → Open → Open again dismisses it. Gatekeeper may also require `xattr -d com.apple.quarantine /Applications/CAIRN.app` the first time.
- **Windows**: SmartScreen warning. Click *More info* → *Run anyway*.
- **Linux**: no warning. Users just `chmod +x` the AppImage.

To sign later:
- macOS: Apple Developer ID ($99/yr) + `tauri.conf.json` > `bundle.macOS.signingIdentity`
- Windows: Azure Trusted Signing or an EV code signing certificate + `bundle.windows.certificateThumbprint`

---

## CI-friendly build

The repo ships a three-OS GitHub Actions matrix at [`.github/workflows/release.yml`](../.github/workflows/release.yml). It triggers on `v*` tag pushes and on manual `workflow_dispatch`; each platform job runs the same wrapper script you'd run locally and uploads the resulting installers to a draft GitHub Release.

To cut a release:

```bash
git tag v2.0.1
git push origin v2.0.1
```

Watch the run in the Actions tab. When all three jobs finish, head to the **Releases** page — there will be a draft pre-populated with the installers. Edit the notes, click Publish.

For an unattended build without cutting a tag, trigger the workflow manually (Actions → Release → Run workflow); artifacts upload via `actions/upload-artifact` in that path.

---

## Known limitations (v0)

- **No auto-restart** of the sidecar if it crashes. In-flight requests fail with `ERR_SIDECAR_GONE`; the user must relaunch the app. Add a restart loop in `src-tauri/src/sidecar.rs::reader_loop` when desired.
- **Single-window only.** Tauri supports multi-window; adding a second window requires passing the `SidecarManager` to it via Tauri state (already `Arc<…>` — trivial).
- **`data/` path is process-relative.** On packaged builds, feedback JSONL may land somewhere unexpected. Resolve an app-data dir via `app.path().app_data_dir()` and pass it to the sidecar via a flag.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `sidecar binary not found at ... and CAIRN_SIDECAR_CMD not set` | Dev without env var, or production without PyInstaller output | Set `CAIRN_SIDECAR_CMD` or build the sidecar binary into `src-tauri/binaries/` |
| White window, no content | `frontendDist` path wrong | Check `tauri.conf.json`: should be `../cairn/gui/static` (relative to `src-tauri/`) |
| `invoke is not defined` in browser mode | Someone opened `index.html` directly as `file://` | Always serve via `cairn gui` (HTTP) or Tauri (invoke) |
| `ERR_SIDECAR_GONE` on every call | Python subprocess died at startup | Run `CAIRN_SIDECAR_CMD` manually in a shell; check the error |
| `cargo tauri icon` fails | Source image is not square | Crop to square before running |

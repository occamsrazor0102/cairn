"""Run the HC3 benchmark grid on Modal's GPU instead of local CPU.

The local machine has no CUDA, so the heavy transformer detectors
(surprisal_dynamics, contrastive_lm, token_cohesiveness, mixed_boundary)
crawl on CPU. Modal's T4 chews through a full HC3 grid in minutes.

This script:
  1. Ensures every requested HC3 subset is downloaded + split (persisted
     on a Modal Volume so re-runs are instant).
  2. Builds a deduplicated feature cache for the union of every test
     split the grid will hit (the heavy GPU step).
  3. Runs `scripts/run_hc3_benchmark.py --features-cache <cache>` to
     produce the per-cell metrics + summary CSV / Markdown.
  4. Returns a tarball with the benchmark artefacts AND the feature
     cache, so follow-on local analyses (per-detector AUROC, LOO,
     loss-filter) skip the GPU work entirely.

Usage:
    modal run scripts/modal_run_hc3_benchmark.py
    modal run scripts/modal_run_hc3_benchmark.py --subsets finance,medicine \\
        --configs configs/.profile_general_hc3.yaml --limit 200

One-time setup mirrors `scripts/modal_cache_features.py`:
    pip install modal
    modal token new
    # ensure HF_TOKEN exists in .env

Cost estimate: T4 at ~$0.50/hr. Full HC3 grid (~6k unique texts) at
~10 samples/sec on GPU = ~10 min = ~$0.10. Re-runs hit the cache.
"""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import modal

app = modal.App("cairn-hc3-benchmark")

# Image: heavy ML deps + HC3 download support. Mounts the cairn/ package,
# configs/, scripts/, and the calibrated general_hc3 profile artefacts so
# the harness runs unchanged inside the container.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "sentence-transformers",
        "scikit-learn",
        "interpret",
        "numpy",
        "pyyaml",
        "huggingface_hub",
        "tqdm",
    )
    .add_local_dir("cairn", remote_path="/root/cairn")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_dir(
        "models/calib2000/profiles/general_hc3",
        remote_path="/root/models/calib2000/profiles/general_hc3",
    )
)

# Shared HF cache — same volume name as scripts/modal_cache_features.py
# so cold-start weight downloads only happen once across both scripts.
hf_cache = modal.Volume.from_name("cairn-hf-cache", create_if_missing=True)

# Persistent HC3 dataset volume — raw HC3 JSONLs and stratified splits
# survive across runs, making re-benchmarks near-instant.
hc3_data = modal.Volume.from_name("cairn-hc3-data", create_if_missing=True)


# ---------------------------------------------------------------------------
# Pure helpers — unit-tested without a Modal token.
# ---------------------------------------------------------------------------

def parse_csv_list(s: str | None) -> list[str]:
    """Split a comma-separated CLI value into a list, stripping whitespace.

    Modal's `local_entrypoint` only takes scalar args, so list-valued flags
    are passed as comma-separated strings.
    """
    if not s:
        return []
    return [item.strip() for item in s.split(",") if item.strip()]


def pack_directory(root: Path, files: list[Path]) -> bytes:
    """Pack `files` into a gzipped tarball. Names are stored relative to root."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in files:
            arcname = f.relative_to(root).as_posix()
            tar.add(f, arcname=arcname)
    return buf.getvalue()


def unpack_tarball(data: bytes, dest: Path) -> list[Path]:
    """Extract a tarball into `dest` and return the list of written files.

    `dest` is created if missing. Path traversal entries (absolute paths,
    `..` segments) are rejected to avoid escaping the destination.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"unsafe tar entry: {name}")
            tar.extract(member, dest)
            if member.isfile():
                written.append(dest / name)
    return written


def collect_outputs(out_dir: Path, cache: Path | None) -> list[Path]:
    """Files to ship back from the remote run."""
    files: list[Path] = []
    if out_dir.exists():
        files.extend(p for p in out_dir.rglob("*") if p.is_file())
    if cache is not None and cache.exists():
        files.append(cache)
    return files


# ---------------------------------------------------------------------------
# Remote function — runs on Modal T4.
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    secrets=[modal.Secret.from_dotenv()],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/data": hc3_data,
    },
)
def run_grid(
    subsets: list[str],
    configs: list[str],
    *,
    limit: int | None = None,
    seed: int = 0,
    max_per_class: int = 2000,
    use_cache: bool = True,
    rebuild: bool = False,
) -> bytes:
    """Run the full HC3 grid remotely. Returns a gzipped tarball of artefacts.

    The local `run_hc3_benchmark.py` script is pure orchestration over
    `download_hc3.py`, `split_dataset.py`, and `run_benchmark.py` — so the
    same code path runs unchanged inside the container.

    Args:
        max_per_class: Cap per (human/llm) class when fetching missing HC3
            subsets. Default 2000 matches the user's local downloads — at
            500 the `all` subset becomes a degenerate slice (drawn
            sequentially from HF and dominated by the first source),
            which sinks the calibrated EBM well below published numbers.
        use_cache: When True, the harness builds a feature cache once and
            replays it across cells. Set False to A/B-test the calibrated
            path against a live pipeline (rules out cache/EBM mismatch).
        rebuild: When True, wipes stale per-subset HC3 splits + the
            feature cache on the persistent volume so a new
            `max_per_class` setting actually takes effect.
    """
    import glob
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/root")
    os.chdir("/root")

    from cairn.dotenv import load_dotenv
    load_dotenv()

    if rebuild:
        # Splits and cache are stale once max_per_class changes — wipe them
        # so the next phase downloads and re-splits under the new cap.
        patterns = [
            "/root/data/hc3_*.jsonl",
            "/root/data/all_features_hc3_modal.jsonl",
            "/root/data/_hc3_cache_input.jsonl",
        ]
        for pat in patterns:
            for p in glob.glob(pat):
                print(f"[rebuild] removing {p}", flush=True)
                Path(p).unlink()
        hc3_data.commit()

    cache_path = Path("/root/data/all_features_hc3_modal.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", "scripts/run_hc3_benchmark.py",
        "--data-dir", "/root/data",
        "--out-dir", "/root/docs/benchmarks/hc3_grid",
        "--summary-csv", "/root/docs/benchmarks/hc3_grid/summary.csv",
        "--summary-md", "/root/docs/benchmarks/hc3_grid/summary.md",
        "--max-per-class", str(max_per_class),
        "--seed", str(seed),
        "--skip-existing",
    ]
    if use_cache:
        cmd += ["--features-cache", str(cache_path)]
    if subsets:
        cmd += ["--subsets", *subsets]
    if configs:
        cmd += ["--configs", *configs]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    print("running:", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"run_hc3_benchmark.py failed (rc={res.returncode})")

    hc3_data.commit()

    out_dir = Path("/root/docs/benchmarks/hc3_grid")
    files = collect_outputs(out_dir, cache_path)
    if not files:
        raise RuntimeError("no output artefacts found")

    print(f"packing {len(files)} files...", flush=True)
    return pack_directory(Path("/root"), files)


# ---------------------------------------------------------------------------
# Local entrypoint — invoked via `modal run`.
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    subsets: str = "",
    configs: str = "",
    limit: int = 0,
    seed: int = 0,
    max_per_class: int = 2000,
    out_dir: str = "docs/benchmarks/hc3_grid_modal",
    no_cache: bool = False,
    rebuild: bool = False,
) -> None:
    """Driver that calls the remote function and writes results locally.

    Args:
        subsets: Comma-separated HC3 subsets, or empty for all six.
        configs: Comma-separated cairn config paths, or empty for the
                 heuristic + general_hc3 defaults.
        limit:   Sub-sample each test split to N items (0 = no limit).
        seed:    Seed forwarded to run_hc3_benchmark.py.
        max_per_class: Cap per (human/llm) class when fetching missing
            subsets. Default 2000 matches the user's local download cap;
            at lower values the `all` subset degenerates and tanks the
            calibrated EBM.
        out_dir: Local destination directory for unpacked artefacts.
        no_cache: Skip the feature cache and run each cell against the
            live pipeline. Use to A/B-test cache-vs-no-cache when the
            calibrated path looks suspiciously low.
        rebuild: Wipe stale per-subset splits + feature cache on the
            persistent Modal volume before running. Required when
            changing `max_per_class` from a previous run.
    """
    subset_list = parse_csv_list(subsets)
    config_list = parse_csv_list(configs)
    eff_limit = limit if limit > 0 else None

    print(
        "dispatching HC3 grid to Modal "
        f"(subsets={subset_list or 'ALL'}, configs={config_list or 'DEFAULTS'}, "
        f"limit={eff_limit}, seed={seed}, max_per_class={max_per_class}, "
        f"use_cache={not no_cache}, rebuild={rebuild})..."
    )
    tar_bytes = run_grid.remote(
        subsets=subset_list,
        configs=config_list,
        limit=eff_limit,
        seed=seed,
        max_per_class=max_per_class,
        use_cache=not no_cache,
        rebuild=rebuild,
    )

    dest = Path(out_dir)
    written = unpack_tarball(tar_bytes, dest)
    print(f"unpacked {len(written)} files into {dest}/")
    summary_md = dest / "docs" / "benchmarks" / "hc3_grid" / "summary.md"
    if summary_md.exists():
        print("\n" + summary_md.read_text(encoding="utf-8"))

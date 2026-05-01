"""Run scripts/cache_pipeline_features.py on Modal's GPU instead of local CPU.

The local machine has no CUDA, so the heavy transformer detectors
(surprisal_dynamics, contrastive_lm, token_cohesiveness, mixed_boundary)
run at ~1 sample/sec on CPU. Modal's T4 GPU runs them at ~10x that rate.

Usage:
    modal run scripts/modal_cache_features.py \\
        --dataset-path data/calib2000_full.jsonl \\
        --out-path data/calib2000_features.jsonl

One-time setup:
    pip install modal
    modal token new            # opens browser, authenticates
    # ensure HF_TOKEN exists in .env (read by modal.Secret.from_dotenv)

Cost estimate: T4 at ~$0.50/hr. 24k unique texts at ~10 samples/sec on
GPU = ~40 min = ~$0.35. Compare with ~10+ hr CPU locally.
"""
from __future__ import annotations

import modal

app = modal.App("cairn-cache-features")

# Container image with all the heavy ML deps preinstalled. Mounts the
# cairn/ package + configs/ directory so the function can import normally.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "sentence-transformers",
        "scikit-learn",
        "numpy",
        "pyyaml",
        "huggingface_hub",
    )
    .add_local_dir("cairn", remote_path="/root/cairn")
    .add_local_dir("configs", remote_path="/root/configs")
)

# Persistent volume for HF model cache so cold starts are fast on
# subsequent runs (first run downloads ~1 GB of GPT-2 + sentence-
# transformer weights; later runs hit the volume).
hf_cache = modal.Volume.from_name("cairn-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_dotenv()],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def cache_features(dataset_jsonl: bytes, config_path: str = "configs/default.yaml") -> bytes:
    """Run pipeline.analyze_detailed on each unique text. Returns JSONL bytes."""
    import io
    import json
    import os
    import sys
    sys.path.insert(0, "/root")
    os.chdir("/root")

    from cairn.dotenv import load_dotenv
    load_dotenv()

    from cairn.feature_cache import text_hash, write_record
    from cairn.pipeline import CairnPipeline, normalize_text

    pipeline = CairnPipeline.from_config_file(config_path)

    out = io.StringIO()
    seen: set[str] = set()
    n_dup = 0
    n_done = 0
    text = dataset_jsonl.decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    print(f"processing {len(lines)} samples...", flush=True)
    for line in lines:
        sample = json.loads(line)
        norm = normalize_text(sample["text"])
        key = text_hash(norm)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        _det, results, rd = pipeline.analyze_detailed(norm)
        write_record(out, key, results, rd)
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done} unique texts cached, {n_dup} duplicates", flush=True)

    print(f"done: {n_done} unique features, {n_dup} duplicates", flush=True)
    return out.getvalue().encode("utf-8")


@app.local_entrypoint()
def main(dataset_path: str, out_path: str, config_path: str = "configs/default.yaml") -> None:
    """Local driver: read dataset, send to GPU, write cache to disk."""
    from pathlib import Path

    src = Path(dataset_path).read_bytes()
    print(f"sending {len(src):,} bytes to Modal GPU function ({dataset_path})...")
    cache_bytes = cache_features.remote(src, config_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(cache_bytes)
    n_lines = cache_bytes.count(b"\n")
    print(f"wrote {n_lines:,} cached features to {out_path}")

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
    # H200 (141 GB HBM3e) -- top-tier inference card. A10G (24 GB) ran
    # at ~0.05-0.09 s/sample on the 110k-row full_coverage cache pass;
    # H200 is ~1.3x faster than H100 and ~3-4x faster than A10G on
    # transformer token-decode (HBM3e bandwidth ~4.8 TB/s vs 3.35 on
    # H100 vs 0.6 on A10G). Projected to land the full cache in
    # ~30-45 min. Cost is higher per hour (~$5-6/hr on H200 vs ~$1.10/hr
    # on A10G) but total spend is roughly net-neutral or only slightly
    # higher because the faster GPU finishes sooner.
    gpu="H200",
    # 6 hours -- 110k unique training rows; H200 should finish much
    # sooner but we keep the ceiling generous in case of cold-start
    # or model-load stalls.
    timeout=21600,
    secrets=[modal.Secret.from_dotenv()],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def cache_features(
    dataset_jsonl: bytes,
    config_path: str = "configs/default.yaml",
    run_id: str = "v8",
) -> bytes:
    """Run pipeline.analyze_detailed on each unique text. Returns JSONL bytes.

    Checkpoints incrementally to /root/.cache/huggingface/cairn_cache_output/
    so Modal worker preemption (auto-restart with same inputs) resumes where
    the previous run left off instead of reprocessing from scratch.
    """
    import json
    import os
    import sys
    sys.path.insert(0, "/root")
    os.chdir("/root")

    from cairn.dotenv import load_dotenv
    load_dotenv()

    from cairn.feature_cache import text_hash, write_record
    from cairn.pipeline import CairnPipeline, normalize_text

    # Persistent checkpoint -- the HF volume survives worker preemption.
    # On restart we read back already-processed text hashes so the next
    # pass only does new work.
    checkpoint_path = f"/root/.cache/huggingface/cairn_cache_output/{run_id}_features.jsonl"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    seen: set[str] = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as cp:
            for ln in cp:
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    # Partial line written when the worker was preempted
                    # mid-record. Safe to skip -- we'll re-process that
                    # text and append a fresh record below.
                    continue
                h = rec.get("hash")
                if h:
                    seen.add(h)
        print(f"resuming from checkpoint: {len(seen)} entries already cached", flush=True)

    pipeline = CairnPipeline.from_config_file(config_path)

    n_dup = 0
    n_done = 0
    text = dataset_jsonl.decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    print(f"processing {len(lines)} samples...", flush=True)

    with open(checkpoint_path, "a", encoding="utf-8") as cp:
        for line in lines:
            sample = json.loads(line)
            norm = normalize_text(sample["text"])
            key = text_hash(norm)
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
            _det, results, rd = pipeline.analyze_detailed(norm)
            write_record(cp, key, results, rd)
            cp.flush()
            n_done += 1
            if n_done % 100 == 0:
                # Periodic volume commit -- writes only become durable
                # across preemption when committed.
                hf_cache.commit()
                print(
                    f"  {n_done} new cached this run, "
                    f"{len(seen)} total in checkpoint, {n_dup} duplicates",
                    flush=True,
                )

    hf_cache.commit()
    print(
        f"done: {n_done} new this run; {len(seen)} total in checkpoint; "
        f"{n_dup} duplicates skipped",
        flush=True,
    )

    with open(checkpoint_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    dataset_path: str,
    out_path: str,
    config_path: str = "configs/default.yaml",
    run_id: str = "v8",
    shards: int = 1,
) -> None:
    """Local driver: read dataset, send to GPU, write cache to disk.

    When ``shards > 1``, splits the input into N contiguous shards and
    runs them in parallel containers via Modal ``starmap``. Each shard
    uses its own resumable checkpoint (``{run_id}_shard{i}``). Outputs
    are merged with hash dedup on the local side.
    """
    import json
    from pathlib import Path

    src = Path(dataset_path).read_bytes()
    if shards <= 1:
        print(f"sending {len(src):,} bytes to Modal GPU function ({dataset_path})...")
        cache_bytes = cache_features.remote(src, config_path, run_id)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(cache_bytes)
        n_lines = cache_bytes.count(b"\n")
        print(f"wrote {n_lines:,} cached features to {out_path}")
        return

    text = src.decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    n = len(lines)
    shard_size = (n + shards - 1) // shards
    shard_args = []
    for i in range(shards):
        start = i * shard_size
        end = min((i + 1) * shard_size, n)
        if start >= end:
            continue
        shard_bytes = ("\n".join(lines[start:end]) + "\n").encode("utf-8")
        shard_args.append((shard_bytes, config_path, f"{run_id}_shard{i}"))
        print(f"  shard {i}: rows [{start}, {end}) -> {len(shard_bytes):,} bytes")

    print(f"fanning out {len(shard_args)} parallel shards to Modal...")
    results = list(cache_features.starmap(shard_args))

    seen: set[str] = set()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_dup_xshard = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for idx, shard_bytes in enumerate(results):
            for raw in shard_bytes.decode("utf-8").splitlines():
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                h = rec.get("hash")
                if not h:
                    continue
                if h in seen:
                    n_dup_xshard += 1
                    continue
                seen.add(h)
                out.write(raw + "\n")
                n_total += 1
            print(f"  shard {idx}: merged ({n_total} unique so far)")
    print(
        f"wrote {n_total:,} unique features to {out_path} "
        f"({n_dup_xshard} cross-shard duplicates merged)"
    )

"""LAPD reproducibility spike on Modal T4.

Loads the Falcon-7B + Falcon-7B-Instruct pair (4-bit quantized via
bitsandbytes), runs LAPDDetector.analyze over `data/ood_raid.jsonl`,
and reports AUROC of the heuristic p_llm + raw S_LAPD against the
0/1 label.

Acceptance gate from `docs/superpowers/specs/2026-05-17-lapd-detector-scoping.md`:
RAID AUROC >= 0.82 → proceed with full integration. < 0.78 → abort
and pivot to SpecDetect++.

Usage:
    modal run scripts/modal_spike_lapd.py \\
        --dataset-path data/ood_raid.jsonl \\
        --out-path docs/benchmarks/calib2000/lapd_spike_raid.json

Cost estimate: A100-40GB at ~$2.50/hr. A10G's 22 GB was insufficient
even with 4-bit quantization because transformers' caching_allocator_warmup
holds fp16-sized buffers during the second model's load (peak ~14 GB
per model). A100-40GB gives 12+ GB of post-load headroom for the
inference. 1,000 RAID rows at ~8 s/row with n_perturbations=50 ≈
2.5 hr ≈ $6.25 per sweep value. Three sweep values (n=20/50/100) ≈
$10-14 total. Falcon pair (~14 GB) is partial-cached in the
persistent HF cache volume from prior runs.
"""
from __future__ import annotations

import modal

app = modal.App("cairn-lapd-spike")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "scikit-learn",
        "numpy",
        "pyyaml",
        "huggingface_hub",
    )
    .add_local_dir("cairn", remote_path="/root/cairn")
    .add_local_dir("configs", remote_path="/root/configs")
)

hf_cache = modal.Volume.from_name("cairn-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=14400,  # 4 hours; enough headroom for all three sweep values
    secrets=[modal.Secret.from_dotenv()],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_spike(
    dataset_jsonl: bytes,
    n_perturbations_list: list[int],
    aligned_model: str = "tiiuae/falcon-7b-instruct",
    base_model: str = "tiiuae/falcon-7b",
    max_samples: int | None = None,
) -> bytes:
    """Run the LAPD spike sweep on Modal. Returns JSON bytes with per-sweep AUROC."""
    import io
    import json
    import math
    import os
    import sys
    import time

    sys.path.insert(0, "/root")
    os.chdir("/root")

    from cairn.detectors.lapd import LAPDDetector

    text = dataset_jsonl.decode("utf-8")
    rows: list[dict] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    # Stratify: interleave label-0 and label-1 rows so that any
    # max_samples cap yields a class-balanced subset. Without this the
    # raw file is sorted by label and `rows[:100]` returns 100 humans →
    # AUROC undefined.
    label0 = [r for r in rows if int(r.get("label", -1)) == 0]
    label1 = [r for r in rows if int(r.get("label", -1)) == 1]
    interleaved: list[dict] = []
    for h, l in zip(label0, label1):
        interleaved.append(h)
        interleaved.append(l)
    # Append any tail-overflow (shouldn't happen for balanced RAID, but safe).
    interleaved.extend(label0[len(label1):])
    interleaved.extend(label1[len(label0):])
    rows = interleaved
    if max_samples is not None:
        rows = rows[:max_samples]
    print(
        f"loaded {len(rows)} RAID rows "
        f"(label-0: {sum(1 for r in rows if int(r['label']) == 0)}, "
        f"label-1: {sum(1 for r in rows if int(r['label']) == 1)})",
        flush=True,
    )

    # Pre-load the pair ONCE; reused across all sweep values.
    detector = LAPDDetector()
    detector._ensure_loaded(aligned_model, base_model, "cuda", quantize_4bit=True)
    print(f"models loaded: {aligned_model} + {base_model}", flush=True)

    sweep_results: list[dict] = []
    for n_perturbations in n_perturbations_list:
        print(f"--- sweep n_perturbations={n_perturbations} ---", flush=True)
        labels: list[int] = []
        p_llms: list[float] = []
        s_lapds: list[float] = []
        skipped: list[dict] = []
        t0 = time.time()

        cfg = {
            "aligned_model": aligned_model,
            "base_model": base_model,
            "quantize_4bit": True,
            "n_perturbations": n_perturbations,
            "perturbation_rate": 0.10,
            "max_tokens": 512,
            "seed": 42,
        }
        for i, row in enumerate(rows):
            lr = detector.analyze(row["text"], cfg)
            if lr.determination == "SKIP":
                skipped.append({"i": i, "reason": lr.signals.get("skipped")})
                continue
            labels.append(int(row["label"]))
            p_llms.append(float(lr.p_llm))
            s_lapds.append(float(lr.signals["s_lapd"]))
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(rows) - i - 1) / rate
                print(
                    f"  {i + 1}/{len(rows)} rows in {elapsed:.0f}s "
                    f"({rate:.2f}/s, ETA {eta:.0f}s, skipped={len(skipped)})",
                    flush=True,
                )

        elapsed = time.time() - t0
        auroc_p = _auroc(labels, p_llms) if labels else float("nan")
        auroc_s = _auroc(labels, s_lapds) if labels else float("nan")
        # If our sign convention disagrees with the paper's, AUROC < 0.5 and
        # the "flipped" orientation is what the gate should be measured
        # against. Report both so the verdict is unambiguous.
        auroc_p_flipped = 1.0 - auroc_p if labels else float("nan")
        auroc_s_flipped = 1.0 - auroc_s if labels else float("nan")
        print(
            f"sweep {n_perturbations}: "
            f"AUROC(p_llm)={auroc_p:.4f} (flipped={auroc_p_flipped:.4f}), "
            f"AUROC(s_lapd)={auroc_s:.4f} (flipped={auroc_s_flipped:.4f}), "
            f"n={len(labels)}, skipped={len(skipped)}, time={elapsed:.0f}s",
            flush=True,
        )

        sweep_results.append({
            "n_perturbations": n_perturbations,
            "n_scored": len(labels),
            "n_skipped": len(skipped),
            "auroc_p_llm": auroc_p,
            "auroc_s_lapd": auroc_s,
            "auroc_p_llm_flipped": auroc_p_flipped,
            "auroc_s_lapd_flipped": auroc_s_flipped,
            "wall_seconds": elapsed,
            "sec_per_sample": elapsed / max(len(labels), 1),
            "skipped_breakdown": _tally_skip_reasons(skipped),
        })

    spec_gate_pass = any(
        max(
            s["auroc_s_lapd"], s["auroc_p_llm"],
            s["auroc_s_lapd_flipped"], s["auroc_p_llm_flipped"],
        ) >= 0.82
        for s in sweep_results
    )

    return json.dumps({
        "aligned_model": aligned_model,
        "base_model": base_model,
        "dataset_n_rows": len(rows),
        "sweeps": sweep_results,
        "spec_gate_threshold": 0.82,
        "spec_gate_pass": spec_gate_pass,
    }, indent=2).encode("utf-8")


def _auroc(labels: list[int], scores: list[float]) -> float:
    """sklearn AUROC -- imported inside to keep the local entrypoint slim."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def _tally_skip_reasons(skipped: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in skipped:
        reason = s.get("reason") or "unknown"
        out[reason] = out.get(reason, 0) + 1
    return out


@app.local_entrypoint()
def main(
    dataset_path: str = "data/ood_raid.jsonl",
    out_path: str = "docs/benchmarks/calib2000/lapd_spike_raid.json",
    n_perturbations: str = "20,50,100",
    aligned_model: str = "tiiuae/falcon-7b-instruct",
    base_model: str = "tiiuae/falcon-7b",
    max_samples: int = -1,
) -> None:
    """Local driver: read dataset, send to GPU, write results to disk."""
    from pathlib import Path

    src = Path(dataset_path).read_bytes()
    sweep = [int(x) for x in n_perturbations.split(",") if x.strip()]
    cap = None if max_samples < 0 else max_samples
    print(
        f"dispatching to Modal: {dataset_path} -> {out_path}, "
        f"sweep={sweep}, cap={cap}"
    )
    out_bytes = run_spike.remote(
        src, sweep, aligned_model, base_model, cap,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(out_bytes)
    print(f"wrote {out_path} ({len(out_bytes):,} bytes)")
    print(out_bytes.decode("utf-8"))

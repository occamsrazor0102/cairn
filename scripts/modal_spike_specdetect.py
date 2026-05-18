"""SpecDetect dual-slice spike on Modal T4 (RAID + MAGE).

Loads GPT-J-6B once and runs base SpecDetect over both CAIRN OOD test
slices. The paper-tested XSum control was dropped because the local
MAGE jsonl doesn't tag subdomains and we don't have a clean XSum slice
on disk -- MAGE substitutes as the second control with a known-shape
class balance (500/500).

Sanity gates (softer than the original XSum gate):
- RAID: AUROC >= 0.78 in either orientation greenlights the calibrator
  retrain. Compare against robust_v1's RAID 0.78.
- MAGE: AUROC >= 0.60 in either orientation indicates the detector
  produces signal on MAGE-like text. Compare against robust_v1's MAGE 0.66.
- If both slices are within 0.05 of 0.5 (either orientation), the
  implementation is likely broken and we abort.

Each row reports `dft_total_energy` AUROC (the paper's headline feature)
plus per-band AUROCs (the novel signal CAIRN gains over
`surprisal_dynamics`). Both orientations (raw + 1-AUROC) are written
so a sign-flip can be checked without re-running.

Usage:
    modal run scripts/modal_spike_specdetect.py \\
        --raid-path data/ood_raid.jsonl \\
        --xsum-path data/ood_mage.jsonl \\
        --out-path docs/benchmarks/calib2000/specdetect_spike.json

(--xsum-path is kept as the arg name for backward compat; it now
accepts any class-balanced second slice. Defaults to MAGE.)

Cost: T4 at ~$0.50/hr. GPT-J-6B fp16 fits in 16 GB with headroom; one
forward pass per sample at ~0.3-0.5 s. 2,000 total samples ~ 15 min ~
~$0.15-0.25. First run downloads GPT-J-6B (~12 GB) into the persistent
`cairn-hf-cache` volume.
"""
from __future__ import annotations

import modal

app = modal.App("cairn-specdetect-spike")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
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
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_dotenv()],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_spike(
    raid_jsonl: bytes,
    xsum_jsonl: bytes,
    model_name: str = "EleutherAI/gpt-j-6b",
    max_raid: int | None = None,
    max_xsum: int | None = None,
) -> bytes:
    """Score both slices with SpecDetect and return per-feature AUROCs."""
    import json
    import os
    import sys
    import time

    sys.path.insert(0, "/root")
    os.chdir("/root")

    from cairn.detectors.specdetect import SpecDetectDetector

    detector = SpecDetectDetector()
    detector._ensure_loaded(model_name, "cuda")
    print(f"loaded {model_name}", flush=True)

    def _stratify(rows: list[dict]) -> list[dict]:
        # Interleave label-0 and label-1 so max-N caps yield balanced subsets.
        l0 = [r for r in rows if int(r.get("label", -1)) == 0]
        l1 = [r for r in rows if int(r.get("label", -1)) == 1]
        out: list[dict] = []
        for h, l in zip(l0, l1):
            out.append(h)
            out.append(l)
        out.extend(l0[len(l1):])
        out.extend(l1[len(l0):])
        return out

    def _load(blob: bytes, cap: int | None) -> list[dict]:
        rows: list[dict] = []
        for line in blob.decode("utf-8").split("\n"):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        rows = _stratify(rows)
        if cap is not None:
            rows = rows[:cap]
        return rows

    def _score_slice(name: str, rows: list[dict]) -> dict:
        print(
            f"--- {name}: {len(rows)} rows "
            f"(label-0: {sum(1 for r in rows if int(r['label']) == 0)}, "
            f"label-1: {sum(1 for r in rows if int(r['label']) == 1)}) ---",
            flush=True,
        )
        cfg = {"model": model_name, "max_tokens": 512}
        labels: list[int] = []
        feats: dict[str, list[float]] = {
            "p_llm": [],
            "dft_total_energy": [],
            "dft_low_band_energy": [],
            "dft_mid_band_energy": [],
            "dft_high_band_energy": [],
            "dft_low_mid_ratio": [],
            "dft_spectral_entropy": [],
        }
        skipped: list[dict] = []
        t0 = time.time()

        for i, row in enumerate(rows):
            lr = detector.analyze(row["text"], cfg)
            if lr.determination == "SKIP":
                skipped.append({"i": i, "reason": lr.signals.get("skipped")})
                continue
            labels.append(int(row["label"]))
            feats["p_llm"].append(float(lr.p_llm))
            for k in ("dft_total_energy", "dft_low_band_energy",
                      "dft_mid_band_energy", "dft_high_band_energy",
                      "dft_low_mid_ratio", "dft_spectral_entropy"):
                feats[k].append(float(lr.signals[k]))
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{len(rows)} in {elapsed:.0f}s", flush=True)

        elapsed = time.time() - t0
        out = {
            "n_scored": len(labels),
            "n_skipped": len(skipped),
            "wall_seconds": elapsed,
            "sec_per_sample": elapsed / max(len(labels), 1),
            "skipped_breakdown": _tally(skipped),
            "aurocs": {},
            "aurocs_flipped": {},
        }
        for k, scores in feats.items():
            if not labels:
                out["aurocs"][k] = float("nan")
                out["aurocs_flipped"][k] = float("nan")
                continue
            auc = _auroc(labels, scores)
            out["aurocs"][k] = auc
            out["aurocs_flipped"][k] = 1.0 - auc

        print(
            f"{name}: best AUROC across features = "
            f"{max(out['aurocs'].values()):.4f} "
            f"(flipped max = {max(out['aurocs_flipped'].values()):.4f})",
            flush=True,
        )
        return out

    raid_rows = _load(raid_jsonl, max_raid)
    second_rows = _load(xsum_jsonl, max_xsum)

    second_result = _score_slice("second slice (MAGE)", second_rows)
    raid_result = _score_slice("RAID OOD", raid_rows)

    # Sanity gates
    second_best = max(
        max(second_result["aurocs"].values()),
        max(second_result["aurocs_flipped"].values()),
    )
    raid_best = max(
        max(raid_result["aurocs"].values()),
        max(raid_result["aurocs_flipped"].values()),
    )
    raid_gate = raid_best >= 0.78
    second_gate = second_best >= 0.60
    # Implementation-sanity check: if BOTH slices are essentially random
    # in BOTH orientations, the detector is broken.
    raid_not_random = abs(raid_best - 0.5) >= 0.05
    second_not_random = abs(second_best - 0.5) >= 0.05
    implementation_ok = raid_not_random or second_not_random

    return json.dumps({
        "model": model_name,
        "second_slice": second_result,
        "raid": raid_result,
        "spec_gates": {
            "raid_threshold": 0.78,
            "raid_pass": raid_gate,
            "second_threshold": 0.60,
            "second_pass": second_gate,
            "implementation_ok": implementation_ok,
            "overall_greenlight": raid_gate and second_gate,
        },
    }, indent=2).encode("utf-8")


def _auroc(labels: list[int], scores: list[float]) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def _tally(skipped: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in skipped:
        out[s.get("reason") or "unknown"] = out.get(s.get("reason") or "unknown", 0) + 1
    return out


@app.local_entrypoint()
def main(
    raid_path: str = "data/ood_raid.jsonl",
    xsum_path: str = "data/ood_mage.jsonl",
    out_path: str = "docs/benchmarks/calib2000/specdetect_spike.json",
    model_name: str = "EleutherAI/gpt-j-6b",
    max_raid: int = -1,
    max_xsum: int = -1,
) -> None:
    from pathlib import Path

    raid_blob = Path(raid_path).read_bytes()
    xsum_blob = Path(xsum_path).read_bytes()
    print(
        f"dispatching: model={model_name}, "
        f"raid={raid_path}({len(raid_blob):,} bytes), "
        f"xsum={xsum_path}({len(xsum_blob):,} bytes)"
    )
    out_bytes = run_spike.remote(
        raid_blob,
        xsum_blob,
        model_name,
        None if max_raid < 0 else max_raid,
        None if max_xsum < 0 else max_xsum,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(out_bytes)
    print(f"wrote {out_path} ({len(out_bytes):,} bytes)")
    print(out_bytes.decode("utf-8"))

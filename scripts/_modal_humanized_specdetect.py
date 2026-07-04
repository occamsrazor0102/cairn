"""Measure SpecDetect (GPT-J-6B) on the humanized-prompt slice, on Modal T4.

Scores data/_humanized_eval.jsonl (102 humanized LLM prompts label=1 + 400 human
prompts >=50w label=0) and reports, for p_llm and each DFT feature: AUC (both
orientations) and TPR at 5/10/20% FPR. Tells us whether SpecDetect's spectral
signal carries the humanized slice where the heuristic ensemble was ~chance.

Usage:  modal run scripts/_modal_humanized_specdetect.py --data-path data/_humanized_eval.jsonl
Cost: T4 ~$0.20 for ~500 prompts.
"""
from __future__ import annotations

import modal

app = modal.App("cairn-humanized-specdetect")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "scikit-learn", "numpy", "pyyaml", "huggingface_hub")
    .add_local_dir("cairn", remote_path="/root/cairn")
    .add_local_dir("configs", remote_path="/root/configs")
)
hf_cache = modal.Volume.from_name("cairn-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="T4", timeout=3600,
              secrets=[modal.Secret.from_dotenv()],
              volumes={"/root/.cache/huggingface": hf_cache})
def run(data_jsonl: bytes, model_name: str = "EleutherAI/gpt-j-6b") -> bytes:
    import json
    import os
    import sys
    import time
    sys.path.insert(0, "/root")
    os.chdir("/root")
    from sklearn.metrics import roc_auc_score
    from cairn.detectors.specdetect import SpecDetectDetector

    det = SpecDetectDetector()
    det._ensure_loaded(model_name, "cuda")
    print(f"loaded {model_name}", flush=True)

    rows = [json.loads(l) for l in data_jsonl.decode("utf-8").split("\n") if l.strip()]
    cfg = {"model": model_name, "max_tokens": 512}
    FEATS = ["p_llm", "dft_total_energy", "dft_low_band_energy", "dft_mid_band_energy",
             "dft_high_band_energy", "dft_low_mid_ratio", "dft_spectral_entropy"]
    labels, words = [], []
    feats = {k: [] for k in FEATS}
    skip = {"pos": 0, "neg": 0}
    t0 = time.time()
    for i, r in enumerate(rows):
        lr = det.analyze(r["text"], cfg)
        if lr.determination == "SKIP":
            skip["pos" if r["label"] == 1 else "neg"] += 1
            continue
        labels.append(int(r["label"])); words.append(len(r["text"].split()))
        feats["p_llm"].append(float(lr.p_llm))
        for k in FEATS[1:]:
            feats[k].append(float(lr.signals.get(k, float("nan"))))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)} in {time.time()-t0:.0f}s", flush=True)

    def tpr_at_fpr(scores, labs, fpr, flip=False):
        s = [-x for x in scores] if flip else scores
        neg = sorted(v for v, l in zip(s, labs) if l == 0)
        if not neg:
            return float("nan")
        thr = neg[max(0, min(len(neg) - 1, int(__import__("math").ceil((1 - fpr) * len(neg))) - 1))]
        pos = [v for v, l in zip(s, labs) if l == 1]
        return sum(1 for v in pos if v > thr) / len(pos) if pos else float("nan")

    out = {"model": model_name, "n_scored": len(labels),
           "n_pos": sum(labels), "n_neg": len(labels) - sum(labels),
           "skipped": skip, "features": {},
           "per_prompt": {"label": labels, "words": words,
                          "dft_total_energy": feats["dft_total_energy"],
                          "p_llm": feats["p_llm"]}}
    for k in FEATS:
        sc = feats[k]
        if len(set(labels)) < 2 or not sc:
            continue
        auc = roc_auc_score(labels, sc)
        best_flip = auc < 0.5
        out["features"][k] = {
            "auc": auc, "auc_oriented": max(auc, 1 - auc),
            "tpr@5": tpr_at_fpr(sc, labels, 0.05, best_flip),
            "tpr@10": tpr_at_fpr(sc, labels, 0.10, best_flip),
            "tpr@20": tpr_at_fpr(sc, labels, 0.20, best_flip),
        }
    return json.dumps(out, indent=2).encode("utf-8")


@app.local_entrypoint()
def main(data_path: str = "data/_humanized_eval.jsonl",
         out_path: str = "scripts/_modal_specdetect_humanized_results.json",
         model_name: str = "EleutherAI/gpt-j-6b") -> None:
    from pathlib import Path
    blob = Path(data_path).read_bytes()
    print(f"dispatching SpecDetect on {data_path} ({len(blob):,} bytes)")
    out = run.remote(blob, model_name)
    Path(out_path).write_bytes(out)
    print(out.decode("utf-8"))
    print(f"\nwrote {out_path}")

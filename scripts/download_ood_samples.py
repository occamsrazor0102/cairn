"""Download balanced OOD test samples from M4, RAID, and MAGE.

Each dataset has its own schema; this script normalizes them to BEET
EvalSample JSONL with 500 human + 500 LLM rows per dataset (configurable
via --per-label). Reads HF_TOKEN from .env via cairn.dotenv.

Usage:
    python scripts/download_ood_samples.py
    python scripts/download_ood_samples.py --per-label 1000 --out-dir data/ood
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import EvalSample, save_dataset


_MIN_TEXT_LEN = 50


def _sample_m4(per_label: int, seed: int) -> list[EvalSample]:
    """M4 dev split: id, source, sub_source, lang, model, label, text.
    label: 0 = human, 1 = AI.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd
    p = hf_hub_download(
        "Jinyan1/COLING_2025_MGT_en",
        "data/dev-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(p, columns=["id", "model", "label", "text", "sub_source"])
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.len() >= _MIN_TEXT_LEN]
    human_pool = df[df["label"] == 0]
    llm_pool = df[df["label"] == 1]
    h = human_pool.sample(min(per_label, len(human_pool)), random_state=seed)
    l = llm_pool.sample(min(per_label, len(llm_pool)), random_state=seed)

    samples: list[EvalSample] = []
    for _, row in h.iterrows():
        sub = row.get("sub_source") or "unknown"
        samples.append(EvalSample(
            id=f"m4_{row['id']}", text=row["text"].strip(),
            label=0, tier="human", source=f"m4_{sub}",
            source_id=str(row["id"]),
        ))
    for _, row in l.iterrows():
        sub = row.get("sub_source") or "unknown"
        samples.append(EvalSample(
            id=f"m4_{row['id']}", text=row["text"].strip(),
            label=1, tier="A0", source=f"m4_{sub}",
            source_id=str(row["id"]),
        ))
    return samples


def _sample_raid(per_label: int, seed: int) -> list[EvalSample]:
    """RAID train.csv: id, adv_source_id, source_id, model, decoding,
    repetition_penalty, attack, domain, title, prompt, generation.
    model='human' marks human rows; everything else is an LLM.
    Filter to attack='none' so we test on clean LLM text, not adversarially
    perturbed text — that's a separate, harder evaluation.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd
    p = hf_hub_download("liamdugan/raid", "train.csv", repo_type="dataset")
    rng = random.Random(seed)

    human_rows: list[dict] = []
    llm_rows: list[dict] = []
    target = per_label * 5  # collect a buffer, sample down

    cols = ["id", "model", "attack", "domain", "generation"]
    for chunk in pd.read_csv(p, chunksize=20000, usecols=cols, dtype=str,
                             on_bad_lines="skip"):
        chunk = chunk[chunk["attack"].fillna("") == "none"]
        chunk = chunk.dropna(subset=["generation"])
        chunk = chunk[chunk["generation"].str.len() >= _MIN_TEXT_LEN]
        h = chunk[chunk["model"] == "human"]
        l = chunk[chunk["model"] != "human"]
        human_rows.extend(h.to_dict("records"))
        llm_rows.extend(l.to_dict("records"))
        if len(human_rows) >= target and len(llm_rows) >= target:
            break

    rng.shuffle(human_rows)
    rng.shuffle(llm_rows)

    samples: list[EvalSample] = []
    for r in human_rows[:per_label]:
        samples.append(EvalSample(
            id=f"raid_{r.get('id', '')}", text=r["generation"].strip(),
            label=0, tier="human",
            source=f"raid_{r.get('domain', 'unknown')}",
            source_id=str(r.get("id", "")),
        ))
    for r in llm_rows[:per_label]:
        samples.append(EvalSample(
            id=f"raid_{r.get('id', '')}", text=r["generation"].strip(),
            label=1, tier="A0",
            source=f"raid_{r.get('domain', 'unknown')}",
            source_id=str(r.get("id", "")),
        ))
    return samples


def _sample_mage(per_label: int, seed: int) -> list[EvalSample]:
    """MAGE test.csv schema is unconfirmed at write-time; tries common
    variants (text/label, generation/label, etc.). Raises a helpful error
    if the columns differ.
    """
    from huggingface_hub import hf_hub_download
    import pandas as pd
    p = hf_hub_download("yaful/MAGE", "test.csv", repo_type="dataset")
    df = pd.read_csv(p, on_bad_lines="skip")

    text_col = next((c for c in ("text", "generation", "content") if c in df.columns), None)
    label_col = next((c for c in ("label", "is_ai", "ai") if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"MAGE: could not find text/label columns. Available: {df.columns.tolist()}"
        )

    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.len() >= _MIN_TEXT_LEN]

    vals = sorted(df[label_col].dropna().unique())
    if len(vals) != 2:
        raise ValueError(f"MAGE: expected 2 label values, got {vals}")
    human_val, llm_val = vals[0], vals[1]

    h_pool = df[df[label_col] == human_val]
    l_pool = df[df[label_col] == llm_val]
    h = h_pool.sample(min(per_label, len(h_pool)), random_state=seed)
    l = l_pool.sample(min(per_label, len(l_pool)), random_state=seed)

    samples: list[EvalSample] = []
    for i, (_, row) in enumerate(h.iterrows()):
        samples.append(EvalSample(
            id=f"mage_h_{i}", text=str(row[text_col]).strip(),
            label=0, tier="human", source="mage",
        ))
    for i, (_, row) in enumerate(l.iterrows()):
        samples.append(EvalSample(
            id=f"mage_l_{i}", text=str(row[text_col]).strip(),
            label=1, tier="A0", source="mage",
        ))
    return samples


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--per-label", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--datasets", nargs="+",
                    default=["m4", "raid", "mage"],
                    choices=["m4", "raid", "mage"])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samplers = {"m4": _sample_m4, "raid": _sample_raid, "mage": _sample_mage}
    for name in args.datasets:
        print(f"\n=== {name} ===", flush=True)
        try:
            samples = samplers[name](args.per_label, args.seed)
            n_h = sum(1 for s in samples if s.label == 0)
            n_l = sum(1 for s in samples if s.label == 1)
            print(f"  collected {len(samples)} ({n_h} human / {n_l} llm)", flush=True)
            out = args.out_dir / f"ood_{name}.jsonl"
            save_dataset(samples, out)
            print(f"  wrote {out}", flush=True)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

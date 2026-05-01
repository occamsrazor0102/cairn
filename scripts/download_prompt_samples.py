"""Download authentic-provenance prompts for human-vs-LLM-prompt detection.

The detector is currently trained on human-vs-LLM RESPONSES. To extend
coverage to detecting LLM-authored PROMPTS, we need a balanced corpus
where authorship is iron-clad on both sides.

HUMAN prompts (target ~700): pre-LLM-era OR human-curated reasoning
sources where users were unambiguously writing without AI assistance.
  - HC3 questions: extracted from Reddit ELI5 / finance / medicine /
    wiki_csai / open_qa, vintage 2018-2022.
  - HH-RLHF: Anthropic 2022, pre-ChatGPT crowdworker prompts.
  - BBH (BIG-bench Hard): hand-curated reasoning challenges, 2023.
  - TruthfulQA: hand-crafted "stump the model" adversarial questions.
  - GPQA: graduate-level domain-expert-written science questions.

LLM-generated prompts (target ~600): documented synthetic instruction
datasets from self-instruct / evol chains.
  - Alpaca: GPT-3.5 self-instruct, Stanford 2023.
  - WizardLM Evol-Instruct V2: GPT-4 evolved.
  - Magpie: Llama-3-Instruct generated.

Output: EvalSample JSONL where source field is prefixed `prompt_*` to
distinguish from response-mode training data. The `text` field is the
prompt itself.

Usage:
    python scripts/download_prompt_samples.py --per-source 100
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import EvalSample, save_dataset


_HC3_FILES = {
    "finance": "finance.jsonl",
    "medicine": "medicine.jsonl",
    "open_qa": "open_qa.jsonl",
    "reddit_eli5": "reddit_eli5.jsonl",
    "wiki_csai": "wiki_csai.jsonl",
}


def _sample_hc3_questions(per_subset: int, seed: int) -> list[EvalSample]:
    """Extract real-user questions from HC3 raw files. Vintage 2018-2022."""
    from huggingface_hub import hf_hub_download
    rng = random.Random(seed)
    samples: list[EvalSample] = []
    for subset, fname in _HC3_FILES.items():
        path = hf_hub_download(
            repo_id="Hello-SimpleAI/HC3", filename=fname, repo_type="dataset",
        )
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                q = (row.get("question") or "").strip()
                if q and 20 <= len(q) <= 2000:
                    rows.append(q)
        rng.shuffle(rows)
        for i, q in enumerate(rows[:per_subset]):
            samples.append(EvalSample(
                id=f"prompt_hc3_{subset}_{i}", text=q,
                label=0, tier="human",
                source=f"prompt_hc3_{subset}",
            ))
    return samples


def _sample_hh_rlhf(target: int, seed: int) -> list[EvalSample]:
    """First human turn from each HH-RLHF conversation. Anthropic 2022."""
    from huggingface_hub import hf_hub_download
    import gzip
    rng = random.Random(seed)
    p = hf_hub_download(
        repo_id="Anthropic/hh-rlhf",
        filename="helpful-base/train.jsonl.gz",
        repo_type="dataset",
    )
    rows = []
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chosen = row.get("chosen", "")
            if "\n\nHuman: " not in chosen:
                continue
            after = chosen.split("\n\nHuman: ", 1)[1]
            prompt = after.split("\n\nAssistant:", 1)[0].strip()
            if 20 <= len(prompt) <= 2000:
                rows.append(prompt)
    rng.shuffle(rows)
    return [
        EvalSample(
            id=f"prompt_hh_rlhf_{i}", text=q,
            label=0, tier="human", source="prompt_hh_rlhf",
        )
        for i, q in enumerate(rows[:target])
    ]


def _sample_bbh(target: int, seed: int) -> list[EvalSample]:
    """BIG-bench Hard: hand-curated reasoning challenges."""
    from huggingface_hub import HfApi, hf_hub_download
    rng = random.Random(seed)
    api = HfApi()
    info = api.dataset_info("lukaemon/bbh")
    json_files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".json")]
    rows: list[str] = []
    for fname in json_files[:8]:  # sample from first 8 BBH subtasks
        try:
            p = hf_hub_download("lukaemon/bbh", fname, repo_type="dataset")
            data = json.loads(open(p, encoding="utf-8").read())
            examples = data.get("examples", [])
            for r in examples:
                q = (r.get("input") or "").strip()
                if q and 20 <= len(q) <= 2000:
                    rows.append(q)
        except Exception:
            continue
    rng.shuffle(rows)
    return [
        EvalSample(
            id=f"prompt_bbh_{i}", text=q,
            label=0, tier="human", source="prompt_bbh",
        )
        for i, q in enumerate(rows[:target])
    ]


def _sample_truthfulqa(target: int, seed: int) -> list[EvalSample]:
    """TruthfulQA: hand-crafted adversarial questions designed to stump LLMs."""
    from huggingface_hub import hf_hub_download
    rng = random.Random(seed)
    try:
        p = hf_hub_download(
            "truthfulqa/truthful_qa", "generation/validation-00000-of-00001.parquet",
            repo_type="dataset",
        )
        import pandas as pd
        df = pd.read_parquet(p, columns=["question"])
    except Exception:
        # fallback: alternative path
        p = hf_hub_download(
            "truthful_qa", "generation/validation-00000-of-00001.parquet",
            repo_type="dataset",
        )
        import pandas as pd
        df = pd.read_parquet(p, columns=["question"])
    qs = [str(q).strip() for q in df["question"].dropna().tolist()
          if 20 <= len(str(q).strip()) <= 2000]
    rng.shuffle(qs)
    return [
        EvalSample(
            id=f"prompt_truthfulqa_{i}", text=q,
            label=0, tier="human", source="prompt_truthfulqa",
        )
        for i, q in enumerate(qs[:target])
    ]


def _sample_gpqa(target: int, seed: int) -> list[EvalSample]:
    """GPQA: graduate-level domain-expert-written science questions."""
    from huggingface_hub import hf_hub_download
    rng = random.Random(seed)
    p = hf_hub_download(
        "Idavidrein/gpqa", "gpqa_main.csv", repo_type="dataset",
    )
    import pandas as pd
    df = pd.read_csv(p)
    text_col = next(
        (c for c in ("Question", "question", "Pre-Revision Question") if c in df.columns),
        None,
    )
    if text_col is None:
        raise ValueError(f"GPQA: no question column found. Cols: {df.columns.tolist()}")
    qs = [str(q).strip() for q in df[text_col].dropna().tolist()
          if 20 <= len(str(q).strip()) <= 2000]
    rng.shuffle(qs)
    return [
        EvalSample(
            id=f"prompt_gpqa_{i}", text=q,
            label=0, tier="human", source="prompt_gpqa",
        )
        for i, q in enumerate(qs[:target])
    ]


def _sample_alpaca(target: int, seed: int) -> list[EvalSample]:
    """Stanford Alpaca instructions. Generated by GPT-3.5 via self-instruct."""
    from huggingface_hub import hf_hub_download
    rng = random.Random(seed)
    p = hf_hub_download(
        "tatsu-lab/alpaca", "alpaca_data.json", repo_type="dataset",
    )
    data = json.loads(open(p, encoding="utf-8").read())
    instructions = []
    for r in data:
        instr = (r.get("instruction") or "").strip()
        inp = (r.get("input") or "").strip()
        full = instr + ("\n\n" + inp if inp else "")
        if 20 <= len(full) <= 2000:
            instructions.append(full)
    rng.shuffle(instructions)
    return [
        EvalSample(
            id=f"prompt_alpaca_{i}", text=q,
            label=1, tier="A0", source="prompt_alpaca",
        )
        for i, q in enumerate(instructions[:target])
    ]


def _sample_wizardlm(target: int, seed: int) -> list[EvalSample]:
    """WizardLM Evol-Instruct V2. GPT-4 evolved instructions."""
    from huggingface_hub import hf_hub_download
    rng = random.Random(seed)
    p = hf_hub_download(
        "WizardLMTeam/WizardLM_evol_instruct_V2_196k",
        "WizardLM_evol_instruct_V2_143k.json",
        repo_type="dataset",
    )
    data = json.loads(open(p, encoding="utf-8").read())
    rows = []
    for r in data:
        convs = r.get("conversations", [])
        if not convs:
            continue
        first = convs[0]
        if first.get("from") != "human":
            continue
        text = (first.get("value") or "").strip()
        if 20 <= len(text) <= 2000:
            rows.append(text)
    rng.shuffle(rows)
    return [
        EvalSample(
            id=f"prompt_wizardlm_{i}", text=q,
            label=1, tier="A0", source="prompt_wizardlm",
        )
        for i, q in enumerate(rows[:target])
    ]


def _sample_magpie(target: int, seed: int) -> list[EvalSample]:
    """Magpie. Llama-3-Instruct generated instructions."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    rng = random.Random(seed)
    p = hf_hub_download(
        "Magpie-Align/Magpie-Pro-300K-Filtered",
        "data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(p, columns=["instruction"])
    df = df.dropna(subset=["instruction"])
    df = df[df["instruction"].str.len().between(20, 2000)]
    df = df.sample(min(target, len(df)), random_state=seed)
    return [
        EvalSample(
            id=f"prompt_magpie_{i}", text=row["instruction"],
            label=1, tier="A0", source="prompt_magpie",
        )
        for i, (_, row) in enumerate(df.iterrows())
    ]


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--per-source", type=int, default=100,
                    help="rows from each source. HC3 takes per-subset (×5). "
                         "HH-RLHF takes 5×per-source. Default 100.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== HUMAN PROMPTS (provenance-clean: pre-LLM era + curated reasoning) ===", flush=True)
    human: list[EvalSample] = []
    sources_human = [
        ("hc3", lambda: _sample_hc3_questions(args.per_source, args.seed)),
        ("hh_rlhf", lambda: _sample_hh_rlhf(args.per_source * 5, args.seed)),
        ("bbh", lambda: _sample_bbh(args.per_source, args.seed)),
        ("truthfulqa", lambda: _sample_truthfulqa(args.per_source, args.seed)),
        ("gpqa", lambda: _sample_gpqa(args.per_source, args.seed)),
    ]
    for name, fn in sources_human:
        try:
            s = fn()
            print(f"  {name}: {len(s)} prompts", flush=True)
            human.extend(s)
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}", flush=True)

    print("\n=== LLM-GENERATED PROMPTS (known synthetic) ===", flush=True)
    llm: list[EvalSample] = []
    sources_llm = [
        ("alpaca", lambda: _sample_alpaca(args.per_source * 2, args.seed)),
        ("wizardlm", lambda: _sample_wizardlm(args.per_source * 2, args.seed)),
        ("magpie", lambda: _sample_magpie(args.per_source * 2, args.seed)),
    ]
    for name, fn in sources_llm:
        try:
            s = fn()
            print(f"  {name}: {len(s)} prompts", flush=True)
            llm.extend(s)
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(human); rng.shuffle(llm)
    n = min(len(human), len(llm))
    human = human[:n]
    llm = llm[:n]
    print(f"\nbalanced: {n} human + {n} LLM = {2 * n} prompts total", flush=True)

    save_dataset(human, args.out_dir / "prompts_human.jsonl")
    save_dataset(llm, args.out_dir / "prompts_llm.jsonl")
    combined = human + llm
    rng.shuffle(combined)
    save_dataset(combined, args.out_dir / "prompts_combined.jsonl")
    print(f"wrote {args.out_dir / 'prompts_combined.jsonl'}", flush=True)


if __name__ == "__main__":
    main()

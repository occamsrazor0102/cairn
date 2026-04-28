"""Build matched-mixed vs matched-single experimental datasets.

Tests the hypothesis: does training on the SAME prompts answered by
multiple providers (matched-mixed) generalize better than training on
more diverse prompts from a single provider (matched-single)?

Inputs (must already exist; produced by scripts/regenerate_llm_outputs.py):
  data/regen_matched_anthropic.jsonl
  data/regen_matched_openai.jsonl
  data/regen_matched_gemini.jsonl

Each input has the same human-side rows (HC3 humans for q1500-1999) but a
different LLM-side. We join by source_id, restrict to questions covered by
ALL three providers, then construct two training sets at identical sample
counts but different LLM-side diversity.

Outputs (split by question, seeded for reproducibility):
  data/matched_mixed_train.jsonl   — N_train questions; each has 1 LLM rotated
                                     across providers (Anthropic / OpenAI /
                                     Gemini in repeating order). 2*N_train rows.
  data/matched_mixed_calib.jsonl   — N_calib questions, same rotation.
  data/matched_mixed_test.jsonl    — N_test questions, all 3 LLMs each.

  data/matched_single_train.jsonl  — same N_train questions; LLM is always
                                     the chosen provider (default: anthropic).
  data/matched_single_calib.jsonl  — same N_calib questions; one provider only.
  data/matched_single_test.jsonl   — IDENTICAL test set to matched_mixed_test
                                     (same held-out questions × 3 LLMs each).

Both profiles are evaluated on identical held-out data, so any difference in
test-set metrics reflects training-side composition, not test-side noise.

Usage:
    python scripts/build_matched_experiment.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from cairn.evaluation.dataset import EvalSample, save_dataset


_PROVIDERS = ["anthropic", "openai", "gemini"]


def _load_matched(provider: str) -> dict[str, dict]:
    """Load one matched file. Returns {source_id: {"human": row, "llm": row}}."""
    path = Path(f"data/regen_matched_{provider}.jsonl")
    if not path.exists():
        raise SystemExit(f"missing input: {path}. Run scripts/regenerate_llm_outputs.py first.")
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = row.get("source_id")
        if not sid:
            continue
        bucket = out.setdefault(sid, {})
        bucket["human" if row["label"] == 0 else "llm"] = row
    return out


def _row_to_sample(row: dict, source_override: str | None = None) -> EvalSample:
    return EvalSample(
        id=row["id"],
        text=row["text"],
        label=int(row["label"]),
        tier=row.get("tier"),
        source=source_override or row.get("source"),
        source_id=row.get("source_id"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--calib-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--single-provider", choices=_PROVIDERS, default="anthropic",
                    help="provider used as the LLM side in the single-provider arm "
                         "(default: anthropic)")
    args = ap.parse_args()

    by_provider = {p: _load_matched(p) for p in _PROVIDERS}

    # Restrict to questions covered by all three providers (with both human + llm rows).
    common_qids = sorted(
        set(by_provider["anthropic"].keys())
        & set(by_provider["openai"].keys())
        & set(by_provider["gemini"].keys())
    )
    common_qids = [
        sid for sid in common_qids
        if all("human" in by_provider[p][sid] and "llm" in by_provider[p][sid] for p in _PROVIDERS)
    ]
    print(f"common qids covered by all 3 providers: {len(common_qids)}")

    rng = random.Random(args.seed)
    rng.shuffle(common_qids)
    n = len(common_qids)
    n_train = int(n * args.train_frac)
    n_calib = int(n * args.calib_frac)
    train_q = common_qids[:n_train]
    calib_q = common_qids[n_train:n_train + n_calib]
    test_q = common_qids[n_train + n_calib:]
    print(f"split: train={len(train_q)} calib={len(calib_q)} test={len(test_q)}")

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    def _build_mixed(qids: list[str], split_name: str) -> list[EvalSample]:
        samples: list[EvalSample] = []
        for idx, sid in enumerate(qids):
            provider = _PROVIDERS[idx % 3]  # rotate
            tag = f"matched_mixed_{split_name}"
            samples.append(_row_to_sample(by_provider[provider][sid]["human"], tag))
            samples.append(_row_to_sample(by_provider[provider][sid]["llm"], tag))
        return samples

    def _build_single(qids: list[str], split_name: str) -> list[EvalSample]:
        provider = args.single_provider
        samples: list[EvalSample] = []
        for sid in qids:
            tag = f"matched_single_{provider}_{split_name}"
            samples.append(_row_to_sample(by_provider[provider][sid]["human"], tag))
            samples.append(_row_to_sample(by_provider[provider][sid]["llm"], tag))
        return samples

    def _build_test(qids: list[str]) -> list[EvalSample]:
        """Identical for both arms — 1 human + 3 LLMs per question."""
        samples: list[EvalSample] = []
        for sid in qids:
            human_row = by_provider["anthropic"][sid]["human"]
            samples.append(_row_to_sample(human_row, "matched_test_human"))
            for provider in _PROVIDERS:
                llm_row = by_provider[provider][sid]["llm"]
                # Suffix id so the same qid_l doesn't collide across providers.
                samples.append(EvalSample(
                    id=f"{llm_row['id']}_{provider}",
                    text=llm_row["text"],
                    label=int(llm_row["label"]),
                    tier=llm_row.get("tier"),
                    source=f"matched_test_{provider}",
                    source_id=llm_row.get("source_id"),
                ))
        return samples

    save_dataset(_build_mixed(train_q, "train"), out_dir / "matched_mixed_train.jsonl")
    save_dataset(_build_mixed(calib_q, "calib"), out_dir / "matched_mixed_calib.jsonl")
    test_rows = _build_test(test_q)
    save_dataset(test_rows, out_dir / "matched_mixed_test.jsonl")

    save_dataset(_build_single(train_q, "train"), out_dir / "matched_single_train.jsonl")
    save_dataset(_build_single(calib_q, "calib"), out_dir / "matched_single_calib.jsonl")
    save_dataset(test_rows, out_dir / "matched_single_test.jsonl")

    print(f"wrote 6 split files to {out_dir}/ matched_{{mixed,single}}_{{train,calib,test}}.jsonl")
    print(f"  train: {len(train_q) * 2} rows ({len(train_q)} questions × human+llm)")
    print(f"  calib: {len(calib_q) * 2} rows ({len(calib_q)} questions × human+llm)")
    print(f"  test:  {len(test_rows)} rows ({len(test_q)} questions × 1 human + 3 LLMs)")


if __name__ == "__main__":
    main()

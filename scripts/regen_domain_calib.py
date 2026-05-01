"""Generate domain-specific calibration pairs from GSM8K + HumanEval + arXiv.

Unlike scripts/regenerate_llm_outputs.py which is tied to HC3, this
script accepts arbitrary (prompt, human_answer) pairs from datasets
where both sides exist:

  * GSM8K: math word problems with chain-of-thought solutions
  * HumanEval: Python function prompts with canonical implementations
  * arXiv: scientific abstracts (paired with LLM-regen abstracts of
    the same paper title)

Round-robin distributes the prompts across the configured providers so
each provider answers a balanced subset (default: 6 providers x ~860
prompts each = ~5,164 calls total). Each call writes a paired (human,
llm) JSONL row to a per-provider output file.

Usage:
    python scripts/regen_domain_calib.py \\
        --gsm8k 2500 --humaneval 164 --arxiv 2500 \\
        --providers anthropic openai gemini mistral deepseek moonshot \\
        --concurrency 8 --max-tokens 2000 \\
        --out-dir data
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import threading
import time
from pathlib import Path

from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import EvalSample


sys.path.insert(0, str(Path(__file__).parent))
from regenerate_llm_outputs import (  # noqa: E402
    _generate_anthropic, _generate_openai, _generate_gemini,
    _generate_mistral, _generate_openai_compat, _resolve_api_key,
)


_PROVIDER_DEFAULTS = {
    "anthropic":  {"model": "claude-sonnet-4-6"},
    "openai":     {"model": "gpt-5.4-mini"},
    "gemini":     {"model": "gemini-3-flash-preview"},
    "mistral":    {"model": "mistral-medium-latest"},
    "deepseek":   {"model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1",
                   "api_key_env": "DEEPSEEK_API_KEY"},
    "moonshot":   {"model": "kimi-k2.6", "base_url": "https://api.moonshot.ai/v1",
                   "api_key_env": "MOONSHOT_API_KEY"},
}


def _generate(provider: str, question: str, api_key: str, model: str, max_tokens: int,
              base_url: str | None = None) -> tuple[str, dict]:
    if provider == "anthropic":
        return _generate_anthropic(question, api_key, model, max_tokens)
    if provider == "openai":
        return _generate_openai(question, api_key, model, max_tokens)
    if provider == "gemini":
        return _generate_gemini(question, api_key, model, max_tokens)
    if provider == "mistral":
        return _generate_mistral(question, api_key, model, max_tokens)
    if provider in ("deepseek", "moonshot"):
        if not base_url:
            raise ValueError(f"{provider} requires base_url")
        return _generate_openai_compat(question, api_key, model, max_tokens, base_url)
    raise ValueError(f"unknown provider: {provider}")


def _load_gsm8k(target: int, seed: int) -> list[tuple[str, str, str, str]]:
    """Returns (qid, question, human_answer, source). 2500 from train."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    rng = random.Random(seed)
    p = hf_hub_download("openai/gsm8k", "main/train-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p, columns=["question", "answer"]).dropna()
    rows = list(df.itertuples(index=False))
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows[:target]):
        q = str(r.question).strip()
        a = str(r.answer).strip()
        if 20 <= len(q) <= 2000 and 20 <= len(a) <= 4000:
            out.append((f"gsm8k_{i}", q, a, "gsm8k"))
    return out


def _load_humaneval(target: int, seed: int) -> list[tuple[str, str, str, str]]:
    from huggingface_hub import hf_hub_download
    import pandas as pd
    rng = random.Random(seed)
    p = hf_hub_download(
        "openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df = pd.read_parquet(p, columns=["prompt", "canonical_solution"]).dropna()
    rows = list(df.itertuples(index=False))
    rng.shuffle(rows)
    out = []
    for i, r in enumerate(rows[:target]):
        q = str(r.prompt).strip()
        a = str(r.canonical_solution).strip()
        if 20 <= len(q) <= 2000 and 20 <= len(a) <= 4000:
            out.append((f"humaneval_{i}", q, a, "humaneval"))
    return out


def _load_arxiv(target: int, seed: int) -> list[tuple[str, str, str, str]]:
    """arXiv abstracts. Pair = (title-as-prompt, abstract-as-human-answer).

    gfissore/arxiv-abstracts-2021 ships as arxiv-abstracts.jsonl.gz —
    one JSON record per line with at least 'title' and 'abstract' keys.
    """
    from huggingface_hub import hf_hub_download
    import gzip
    rng = random.Random(seed)
    p = hf_hub_download(
        "gfissore/arxiv-abstracts-2021", "arxiv-abstracts.jsonl.gz",
        repo_type="dataset",
    )
    rows: list[tuple[str, str]] = []
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = (r.get("title") or "").strip().replace("\n", " ")
            abstract = (r.get("abstract") or "").strip()
            if 20 <= len(title) <= 500 and 100 <= len(abstract) <= 4000:
                rows.append((title, abstract))
            # the file is millions of rows; take a generous superset to
            # sample from without scanning the whole thing
            if len(rows) >= max(target * 5, 20000):
                break
    rng.shuffle(rows)
    out = []
    for i, (title, abstract) in enumerate(rows[:target]):
        prompt = (f"Write a scientific abstract (~150-300 words) for a paper "
                  f"titled: '{title}'.")
        out.append((f"arxiv_{i}", prompt, abstract, "arxiv"))
    return out


def _process_one(provider: str, model: str, base_url: str | None,
                 max_tokens: int, api_key: str,
                 prompt_tuple: tuple[str, str, str, str]) -> tuple[str, list[EvalSample], str]:
    qid, question, human_answer, source = prompt_tuple
    try:
        llm_text, _usage = _generate(provider, question, api_key, model, max_tokens, base_url)
    except Exception as e:
        return ("failed", [], f"{type(e).__name__}: {str(e)[:200]}")
    if not llm_text or not llm_text.strip():
        return ("empty", [], "model returned no text")
    source_tag = f"domain_{source}_regen_{model}"
    rows = [
        EvalSample(id=f"{qid}_h", text=human_answer, label=0, tier="human",
                   source=source_tag, source_id=qid),
        EvalSample(id=f"{qid}_l", text=llm_text.strip(), label=1, tier="A0",
                   source=source_tag, source_id=qid),
    ]
    return ("ok", rows, "")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gsm8k", type=int, default=2500)
    ap.add_argument("--humaneval", type=int, default=164)
    ap.add_argument("--arxiv", type=int, default=2500)
    ap.add_argument("--providers", nargs="+",
                    default=list(_PROVIDER_DEFAULTS.keys()))
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== sourcing prompts ===", flush=True)
    all_prompts: list[tuple[str, str, str, str]] = []
    if args.gsm8k > 0:
        s = _load_gsm8k(args.gsm8k, args.seed)
        print(f"  gsm8k: {len(s)}")
        all_prompts.extend(s)
    if args.humaneval > 0:
        s = _load_humaneval(args.humaneval, args.seed)
        print(f"  humaneval: {len(s)}")
        all_prompts.extend(s)
    if args.arxiv > 0:
        s = _load_arxiv(args.arxiv, args.seed)
        print(f"  arxiv: {len(s)}")
        all_prompts.extend(s)
    rng = random.Random(args.seed)
    rng.shuffle(all_prompts)
    print(f"  total: {len(all_prompts)} prompts", flush=True)

    # Each provider gets the FULL prompt set (not round-robin).
    by_provider: dict[str, list] = {p: list(all_prompts) for p in args.providers}
    print()
    for p in args.providers:
        print(f"  {p}: {len(by_provider[p])} prompts assigned (full fan-out)", flush=True)

    write_lock = threading.Lock()
    grand_ok = grand_failed = grand_empty = 0
    for provider in args.providers:
        prompts = by_provider[provider]
        if not prompts:
            continue
        cfg = _PROVIDER_DEFAULTS[provider]
        api_env = cfg.get("api_key_env") or f"{provider.upper()}_API_KEY"
        try:
            api_key = _resolve_api_key(provider, env_override=api_env)
        except SystemExit as e:
            print(f"  SKIP {provider}: {e}", flush=True)
            continue
        model = cfg["model"]
        base_url = cfg.get("base_url")
        out_path = args.out_dir / f"calib_v6_{provider}.jsonl"

        # Resume: skip prompts whose source_id has already been generated.
        # This lets a crashed/aborted run pick up where it left off.
        done_ids: set[str] = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("label") == 1 and r.get("source_id"):
                        done_ids.add(r["source_id"])
        if done_ids:
            before = len(prompts)
            prompts = [p for p in prompts if p[0] not in done_ids]
            print(f"  resuming: {len(done_ids)} already done, {len(prompts)} remaining (was {before})",
                  flush=True)

        print(f"\n=== {provider} ({len(prompts)} prompts -> {out_path}) ===", flush=True)
        t0 = time.monotonic()
        n_ok = n_failed = n_empty = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            future_to = {
                ex.submit(_process_one, provider, model, base_url, args.max_tokens,
                          api_key, t): t
                for t in prompts
            }
            with open(out_path, "a", encoding="utf-8") as out_f:
                for fut in concurrent.futures.as_completed(future_to):
                    status, rows, msg = fut.result()
                    if status == "ok":
                        with write_lock:
                            for r in rows:
                                out_f.write(json.dumps(
                                    {k: v for k, v in r.__dict__.items() if v is not None}
                                ) + "\n")
                        n_ok += 1
                    elif status == "failed":
                        n_failed += 1
                    elif status == "empty":
                        n_empty += 1
        dt = time.monotonic() - t0
        print(f"  ok={n_ok} failed={n_failed} empty={n_empty}  {dt:.1f}s", flush=True)
        grand_ok += n_ok; grand_failed += n_failed; grand_empty += n_empty

    print(f"\n=== TOTAL: ok={grand_ok} failed={grand_failed} empty={grand_empty} ===", flush=True)


if __name__ == "__main__":
    main()

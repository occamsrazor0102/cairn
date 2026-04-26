"""Regenerate the LLM side of an HC3 subset using a current model.

HC3's `chatgpt_answers` are early-2023 gpt-3.5 outputs. Detectors
calibrated against them learn "ChatGPT-circa-2023" patterns, not
modern-LLM patterns. This script keeps HC3's real human answers (the
human side stays human) and replaces the LLM side with fresh
generations from a current Anthropic OR OpenAI model.

Output is BEET EvalSample JSONL — feeds straight into
`scripts/calibrate_detectors.py` / `train_fusion.py` /
`calibrate_conformal.py` via their `--dataset` arg.

The prompt is intentionally minimal:
    System: You are a helpful assistant. Answer the user's question
            accurately and helpfully.
    User:   <HC3 question, verbatim>

No "be concise / be human / write in style X" — those would teach the
detector to spot prompt-engineering artifacts instead of native model
output. Mirrors how a real user would query Claude.ai or ChatGPT.

The script is RESUMABLE: it appends to --out and skips questions that
already have a generated llm-side row. A crash or rate-limit doesn't
waste credits.

Usage (dry-run, ~$0.03):
    export ANTHROPIC_API_KEY=...
    python scripts/regenerate_llm_outputs.py \\
        --subset finance --max-questions 10 \\
        --provider anthropic --model claude-haiku-4-5-20251001 \\
        --out data/hc3_finance_regen_haiku.jsonl

Usage (full ~2000-question regeneration, ~$1.50 on Haiku):
    python scripts/regenerate_llm_outputs.py \\
        --subset all --max-questions 1000 \\
        --provider anthropic --model claude-haiku-4-5-20251001 \\
        --out data/hc3_all_regen_haiku.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from beet.evaluation.dataset import EvalSample


_SUBSET_FILES = {
    "all": "all.jsonl",
    "finance": "finance.jsonl",
    "medicine": "medicine.jsonl",
    "open_qa": "open_qa.jsonl",
    "reddit_eli5": "reddit_eli5.jsonl",
    "wiki_csai": "wiki_csai.jsonl",
}

_SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question accurately and helpfully."


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from `.env` at the repo root into os.environ.

    Existing env vars take precedence (an explicit `export` wins over
    .env). No-op if .env doesn't exist. Hand-rolled to avoid a hard
    dependency on python-dotenv.

    Format:
      # comments and blank lines OK
      KEY=value
      KEY="value with spaces"
      KEY='value'
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


def _resolve_api_key(provider: str) -> str:
    env_var = f"{provider.upper()}_API_KEY"
    key = os.environ.get(env_var)
    if not key:
        raise SystemExit(
            f"missing {env_var} in environment. "
            f"either `export {env_var}=...` in your shell, "
            f"or copy .env.example -> .env and put the key there."
        )
    return key


def _load_existing(out_path: Path) -> tuple[list[dict], set[str]]:
    """Return (existing_rows, set_of_completed_question_ids).

    A question is 'completed' if its llm-side row exists in the file.
    """
    if not out_path.exists():
        return [], set()
    rows: list[dict] = []
    done: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append(r)
        if r.get("label") == 1 and r.get("source_id"):
            done.add(r["source_id"])
    return rows, done


def _generate_anthropic(question: str, api_key: str, model: str, max_tokens: int) -> tuple[str, dict]:
    import anthropic  # type: ignore
    client = anthropic.Anthropic(api_key=api_key)
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": _SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": question}],
    }
    # Sonnet/Opus 4.x can do extended thinking; explicitly disable so we
    # capture clean answer output (and don't pay for hidden reasoning).
    # Haiku doesn't support the thinking parameter and would 400.
    if any(t in model for t in ("sonnet-4", "opus-4")):
        kwargs["thinking"] = {"type": "disabled"}
    msg = client.messages.create(**kwargs)
    text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
    usage = {
        "input_tokens": getattr(msg.usage, "input_tokens", 0),
        "output_tokens": getattr(msg.usage, "output_tokens", 0),
    }
    return text, usage


def _generate_openai(
    question: str, api_key: str, model: str, max_tokens: int,
    prompt_id: str | None = None,
    prompt_version: str = "1",
    template_variable: str | None = None,
) -> tuple[str, dict]:
    """Two modes:

    * `prompt_id` set  → use the OpenAI Responses API with a saved
      prompt template (the one you built in platform.openai.com →
      Prompts). The HC3 question is passed either as `input` or
      assigned to a named variable on the template, depending on
      `template_variable`.
    * `prompt_id` None → fall back to chat.completions with the
      minimal "you are a helpful assistant" system prompt baked in.
    """
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=api_key)

    if prompt_id is not None:
        prompt_arg: dict = {"id": prompt_id, "version": prompt_version}
        kwargs: dict = {"prompt": prompt_arg, "max_output_tokens": max_tokens}
        if template_variable:
            prompt_arg["variables"] = {template_variable: question}
        else:
            kwargs["input"] = question
        resp = client.responses.create(**kwargs)
        # Responses API exposes a convenience `output_text` that joins all
        # text segments; fall back to walking `output` if that's missing.
        text = getattr(resp, "output_text", None)
        if text is None:
            parts: list[str] = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
            text = "".join(parts)
        u = getattr(resp, "usage", None)
        usage = {
            "input_tokens": getattr(u, "input_tokens", 0) if u else 0,
            "output_tokens": getattr(u, "output_tokens", 0) if u else 0,
        }
        return text or "", usage

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    text = resp.choices[0].message.content or ""
    u = resp.usage
    usage = {
        "input_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
        "output_tokens": getattr(u, "completion_tokens", 0) if u else 0,
    }
    return text, usage


def _generate(provider: str, question: str, api_key: str, model: str, max_tokens: int,
              prompt_id: str | None = None,
              prompt_version: str = "1",
              template_variable: str | None = None,
              max_retries: int = 3) -> tuple[str, dict]:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                return _generate_anthropic(question, api_key, model, max_tokens)
            return _generate_openai(
                question, api_key, model, max_tokens,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                template_variable=template_variable,
            )
        except Exception as e:
            last_err = e
            backoff = 2 ** attempt
            print(f"  retry {attempt + 1}/{max_retries} after {backoff}s: {type(e).__name__}: {str(e)[:120]}",
                  file=sys.stderr)
            time.sleep(backoff)
    raise RuntimeError(f"all retries failed: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--subset", default="finance", choices=sorted(_SUBSET_FILES.keys()))
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="model id (anthropic: e.g. claude-haiku-4-5-20251001; openai: e.g. gpt-4o-mini)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-questions", type=int, default=10,
                    help="cap questions processed (default: 10 dry-run; set higher for production)")
    ap.add_argument("--start-question", type=int, default=0,
                    help="skip the first N HC3 rows before processing — use with multi-provider runs to "
                         "carve non-overlapping slices (e.g. provider A: --start-question 0 --max-questions 500, "
                         "provider B: --start-question 500 --max-questions 500). IDs use absolute row index, "
                         "so non-overlapping slices never collide.")
    ap.add_argument("--max-tokens", type=int, default=600,
                    help="max output tokens per generation (default: 600 ~= 450 words)")
    ap.add_argument("--openai-prompt-id", type=str, default=None,
                    help="when set, switches the OpenAI provider to the Responses API and uses the "
                         "saved prompt template with this ID (e.g. pmpt_abc123…). Ignored for Anthropic.")
    ap.add_argument("--openai-prompt-version", type=str, default="1",
                    help="version of the saved OpenAI prompt template (default: 1)")
    ap.add_argument("--openai-template-variable", type=str, default=None,
                    help="if your OpenAI prompt template uses {{name}} variable substitution, pass the "
                         "variable name here (e.g. 'question'). The HC3 question is bound to it. "
                         "If unset, the question is passed via the Responses API `input` parameter.")
    args = ap.parse_args()

    _load_dotenv()

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:
        raise SystemExit("requires `huggingface_hub`. install: pip install huggingface_hub") from e

    api_key = _resolve_api_key(args.provider)

    print(f"fetching Hello-SimpleAI/HC3/{_SUBSET_FILES[args.subset]} ...", file=sys.stderr)
    raw_path = hf_hub_download(
        repo_id="Hello-SimpleAI/HC3",
        filename=_SUBSET_FILES[args.subset],
        repo_type="dataset",
    )

    rows, done = _load_existing(args.out)
    if done:
        print(f"resuming: {len(done)} questions already generated in {args.out}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    source_tag = f"hc3_{args.subset}_regen_{args.model}"
    total_in = 0
    total_out = 0
    n_processed = 0
    n_skipped = 0

    with open(raw_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i < args.start_question:
                continue
            if n_processed >= args.max_questions:
                break
            row = json.loads(line)
            question = (row.get("question") or "").strip()
            human_answers = row.get("human_answers", []) or []
            if not question or not human_answers:
                n_skipped += 1
                continue

            qid = f"hc3_{args.subset}_q{i:06d}"
            if qid in done:
                continue

            human_text = next((a.strip() for a in human_answers if isinstance(a, str) and a.strip()), None)
            if not human_text:
                n_skipped += 1
                continue

            try:
                llm_text, usage = _generate(
                    args.provider, question, api_key, args.model, args.max_tokens,
                    prompt_id=args.openai_prompt_id,
                    prompt_version=args.openai_prompt_version,
                    template_variable=args.openai_template_variable,
                )
            except Exception as e:
                print(f"FAILED q{i}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
                continue

            total_in += usage["input_tokens"]
            total_out += usage["output_tokens"]
            n_processed += 1

            human_row = EvalSample(
                id=f"{qid}_h",
                text=human_text,
                label=0,
                tier="human",
                source=source_tag,
                source_id=qid,
            )
            llm_row = EvalSample(
                id=f"{qid}_l",
                text=llm_text.strip(),
                label=1,
                tier="A0",
                source=source_tag,
                source_id=qid,
            )

            print(
                f"q{i:06d} ok  in={usage['input_tokens']:>5} out={usage['output_tokens']:>5} "
                f"running_total in={total_in} out={total_out}",
                file=sys.stderr,
            )

            # Append incrementally so a crash doesn't waste credits.
            with open(args.out, "a", encoding="utf-8") as out_f:
                for r in (human_row, llm_row):
                    out_f.write(json.dumps({k: v for k, v in r.__dict__.items() if v is not None}) + "\n")

    final_rows, _ = _load_existing(args.out)
    print(
        f"\ndone. processed {n_processed} questions ({n_skipped} skipped/blank). "
        f"file now has {len(final_rows)} rows ({sum(1 for r in final_rows if r['label']==0)} human / "
        f"{sum(1 for r in final_rows if r['label']==1)} llm).",
        file=sys.stderr,
    )
    print(f"tokens this run: input={total_in:,} output={total_out:,}", file=sys.stderr)
    print(f"wrote/appended to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

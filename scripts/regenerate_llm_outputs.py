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
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import EvalSample


_SUBSET_FILES = {
    "all": "all.jsonl",
    "finance": "finance.jsonl",
    "medicine": "medicine.jsonl",
    "open_qa": "open_qa.jsonl",
    "reddit_eli5": "reddit_eli5.jsonl",
    "wiki_csai": "wiki_csai.jsonl",
}

_SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question accurately and helpfully."


def _resolve_api_key(provider: str, env_override: str | None = None) -> str:
    env_var = env_override or f"{provider.upper()}_API_KEY"
    key = os.environ.get(env_var)
    if not key:
        raise SystemExit(
            f"missing {env_var} in environment. "
            f"either `export {env_var}=...` in your shell, "
            f"or copy .env.example -> .env and put the key there."
        )
    return key


def _generate_gemini(question: str, api_key: str, model: str, max_tokens: int) -> tuple[str, dict]:
    """Generate via Google's google-genai SDK.

    thinking_level=HIGH for extended reasoning. No grounding tool —
    GoogleSearch grounding has its own per-day quota that's much
    smaller than the model's RPM/TPM budget, and the bare model run
    matches the tool-free conditioning used by Anthropic/OpenAI/etc.
    """
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)],
        )
    ]
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        max_output_tokens=max_tokens,
        system_instruction=_SYSTEM_PROMPT,
    )
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    text = getattr(resp, "text", None) or ""
    u = getattr(resp, "usage_metadata", None)
    usage = {
        "input_tokens": getattr(u, "prompt_token_count", 0) if u else 0,
        # Gemini reports candidates_token_count (final answer tokens) and
        # thoughts_token_count (reasoning); we sum both because billing
        # treats them together and we want the running total to be honest.
        "output_tokens": (
            (getattr(u, "candidates_token_count", 0) or 0) +
            (getattr(u, "thoughts_token_count", 0) or 0)
        ) if u else 0,
    }
    return text, usage


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
    """Two modes (both use the Responses API):

    * `prompt_id` set  → call with a saved prompt template (the one you
      built in platform.openai.com → Prompts). The HC3 question is
      passed either as `input` or assigned to a named variable on the
      template, depending on `template_variable`.
    * `prompt_id` None → call the raw model directly with reasoning
      effort=medium, verbosity=medium, and the helpful-assistant
      `instructions`. Mirrors how a real API user invokes a reasoning
      model (gpt-5.x family).
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
    else:
        resp = client.responses.create(
            model=model,
            input=question,
            instructions=_SYSTEM_PROMPT,
            text={"format": {"type": "text"}, "verbosity": "medium"},
            reasoning={"effort": "medium", "summary": "auto"},
            max_output_tokens=max_tokens,
            store=True,
            tools=[],
        )
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


def _generate_mistral(
    question: str, api_key: str, model: str, max_tokens: int,
) -> tuple[str, dict]:
    """Generate via Mistral SDK's beta Conversations API.

    Uses `client.beta.conversations.start()` per the user-provided
    reference. The system prompt is passed via the top-level
    `instructions` arg (Conversations API equivalent of `system`).
    """
    try:
        from mistralai import Mistral  # type: ignore
    except ImportError:  # older path
        from mistralai.client import Mistral  # type: ignore
    client = Mistral(api_key=api_key)
    response = client.beta.conversations.start(
        inputs=[{"role": "user", "content": question}],
        model=model,
        instructions=_SYSTEM_PROMPT,
        completion_args={"temperature": 0.89, "max_tokens": max_tokens, "top_p": 1},
        tools=[],
    )
    parts: list[str] = []
    for out in getattr(response, "outputs", None) or []:
        if isinstance(out, dict):
            t = out.get("type")
            if t not in ("message.output", "message.text", None):
                continue
            content = out.get("content")
        else:
            t = getattr(out, "type", None)
            if t not in ("message.output", "message.text", None):
                continue
            content = getattr(out, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for chunk in content:
                if isinstance(chunk, dict):
                    if chunk.get("type") in (None, "text"):
                        txt = chunk.get("text")
                    else:
                        txt = None
                else:
                    if getattr(chunk, "type", "text") == "text":
                        txt = getattr(chunk, "text", None)
                    else:
                        txt = None
                if isinstance(txt, str):
                    parts.append(txt)
    text = "".join(parts)
    u = getattr(response, "usage", None)
    if u is None and isinstance(response, dict):
        u = response.get("usage")
    if isinstance(u, dict):
        usage = {
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
        }
    else:
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
            "output_tokens": getattr(u, "completion_tokens", 0) if u else 0,
        }
    return text, usage


def _generate_openai_compat(
    question: str, api_key: str, model: str, max_tokens: int, base_url: str,
) -> tuple[str, dict]:
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    raw = resp.choices[0].message.content or ""
    if isinstance(raw, list):
        text = "".join(
            block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            for block in raw
            if (isinstance(block, dict) and block.get("type") == "text")
            or (hasattr(block, "type") and getattr(block, "type") == "text")
        )
    else:
        text = raw
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
              base_url: str | None = None,
              max_retries: int = 3) -> tuple[str, dict]:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            if provider == "anthropic":
                return _generate_anthropic(question, api_key, model, max_tokens)
            if provider == "gemini":
                return _generate_gemini(question, api_key, model, max_tokens)
            if provider == "mistral":
                return _generate_mistral(question, api_key, model, max_tokens)
            if provider == "openai-compat":
                if not base_url:
                    raise ValueError("openai-compat provider requires --base-url")
                return _generate_openai_compat(question, api_key, model, max_tokens, base_url)
            return _generate_openai(
                question, api_key, model, max_tokens,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                template_variable=template_variable,
            )
        except Exception as e:
            last_err = e
            # Jitter spreads concurrent-thread retries so a transient rate-
            # limit doesn't dogpile. Without it, all N workers retry in
            # lockstep and re-trigger the same 429.
            backoff = (2 ** attempt) + random.uniform(0, 1.5)
            print(f"  retry {attempt + 1}/{max_retries} after {backoff:.1f}s: {type(e).__name__}: {str(e)[:120]}",
                  file=sys.stderr)
            time.sleep(backoff)
    raise RuntimeError(f"all retries failed: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--subset", default="finance", choices=sorted(_SUBSET_FILES.keys()))
    ap.add_argument("--provider",
                    choices=["anthropic", "openai", "gemini", "mistral", "openai-compat"],
                    default="anthropic")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="model id. anthropic: e.g. claude-haiku-4-5-20251001 / claude-sonnet-4-6; "
                         "openai: e.g. gpt-4o-mini (or use --openai-prompt-id with a saved template); "
                         "gemini: e.g. gemini-3-flash-preview")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-questions", type=int, default=10,
                    help="cap questions processed (default: 10 dry-run; set higher for production)")
    ap.add_argument("--start-question", type=int, default=0,
                    help="skip the first N HC3 rows before processing — use with multi-provider runs to "
                         "carve non-overlapping slices (e.g. provider A: --start-question 0 --max-questions 500, "
                         "provider B: --start-question 500 --max-questions 500). IDs use absolute row index, "
                         "so non-overlapping slices never collide.")
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="max output tokens per generation (default: 4000). For Anthropic this caps "
                         "the message; for OpenAI Responses it caps reasoning+message combined, so "
                         "keep it generous when the template uses a reasoning model. Billing is per "
                         "generated token, not per cap.")
    ap.add_argument("--openai-prompt-id", type=str, default=None,
                    help="when set, switches the OpenAI provider to the Responses API and uses the "
                         "saved prompt template with this ID (e.g. pmpt_abc123…). Ignored for Anthropic.")
    ap.add_argument("--openai-prompt-version", type=str, default="1",
                    help="version of the saved OpenAI prompt template (default: 1)")
    ap.add_argument("--openai-template-variable", type=str, default=None,
                    help="if your OpenAI prompt template uses {{name}} variable substitution, pass the "
                         "variable name here (e.g. 'question'). The HC3 question is bound to it. "
                         "If unset, the question is passed via the Responses API `input` parameter.")
    ap.add_argument("--base-url", type=str, default=None,
                    help="base URL for openai-compat provider (e.g. https://api.deepseek.com/v1)")
    ap.add_argument("--api-key-env", type=str, default=None,
                    help="override which env var to read the API key from "
                         "(e.g. DEEPSEEK_API_KEY, KIMI_API_KEY). Default: {PROVIDER}_API_KEY.")
    ap.add_argument("--shuffle", action="store_true",
                    help="randomly sample questions instead of taking a sequential slice. "
                         "Use with --seed to get the same sample across multiple provider runs.")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for --shuffle (default: 42). Same seed = same questions.")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel API workers (default: 8). Bump higher if your provider tier "
                         "allows more RPM (e.g. 32 for tier-2 OpenAI, 64+ for higher tiers). Drop "
                         "to 1 for strictly sequential execution / debugging.")
    args = ap.parse_args()

    load_dotenv()

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:
        raise SystemExit("requires `huggingface_hub`. install: pip install huggingface_hub") from e

    api_key = _resolve_api_key(args.provider, env_override=args.api_key_env)

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

    if args.openai_prompt_id and args.provider == "openai":
        # Template specifies its own model; --model is unused. Tag with a
        # short prompt-id so the source field identifies the corpus.
        source_tag = f"hc3_{args.subset}_regen_openai_{args.openai_prompt_id[:20]}"
    else:
        source_tag = f"hc3_{args.subset}_regen_{args.model}"

    # Phase 1: collect work items. Cheap (just file scan + dict lookups).
    all_eligible: list[tuple[str, int, str, str]] = []  # (qid, i, question, human_text)
    n_skipped = 0
    with open(raw_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if not args.shuffle:
                if i < args.start_question:
                    continue
                if i >= args.start_question + args.max_questions:
                    break
            row = json.loads(line)
            question = (row.get("question") or "").strip()
            human_answers = row.get("human_answers", []) or []
            if not question or not human_answers:
                n_skipped += 1
                continue
            human_text = next(
                (a.strip() for a in human_answers if isinstance(a, str) and a.strip()),
                None,
            )
            if not human_text:
                n_skipped += 1
                continue
            qid = f"hc3_{args.subset}_q{i:06d}"
            all_eligible.append((qid, i, question, human_text))

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(all_eligible)
        all_eligible = all_eligible[:args.max_questions]

    work_items = [item for item in all_eligible if item[0] not in done]

    if not work_items:
        print(
            f"nothing to do (resumed: {len(done)} done; skipped {n_skipped} blank/empty).",
            file=sys.stderr,
        )
        return

    # Phase 2: submit to a worker pool. Each task is one API call; result is
    # either two rows (human + llm) to append, or a status that gets logged
    # but not written. Output ordering is by completion, not submission —
    # downstream calibration reads by id, so order doesn't matter.
    write_lock = threading.Lock()
    counters_lock = threading.Lock()
    counters = {"in": 0, "out": 0, "ok": 0, "failed": 0, "empty": 0}

    def _process(qid: str, i: int, question: str, human_text: str) -> tuple[str, list[EvalSample], dict, str]:
        try:
            llm_text, usage = _generate(
                args.provider, question, api_key, args.model, args.max_tokens,
                prompt_id=args.openai_prompt_id,
                prompt_version=args.openai_prompt_version,
                template_variable=args.openai_template_variable,
                base_url=args.base_url,
            )
        except Exception as e:
            return ("failed", [], {"input_tokens": 0, "output_tokens": 0},
                    f"{type(e).__name__}: {str(e)[:200]}")
        if not llm_text or not llm_text.strip():
            return ("empty", [], usage, "model returned no text")
        rows = [
            EvalSample(id=f"{qid}_h", text=human_text, label=0, tier="human",
                       source=source_tag, source_id=qid),
            EvalSample(id=f"{qid}_l", text=llm_text.strip(), label=1, tier="A0",
                       source=source_tag, source_id=qid),
        ]
        return ("ok", rows, usage, "")

    print(
        f"submitting {len(work_items)} questions across {args.concurrency} workers "
        f"(resumed: {len(done)} done; skipped {n_skipped} blank/empty)",
        file=sys.stderr,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        future_to_qi = {
            ex.submit(_process, qid, i, q, h): (qid, i)
            for qid, i, q, h in work_items
        }
        for fut in concurrent.futures.as_completed(future_to_qi):
            qid, i = future_to_qi[fut]
            status, rows, usage, msg = fut.result()
            with counters_lock:
                counters["in"] += usage.get("input_tokens", 0)
                counters["out"] += usage.get("output_tokens", 0)
                if status == "ok":
                    counters["ok"] += 1
                elif status == "failed":
                    counters["failed"] += 1
                elif status == "empty":
                    counters["empty"] += 1
                snap = dict(counters)
            if status == "failed":
                print(f"FAILED q{i:06d}: {msg}", file=sys.stderr)
                continue
            if status == "empty":
                print(
                    f"EMPTY q{i:06d}: in={usage['input_tokens']} out={usage['output_tokens']} ({msg}). "
                    f"Try a higher --max-tokens or check the prompt template.",
                    file=sys.stderr,
                )
                continue
            # status == "ok"
            with write_lock:
                with open(args.out, "a", encoding="utf-8") as out_f:
                    for r in rows:
                        out_f.write(
                            json.dumps({k: v for k, v in r.__dict__.items() if v is not None}) + "\n"
                        )
            print(
                f"q{i:06d} ok  in={usage['input_tokens']:>5} out={usage['output_tokens']:>5} "
                f"({snap['ok']}/{snap['ok'] + snap['failed'] + snap['empty']} done, "
                f"running_total in={snap['in']} out={snap['out']})",
                file=sys.stderr,
            )

    n_processed = counters["ok"]
    n_attempted = counters["ok"] + counters["failed"] + counters["empty"]
    total_in = counters["in"]
    total_out = counters["out"]

    final_rows, _ = _load_existing(args.out)
    print(
        f"\ndone. attempted {n_attempted}, succeeded {n_processed} "
        f"({n_attempted - n_processed} failed, {n_skipped} skipped/blank). "
        f"file now has {len(final_rows)} rows ({sum(1 for r in final_rows if r['label']==0)} human / "
        f"{sum(1 for r in final_rows if r['label']==1)} llm).",
        file=sys.stderr,
    )
    print(f"tokens this run: input={total_in:,} output={total_out:,}", file=sys.stderr)
    print(f"wrote/appended to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

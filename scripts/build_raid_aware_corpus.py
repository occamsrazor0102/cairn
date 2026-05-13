# scripts/build_raid_aware_corpus.py
"""A2 step 3 — build the RAID-aware adversarial training slice.

Loads LLM-labeled rows from a source corpus (default
``data/calib2000_v6_train.jsonl``), applies the expanded transform set
defined in ``cairn/adversarial/transforms.py``, and writes the
adversarial-augmented JSONL with ``source="adversarial_raid_v1"``,
``tier="A2"``.

Defaults are **zero-spend, zero-download**:

- Only the seven pure-Python transforms run by default
  (``strip_preamble``, ``inject_typos``, ``casualize``, ``synonym_swap``,
  ``mix_human``, ``burstiness_injection``, ``emoji_noise``).
- ``translation_pivot`` is gated by ``--include-translation-pivot``
  (~1.5 GB NLLB-200 download on first run).
- The LM-driven attacks (``coached_casual``, ``paraphrase_launder``,
  ``grammar_normalize``, ``style_transfer``, ``summarization_attack``)
  are gated by ``--include-prompt-attacks`` + ``--provider`` (no provider
  means: skip with a stderr note).
- ``humanizer_tools`` is gated by ``--include-humanizer`` plus
  ``CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1`` (same gate as A1).

Provider-stratified sampling parses the source field for known provider
keywords (anthropic, openai, gpt, claude, gemini, deepseek, mistral,
moonshot, kimi). Rows whose source doesn't match fall into the
``unknown`` bucket — kept unless ``--skip-unknown-provider``.

Per-cell cap (``--max-per-provider-per-attack``) bounds output size and
spend. With 7 pure-Python attacks x 6 providers x 100 rows ~ 4200 rows
(the spec calls for ~15k total once prompt-category attacks are folded
in via a follow-on opt-in run).

This script writes the augmentation file only. It does NOT append to
``data/calib2000_v6_train.jsonl`` -- that's A2 step 4, a separate
decision.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

# `cairn.evaluation.dataset` must import before `cairn.adversarial.registry`
# to bootstrap the evaluation package -- same ordering issue documented in
# scripts/run_per_attack_baseline.py.
from cairn.dotenv import load_dotenv
from cairn.evaluation.dataset import EvalSample, load_dataset, save_dataset
from cairn.adversarial.generator import generate
from cairn.adversarial.registry import get_attack


PURE_PYTHON_ATTACKS = (
    "strip_preamble",
    "inject_typos",
    "casualize",
    "synonym_swap",
    "mix_human",
    "burstiness_injection",
    "emoji_noise",
)

# Local-model attack: NLLB-200 download on first run, then offline.
LOCAL_MODEL_ATTACKS = ("translation_pivot",)

# Provider-required attacks: each row triggers an LLM call.
PROMPT_ATTACKS = (
    "coached_casual",
    "paraphrase_launder",
    "grammar_normalize",
    "style_transfer",
    "summarization_attack",
)

# Commercial gate -- same env var as the A1 commercial adapters.
COMMERCIAL_ATTACKS = ("humanizer_tools",)

# Order matters: heuristic keyword match falls through left-to-right.
# "anthropic" before "claude" because the project labels regen_claude.jsonl
# rows as the anthropic provider (see scripts/regenerate_llm_outputs.py).
_PROVIDER_KEYWORDS = (
    ("anthropic", "anthropic"),
    ("claude", "anthropic"),
    ("openai", "openai"),
    ("gpt", "openai"),
    ("gemini", "gemini"),
    ("deepseek", "deepseek"),
    ("mistral", "mistral"),
    ("moonshot", "moonshot"),
    ("kimi", "moonshot"),
)


def _extract_provider(source: str | None) -> str:
    if not source:
        return "unknown"
    lower = source.lower()
    for needle, bucket in _PROVIDER_KEYWORDS:
        if needle in lower:
            return bucket
    return "unknown"


def _make_provider(name: str | None) -> Callable | None:
    if name is None:
        return None
    if name == "anthropic":
        from cairn.providers.anthropic import provider_callable
        return provider_callable()
    raise ValueError(f"unknown provider {name!r}; supported: anthropic")


def _resolve_attacks(
    *,
    include_translation: bool,
    include_prompt: bool,
    include_humanizer: bool,
    provider: Callable | None,
) -> list[str]:
    """Return the ordered list of attack names to run, filtered by gates.

    Each gate prints to stderr when it drops a category so the operator
    sees why the output is missing rows.
    """
    out = list(PURE_PYTHON_ATTACKS)
    if include_translation:
        out.extend(LOCAL_MODEL_ATTACKS)
    else:
        print(
            "[skip] translation_pivot (needs --include-translation-pivot)",
            file=sys.stderr,
        )
    if include_prompt:
        if provider is None:
            print(
                "[skip] prompt-category attacks (--include-prompt-attacks "
                "given but --provider missing)",
                file=sys.stderr,
            )
        else:
            out.extend(PROMPT_ATTACKS)
    else:
        print(
            "[skip] prompt-category attacks (needs --include-prompt-attacks "
            "+ --provider)",
            file=sys.stderr,
        )
    if include_humanizer:
        if os.environ.get("CAIRN_ALLOW_COMMERCIAL_BENCHMARK") != "1":
            print(
                "[skip] humanizer_tools (CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1 "
                "not set)",
                file=sys.stderr,
            )
        else:
            out.extend(COMMERCIAL_ATTACKS)
    else:
        print(
            "[skip] humanizer_tools (needs --include-humanizer)",
            file=sys.stderr,
        )
    # Drop any attacks whose registration is missing -- defends against
    # someone deleting an attack from transforms.py without updating the
    # constant tuples above.
    out = [name for name in out if get_attack(name) is not None]
    return out


def _stratified_sample(
    samples: list[EvalSample],
    *,
    per_provider: int,
    seed: int,
    skip_unknown_provider: bool,
) -> list[EvalSample]:
    """Return at most ``per_provider`` LLM-labeled rows per provider bucket."""
    rng = random.Random(seed)
    by_provider: dict[str, list[EvalSample]] = defaultdict(list)
    for s in samples:
        if s.label != 1:
            continue
        bucket = _extract_provider(s.source)
        if skip_unknown_provider and bucket == "unknown":
            continue
        by_provider[bucket].append(s)
    out: list[EvalSample] = []
    for bucket, rows in sorted(by_provider.items()):
        rng.shuffle(rows)
        out.extend(rows[:per_provider])
    return out


def build_corpus(
    *,
    source_path: Path,
    output_path: Path,
    attacks: list[str],
    per_provider_per_attack: int,
    seed: int,
    skip_unknown_provider: bool,
    provider: Callable | None,
) -> list[EvalSample]:
    """Run the augmentation. Returns the list of generated samples."""
    samples = load_dataset(source_path)
    seeds = _stratified_sample(
        samples,
        per_provider=per_provider_per_attack,
        seed=seed,
        skip_unknown_provider=skip_unknown_provider,
    )
    provider_buckets = sorted({_extract_provider(s.source) for s in seeds})
    print(
        f"[info] sampled {len(seeds)} LLM rows across "
        f"{len(provider_buckets)} provider buckets: {provider_buckets}",
        file=sys.stderr,
    )

    generated: list[EvalSample] = []
    for attack_name in attacks:
        slice_out = generate(seeds, [attack_name], provider=provider, seed=seed)
        # Re-tag for the calib2000 schema: source becomes the adversarial
        # corpus tag, tier becomes "A2"; original provenance is preserved
        # on attack_name / attack_category / source_id (already filled by
        # generate()).
        retagged = [
            EvalSample(
                id=s.id,
                text=s.text,
                label=s.label,
                tier="A2",
                source="adversarial_raid_v1",
                attack_name=s.attack_name,
                attack_category=s.attack_category,
                source_id=s.source_id,
            )
            for s in slice_out
        ]
        print(
            f"[info] {attack_name}: {len(retagged)} rows generated",
            file=sys.stderr,
        )
        generated.extend(retagged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(generated, output_path)
    print(
        f"[done] wrote {len(generated)} rows -> {output_path}",
        file=sys.stderr,
    )
    return generated


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--source",
        type=Path,
        default=Path("data/calib2000_v6_train.jsonl"),
        help="Source corpus to draw LLM-labeled rows from.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/adversarial_raid_v1.jsonl"),
        help="Output JSONL path.",
    )
    p.add_argument(
        "--max-per-provider-per-attack",
        type=int,
        default=100,
        help="Cap rows-per-provider-per-attack. Spec target ~15k total "
             "across 12 attacks x 6 providers (~208/cell).",
    )
    p.add_argument("--seed", type=int, default=20260513)
    p.add_argument("--skip-unknown-provider", action="store_true")
    p.add_argument("--include-translation-pivot", action="store_true")
    p.add_argument("--include-prompt-attacks", action="store_true")
    p.add_argument(
        "--provider",
        choices=("anthropic",),
        default=None,
        help="Required when --include-prompt-attacks is set.",
    )
    p.add_argument("--include-humanizer", action="store_true")
    args = p.parse_args(argv)

    provider = _make_provider(args.provider)
    attacks = _resolve_attacks(
        include_translation=args.include_translation_pivot,
        include_prompt=args.include_prompt_attacks,
        include_humanizer=args.include_humanizer,
        provider=provider,
    )
    if not attacks:
        print("[error] no attacks left to run", file=sys.stderr)
        return 2

    build_corpus(
        source_path=args.source,
        output_path=args.output,
        attacks=attacks,
        per_provider_per_attack=args.max_per_provider_per_attack,
        seed=args.seed,
        skip_unknown_provider=args.skip_unknown_provider,
        provider=provider,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Structural + behavioral tests for scripts/build_raid_aware_corpus.py.

AST inspection guards against drift of the argparse flag set and the
attack-class constants. A behavioral test drives the in-process
``build_corpus`` function against a tiny synthetic 3-provider corpus to
verify provider-stratified sampling, per-cell capping, and the output
JSONL schema match the calib2000 source-tag contract.

Pure imports only -- the script's ``if __name__ == '__main__'`` guard
keeps importlib safe. No ``exec_module`` or argparse monkeypatching.
"""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_raid_aware_corpus.py"


# --- AST inspection ---------------------------------------------------------

def _argparse_flags() -> set[str]:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        val = node.args[0].value
        if isinstance(val, str) and val.startswith("--"):
            flags.add(val)
    return flags


def _string_tuple_constant(name: str) -> tuple[str, ...] | None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name):
            continue
        if not isinstance(node.value, ast.Tuple):
            continue
        return tuple(
            elt.value for elt in node.value.elts
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
        )
    return None


def test_script_parses():
    ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))


def test_argparse_signature():
    flags = _argparse_flags()
    expected = {
        "--source", "--output", "--max-per-provider-per-attack", "--seed",
        "--skip-unknown-provider", "--include-translation-pivot",
        "--include-prompt-attacks", "--provider", "--include-humanizer",
        "--concurrency",
    }
    assert expected.issubset(flags), (
        f"missing argparse flags; got {flags}, need {expected}"
    )


def test_pure_python_attack_set_locked():
    # Spec calls out exactly these seven transforms as zero-API-spend.
    got = _string_tuple_constant("PURE_PYTHON_ATTACKS")
    assert got == (
        "strip_preamble", "inject_typos", "casualize", "synonym_swap",
        "mix_human", "burstiness_injection", "emoji_noise",
    ), got


def test_prompt_attack_set_locked():
    got = _string_tuple_constant("PROMPT_ATTACKS")
    assert got == (
        "coached_casual", "paraphrase_launder", "grammar_normalize",
        "style_transfer", "summarization_attack",
    ), got


def test_local_and_commercial_attack_sets():
    assert _string_tuple_constant("LOCAL_MODEL_ATTACKS") == ("translation_pivot",)
    assert _string_tuple_constant("COMMERCIAL_ATTACKS") == ("humanizer_tools",)


# --- behavioral tests -------------------------------------------------------

def _import_module():
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        return importlib.import_module("build_raid_aware_corpus")
    finally:
        sys.path.pop(0)


def test_extract_provider_keyword_mapping():
    mod = _import_module()
    cases = {
        "hc3_finance_regen_deepseek-chat": "deepseek",
        "hc3_finance_regen_anthropic-claude-sonnet-4-6": "anthropic",
        "hc3_finance_regen_claude-haiku": "anthropic",
        "hc3_finance_regen_openai-gpt-4.1": "openai",
        "hc3_finance_regen_gpt-4o": "openai",
        "hc3_finance_regen_gemini-2.5-pro": "gemini",
        "hc3_finance_regen_mistral-small": "mistral",
        "hc3_finance_regen_moonshot-v1": "moonshot",
        "hc3_finance_regen_kimi-k2": "moonshot",
        "hc3_finance_human_writer": "unknown",
        None: "unknown",
    }
    for source, expected in cases.items():
        assert mod._extract_provider(source) == expected, source


def test_stratified_sample_caps_per_provider():
    mod = _import_module()
    from cairn.evaluation.dataset import EvalSample
    samples = []
    # 3 providers x 5 LLM rows + 5 human rows (label=0) that must NOT be
    # picked.
    for provider in ("openai", "anthropic", "gemini"):
        for i in range(5):
            samples.append(EvalSample(
                id=f"{provider}_{i}", text=f"llm text {i}", label=1,
                source=f"hc3_finance_regen_{provider}-stub",
                source_id=f"{provider}_{i}",
            ))
    for i in range(5):
        samples.append(EvalSample(
            id=f"human_{i}", text=f"human text {i}", label=0,
            source="hc3_finance_human", source_id=f"human_{i}",
        ))
    out = mod._stratified_sample(
        samples, per_provider=2, seed=7, skip_unknown_provider=False,
    )
    assert len(out) == 6, [s.id for s in out]
    by_provider: dict[str, int] = {}
    for s in out:
        bucket = mod._extract_provider(s.source)
        by_provider[bucket] = by_provider.get(bucket, 0) + 1
    assert by_provider == {"anthropic": 2, "gemini": 2, "openai": 2}, by_provider
    assert all(s.label == 1 for s in out)


def test_build_corpus_writes_calib2000_schema(tmp_path):
    """End-to-end with pure-Python attacks: output JSONL matches the
    calib2000 source-tag contract and preserves provenance."""
    mod = _import_module()
    from cairn.evaluation.dataset import EvalSample, save_dataset

    seed_samples = [
        EvalSample(
            id=f"seed_{i}", text=("This is a sample sentence. " * 8).strip(),
            label=1,
            source=f"hc3_finance_regen_{provider}-stub",
            source_id=f"seed_{i}",
        )
        for i, provider in enumerate(["openai", "anthropic", "gemini"])
    ]
    src = tmp_path / "source.jsonl"
    save_dataset(seed_samples, src)
    out = tmp_path / "adversarial_raid_v1.jsonl"

    generated = mod.build_corpus(
        source_path=src,
        output_path=out,
        attacks=["strip_preamble", "inject_typos"],
        per_provider_per_attack=1,
        seed=11,
        skip_unknown_provider=False,
        provider=None,
    )

    # 3 providers x 2 attacks x 1 row = 6 generated rows
    assert len(generated) == 6, len(generated)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 6
    for row in rows:
        assert row["label"] == 1
        assert row["tier"] == "A2"
        assert row["source"] == "adversarial_raid_v1"
        assert row["attack_name"] in {"strip_preamble", "inject_typos"}
        assert row["attack_category"] == "transform"
        assert row["source_id"].startswith("seed_")
        assert row["id"].endswith("__" + row["attack_name"])


def test_resolve_attacks_default_is_pure_python_only():
    mod = _import_module()
    attacks = mod._resolve_attacks(
        include_translation=False,
        include_prompt=False,
        include_humanizer=False,
        provider=None,
    )
    assert tuple(attacks) == mod.PURE_PYTHON_ATTACKS


def test_resolve_attacks_prompt_without_provider_is_dropped():
    mod = _import_module()
    attacks = mod._resolve_attacks(
        include_translation=False,
        include_prompt=True,
        include_humanizer=False,
        provider=None,  # missing
    )
    # Prompt attacks must NOT appear when provider is None, even with the
    # --include-prompt-attacks flag.
    assert tuple(attacks) == mod.PURE_PYTHON_ATTACKS


def test_parallel_path_used_for_prompt_attacks_and_preserves_order(tmp_path, monkeypatch):
    """Prompt-category attacks must route through _generate_parallel
    when concurrency > 1, and output order must match input order
    (deterministic for a given seed). Uses the registered prompt attack
    'paraphrase_launder' with a fake provider so no real API calls fire.
    """
    mod = _import_module()
    from cairn.evaluation.dataset import EvalSample, save_dataset

    # Fake provider: deterministic identity-ish function so we can assert
    # both that it was called and that order is preserved.
    def fake_provider(prompt: str) -> str:
        return f"PARAPHRASED::{prompt[-40:]}"

    seeds = [
        EvalSample(
            id=f"seed_{i}", text=f"row {i} content text",
            label=1, source=f"hc3_finance_regen_openai-stub-{i}",
            source_id=f"seed_{i}",
        )
        for i in range(8)
    ]
    src = tmp_path / "source.jsonl"
    save_dataset(seeds, src)
    out = tmp_path / "out.jsonl"

    # Capture both that _generate_parallel was invoked AND the seed
    # order it received -- order through the thread pool must match the
    # post-stratified-sample input order.
    parallel_args: list[dict] = []
    real_parallel = mod._generate_parallel

    def _spy(seeds_arg, attack, *, provider, seed, max_workers):
        parallel_args.append({
            "max_workers": max_workers,
            "seed_ids": [s.id for s in seeds_arg],
        })
        return real_parallel(
            seeds_arg, attack,
            provider=provider, seed=seed, max_workers=max_workers,
        )

    monkeypatch.setattr(mod, "_generate_parallel", _spy)

    generated = mod.build_corpus(
        source_path=src,
        output_path=out,
        attacks=["paraphrase_launder"],
        per_provider_per_attack=8,
        seed=42,
        skip_unknown_provider=False,
        provider=fake_provider,
        concurrency=4,
    )

    # Parallel dispatch was taken (once, max_workers=4).
    assert len(parallel_args) == 1
    assert parallel_args[0]["max_workers"] == 4
    # Ordering preserved THROUGH the thread pool: outputs match the
    # order the spy saw on input. This is what ThreadPoolExecutor.map
    # guarantees and what the test exists to assert.
    expected_ids = [sid + "__paraphrase_launder" for sid in parallel_args[0]["seed_ids"]]
    assert [s.id for s in generated] == expected_ids
    # All rows have the expected attack metadata + transformed text shape.
    for s in generated:
        assert s.attack_name == "paraphrase_launder"
        assert s.attack_category == "prompt"
        assert s.text.startswith("PARAPHRASED::")


def test_concurrency_one_keeps_serial_path(tmp_path, monkeypatch):
    """concurrency=1 must NOT hit _generate_parallel even for prompt
    attacks -- it falls back to the existing generate() function."""
    mod = _import_module()
    from cairn.evaluation.dataset import EvalSample, save_dataset

    def fake_provider(prompt: str) -> str:
        return "OK"

    seeds = [EvalSample(
        id="seed_0", text="row 0 content text",
        label=1, source="hc3_finance_regen_openai-stub", source_id="seed_0",
    )]
    src = tmp_path / "source.jsonl"
    save_dataset(seeds, src)

    parallel_calls: list[int] = []
    monkeypatch.setattr(
        mod, "_generate_parallel",
        lambda *a, **k: parallel_calls.append(k.get("max_workers", -1)) or [],
    )

    mod.build_corpus(
        source_path=src,
        output_path=tmp_path / "out.jsonl",
        attacks=["paraphrase_launder"],
        per_provider_per_attack=1,
        seed=42,
        skip_unknown_provider=False,
        provider=fake_provider,
        concurrency=1,
    )
    assert parallel_calls == [], parallel_calls


def test_resolve_attacks_humanizer_requires_env_gate(monkeypatch):
    mod = _import_module()
    monkeypatch.delenv("CAIRN_ALLOW_COMMERCIAL_BENCHMARK", raising=False)
    attacks = mod._resolve_attacks(
        include_translation=False,
        include_prompt=False,
        include_humanizer=True,  # flag set
        provider=None,
    )
    assert "humanizer_tools" not in attacks
    monkeypatch.setenv("CAIRN_ALLOW_COMMERCIAL_BENCHMARK", "1")
    attacks_env = mod._resolve_attacks(
        include_translation=False,
        include_prompt=False,
        include_humanizer=True,
        provider=None,
    )
    assert "humanizer_tools" in attacks_env

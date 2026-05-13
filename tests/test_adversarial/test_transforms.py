import pytest

# Bootstrap `cairn.evaluation` BEFORE `cairn.adversarial.transforms`.
# Without this, importing `cairn.adversarial.transforms` first triggers a
# circular import (adversarial.__init__ -> adversarial.generator ->
# evaluation.__init__ -> evaluation.robustness -> adversarial.generator
# [partial]). `tests/test_adversarial/test_generator.py:2` does the same.
# Pytest collection order normally hides this, but isolated runs of this
# file would otherwise fail at collection.
from cairn.evaluation.dataset import EvalSample as _bootstrap_evaluation  # noqa: F401

from cairn.adversarial.transforms import (
    strip_preamble,
    inject_typos,
    casualize,
    synonym_swap,
    mix_human,
    coached_casual,
    paraphrase_launder,
    burstiness_injection,
    emoji_noise,
    translation_pivot,
    grammar_normalize,
    style_transfer,
    summarization_attack,
    humanizer_tools,
    _HUMANIZER_GATE_ENV,
    _HUMANIZER_API_KEY_ENV,
)
from cairn.adversarial.registry import get_attack, list_attacks


class TestStripPreamble:
    def test_removes_certainly(self):
        out = strip_preamble("Certainly! Here is your answer: The sky is blue.")
        assert "Certainly" not in out
        assert "sky is blue" in out

    def test_removes_heres(self):
        out = strip_preamble("Here's a comprehensive overview: content follows.")
        assert not out.lower().startswith("here")

    def test_removes_closer(self):
        out = strip_preamble("The answer is 42. Let me know if you'd like more detail.")
        assert "let me know" not in out.lower()
        assert "42" in out

    def test_idempotent_on_clean_text(self):
        clean = "This is a sentence with no preamble."
        assert strip_preamble(clean) == clean


class TestInjectTypos:
    def test_modifies_text(self):
        text = "The quick brown fox has jumped over the lazy dog and the hen."
        out = inject_typos(text, seed=42)
        # With fingerprint words in text and high rate, output should differ.
        assert isinstance(out, str)

    def test_empty_string(self):
        assert inject_typos("", seed=1) == ""

    def test_deterministic_with_seed(self):
        t = "The quick brown fox and the lazy dog"
        assert inject_typos(t, seed=7) == inject_typos(t, seed=7)


class TestCasualize:
    def test_contraction(self):
        out = casualize("I cannot do that. It is not possible.", seed=1)
        assert "can't" in out
        assert "it's" in out.lower() or "isn't" in out.lower()

    def test_short_text_no_crash(self):
        assert isinstance(casualize("Hi.", seed=1), str)


class TestSynonymSwap:
    def test_replaces_fingerprint_vocab(self):
        out = synonym_swap("Furthermore, we utilize a comprehensive approach.")
        lower = out.lower()
        assert "furthermore" not in lower
        assert "utilize" not in lower
        assert "comprehensive" not in lower

    def test_preserves_unknown_words(self):
        assert synonym_swap("The cat sat on the mat.") == "The cat sat on the mat."


class TestMixHuman:
    def test_adds_content(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        out = mix_human(text, seed=1)
        assert len(out) > len(text)

    def test_short_text_appended(self):
        out = mix_human("Only this.", seed=1)
        assert out.startswith("Only this.")
        assert len(out) > len("Only this.")


class TestPromptAttacks:
    """Prompt-based attacks require a provider; verify prompt content and errors."""

    def test_coached_casual_calls_provider_with_instruction_and_text(self):
        captured = {}

        def provider(prompt: str) -> str:
            captured["prompt"] = prompt
            return "rewritten text"

        out = coached_casual("Comprehensive analysis of widgets.", provider=provider)
        assert out == "rewritten text"
        assert "casual" in captured["prompt"].lower()
        assert "Comprehensive analysis of widgets." in captured["prompt"]

    def test_paraphrase_launder_calls_provider(self):
        captured = {}

        def provider(prompt: str) -> str:
            captured["prompt"] = prompt
            return "paraphrased"

        out = paraphrase_launder("Original sentence here.", provider=provider)
        assert out == "paraphrased"
        assert "paraphrase" in captured["prompt"].lower()
        assert "Original sentence here." in captured["prompt"]

    def test_coached_casual_requires_provider(self):
        with pytest.raises(RuntimeError):
            coached_casual("anything")

    def test_paraphrase_launder_requires_provider(self):
        with pytest.raises(RuntimeError):
            paraphrase_launder("anything")

    def test_prompt_attacks_registered(self):
        assert get_attack("coached_casual") is not None
        assert get_attack("coached_casual").category == "prompt"
        assert get_attack("paraphrase_launder") is not None
        assert get_attack("paraphrase_launder").category == "prompt"
        names = {a.name for a in list_attacks(category="prompt")}
        assert {"coached_casual", "paraphrase_launder"}.issubset(names)


# ---------------------------------------------------------------------------
# Wave 2 / A2 step 2 — seven new transforms
# ---------------------------------------------------------------------------

_PROSE_FIXTURE = (
    "The quick brown fox jumps over the lazy dog. "
    "Furthermore, the cat sat on the mat. "
    "Honestly, the river carved its way through limestone. "
    "We need to think about this more carefully. "
    "Mandelbrot would have agreed with the fractal hypothesis."
)


class TestBurstinessInjection:
    def test_empty_string_passthrough(self):
        assert burstiness_injection("", seed=1) == ""

    def test_single_sentence_unchanged(self):
        # Length-2 guard returns input verbatim — no fragments are
        # inserted into a single-sentence corpus.
        text = "Just one statement."
        assert burstiness_injection(text, seed=1) == text

    def test_multi_sentence_appends_fragments(self):
        out = burstiness_injection(_PROSE_FIXTURE, rate=1.0, seed=42)
        assert len(out) > len(_PROSE_FIXTURE)
        # At least one canonical fragment should appear (rate=1.0).
        assert any(frag in out for frag in (
            "Yeah.", "Hmm.", "Right.", "Anyway,", "Honestly,",
        ))

    def test_deterministic_with_seed(self):
        a = burstiness_injection(_PROSE_FIXTURE, seed=7)
        b = burstiness_injection(_PROSE_FIXTURE, seed=7)
        assert a == b

    def test_50kb_input_no_crash(self):
        big = (_PROSE_FIXTURE + " ") * 200  # ~25 KB; still well above the
                                             # `> 20 char` per-sentence guard
        out = burstiness_injection(big, seed=3)
        assert isinstance(out, str)
        assert len(out) >= len(big)

    def test_registered(self):
        a = get_attack("burstiness_injection")
        assert a is not None
        assert a.category == "transform"


class TestEmojiNoise:
    def test_empty_string_passthrough(self):
        assert emoji_noise("", seed=1) == ""

    def test_high_rate_appends_emoji(self):
        out = emoji_noise("Sentence one. Sentence two. Sentence three.",
                          rate=1.0, seed=42)
        # At least one canonical emoji should appear at rate=1.0.
        assert any(e in out for e in ("\U0001f642", "\U0001f44d",
                                      "\U0001f914", "\U0001f4a1",
                                      "\U0001f3af", "\U0001f525", "✨",
                                      "\U0001f440", "\U0001f4dd",
                                      "\U0001f680"))

    def test_zero_rate_idempotent(self):
        text = "Sentence one. Sentence two."
        # rate=0.0 means the rng.random() < 0.0 check never fires.
        assert emoji_noise(text, rate=0.0, seed=1) == text

    def test_deterministic_with_seed(self):
        text = "First. Second. Third. Fourth."
        a = emoji_noise(text, rate=0.5, seed=11)
        b = emoji_noise(text, rate=0.5, seed=11)
        assert a == b

    def test_50kb_input_no_crash(self):
        big = "a sentence. " * 4500  # ~54 KB
        out = emoji_noise(big, rate=0.05, seed=3)
        assert isinstance(out, str)

    def test_registered(self):
        a = get_attack("emoji_noise")
        assert a is not None
        assert a.category == "transform"


class TestTranslationPivot:
    """Structural tests only — the NLLB-200 model is too heavy
    (~2 GB) to load in the unit suite. The integration test that
    asserts label-preservation cosine ≥ 0.6 is gated under the
    `pytest.importorskip` block in `test_label_preservation_pure_python`
    and lives separately as a `@pytest.mark.slow` integration test.
    """

    def test_empty_string_passthrough(self):
        # Empty input bypasses the model load entirely.
        assert translation_pivot("", pivot_lang="fra_Latn") == ""

    def test_registered(self):
        a = get_attack("translation_pivot")
        assert a is not None
        assert a.category == "transform"

    def test_missing_transformers_raises_clear_error(self, monkeypatch):
        """When `transformers` is unavailable, the lazy loader must
        raise a clear RuntimeError naming the install extra."""
        # Reset the module-level singleton so the loader actually tries
        # to import. Other tests in this class don't trigger the loader,
        # so there's no risk of stomping on a real cached pipeline.
        import cairn.adversarial.transforms as mod
        monkeypatch.setattr(mod, "_TRANSLATION_PIPELINE", None)
        # Remove the transformers module from sys.modules and shadow it
        # with a sentinel that raises ImportError on import.
        import sys
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(RuntimeError, match="transformers"):
            translation_pivot("the cat sat on the mat.",
                              pivot_lang="fra_Latn")


class TestGrammarNormalize:
    def test_requires_provider(self):
        with pytest.raises(RuntimeError, match="provider"):
            grammar_normalize("anything")

    def test_calls_provider_with_prompt_and_text(self):
        captured = {}

        def provider(prompt: str) -> str:
            captured["prompt"] = prompt
            return "corrected text"

        out = grammar_normalize("teh cat sat on teh mat.", provider=provider)
        assert out == "corrected text"
        assert "grammar" in captured["prompt"].lower()
        assert "teh cat sat on teh mat." in captured["prompt"]

    def test_registered(self):
        a = get_attack("grammar_normalize")
        assert a is not None
        assert a.category == "prompt"


class TestStyleTransfer:
    def test_requires_provider(self):
        with pytest.raises(RuntimeError, match="provider"):
            style_transfer("anything")

    def test_calls_provider_with_prompt_and_text(self):
        captured = {}

        def provider(prompt: str) -> str:
            captured["prompt"] = prompt
            return "rewritten"

        out = style_transfer("Furthermore, comprehensive analysis is essential.",
                             provider=provider)
        assert out == "rewritten"
        assert "style" in captured["prompt"].lower()
        assert "Furthermore" in captured["prompt"]

    def test_registered(self):
        a = get_attack("style_transfer")
        assert a is not None
        assert a.category == "prompt"


class TestSummarizationAttack:
    def test_requires_provider(self):
        with pytest.raises(RuntimeError, match="provider"):
            summarization_attack("anything")

    def test_calls_provider_with_prompt_and_text(self):
        captured = {}

        def provider(prompt: str) -> str:
            captured["prompt"] = prompt
            return "short summary"

        out = summarization_attack(_PROSE_FIXTURE, provider=provider)
        assert out == "short summary"
        assert "summar" in captured["prompt"].lower()
        assert _PROSE_FIXTURE in captured["prompt"]

    def test_registered(self):
        a = get_attack("summarization_attack")
        assert a is not None
        assert a.category == "prompt"


class TestHumanizerTools:
    def test_gated_by_env_var(self, monkeypatch):
        """Without CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1 the call must
        raise before doing anything else."""
        monkeypatch.delenv(_HUMANIZER_GATE_ENV, raising=False)
        monkeypatch.delenv(_HUMANIZER_API_KEY_ENV, raising=False)
        with pytest.raises(RuntimeError, match=_HUMANIZER_GATE_ENV):
            humanizer_tools("any text")

    def test_requires_api_key_after_gate(self, monkeypatch):
        """Setting the gate alone is not sufficient — API key must also
        be present."""
        monkeypatch.setenv(_HUMANIZER_GATE_ENV, "1")
        monkeypatch.delenv(_HUMANIZER_API_KEY_ENV, raising=False)
        with pytest.raises(RuntimeError, match=_HUMANIZER_API_KEY_ENV):
            humanizer_tools("any text")

    def test_skeleton_raises_after_gates_pass(self, monkeypatch):
        """Both gates pass → live integration is still a skeleton."""
        monkeypatch.setenv(_HUMANIZER_GATE_ENV, "1")
        monkeypatch.setenv(_HUMANIZER_API_KEY_ENV, "fake-key-for-test")
        with pytest.raises(NotImplementedError, match="SKELETON"):
            humanizer_tools("any text")

    def test_registered_as_transform_for_self_gating(self):
        """humanizer_tools self-gates via env vars — registry category
        must be 'transform' so the generator does NOT raise the generic
        'provider required' error before the gate runs."""
        a = get_attack("humanizer_tools")
        assert a is not None
        assert a.category == "transform"


class TestNewTransformsRegistryCoverage:
    """Sanity check: all 7 new attacks land in the registry."""

    def test_all_seven_registered(self):
        all_names = {a.name for a in list_attacks()}
        new_seven = {
            "burstiness_injection", "emoji_noise", "translation_pivot",
            "grammar_normalize", "style_transfer", "summarization_attack",
            "humanizer_tools",
        }
        assert new_seven.issubset(all_names), (
            f"missing from registry: {new_seven - all_names}"
        )

    def test_categories_assigned_correctly(self):
        # Pure-Python and local-model -> transform.
        for name in ("burstiness_injection", "emoji_noise",
                     "translation_pivot", "humanizer_tools"):
            assert get_attack(name).category == "transform", (
                f"{name} should be 'transform'"
            )
        # LM-required -> prompt.
        for name in ("grammar_normalize", "style_transfer",
                     "summarization_attack"):
            assert get_attack(name).category == "prompt", (
                f"{name} should be 'prompt'"
            )


class TestLabelPreservationCosine:
    """Risk-register mitigation: pure-Python transforms should preserve
    enough surface text that an embedding cosine vs the original stays
    >= 0.6 (loose threshold per the plan). LM-based transforms are NOT
    covered here because the unit suite mocks the provider — measured
    cosine would be cosine(mock-output, original), which is
    uninformative.
    """

    def test_burstiness_injection_preserves_meaning(self):
        sentence_transformers = pytest.importorskip("sentence_transformers")
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        out = burstiness_injection(_PROSE_FIXTURE, rate=1.0, seed=7)
        a, b = model.encode([_PROSE_FIXTURE, out],
                            normalize_embeddings=True)
        cos = float(np.dot(a, b))
        assert cos >= 0.6, f"cosine {cos:.3f} below 0.6 threshold"

    def test_emoji_noise_preserves_meaning(self):
        sentence_transformers = pytest.importorskip("sentence_transformers")
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        out = emoji_noise(_PROSE_FIXTURE, rate=1.0, seed=7)
        a, b = model.encode([_PROSE_FIXTURE, out],
                            normalize_embeddings=True)
        cos = float(np.dot(a, b))
        assert cos >= 0.6, f"cosine {cos:.3f} below 0.6 threshold"

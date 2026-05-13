"""Programmatic post-processing adversarial transforms.

Each `transform`-category attack is a pure function: str -> str, no API calls.
`prompt`-category attacks require a `provider` callable (str -> str) injected by
the harness; they re-generate content via an LLM.
"""
from __future__ import annotations

import os
import random
import re
import threading
from typing import Any, Callable

from cairn.adversarial.registry import Attack, register


_PREAMBLE_OPENERS = (
    r"(?i)^\s*(certainly!?|sure!?|here(?:'s|\s+is)|of course!?|absolutely!?|"
    r"let\s+me|below\s+is|i'?d\s+be\s+happy\s+to|great\s+question!?)[^\n.!?:]*[.!?:]\s*"
)
_PREAMBLE_CLOSERS = (
    r"(?i)\s*(let\s+me\s+know\s+if\s+you'?d\s+like[^.!?]*[.!?]|"
    r"hope\s+this\s+helps[^.!?]*[.!?]|"
    r"feel\s+free\s+to\s+(?:ask|let\s+me\s+know)[^.!?]*[.!?])\s*$"
)


def strip_preamble(text: str, **_: object) -> str:
    """Remove common LLM opener/closer phrasing."""
    out = re.sub(_PREAMBLE_OPENERS, "", text, count=1)
    out = re.sub(_PREAMBLE_CLOSERS, "", out)
    return out.strip()


_TYPO_SWAPS = {"the": "teh", "and": "adn", "you": "yuo", "with": "wiht",
               "that": "taht", "this": "tihs", "have": "ahve", "from": "form"}


def inject_typos(text: str, *, rate: float = 0.15, seed: int | None = None, **_: object) -> str:
    """Introduce casual misspellings and occasional punctuation drops.

    The two branches have deliberately asymmetric multipliers on `rate`:
    - ``rate * 3`` for typo swaps, because typos only fire on the small
      fraction of words present in _TYPO_SWAPS — boosting here keeps the
      effective per-word typo rate close to the nominal ``rate``.
    - ``rate / 3`` for punctuation drops, because that branch fires on
      every word ending in `.` / `,` / `;` (a much larger population) —
      dampening keeps the punctuation-loss rate comparable.
    Both are empirical starting points; treat as knobs, not physics.
    """
    rng = random.Random(seed)
    words = text.split(" ")
    out = []
    for w in words:
        lw = w.lower()
        key = re.sub(r"[^a-z]", "", lw)
        if key in _TYPO_SWAPS and rng.random() < rate * 3:
            swapped = _TYPO_SWAPS[key]
            if w and w[0].isupper():
                swapped = swapped.capitalize()
            trailing = re.sub(r"[a-zA-Z]", "", w)
            out.append(swapped + trailing)
        elif rng.random() < rate / 3 and w.endswith((".", ",", ";")):
            out.append(w[:-1])
        else:
            out.append(w)
    return " ".join(out)


_CONTRACTIONS = [
    (r"\bdo not\b", "don't"),
    (r"\bcannot\b", "can't"),
    (r"\bis not\b", "isn't"),
    (r"\bwill not\b", "won't"),
    (r"\bit is\b", "it's"),
    (r"\bI am\b", "I'm"),
    (r"\byou are\b", "you're"),
    (r"\bthat is\b", "that's"),
    (r"\bare not\b", "aren't"),
    (r"\bhave not\b", "haven't"),
]
_FILLERS = [" honestly", " like", " basically", " kind of"]


def casualize(text: str, *, seed: int | None = None, **_: object) -> str:
    """Lower formality: contractions + sprinkled filler words."""
    rng = random.Random(seed)
    out = text
    for pat, repl in _CONTRACTIONS:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    # Occasionally splice a filler at sentence starts
    sentences = re.split(r"(?<=[.!?])\s+", out)
    for i in range(len(sentences)):
        if rng.random() < 0.25 and len(sentences[i]) > 10:
            filler = rng.choice(_FILLERS).strip()
            sentences[i] = sentences[i][0].lower() + sentences[i][1:]
            sentences[i] = f"{filler}, " + sentences[i]
    return " ".join(sentences)


_SYNONYMS = {
    "furthermore": "also",
    "moreover": "also",
    "comprehensive": "full",
    "utilize": "use",
    "commence": "start",
    "facilitate": "help",
    "numerous": "many",
    "demonstrate": "show",
    "implement": "do",
    "subsequently": "then",
}


def synonym_swap(text: str, **_: object) -> str:
    """Replace common LLM fingerprint vocabulary with plainer synonyms."""
    out = text
    for word, repl in _SYNONYMS.items():
        out = re.sub(rf"\b{word}\b", repl, out, flags=re.IGNORECASE)
    return out


_HUMAN_FRAGMENTS = [
    "honestly this took me three tries to get right",
    "anyway, moving on",
    "idk if this is what you wanted",
    "i'll circle back to this next week",
    "not sure that matters but whatever",
]


def mix_human(text: str, *, seed: int | None = None, **_: object) -> str:
    """Splice short human-style fragments into LLM text at random positions."""
    rng = random.Random(seed)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 2:
        return text + " " + rng.choice(_HUMAN_FRAGMENTS) + "."
    n_insertions = max(1, len(sentences) // 4)
    for _ in range(n_insertions):
        pos = rng.randint(1, len(sentences) - 1)
        sentences.insert(pos, rng.choice(_HUMAN_FRAGMENTS) + ".")
    return " ".join(sentences)


_COACHED_CASUAL_PROMPT = (
    "Rewrite the following text in a casual, conversational tone. "
    "Include occasional typos and informal language. "
    "Avoid numbered lists, bullet points, and words like 'ensure', "
    "'comprehensive', 'robust', or 'facilitate'. "
    "Make it sound like a real person wrote it quickly.\n\n"
)

_PARAPHRASE_LAUNDER_PROMPT = (
    "Paraphrase the following text completely. Use different sentence "
    "structures, vocabulary, and phrasing while preserving the meaning. "
    "Do not add any preamble.\n\n"
)


def coached_casual(
    text: str, *, provider: Callable[[str], str] | None = None, **_: object
) -> str:
    """Re-generate via LLM provider with casual-tone coaching prompt."""
    if provider is None:
        raise RuntimeError("coached_casual attack requires a 'provider' callable")
    return provider(_COACHED_CASUAL_PROMPT + text)


def paraphrase_launder(
    text: str, *, provider: Callable[[str], str] | None = None, **_: object
) -> str:
    """Re-generate via LLM provider with paraphrase instruction."""
    if provider is None:
        raise RuntimeError("paraphrase_launder attack requires a 'provider' callable")
    return provider(_PARAPHRASE_LAUNDER_PROMPT + text)


# Register all transforms
register(Attack(
    name="strip_preamble",
    category="transform",
    description="Remove 'Certainly!', 'Here's...' openers and closers",
    severity="basic",
    apply=strip_preamble,
))
register(Attack(
    name="inject_typos",
    category="transform",
    description="Random casual misspellings and missing punctuation",
    severity="basic",
    apply=inject_typos,
))
register(Attack(
    name="casualize",
    category="transform",
    description="Lower formality via contractions and filler words",
    severity="moderate",
    apply=casualize,
))
register(Attack(
    name="synonym_swap",
    category="transform",
    description="Replace fingerprint vocabulary with common synonyms",
    severity="moderate",
    apply=synonym_swap,
))
register(Attack(
    name="mix_human",
    category="transform",
    description="Splice human-style sentences into LLM text",
    severity="advanced",
    apply=mix_human,
))
register(Attack(
    name="coached_casual",
    category="prompt",
    description="LLM re-generates with casual tone coaching",
    severity="moderate",
    apply=coached_casual,
))
register(Attack(
    name="paraphrase_launder",
    category="prompt",
    description="LLM paraphrases to strip stylistic fingerprints",
    severity="advanced",
    apply=paraphrase_launder,
))


# ---------------------------------------------------------------------------
# Wave 2 / A2 step 2 — seven new transforms
# ---------------------------------------------------------------------------
# Two pure-Python (burstiness_injection, emoji_noise), one local-model
# (translation_pivot, NLLB-200), three provider-required (grammar_normalize,
# style_transfer, summarization_attack), and one gated commercial skeleton
# (humanizer_tools). See
# docs/superpowers/specs/2026-05-07-post-honest-calibration-roadmap.md
# Wave 2 step 2 for the rationale and per-attack mitigation register.

# --- burstiness_injection ---------------------------------------------------

_BURSTINESS_FRAGMENTS = (
    "Yeah.",
    "Hmm.",
    "Right.",
    "I think so.",
    "Anyway,",
    "Honestly,",
    "Wait.",
    "Sort of.",
)


def burstiness_injection(
    text: str, *, rate: float = 0.20, seed: int | None = None, **_: object
) -> str:
    """Increase sentence-length variance by interleaving short fragments.

    Targets perplexity / sentence-length-distribution detectors that flag
    uniformly-medium LLM cadence. Fragments are inserted *after* a
    sentence (not in the middle) to preserve readability and meaning.
    """
    rng = random.Random(seed)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) < 2:
        return text
    out: list[str] = []
    for s in sentences:
        out.append(s)
        # Only inject after meaningfully-long sentences so a haiku-length
        # corpus doesn't get drowned in fillers.
        if len(s) > 20 and rng.random() < rate:
            out.append(rng.choice(_BURSTINESS_FRAGMENTS))
    return " ".join(out)


# --- emoji_noise ------------------------------------------------------------

# Visible-but-low-frequency emojis. Restricted to Unicode that has been
# stable across emoji versions to avoid render quirks in test fixtures.
_EMOJIS = ("\U0001f642", "\U0001f44d", "\U0001f914", "\U0001f4a1",
           "\U0001f3af", "\U0001f525", "✨", "\U0001f440",
           "\U0001f4dd", "\U0001f680")


def emoji_noise(
    text: str, *, rate: float = 0.10, seed: int | None = None, **_: object
) -> str:
    """Append emojis at sentence boundaries.

    Targets fingerprint-vocab and surface-pattern detectors that key on a
    clean text-only stream. Emojis are appended *after* the terminal
    punctuation so detectors that segment on punctuation still see clean
    sentences.
    """
    rng = random.Random(seed)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for s in sentences:
        if not s.strip():
            continue
        if rng.random() < rate:
            s = s + " " + rng.choice(_EMOJIS)
        out.append(s)
    return " ".join(out)


# --- translation_pivot ------------------------------------------------------

# NLLB-200-distilled-600M is the smallest official NLLB checkpoint
# (~2 GB on disk). Round-trip translation strips most stylistic
# fingerprints — formality cues, function-word distributions,
# punctuation idioms — while preserving content. Lazy + thread-safe
# singleton so the model is loaded at most once per process.
_NLLB_DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"
_TRANSLATION_PIPELINE: tuple[Any, Any] | None = None
_TRANSLATION_LOCK = threading.Lock()


def _load_translation_pipeline(model_name: str) -> tuple[Any, Any]:
    """Return (model, tokenizer) for the requested NLLB checkpoint.

    Cached at module level — re-requesting the same model name returns
    the cached objects. Re-requesting a *different* model name returns
    the cached pair without reloading; pass `model_name=...` only on
    the first call to override the default.
    """
    global _TRANSLATION_PIPELINE
    if _TRANSLATION_PIPELINE is not None:
        return _TRANSLATION_PIPELINE
    with _TRANSLATION_LOCK:
        if _TRANSLATION_PIPELINE is not None:  # double-checked
            return _TRANSLATION_PIPELINE
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "translation_pivot requires `transformers` "
                "(install with `pip install 'cairn[tier2]'`)."
            ) from e
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _TRANSLATION_PIPELINE = (model, tokenizer)
        return _TRANSLATION_PIPELINE


def translation_pivot(
    text: str,
    *,
    pivot_lang: str = "fra_Latn",
    source_lang: str = "eng_Latn",
    model_name: str = _NLLB_DEFAULT_MODEL,
    max_tokens: int = 512,
    **_: object,
) -> str:
    """Two-hop translation (en → pivot → en) via NLLB-200.

    Default pivot is French (`fra_Latn`); override via `pivot_lang` for
    a different language pair. Pivot choice trades off label-preservation
    cosine (closer languages preserve more) against fingerprint-stripping
    strength (more distant languages strip more style).
    """
    if not text.strip():
        return text
    model, tok = _load_translation_pipeline(model_name)

    def _translate(src: str, src_lang: str, tgt_lang: str) -> str:
        tok.src_lang = src_lang
        ins = tok(src, return_tensors="pt", truncation=True, max_length=max_tokens)
        bos_id = tok.convert_tokens_to_ids(tgt_lang)
        out_ids = model.generate(
            **ins, forced_bos_token_id=bos_id, max_new_tokens=max_tokens,
        )
        return tok.batch_decode(out_ids, skip_special_tokens=True)[0]

    pivoted = _translate(text, source_lang, pivot_lang)
    return _translate(pivoted, pivot_lang, source_lang)


# --- grammar_normalize, style_transfer, summarization_attack ----------------

_GRAMMAR_NORMALIZE_PROMPT = (
    "Correct any grammar, punctuation, or spelling errors in the "
    "following text. Preserve the meaning and approximate length. "
    "Do not add any preamble or commentary; return only the corrected "
    "text.\n\n"
)

_STYLE_TRANSFER_PROMPT = (
    "Rewrite the following text in a different writing style — switch "
    "from formal to casual or vice versa. Preserve the factual content "
    "and the approximate length. Do not add preamble; return only the "
    "rewritten text.\n\n"
)

_SUMMARIZATION_ATTACK_PROMPT = (
    "Summarize the following text in roughly half the length while "
    "preserving the key claims. Use natural sentence flow. Do not add "
    "preamble; return only the summary.\n\n"
)


def grammar_normalize(
    text: str, *, provider: Callable[[str], str] | None = None, **_: object
) -> str:
    """LM-based grammar / spelling normalization.

    Strips idiosyncratic spelling and punctuation that sometimes flag
    LLM-generated text by polishing the grammar surface. Plan A2 step 2
    calls for a local Llama-3.2-1B-Instruct fallback by default; until
    that local-LM provider lands, callers must supply a `provider`
    callable explicitly (anthropic / openai / etc.).
    """
    if provider is None:
        raise RuntimeError(
            "grammar_normalize requires a 'provider' callable. "
            "Pass --include-prompt-attacks --provider <name> in the "
            "runner. Local-LM default routing (Llama-3.2-1B-Instruct) "
            "is a follow-up per A2 step 2."
        )
    return provider(_GRAMMAR_NORMALIZE_PROMPT + text)


def style_transfer(
    text: str, *, provider: Callable[[str], str] | None = None, **_: object
) -> str:
    """LM-based register switch (formal ↔ casual). Targets
    voice / fingerprint detectors that key on register-specific
    vocabulary (`furthermore`, `comprehensive`, etc.)."""
    if provider is None:
        raise RuntimeError(
            "style_transfer requires a 'provider' callable. "
            "Pass --include-prompt-attacks --provider <name> in the "
            "runner."
        )
    return provider(_STYLE_TRANSFER_PROMPT + text)


def summarization_attack(
    text: str, *, provider: Callable[[str], str] | None = None, **_: object
) -> str:
    """LM-based compression + paraphrase via summarization. Targets
    length-aware EBM features and rephrases away from fingerprint
    vocabulary by forcing a re-expression."""
    if provider is None:
        raise RuntimeError(
            "summarization_attack requires a 'provider' callable. "
            "Pass --include-prompt-attacks --provider <name> in the "
            "runner."
        )
    return provider(_SUMMARIZATION_ATTACK_PROMPT + text)


# --- humanizer_tools (commercial skeleton, double-gated) --------------------

# Mirrors the gating pattern in cairn/benchmarks/originality.py: env-var
# opt-in PLUS API key required. Live HTTP integration is deferred until
# snapshot fixtures and a privacy-review checkpoint are in place — until
# then, calling this attack raises a clear RuntimeError chain.
_HUMANIZER_GATE_ENV = "CAIRN_ALLOW_COMMERCIAL_BENCHMARK"
_HUMANIZER_API_KEY_ENV = "HUMANIZER_API_KEY"


def humanizer_tools(text: str, **_: object) -> str:
    """Commercial humanizer API (skeleton).

    Wraps a third-party humanizer service (e.g., Undetectable.ai,
    Quillbot). Sends the *full submission text* to a third-party server
    — read the privacy implications before enabling. Double-gated by
    `CAIRN_ALLOW_COMMERCIAL_BENCHMARK=1` and a populated
    `HUMANIZER_API_KEY` env var to make the trade-off explicit.

    Live integration is deferred until snapshot fixtures land (matches
    `cairn/benchmarks/originality.py`'s skeleton policy).
    """
    if os.environ.get(_HUMANIZER_GATE_ENV) != "1":
        raise RuntimeError(
            f"humanizer_tools is gated behind {_HUMANIZER_GATE_ENV}=1 "
            "for privacy and budget reasons. Set the env var explicitly "
            "to opt in."
        )
    if not os.environ.get(_HUMANIZER_API_KEY_ENV):
        raise RuntimeError(
            f"humanizer_tools requires {_HUMANIZER_API_KEY_ENV} in the "
            "environment. See A1's commercial-adapter pattern in "
            "cairn/benchmarks/originality.py."
        )
    raise NotImplementedError(
        "humanizer_tools is a SKELETON; live HTTP integration is "
        "deferred until snapshot fixtures and a privacy-review "
        "checkpoint are in place. See module docstring."
    )


# --- registrations ----------------------------------------------------------

register(Attack(
    name="burstiness_injection",
    category="transform",
    description="Interleave short filler fragments to increase sentence-length variance",
    severity="moderate",
    apply=burstiness_injection,
))
register(Attack(
    name="emoji_noise",
    category="transform",
    description="Sprinkle emojis at sentence boundaries to disrupt token-level fingerprints",
    severity="basic",
    apply=emoji_noise,
))
register(Attack(
    name="translation_pivot",
    category="transform",
    description="Round-trip translation via NLLB-200 (en -> fr -> en) to strip stylistic fingerprints",
    severity="advanced",
    apply=translation_pivot,
))
register(Attack(
    name="grammar_normalize",
    category="prompt",
    description="LM-based grammar / spelling normalization to remove idiosyncratic marks",
    severity="moderate",
    apply=grammar_normalize,
))
register(Attack(
    name="style_transfer",
    category="prompt",
    description="LM-based register switch (formal <-> casual) preserving content",
    severity="advanced",
    apply=style_transfer,
))
register(Attack(
    name="summarization_attack",
    category="prompt",
    description="LM-based compression + paraphrase via summarization",
    severity="advanced",
    apply=summarization_attack,
))
register(Attack(
    # humanizer_tools self-gates via env vars (no provider callable needed
    # from the runner). Marked "transform" so the registry doesn't raise
    # the generic "provider required" error before the gate runs.
    name="humanizer_tools",
    category="transform",
    description="Commercial humanizer API (skeleton; gated by CAIRN_ALLOW_COMMERCIAL_BENCHMARK)",
    severity="advanced",
    apply=humanizer_tools,
))

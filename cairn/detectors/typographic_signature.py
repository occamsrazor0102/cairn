"""Typographic signature -- per-character formatting habits.

Motivation: `docs/SESSION_RECAP.md` records that OpenAI Responses-API
output at `verbosity=low` costs the ensemble 15-24 AUROC points. The
mechanism is structural rather than adversarial -- every LM-backed
detector carries a `min_text_length` floor (50-60 words), so short input
falls through to the six Phase-1 lexical detectors, four of which sit at
chance on the per-detector table.

Typographic habits are measured as ratios -- which convention was
chosen where both were available -- so they are stable at 30 words where
likelihood-based statistics are not, and they survive the paraphrase and
synonym-swap attacks that strip lexical fingerprints: a humanizer
rewrites words, not punctuation conventions. Nothing in CAIRN measures
typography today -- `fingerprint_vocab` is lexical and `nssi` is
phrasal.

CONSTRAINED TO SIGNALS THAT SURVIVE `normalize_text`. This is a real
constraint, not a stylistic one: `cairn/pipeline.py` normalizes before
any detector runs, and `cairn/feature_cache.py` keys the feature cache
on `sha256(normalize_text(text))`. A detector that read raw text would
break that key -- two different raw inputs that normalize identically
would collide and share cached detector output. Checked against
`normalize_text` directly:

  survives   em-dash / en-dash, curly quotes, non-ASCII bullet glyphs,
             Oxford comma, heading title-case (all pure ASCII or
             NFKC-stable)
  destroyed  ellipsis form (NFKC folds U+2026 to "..."), double space
             after a period (whitespace collapse), homoglyphs,
             zero-width characters, ligatures

Only the surviving set is implemented. The destroyed signals are
genuinely informative and are recorded as deferred work in the scoping
note; they need a cache-key change first.

Deterministic and pure in `text`: no RNG, no global state.
"""
from __future__ import annotations

import re

from cairn.contracts import LayerResult
from cairn.detectors._utils import interpolate as _interpolate

_TYPOGRAPHIC_DASH = re.compile(r"[—–]")
# The same rhetorical dash, typed on an ASCII keyboard.
_ASCII_DASH = re.compile(r"(?<=\w)\s*--\s*(?=\w)|(?<=\s)-(?=\s)")
_CURLY_QUOTE = re.compile(r"[‘’“”]")
_STRAIGHT_QUOTE = re.compile(r"[\"']")
# Bullet glyphs a model emits directly, as opposed to the "-" or "*" a
# person types in a plain-text field.
_FANCY_BULLET = re.compile(r"^\s*[•◦‣⁃·]\s+", re.MULTILINE)
_ASCII_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
# Oxford comma opportunities: "a, b, and c" vs "a, b and c".
_OXFORD_PRESENT = re.compile(r",\s+(?:and|or)\s+\w")
_LIST_JOIN_BARE = re.compile(r"\w\s+(?:and|or)\s+\w")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s+(.+)$", re.MULTILINE)
_BOLD_HEADING = re.compile(r"^\s*\*\*([^*\n]{2,80})\*\*:?\s*$", re.MULTILINE)

_WORD_RE = re.compile(r"[\w']+")

# A ratio computed from one or two observations measures nothing -- a
# single em-dash in a 30-word note would otherwise saturate its term to
# 1.0, which is the length artefact this detector exists to avoid. Every
# ratio term below therefore reports 0.0 ("no evidence") until it has
# enough events to describe a habit rather than an incident.
_MIN_EVENTS_FOR_RATIO = 3

# Low floor on purpose: this detector exists to cover the short-form
# regime the tier-2 detectors abandon. Below 25 words even punctuation
# rates are single-event noise.
_MIN_WORDS = 25

# HEURISTIC mapping from the composite typographic score to p(LLM).
# Hand-picked, NOT fitted -- `_HEURISTIC_*` per CLAUDE.md. Input is a
# bounded mean of clamped [0, 1] terms; the table spans that full range
# so no input can saturate an endpoint.
_HEURISTIC_TYPOGRAPHY_TO_P_LLM = [
    (0.00, 0.06),
    (0.15, 0.18),
    (0.30, 0.38),
    (0.45, 0.58),
    (0.60, 0.74),
    (0.80, 0.87),
    (1.00, 0.92),
]


def _skip(detector, signals: dict) -> LayerResult:
    return LayerResult(
        layer_id=detector.id, domain=detector.domain,
        raw_score=0.0, p_llm=0.5, confidence=0.0,
        signals=signals,
        determination="SKIP",
        attacker_tiers=["A0", "A1", "A2"],
        compute_cost=detector.compute_cost,
        min_text_length=_MIN_WORDS,
    )


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _ratio(numerator: int, total: int, min_total: int = _MIN_EVENTS_FOR_RATIO) -> float:
    """Bounded style ratio, or 0.0 when the evidence is too thin.

    0.0 means "no evidence", never "human" -- absence of a habit is not
    itself a signal, and the EBM sees the raw counts separately.
    """
    if total < min_total:
        return 0.0
    return numerator / total


def _title_case_ratio(headings: list[str]) -> float:
    """Fraction of headings whose content words are capitalised.

    Models title-case headings far more consistently than people do in
    running notes. Returns 0.0 below two headings, so absence of
    structure -- or a single heading -- never reads as evidence.
    """
    if len(headings) < 2:
        return 0.0
    titled = 0
    for h in headings:
        words = [w for w in _WORD_RE.findall(h) if len(w) > 3]
        if len(words) < 2:
            continue
        if all(w[0].isupper() for w in words):
            titled += 1
    return titled / len(headings)


class TypographicSignatureDetector:
    id = "typographic_signature"
    domain = "universal"
    compute_cost = "trivial"

    def analyze(self, text: str, config: dict) -> LayerResult:
        n_words = len(_WORD_RE.findall(text))
        if n_words < _MIN_WORDS:
            return _skip(self, {"skipped": "too_short", "n_words": n_words})

        spans: list[dict] = []

        # Every term is a style RATIO -- which convention was chosen when
        # both were available -- rather than a per-1000-words rate. Rates
        # are the length trap: one em-dash in a 30-word note is 33 per
        # 1000 words and would saturate any fixed normalizer, while the
        # same dash in a 3000-word essay is 0.33. Ratios describe the
        # habit and are bounded in [0, 1] by construction.
        n_typo_dash = 0
        for m in _TYPOGRAPHIC_DASH.finditer(text):
            n_typo_dash += 1
            spans.append({
                "start": m.start(), "end": m.end(), "kind": "typography",
                "note": "typographic dash",
            })
        n_ascii_dash = len(_ASCII_DASH.findall(text))
        total_dashes = n_typo_dash + n_ascii_dash
        dash_style_ratio = _ratio(n_typo_dash, total_dashes)

        n_curly = len(_CURLY_QUOTE.findall(text))
        n_straight = len(_STRAIGHT_QUOTE.findall(text))
        total_quotes = n_curly + n_straight
        curly_ratio = _ratio(n_curly, total_quotes)

        n_fancy = 0
        for m in _FANCY_BULLET.finditer(text):
            n_fancy += 1
            spans.append({
                "start": m.start(), "end": m.end(), "kind": "typography",
                "note": "typographic bullet glyph",
            })
        n_ascii_bullet = len(_ASCII_BULLET.findall(text))
        total_bullets = n_fancy + n_ascii_bullet
        fancy_bullet_ratio = _ratio(n_fancy, total_bullets)

        # Of the coordinations that could carry a serial comma, how many
        # do. The denominator is every coordination, comma'd or not --
        # `_LIST_JOIN_BARE` cannot match a comma'd one (the character
        # before " and" is ",", not a word character), so the two counts
        # partition the coordinations rather than nesting.
        n_oxford = len(_OXFORD_PRESENT.findall(text))
        n_bare_join = len(_LIST_JOIN_BARE.findall(text))
        oxford_ratio = _ratio(n_oxford, n_oxford + n_bare_join)

        headings = _MD_HEADING.findall(text) + _BOLD_HEADING.findall(text)
        heading_title_case = _title_case_ratio(headings)

        terms = [
            dash_style_ratio,
            curly_ratio,
            fancy_bullet_ratio,
            oxford_ratio,
            heading_title_case,
        ]
        typography_score = sum(terms) / len(terms)

        p_llm = _interpolate(typography_score, _HEURISTIC_TYPOGRAPHY_TO_P_LLM)

        if p_llm >= 0.75:
            determination = "RED"
        elif p_llm >= 0.50:
            determination = "AMBER"
        elif p_llm >= 0.25:
            determination = "YELLOW"
        else:
            determination = "GREEN"

        spans.sort(key=lambda s: s["start"])

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=typography_score, p_llm=p_llm,
            # Deliberately modest, and flat rather than length-scaled:
            # the point of this detector is the short-text regime, so it
            # must not claim more confidence merely because a sample is
            # long. Keeping it low also limits how far it can move
            # `cascade.aggregate_p_llm`, which is confidence-weighted and
            # gates phases 2 and 3.
            confidence=min(0.25 + n_words / 2000, 0.65),
            signals={
                "dash_style_ratio": round(dash_style_ratio, 4),
                "n_typographic_dashes": n_typo_dash,
                "curly_quote_ratio": round(curly_ratio, 4),
                "fancy_bullet_ratio": round(fancy_bullet_ratio, 4),
                "oxford_comma_ratio": round(oxford_ratio, 4),
                "heading_title_case": round(heading_title_case, 4),
                "typography_score": round(typography_score, 4),
                "n_headings": len(headings),
                "n_words": n_words,
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2"],
            compute_cost=self.compute_cost,
            min_text_length=_MIN_WORDS,
            spans=spans,
        )

    def calibrate(self, labeled_data: list) -> None:
        pass


DETECTOR = TypographicSignatureDetector()

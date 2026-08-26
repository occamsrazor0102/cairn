"""Compression profile -- redundancy as a Kolmogorov-complexity proxy.

Motivation: `docs/benchmarks/attack_coverage.md` records
`summarization_attack` at AUROC 0.4167 -- the one cell in the per-attack
grid that sits *below* the random-classifier floor, meaning the ensemble
systematically calls summarize-then-re-expand output human.

Every current CAIRN detector reads either register (lexical / phrasal
patterns) or likelihood (per-token surprisal under a proxy LM).
Summarize-then-re-expand changes neither much: it pads a compressed
semantic core back out to length using the same register the model
always uses. What it *does* change is redundancy -- the re-expansion
restates the core rather than adding information.

This detector measures redundancy directly, via compressibility and
lexical-repetition statistics. That is an orthogonal axis to everything
already in the cascade, and it needs no model, no network, and no
optional dependency.

DESIGN NOTE -- every signal is bounded in [0, 1] by construction. The
SpecDetect integration failed because `_HEURISTIC_TOTAL_ENERGY_TO_P_LLM`
was anchored over a guessed empirical band (0.5..5.0) while the corpus
median sat at 5.75, so 70.9% of samples saturated to the endpoint and
isotonic calibration could not recover the lost information (see
`docs/benchmarks/robust_v2_validation/HYPOTHESIS_INVESTIGATION.md`).
Bounded scores make that failure mode unreachable: the heuristic table
below spans the full constructible range of its input.

Deterministic and pure in `text` -- no RNG, no global state. (The
`token_cohesiveness` detector drew from the global `random` module and
silently miscalibrated production for months; see the same document.)
"""
from __future__ import annotations

import lzma
import re
import zlib

from cairn.contracts import LayerResult
from cairn.detectors._utils import interpolate as _interpolate

_WORD_RE = re.compile(r"[\w']+")

# Minimum words before compression statistics mean anything. Below this,
# container/header overhead dominates the compressed size and the ratio
# is noise. Chosen to sit under the tier-2 floors (50-60 words) so this
# detector still fires on the short-form inputs where every LM-backed
# detector is gated off -- that is half its value.
_MIN_WORDS = 40

# Window for the moving-average type-token ratio. Plain TTR falls
# monotonically with length, which would make the signal a length proxy;
# MATTR is length-robust because every window is the same size.
_MATTR_WINDOW = 100

_NGRAM_N = 4

# HEURISTIC mapping from the composite redundancy score to p(LLM).
# Hand-picked, NOT fitted to labelled data -- named `_HEURISTIC_*` per
# CLAUDE.md's honest-calibration rule. Replaced by isotonic calibration
# in `scripts/calibrate_detectors.py` once a cache carries this
# detector.
#
# The input is bounded in [0, 1] by construction and the table spans
# that full range, so no input can saturate an endpoint. See the design
# note above for why that matters.
_HEURISTIC_REDUNDANCY_TO_P_LLM = [
    (0.00, 0.05),
    (0.20, 0.15),
    (0.40, 0.35),
    (0.55, 0.55),
    (0.70, 0.75),
    (0.85, 0.88),
    (1.00, 0.93),
]


def _skip(detector, signals: dict) -> LayerResult:
    return LayerResult(
        layer_id=detector.id, domain=detector.domain,
        raw_score=0.0, p_llm=0.5, confidence=0.0,
        signals=signals,
        determination="SKIP",
        attacker_tiers=["A0", "A1", "A2", "A3"],
        compute_cost=detector.compute_cost,
        min_text_length=_MIN_WORDS,
    )


def _mattr(tokens: list[str], window: int) -> float:
    """Moving-average type-token ratio. Returns 1.0 for all-unique text."""
    if len(tokens) <= window:
        return len(set(tokens)) / max(len(tokens), 1)
    total = 0.0
    n_windows = len(tokens) - window + 1
    for i in range(n_windows):
        chunk = tokens[i:i + window]
        total += len(set(chunk)) / window
    return total / n_windows


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


class CompressionProfileDetector:
    id = "compression_profile"
    domain = "universal"
    compute_cost = "trivial"

    def analyze(self, text: str, config: dict) -> LayerResult:
        tokens = [t.lower() for t in _WORD_RE.findall(text)]
        n_words = len(tokens)
        if n_words < _MIN_WORDS:
            return _skip(self, {"skipped": "too_short", "n_words": n_words})

        payload = text.encode("utf-8")
        n_bytes = len(payload)
        if n_bytes == 0:
            return _skip(self, {"skipped": "empty", "n_words": n_words})

        # Both compressors, because their windows differ: zlib's is 32 KB
        # and catches local restatement; lzma's is far larger and catches
        # the long-range "restate the introduction in the conclusion"
        # pattern that re-expansion produces.
        zlib_ratio = len(zlib.compress(payload, 9)) / n_bytes
        lzma_ratio = len(lzma.compress(payload, preset=6)) / n_bytes

        mattr = _mattr(tokens, _MATTR_WINDOW)

        grams = _ngrams(tokens, _NGRAM_N)
        if grams:
            counts: dict[tuple[str, ...], int] = {}
            for g in grams:
                counts[g] = counts.get(g, 0) + 1
            distinct_4 = len(counts) / len(grams)
            max_repeat = max(counts.values()) / len(grams)
        else:
            counts = {}
            distinct_4 = 1.0
            max_repeat = 0.0

        # The composite deliberately uses ONLY the two length-robust
        # terms, and uses them raw rather than rescaled onto a
        # hand-picked band.
        #
        # The compression ratios are strongly length-dependent (short
        # UTF-8 text compresses to ~0.65-0.75, long text to ~0.30-0.40),
        # so any fixed rescale band saturates for one length regime --
        # which is precisely the SpecDetect failure. They ship as
        # `signals` instead, where the EBM can combine them with the
        # `log_word_count` / `is_short` / `is_medium` / `is_long`
        # features that exist for exactly this purpose.
        #
        # `1 - mattr` and `1 - distinct_4` are already fractions in
        # [0, 1] and need no anchoring, so the composite is bounded by
        # construction and no input can saturate the heuristic table.
        # The mapping is coarse on purpose: honest heuristic mode, with
        # the real mapping supplied by isotonic calibration.
        redundancy = 0.6 * (1.0 - mattr) + 0.4 * (1.0 - distinct_4)

        p_llm = _interpolate(redundancy, _HEURISTIC_REDUNDANCY_TO_P_LLM)

        spans = self._repetition_spans(text, counts)

        if p_llm >= 0.75:
            determination = "RED"
        elif p_llm >= 0.50:
            determination = "AMBER"
        elif p_llm >= 0.25:
            determination = "YELLOW"
        else:
            determination = "GREEN"

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=redundancy, p_llm=p_llm,
            # Compression statistics stabilise with length; stay modest
            # on short input rather than claiming confidence the sample
            # size does not support. This also limits how much the
            # detector can move `cascade.aggregate_p_llm`, which is
            # confidence-weighted and gates phases 2 and 3.
            confidence=min(0.30 + n_words / 1200, 0.80),
            signals={
                "zlib_ratio": round(zlib_ratio, 4),
                "lzma_ratio": round(lzma_ratio, 4),
                "mattr_100": round(mattr, 4),
                "distinct_4": round(distinct_4, 4),
                "max_ngram_repeat": round(max_repeat, 5),
                "redundancy": round(redundancy, 4),
                "n_words": n_words,
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2", "A3"],
            compute_cost=self.compute_cost,
            min_text_length=_MIN_WORDS,
            spans=spans,
        )

    @staticmethod
    def _repetition_spans(text: str, counts: dict) -> list[dict]:
        """Highlight the most-repeated 4-grams so a reviewer can see the
        restatement the score is reacting to.

        Only n-grams occurring 3+ times are worth a highlight; below that
        the repetition is ordinary English. Capped at the top 5 phrases
        to keep the UI readable.
        """
        repeated = [(g, c) for g, c in counts.items() if c >= 3]
        if not repeated:
            return []
        repeated.sort(key=lambda gc: (-gc[1], gc[0]))
        spans: list[dict] = []
        lowered = text.lower()
        for gram, count in repeated[:5]:
            phrase = " ".join(gram)
            start = lowered.find(phrase)
            if start < 0:
                # Token joining does not always reproduce the source
                # spacing (punctuation, newlines). Skip rather than
                # emit a span whose offsets do not match the text.
                continue
            spans.append({
                "start": start,
                "end": start + len(phrase),
                "kind": "repetition",
                "note": f"phrase repeated {count}x: '{phrase}'",
            })
        spans.sort(key=lambda s: s["start"])
        # Sliding 4-grams overlap by construction ("the key point here"
        # and "key point here is"). The GUI drops overlaps and reports
        # the count in the legend, which would read as a defect -- so
        # drop them here and emit only disjoint highlights.
        disjoint: list[dict] = []
        for s in spans:
            if disjoint and s["start"] < disjoint[-1]["end"]:
                continue
            disjoint.append(s)
        return disjoint

    def calibrate(self, labeled_data: list) -> None:
        pass


DETECTOR = CompressionProfileDetector()

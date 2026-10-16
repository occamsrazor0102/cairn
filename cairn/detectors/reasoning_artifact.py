"""Reasoning artifacts -- surface traces of hidden chain-of-thought.

Implements Track A3 of
`docs/superpowers/specs/2026-05-07-post-honest-calibration-roadmap.md`.

Motivation: the matched-provider evaluation in `README.md` puts OpenAI
Responses-API reasoning output at AUROC 0.9358 under `calibrated_full`
against Gemini's 0.9961 -- roughly six AUROC points stealthier than
every other frontier provider. Models that reason before answering leak
a distinctive surface register into the answer itself: deliberation
phrasing carried over from the hidden trace, clauses that announce an
intent before performing it, and an unusually uniform register across
what would otherwise be topic boundaries.

None of that is register-generic enough for `nssi` (which keys on
formulaic *conclusion* phrasing) or lexical enough for
`fingerprint_vocab`. This detector targets it directly.

Pure Python -- no model, no network, no optional dependency. Emits
character spans, which makes it the first Phase-2 detector to give the
UI positional evidence; the six existing span emitters are all Phase 1.

Deterministic and pure in `text`: no RNG, no global state.

OVER-FIT RISK, and the gate that guards it: this keys on artefacts of
one provider's *current* output style. The A3 spec's acceptance gate is
therefore two-sided -- OpenAI per-provider AUROC must reach >= 0.96
AND Gemini must not fall by more than 0.02. A detector that lifts one
provider by learning its UX quirks, at the cost of the others, has
failed even if the aggregate improves. Both numbers come from
`EvalReport.per_source` (`cairn/evaluation/runner.py`).
"""
from __future__ import annotations

import re
import statistics

from cairn.contracts import LayerResult
from cairn.detectors._utils import interpolate as _interpolate

# Deliberation phrasing carried out of a hidden reasoning trace into the
# answer. Deliberately narrow: these are first-person process
# announcements, not general discourse markers (those belong to `nssi`).
_DELIBERATION = re.compile(
    r"\b(?:let me (?:think|see|check|start|begin|work|break|walk|consider)"
    r"|let's (?:think|see|break|walk|start|begin|consider|work)"
    r"|i(?:'| a)?ll (?:start|begin|work|walk|break|go)"
    r"|first,? (?:i|let)"
    r"|looking at this"
    r"|thinking (?:about|through) this"
    r"|to (?:approach|tackle) this"
    r"|working through this"
    r"|step by step)\b",
    re.IGNORECASE,
)

# Self-correction / mid-stream revision. Strong hidden-reasoning tell:
# the model surfaces a course change it made while deliberating.
_SELF_CORRECTION = re.compile(
    r"(?:^|(?<=[.!?;]\s))\s*(?:wait|actually|hold on|on second thought"
    r"|but wait|hmm|correction|scratch that|re-?reading)\b[,.]?",
    re.IGNORECASE,
)

# Clauses that state an intent instead of performing the action. High
# ratios of these to plain declaratives mark narrated reasoning.
_INTENT_CLAUSE = re.compile(
    r"\b(?:i(?:'| a)?ll |i will |i need to |i should |we (?:need|should|can) "
    r"|the (?:next|first|key) step is |what we (?:need|want) (?:to|is) "
    r"|my (?:approach|plan|goal) (?:is|here)|the (?:goal|idea|plan) here is)\b",
    re.IGNORECASE,
)

# Enumeration scaffolding around a reasoning trace.
_ENUM_SCAFFOLD = re.compile(
    r"^\s*(?:step\s+\d+\b|\d+[.)]\s+\*\*|\*\*step\s|\#{1,4}\s*step\s)",
    re.IGNORECASE | re.MULTILINE,
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

_MIN_WORDS = 60

# HEURISTIC mapping from the composite artifact score to p(LLM).
# Hand-picked, NOT fitted -- `_HEURISTIC_*` per CLAUDE.md. The input is
# a bounded mean of clamped [0, 1] terms and the table spans that full
# range, so no input saturates an endpoint.
_HEURISTIC_ARTIFACT_TO_P_LLM = [
    (0.00, 0.05),
    (0.15, 0.20),
    (0.30, 0.42),
    (0.45, 0.62),
    (0.60, 0.78),
    (0.80, 0.89),
    (1.00, 0.94),
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


class ReasoningArtifactDetector:
    id = "reasoning_artifact"
    domain = "prose"
    compute_cost = "cheap"

    def analyze(self, text: str, config: dict) -> LayerResult:
        words = text.split()
        n_words = len(words)
        if n_words < _MIN_WORDS:
            return _skip(self, {"skipped": "too_short", "n_words": n_words})

        spans: list[dict] = []

        def _collect(pattern: re.Pattern, note: str) -> int:
            n = 0
            for m in pattern.finditer(text):
                if not m.group(0).strip():
                    continue
                n += 1
                spans.append({
                    "start": m.start(),
                    "end": m.end(),
                    "kind": "reasoning",
                    "note": f"{note} '{m.group(0).strip()}'",
                })
            return n

        n_deliberation = _collect(_DELIBERATION, "deliberation marker")
        n_self_correction = _collect(_SELF_CORRECTION, "self-correction")
        n_intent = _collect(_INTENT_CLAUSE, "intent clause")
        n_enum = len(_ENUM_SCAFFOLD.findall(text))

        per_1k = 1000.0 / n_words
        deliberation_density = n_deliberation * per_1k
        self_correction_density = n_self_correction * per_1k
        intent_density = n_intent * per_1k
        enum_density = n_enum * per_1k

        # Register uniformity: hidden-reasoning output tends to hold a
        # flat sentence-length profile across the whole answer, where
        # human prose varies. Expressed as 1 - normalized coefficient of
        # variation so that "more uniform" reads as "more LLM-like", and
        # bounded in [0, 1].
        sentences = [s for s in _SENT_SPLIT.split(text) if len(s.split()) >= 3]
        if len(sentences) >= 4:
            lengths = [len(s.split()) for s in sentences]
            mean_len = statistics.fmean(lengths)
            cv = (statistics.pstdev(lengths) / mean_len) if mean_len else 0.0
            register_uniformity = _clamp(1.0 - cv / 0.75)
        else:
            register_uniformity = 0.0

        # Each density term is normalized by a rate that represents
        # "saturated" for that marker class, then clamped. Rates are
        # per-1000-words and coarse by design; the real mapping comes
        # from isotonic calibration once a cache carries this detector.
        terms = [
            _clamp(deliberation_density / 4.0),
            _clamp(self_correction_density / 3.0),
            _clamp(intent_density / 8.0),
            _clamp(enum_density / 6.0),
            register_uniformity,
        ]
        artifact_score = sum(terms) / len(terms)

        p_llm = _interpolate(artifact_score, _HEURISTIC_ARTIFACT_TO_P_LLM)

        if p_llm >= 0.75:
            determination = "RED"
        elif p_llm >= 0.50:
            determination = "AMBER"
        elif p_llm >= 0.25:
            determination = "YELLOW"
        else:
            determination = "GREEN"

        spans.sort(key=lambda s: s["start"])
        disjoint: list[dict] = []
        for s in spans:
            if disjoint and s["start"] < disjoint[-1]["end"]:
                continue
            disjoint.append(s)

        return LayerResult(
            layer_id=self.id, domain=self.domain,
            raw_score=artifact_score, p_llm=p_llm,
            confidence=min(0.35 + n_words / 1500, 0.82),
            signals={
                "deliberation_density": round(deliberation_density, 3),
                "self_correction_density": round(self_correction_density, 3),
                "intent_density": round(intent_density, 3),
                "enum_scaffold_density": round(enum_density, 3),
                "register_uniformity": round(register_uniformity, 4),
                "artifact_score": round(artifact_score, 4),
                "n_sentences": len(sentences),
                "n_words": n_words,
            },
            determination=determination,
            attacker_tiers=["A0", "A1"],
            compute_cost=self.compute_cost,
            min_text_length=_MIN_WORDS,
            spans=disjoint,
        )

    def calibrate(self, labeled_data: list) -> None:
        pass


DETECTOR = ReasoningArtifactDetector()

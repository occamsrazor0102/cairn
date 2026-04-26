import re
from beet.contracts import LayerResult
from beet.config import get_pattern_list
from beet.detectors._utils import interpolate as _interpolate

# HEURISTIC mapping from fingerprint hits-per-1000 to p(LLM). Hand-picked
# piecewise-linear table — NOT derived from a labeled dataset. Replace with
# isotonic calibration from beet.calibration once a training set exists.
_HEURISTIC_P_LLM_MAPPING = [
    (0.0, 0.05), (2.0, 0.25), (5.0, 0.45), (8.0, 0.60),
    (12.0, 0.75), (18.0, 0.85), (25.0, 0.93),
]

class FingerprintVocabDetector:
    id = "fingerprint_vocab"
    domain = "universal"
    compute_cost = "trivial"

    def __init__(self):
        from beet.config import _load_yaml, PATTERNS_DIR
        from pathlib import Path
        path = PATTERNS_DIR / "fingerprint_words.yaml"
        data = _load_yaml(path)
        words = data.get("words", [])
        bigrams_raw = data.get("bigrams", [])
        self._word_patterns = [(w, re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE)) for w in words]
        self._bigram_patterns = [(b, re.compile(re.escape(b), re.IGNORECASE)) for b in bigrams_raw]

    def analyze(self, text: str, config: dict) -> LayerResult:
        words = text.split()
        word_count = max(len(words), 1)
        matched_words = []
        spans: list[dict] = []
        for word, pattern in self._word_patterns:
            for m in pattern.finditer(text):
                matched_words.append(word)
                spans.append({
                    "start": m.start(), "end": m.end(),
                    "kind": "fingerprint",
                    "note": f"fingerprint word '{word}'",
                })
        bigram_hits = 0
        matched_bigrams = []
        for bigram, pattern in self._bigram_patterns:
            m = pattern.search(text)
            if m:
                bigram_hits += 1
                matched_bigrams.append(bigram)
                spans.append({
                    "start": m.start(), "end": m.end(),
                    "kind": "fingerprint",
                    "note": f"fingerprint bigram '{bigram}'",
                })
        hits_per_1000 = (len(matched_words) + bigram_hits * 1.5) / word_count * 1000
        p_llm = _interpolate(hits_per_1000, _HEURISTIC_P_LLM_MAPPING)
        if p_llm >= 0.75: determination = "RED"
        elif p_llm >= 0.50: determination = "AMBER"
        elif p_llm >= 0.25: determination = "YELLOW"
        else: determination = "GREEN"
        return LayerResult(
            layer_id=self.id, domain=self.domain, raw_score=hits_per_1000, p_llm=p_llm,
            confidence=min(0.5 + word_count / 2000, 0.90),
            signals={"hits_per_1000": round(hits_per_1000, 2), "matched_words": list(set(matched_words)),
                     "word_hit_count": len(matched_words), "bigram_hits": bigram_hits, "matched_bigrams": matched_bigrams},
            determination=determination, attacker_tiers=["A0", "A1"],
            compute_cost=self.compute_cost, min_text_length=30,
            spans=spans,
        )

    def calibrate(self, labeled_data: list) -> None: pass

DETECTOR = FingerprintVocabDetector()

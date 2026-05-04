"""Semantic-kinematics detector — trajectory shape of meaning in embedding space.

Macro-feature detector that fingerprints the *topological shape of the
argument* in embedding space. Sentence-to-sentence cosine-distance
trajectory, plus its first/second/third derivatives (velocity,
acceleration, jerk), distinguishes the human "drunken walk" pattern of
episodic cognition from the LLM "frictionless gradient" pattern enforced
by self-attention.

The macro-trajectory survives paraphrase because most paraphrasers
operate sentence-locally; they don't reshape the global flow of meaning.
It survives "burstiness" prompt-engineering because that controls
token-level entropy, not embedding-level trajectory.
"""
from __future__ import annotations

from cairn.contracts import LayerResult


# _HEURISTIC_ priors per CLAUDE.md naming. Hand-tuned starting points only;
# the EBM picks up real weights once Phase 4/5 retrains include the new
# semantic_kinematics_* features.
_HEURISTIC_HUMAN_VELOCITY_STD = 0.18
_HEURISTIC_HUMAN_JERK_STD = 0.20
_HEURISTIC_SCALE = 6.0
_MIN_SENTENCES = 5
_MIN_TEXT_LENGTH = 120

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _segment(text: str) -> list[str]:
    try:
        from cairn.normalizer import split_sentences  # type: ignore[attr-defined]
        out = [s.strip() for s in split_sentences(text) if s.strip()]
        if out:
            return out
    except Exception:
        pass
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _cosine_distances(vecs) -> list[float]:
    import numpy as np
    out: list[float] = []
    for i in range(len(vecs) - 1):
        a = vecs[i]
        b = vecs[i + 1]
        out.append(float(1.0 - np.dot(a, b)))
    return out


def _kinematics(distances: list[float]) -> dict[str, float]:
    import numpy as np
    if len(distances) < 3:
        return {
            "velocity_mean": float(np.mean(distances)) if distances else 0.0,
            "velocity_std": 0.0,
            "acceleration_mean": 0.0, "acceleration_std": 0.0,
            "jerk_mean": 0.0, "jerk_std": 0.0,
        }
    v = np.asarray(distances, dtype=float)
    a = np.diff(v)
    j = np.diff(a)
    return {
        "velocity_mean": float(np.mean(v)),
        "velocity_std": float(np.std(v)),
        "acceleration_mean": float(np.mean(np.abs(a))),
        "acceleration_std": float(np.std(a)),
        "jerk_mean": float(np.mean(np.abs(j))) if len(j) else 0.0,
        "jerk_std": float(np.std(j)) if len(j) else 0.0,
    }


def _heuristic_p_llm(k: dict[str, float]) -> float:
    # Low std + low jerk → high p_llm.
    vz = max(0.0, _HEURISTIC_HUMAN_VELOCITY_STD - k["velocity_std"]) / max(_HEURISTIC_HUMAN_VELOCITY_STD, 1e-6)
    jz = max(0.0, _HEURISTIC_HUMAN_JERK_STD - k["jerk_std"]) / max(_HEURISTIC_HUMAN_JERK_STD, 1e-6)
    z = (vz + jz) / 2.0
    return float(min(1.0, z * (_HEURISTIC_SCALE / 6.0)))


class SemanticKinematicsDetector:
    id = "semantic_kinematics"
    domain = "prose"
    compute_cost = "moderate"

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "SemanticKinematicsDetector requires sentence-transformers. "
                "Install with: pip install 'cairn[retrieval]'"
            ) from e
        self._model = SentenceTransformer(model_name)

    def analyze(self, text: str, config: dict) -> LayerResult:
        if len(text) < _MIN_TEXT_LENGTH:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0,
                signals={"reason": "too_short"},
                determination="SKIP", attacker_tiers=["A0", "A1", "A2"],
                compute_cost=self.compute_cost, min_text_length=_MIN_TEXT_LENGTH,
            )
        sentences = _segment(text)
        if len(sentences) < _MIN_SENTENCES:
            return LayerResult(
                layer_id=self.id, domain=self.domain, raw_score=0.0,
                p_llm=0.5, confidence=0.0,
                signals={"reason": "too_few_sentences", "n_sentences": len(sentences)},
                determination="SKIP", attacker_tiers=["A0", "A1", "A2"],
                compute_cost=self.compute_cost, min_text_length=_MIN_TEXT_LENGTH,
            )
        embs = self._model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        distances = _cosine_distances(embs)
        k = _kinematics(distances)
        p_llm = _heuristic_p_llm(k)
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
            raw_score=k["velocity_std"],
            p_llm=p_llm,
            confidence=min(0.40 + len(sentences) / 50.0, 0.85),
            signals={
                "n_sentences": len(sentences),
                "sk_velocity_mean": round(k["velocity_mean"], 4),
                "sk_velocity_std": round(k["velocity_std"], 4),
                "sk_acceleration_mean": round(k["acceleration_mean"], 4),
                "sk_acceleration_std": round(k["acceleration_std"], 4),
                "sk_jerk_mean": round(k["jerk_mean"], 4),
                "sk_jerk_std": round(k["jerk_std"], 4),
            },
            determination=determination,
            attacker_tiers=["A0", "A1", "A2"],
            compute_cost=self.compute_cost,
            min_text_length=_MIN_TEXT_LENGTH,
        )

    def calibrate(self, labeled_data: list) -> None:
        # Calibration is owned by the EBM after Phase 4/5 retrain.
        pass


class _LazySemanticKinematics:
    id = "semantic_kinematics"
    domain = "prose"
    compute_cost = "moderate"
    _instance = None

    def analyze(self, text: str, config: dict) -> LayerResult:
        if self._instance is None:
            self._instance = SemanticKinematicsDetector(
                model_name=(config or {}).get("model", _DEFAULT_MODEL),
            )
        return self._instance.analyze(text, config)

    def calibrate(self, labeled_data: list) -> None:
        pass


DETECTOR = _LazySemanticKinematics()

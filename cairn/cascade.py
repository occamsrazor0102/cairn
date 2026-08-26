from cairn.contracts import LayerResult

# Phase membership uses tuples so iteration order is deterministic across
# process restarts. Set iteration is hash-randomised in CPython, which
# leaks into report ordering and detector_errors sequencing.
# compression_profile (redundancy / Kolmogorov proxy) and
# typographic_signature (formatting habits) are stdlib-only and both
# fire on short input, where every tier-2 detector is gated off by its
# min_text_length floor. Both ship disabled in config pending a retrain.
PHASE1_DETECTORS = ("preamble", "fingerprint_vocab", "prompt_structure", "voice_spec", "instruction_density", "nssi", "compression_profile", "typographic_signature")
# mixed_boundary operates on sentence-level style dissonance within a single
# submission, so it's a Phase-2 detector despite earlier being parked in
# Phase 4 (batch-only) where it never fired.
# fast_detectgpt (analytic conditional-probability curvature) is the
# successor to the classic-DetectGPT `perturbation` detector and runs in
# Phase 2 alongside the other single-pass LM-scoring detectors. The
# expensive perturbation detector stays declared in Phase 3 but ships
# disabled in config (superseded; kept for reproducibility of old runs).
# reasoning_artifact targets the hidden-reasoning stealth gap (Track A3
# of the post-honest-calibration roadmap). Pure Python, but Phase 2
# per that spec: it is a register judgement on longer prose, not a
# cheap gate, and Phase 1 short-circuits on aggregate p_llm.
PHASE2_DETECTORS = ("surprisal_dynamics", "contrastive_lm", "fast_detectgpt", "token_cohesiveness", "mixed_boundary", "semantic_kinematics", "specdetect", "reasoning_artifact")
PHASE3_DETECTORS = ("perturbation", "contrastive_gen", "dna_gpt", "lapd")
# Phase 4 is batch-only; contributor_graph is invoked separately via
# pipeline.analyze_contributors, not through the normal cascade.
PHASE4_DETECTORS = ("cross_similarity",)

class CascadeScheduler:
    def __init__(self, config: dict):
        cascade = config.get("cascade", {})
        self._p1_high = cascade.get("phase1_short_circuit_high", 0.85)
        self._p1_low = cascade.get("phase1_short_circuit_low", 0.10)
        self._p2_high = cascade.get("phase2_short_circuit_high", 0.80)
        self._p2_low = cascade.get("phase2_short_circuit_low", 0.15)
        self._force_phase3 = cascade.get("phase3_always_run", False)

    def aggregate_p_llm(self, results: list[LayerResult]) -> float:
        active = [r for r in results if r.determination != "SKIP" and r.confidence > 0]
        if not active: return 0.5
        total_weight = sum(r.confidence for r in active)
        return sum(r.p_llm * r.confidence for r in active) / total_weight

    def should_run_phase2(self, phase1_results: list[LayerResult]) -> bool:
        agg = self.aggregate_p_llm(phase1_results)
        return self._p1_low < agg < self._p1_high

    def should_run_phase3(self, all_results: list[LayerResult]) -> bool:
        if self._force_phase3: return True
        agg = self.aggregate_p_llm(all_results)
        return self._p2_low < agg < self._p2_high

    def detectors_for_phase(self, phase: int) -> tuple[str, ...]:
        return {1: PHASE1_DETECTORS, 2: PHASE2_DETECTORS, 3: PHASE3_DETECTORS, 4: PHASE4_DETECTORS}.get(phase, ())

import pytest


def test_semantic_kinematics_skips_short_text():
    pytest.importorskip("sentence_transformers")
    from cairn.detectors.semantic_kinematics import SemanticKinematicsDetector
    det = SemanticKinematicsDetector()
    res = det.analyze("Just one sentence.", config={})
    assert res.determination == "SKIP"
    assert res.signals.get("reason") in ("too_short", "too_few_sentences")


def test_semantic_kinematics_emits_velocity_jerk_signals():
    pytest.importorskip("sentence_transformers")
    from cairn.detectors.semantic_kinematics import SemanticKinematicsDetector
    text = (
        "The river carved its way through limestone over millennia. "
        "Geologists call this karst topography. "
        "I once visited a karst cave system in Slovenia. "
        "It smelled like wet stone and old air. "
        "Anyway, the deeper question is whether erosion is fractal at every scale. "
        "Mandelbrot would have said yes. "
        "I'm less sure. "
        "The math gets tangled when you try to bound the dimension empirically."
    )
    det = SemanticKinematicsDetector()
    res = det.analyze(text, config={})
    assert res.determination != "SKIP"
    assert "sk_velocity_std" in res.signals
    assert "sk_jerk_std" in res.signals
    assert "n_sentences" in res.signals
    assert 0.0 <= res.p_llm <= 1.0


def test_semantic_kinematics_low_variance_text_yields_higher_p_llm():
    pytest.importorskip("sentence_transformers")
    from cairn.detectors.semantic_kinematics import SemanticKinematicsDetector
    smooth = (
        "Climate change is a pressing global issue. "
        "The pressing global issue requires immediate action. "
        "Immediate action must be coordinated internationally. "
        "International coordination requires shared frameworks. "
        "Shared frameworks enable consistent policy. "
        "Consistent policy drives measurable outcomes."
    )
    jagged = (
        "Climate change is real. "
        "My dog ate a bee yesterday and I felt bad for both of them. "
        "Anyway, the IPCC report is grim. "
        "Have you ever seen a mountain at dawn? "
        "We need carbon pricing. "
        "I should call my mother."
    )
    det = SemanticKinematicsDetector()
    r_smooth = det.analyze(smooth, config={})
    r_jagged = det.analyze(jagged, config={})
    assert r_smooth.p_llm > r_jagged.p_llm


# ---------------------------------------------------------------------------
# D2 — heuristic-mode suppression flag
# ---------------------------------------------------------------------------
# `_heuristic_p_llm` is anti-correlated with p(LLM) on calib2000 (AUROC
# 0.17/0.24 — see docs/benchmarks/full_coverage_holdout_500.md). Until
# the per-detector calibrator includes semantic_kinematics, the
# `heuristic_suppress` flag neutralises the misleading point estimate
# while keeping raw kinematic signals flowing into the EBM feature cache.

_KINEMATICS_FIXTURE = (
    "The river carved its way through limestone over millennia. "
    "Geologists call this karst topography. "
    "I once visited a karst cave system in Slovenia. "
    "It smelled like wet stone and old air. "
    "The math gets tangled when you try to bound the dimension empirically."
)


def test_semantic_kinematics_suppressed_when_flag_true():
    """heuristic_suppress=True → p_llm=0.5, confidence=0.0, SKIP. Raw
    kinematic signals still populated for downstream EBM consumption."""
    pytest.importorskip("sentence_transformers")
    from cairn.detectors.semantic_kinematics import SemanticKinematicsDetector
    det = SemanticKinematicsDetector()
    res = det.analyze(_KINEMATICS_FIXTURE, config={"heuristic_suppress": True})
    assert res.p_llm == 0.5
    assert res.confidence == 0.0
    assert res.determination == "SKIP"
    for key in (
        "n_sentences",
        "sk_velocity_mean",
        "sk_velocity_std",
        "sk_acceleration_mean",
        "sk_acceleration_std",
        "sk_jerk_mean",
        "sk_jerk_std",
    ):
        assert key in res.signals, f"raw kinematic signal missing: {key}"
    assert res.signals.get("reason") == "heuristic_suppress"
    assert "calibrator" in res.signals.get("notes", "").lower()


def test_semantic_kinematics_not_suppressed_when_flag_false():
    """heuristic_suppress=False (or absent) → existing heuristic mapping."""
    pytest.importorskip("sentence_transformers")
    from cairn.detectors.semantic_kinematics import SemanticKinematicsDetector
    det = SemanticKinematicsDetector()
    res_off = det.analyze(_KINEMATICS_FIXTURE, config={"heuristic_suppress": False})
    res_default = det.analyze(_KINEMATICS_FIXTURE, config={})
    assert res_off.determination != "SKIP"
    assert res_default.determination != "SKIP"
    assert res_off.p_llm == res_default.p_llm
    assert res_off.confidence == res_default.confidence
    assert "reason" not in res_off.signals
    assert "notes" not in res_off.signals


def test_semantic_kinematics_lazy_wrapper_honours_suppression():
    """Lazy wrapper must thread the flag through to the underlying
    analyze() — otherwise pipeline-level callers get the unsuppressed
    behaviour when the detector loads via the registry."""
    pytest.importorskip("sentence_transformers")
    from cairn.detectors.semantic_kinematics import _LazySemanticKinematics
    _LazySemanticKinematics._instance = None  # cold-start
    lazy = _LazySemanticKinematics()
    res = lazy.analyze(_KINEMATICS_FIXTURE, config={"heuristic_suppress": True})
    assert res.p_llm == 0.5
    assert res.confidence == 0.0
    assert res.determination == "SKIP"
    assert res.signals.get("reason") == "heuristic_suppress"


def test_lazy_singleton_thread_safe_first_call():
    """Concurrent first-call to DETECTOR.analyze must not double-instantiate the model."""
    pytest.importorskip("sentence_transformers")
    import threading
    from cairn.detectors.semantic_kinematics import _LazySemanticKinematics

    # Reset the class-level singleton to force a cold start in this test
    _LazySemanticKinematics._instance = None

    instances_observed: set[int] = set()
    errors: list[Exception] = []
    barrier = threading.Barrier(4)

    text = (
        "The river carved its way through limestone over millennia. "
        "Geologists call this karst topography. "
        "I once visited a karst cave system in Slovenia. "
        "It smelled like wet stone and old air. "
        "The math gets tangled when you try to bound the dimension empirically."
    )

    def worker():
        try:
            barrier.wait()  # release all 4 threads simultaneously
            det = _LazySemanticKinematics()
            det.analyze(text, config={})
            instances_observed.add(id(det._instance))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent first-call raised: {errors}"
    assert len(instances_observed) == 1, (
        f"expected exactly 1 SemanticKinematicsDetector instance across 4 threads, "
        f"saw {len(instances_observed)}"
    )

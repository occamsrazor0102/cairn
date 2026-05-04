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

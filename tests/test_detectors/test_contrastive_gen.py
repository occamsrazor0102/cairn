"""Contrastive generation detector tests."""
from unittest.mock import patch

import pytest

from cairn.detectors.contrastive_gen import ContrastiveGenDetector


@pytest.fixture
def detector():
    return ContrastiveGenDetector()


def test_contrastive_gen_registered():
    import cairn.detectors as registry
    assert "contrastive_gen" in registry.get_all_detectors()


def test_contrastive_gen_placed_in_phase3():
    from cairn.cascade import PHASE3_DETECTORS
    assert "contrastive_gen" in PHASE3_DETECTORS


def test_skips_without_task_metadata(detector):
    r = detector.analyze("Some text " * 20, {})
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "no_task_metadata"


def test_skips_without_provider(detector):
    r = detector.analyze("Some text " * 20, {"task_description": "Write a report."})
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "no_provider_configured"


def test_skips_without_api_key(detector, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = detector.analyze(
        "Some text " * 20,
        {"task_description": "Write a report.", "provider": "anthropic"},
    )
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "no_api_key"


def test_skips_on_short_text(detector):
    r = detector.analyze(
        "short.",
        {"task_description": "Write a report.", "provider": "anthropic",
         "api_key": "test-key"},
    )
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "too_short"


def test_high_similarity_maps_to_high_p_llm(detector):
    """When baselines are near-duplicates of the submission, mean similarity
    should be high and determination should reflect it."""
    submission = (
        "The study enrolled one hundred adult participants in a randomised "
        "double-blind trial comparing drug X against placebo. Primary endpoint "
        "was reduction in systolic blood pressure after twelve weeks. Secondary "
        "endpoints included adverse event profile and patient-reported quality "
        "of life scores assessed at baseline and at each follow-up visit. "
        "Statistical analysis used intention-to-treat principles with a "
        "prespecified noninferiority margin of five percent."
    )
    # Near-duplicate baselines — lexical Jaccard will be high.
    baselines = [submission, submission + " Secondary outcome included tolerability."]
    with patch.object(detector, "_generate_baselines", return_value=(baselines, [])):
        r = detector.analyze(
            submission,
            {"task_description": "Write a clinical trial summary.",
             "provider": "anthropic", "api_key": "test-key", "n_baselines": 2},
        )
    assert r.determination != "SKIP"
    assert r.signals["mean_similarity"] > 0.5
    assert r.p_llm >= 0.50


def test_low_similarity_maps_to_low_p_llm(detector):
    submission = (
        "Yesterday I hiked up the canyon trail with the dog and my sister. "
        "We saw two hawks and a coyote near the creek before it started "
        "raining. The dog chased a rabbit into the scrub and came back "
        "covered in burrs. My sister lost her water bottle somewhere near "
        "the switchbacks and we shared mine for the rest of the afternoon. "
        "On the drive home the car radio kept cutting out in the canyons."
    )
    baselines = [
        "The quarterly earnings report highlights significant revenue growth "
        "in the third quarter driven by strong subscription renewals and "
        "enterprise account expansion across all major regions.",
        "Clinical protocols must enumerate inclusion and exclusion criteria "
        "alongside prespecified primary and secondary endpoints and a full "
        "statistical analysis plan before enrolment begins.",
    ]
    with patch.object(detector, "_generate_baselines", return_value=(baselines, [])):
        r = detector.analyze(
            submission,
            {"task_description": "Write a hike account.",
             "provider": "anthropic", "api_key": "test-key", "n_baselines": 2},
        )
    assert r.determination != "SKIP"
    assert r.signals["mean_similarity"] < 0.3


def test_all_baseline_errors_yields_skip(detector):
    with patch.object(detector, "_generate_baselines", return_value=([], ["err1", "err2"])):
        r = detector.analyze(
            "Some long-enough submission text goes here " * 15,
            {"task_description": "Whatever.", "provider": "anthropic",
             "api_key": "test-key"},
        )
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "provider_error"


def test_skips_when_task_description_equals_submission(detector):
    """Privacy guard: if the caller passes the submission as the task
    description, refuse — otherwise the submission would ship to the provider."""
    submission = "This is the user's actual submission text " * 6  # 48 words
    r = detector.analyze(
        submission,
        {"task_description": submission, "provider": "anthropic", "api_key": "k"},
    )
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "task_description_overlaps_submission"


def test_skips_when_task_description_is_long_prefix_of_submission(detector):
    """Task descriptions are short by convention; a too-long task_description
    is a misuse signal."""
    submission = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z " * 4
    # A long task_description (~200 chars) that's also a prefix of submission.
    td = submission[:300]
    r = detector.analyze(
        submission,
        {"task_description": td, "provider": "anthropic", "api_key": "k"},
    )
    assert r.determination == "SKIP"
    assert r.signals.get("skipped") == "task_description_overlaps_submission"


def test_never_sends_submission_to_provider(detector):
    """Safety: _generate_baselines must receive the task description,
    never the submission text."""
    captured = {}

    def fake_gen(task_description, api_key, provider, model, n, config=None):
        captured["task"] = task_description
        return (["baseline one", "baseline two"], [])

    submission = "the secret confidential submission text goes here for " + ("xx " * 60)
    with patch.object(detector, "_generate_baselines", side_effect=fake_gen):
        detector.analyze(
            submission,
            {"task_description": "summarize a meeting", "provider": "anthropic",
             "api_key": "test-key"},
        )
    assert captured["task"] == "summarize a meeting"
    assert submission not in captured["task"]
    assert "confidential" not in captured["task"]


def test_openai_compat_without_base_url_does_not_crash(detector):
    """Regression: openai-compat path with missing base_url must return a
    friendly error, not NameError. The original bug was that _generate_baselines
    referenced `config` without it being a parameter, so the path crashed on
    `config.get("base_url")` with NameError before any base_url check ran."""
    baselines, errors = detector._generate_baselines(
        task_description="summarize a meeting",
        api_key="test-key",
        provider="openai-compat",
        model="gpt-4o-mini",
        n=2,
        config={},
    )
    assert baselines == []
    assert errors  # either "openai package not installed" or the base_url message — both fine


def test_openai_compat_omitted_config_kwarg_does_not_crash(detector):
    """Regression: _generate_baselines must accept an omitted config kwarg
    (defaults to None) without NameError. Guards the case where any future
    caller forgets to thread config through — the helper degrades gracefully
    rather than crashing."""
    baselines, errors = detector._generate_baselines(
        task_description="summarize a meeting",
        api_key="test-key",
        provider="openai-compat",
        model="gpt-4o-mini",
        n=2,
    )
    assert baselines == []
    assert errors


def test_analyze_threads_config_into_baseline_generation(detector):
    """Regression: analyze() must thread the full config dict through to
    _generate_baselines so the openai-compat base_url (and any future provider
    options) reaches the helper. The original bug elided this entirely."""
    captured = {}

    def fake_gen(task_description, api_key, provider, model, n, config=None):
        captured["config"] = config
        return [], ["mocked"]

    with patch.object(detector, "_generate_baselines", side_effect=fake_gen):
        detector.analyze(
            "Some long enough submission text " * 20,
            {"task_description": "summarize a meeting",
             "provider": "openai-compat",
             "api_key": "test-key",
             "base_url": "https://api.example.com/v1"},
        )
    assert captured["config"] is not None
    assert captured["config"].get("base_url") == "https://api.example.com/v1"

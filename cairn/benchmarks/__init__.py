"""CAIRN competitor adapter shelf.

Houses adapters for third-party LLM-text detectors so we can run
head-to-head evaluations against CAIRN on the same datasets via
``cairn.evaluation.runner.run_eval``. Each adapter is gated on its
optional dependencies / API keys via ``is_available()`` so importing
the package is cheap.

See ``docs/superpowers/specs/2026-05-07-post-honest-calibration-roadmap.md``
section A1 for the canonical spec.
"""

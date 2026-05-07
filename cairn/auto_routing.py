"""Profile auto-routing recommendation layer (B3).

Maps a :class:`cairn.contracts.RouterDecision` (already classified
prompt/prose/mixed/insufficient by :mod:`cairn.router`) and optional
task metadata to a calibrated profile recommendation.

Manual profile selection remains the default — the recommendation is
exposed via the ``recommend_profile`` RPC for opt-in use under
``runtime.auto_route: true``. Today this module only computes the
recommendation; downstream wiring into
:meth:`cairn.runtime.RuntimeContext.switch_profile` is a follow-up.

Logic (see ``docs/superpowers/specs/2026-05-07-post-honest-calibration-roadmap.md``
section B3):

* ``domain == "prompt"``        -> ``prompts_only``        (~0.95)
* ``domain == "prose"`` & wc>=50 -> ``master_minus_arxiv``  (~0.90)
* ``domain == "prose"`` & wc<50  -> ``master_minus_arxiv``  (~0.6, caveat short_text)
* ``domain == "mixed"``         -> ``master_minus_arxiv``  (~0.7, caveat mixed_input)
* ``domain == "insufficient"``  -> ``master_minus_arxiv``  (~0.4, caveat insufficient_text)
* ``task_metadata["language"]`` set and not English -> add caveat ``non-english``
  (a calibrated multilingual profile is not yet shipped per A4)
* If the recommended profile is not in ``available_profiles`` the
  function walks a priority order and adds the ``fallback`` caveat,
  lowering confidence by 0.2.
* If ``available_profiles`` is empty the function returns a zero-confidence
  ``default`` recommendation tagged ``no_profiles``.

The returned :class:`ProfileRecommendation` is honest about uncertainty:
the ``confidence`` field is bounded to ``[0, 1]`` and the ``caveats``
tuple is the structural channel for the UI / downstream code to decide
whether to surface a banner.
"""
from __future__ import annotations

from dataclasses import dataclass

from cairn.contracts import RouterDecision


# Priority order used when the recommended profile is unavailable.
# ``master_minus_arxiv`` is the production prose default; the remaining
# entries are progressively-less-calibrated fallbacks ending at
# ``default`` (no calibration artifacts loaded).
_FALLBACK_ORDER: tuple[str, ...] = (
    "master_minus_arxiv",
    "calibrated_calib2000",
    "general_hc3",
    "default",
)

# Languages we treat as English for routing purposes. ``None`` and
# the empty string mean "unspecified" -> assume English so callers
# don't get a ``non-english`` caveat for unset metadata.
_ENGLISH_TOKENS: frozenset[str] = frozenset({"en", "english", ""})

# Word-count threshold below which prose carries a ``short_text`` caveat
# and reduced confidence. Sourced from the spec.
_SHORT_PROSE_WORDS: int = 50


@dataclass(frozen=True)
class ProfileRecommendation:
    """Auto-routing recommendation for a single submission.

    Attributes:
        profile:    Calibrated profile name to load.
        confidence: Confidence in ``[0, 1]``. ``0.0`` means the
                    recommendation is a structural placeholder, not a
                    picked-with-evidence answer.
        reason:     Human-readable rationale, shown by the UI when
                    auto-routing surfaces the recommendation.
        caveats:    Tuple of well-known caveat tokens. Currently
                    defined: ``short_text``, ``mixed_input``,
                    ``insufficient_text``, ``non-english``,
                    ``fallback``, ``no_profiles``.
    """

    profile: str
    confidence: float
    reason: str
    caveats: tuple[str, ...]


def _is_non_english(task_metadata: dict | None) -> bool:
    if not task_metadata:
        return False
    lang = task_metadata.get("language")
    # Treat None / non-strings as "unspecified" -> English.
    if not isinstance(lang, str):
        return False
    return lang.strip().lower() not in _ENGLISH_TOKENS


def _apply_fallback(
    desired: str,
    base_confidence: float,
    base_reason: str,
    base_caveats: tuple[str, ...],
    available_profiles: list[str],
) -> ProfileRecommendation:
    """Pick the first profile in priority order that's actually available.

    Adds the ``fallback`` caveat and drops confidence by 0.2 (floored at
    0). The returned ``reason`` notes both the originally-recommended
    profile and the chosen fallback.
    """
    chosen: str | None = None
    for candidate in _FALLBACK_ORDER:
        if candidate in available_profiles:
            chosen = candidate
            break
    if chosen is None:
        # ``available_profiles`` is non-empty but contains only profiles
        # we don't recognise. Pick the first one alphabetically so
        # callers get a deterministic answer; flag heavily.
        chosen = sorted(available_profiles)[0]

    fallback_caveats = base_caveats + ("fallback",)
    fallback_confidence = max(0.0, base_confidence - 0.2)
    fallback_reason = (
        f"{base_reason}; recommended '{desired}' is unavailable, "
        f"falling back to '{chosen}'"
    )
    return ProfileRecommendation(
        profile=chosen,
        confidence=fallback_confidence,
        reason=fallback_reason,
        caveats=fallback_caveats,
    )


def recommend_profile(
    router_decision: RouterDecision,
    task_metadata: dict | None,
    available_profiles: list[str],
) -> ProfileRecommendation:
    """Pick a calibrated profile for the given router decision.

    See module docstring for the full decision table. The function is
    pure — it does not load profiles, switch contexts, or touch disk.

    Args:
        router_decision:    Output of :meth:`cairn.router.TextRouter.route`.
        task_metadata:      Optional caller-supplied metadata. Today the
                            only key the function consults is ``language``;
                            unrecognised keys are ignored.
        available_profiles: List of profile names actually loadable on
                            this host. Empty -> structural fallback to
                            ``"default"`` with zero confidence.

    Returns:
        :class:`ProfileRecommendation` with the chosen profile, an honest
        confidence in ``[0, 1]``, a human-readable reason, and a tuple
        of caveat tokens for downstream UI / decision use.
    """
    # No profiles loaded -> structural empty answer. The caller can
    # decide whether to surface this as an error or a "configure
    # calibration" banner.
    if not available_profiles:
        return ProfileRecommendation(
            profile="default",
            confidence=0.0,
            reason="no profiles available",
            caveats=("no_profiles",),
        )

    domain = router_decision.domain
    word_count = router_decision.word_count

    if domain == "prompt":
        desired = "prompts_only"
        confidence = 0.95
        reason = "domain classified as prompt"
        caveats: tuple[str, ...] = ()
    elif domain == "prose":
        desired = "master_minus_arxiv"
        if word_count >= _SHORT_PROSE_WORDS:
            confidence = 0.90
            reason = f"long prose ({word_count} words)"
            caveats = ()
        else:
            confidence = 0.6
            reason = f"short prose ({word_count} words)"
            caveats = ("short_text",)
    elif domain == "mixed":
        desired = "master_minus_arxiv"
        confidence = 0.7
        reason = "mixed prompt/prose input"
        caveats = ("mixed_input",)
    elif domain == "insufficient":
        desired = "master_minus_arxiv"
        confidence = 0.4
        reason = "insufficient text for confident routing"
        caveats = ("insufficient_text",)
    else:
        # Defensive: unknown domain. Treat like insufficient with an
        # extra caveat so the surface explains what happened.
        desired = "master_minus_arxiv"
        confidence = 0.3
        reason = f"unrecognised router domain '{domain}'"
        caveats = ("insufficient_text",)

    # Non-English carries a caveat regardless of domain. We do NOT
    # switch to a multilingual profile because none ships today (A4 is
    # still a probe). Leave the chosen profile alone; the UI surfaces
    # the caveat so users know the calibration claim is weaker.
    if _is_non_english(task_metadata):
        caveats = caveats + ("non-english",)

    if desired not in available_profiles:
        return _apply_fallback(
            desired,
            base_confidence=confidence,
            base_reason=reason,
            base_caveats=caveats,
            available_profiles=available_profiles,
        )

    # Clamp defensively in case future logic edits push out of [0, 1].
    confidence = max(0.0, min(1.0, confidence))
    return ProfileRecommendation(
        profile=desired,
        confidence=confidence,
        reason=reason,
        caveats=caveats,
    )

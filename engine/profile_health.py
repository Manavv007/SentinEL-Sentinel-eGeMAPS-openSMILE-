"""Profile separation and collapse diagnostics."""

from __future__ import annotations

from typing import Any

import config

from engine.profile_memory import BehavioralProfile


def profile_health(
    script: BehavioralProfile,
    natural: BehavioralProfile,
    *,
    sample_features: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute script/natural strength and distance; warn on collapse."""
    script_strength = script.profile_strength()
    natural_strength = natural.profile_strength()

    distance = script.profile_distance(natural)
    if sample_features and natural.sample_count > 0:
        distance = max(
            distance,
            1.0
            - (
                script.similarity(sample_features)
                + natural.similarity_mature(sample_features)
            )
            / 2.0,
        )

    collapse_warning = None
    if (
        natural.sample_count >= config.NATURAL_PROFILE_MIN_SAMPLES
        and distance < config.PROFILE_COLLAPSE_DISTANCE
    ):
        collapse_warning = "POSSIBLE PROFILE COLLAPSE — NATURAL profile dilution"

    return {
        "script_profile_strength": round(script_strength, 6),
        "natural_profile_strength": round(natural_strength, 6),
        "profile_distance": round(distance, 6),
        "collapse_warning": collapse_warning,
    }

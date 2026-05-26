"""Negative evidence — signals that suppress script-reading suspicion."""

from __future__ import annotations

from typing import Any

import config


def spontaneity_suppression(
    features: dict[str, float],
    naturality_breakdown: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """
    Return (suppression in [0,1], component breakdown).

    Higher suppression => reduce final contrastive suspicion.
    """
    components = {
        "filler_burst": _filler_burst(features),
        "retrieval_pause": _retrieval_pause(features),
        "self_correction": _self_correction(features, naturality_breakdown),
        "irregular_pacing": _irregular_pacing(features),
        "rate_fluctuation": _rate_fluctuation(features),
        "hesitation_cluster": _hesitation_cluster(features),
        "semantic_repair": _semantic_repair(features),
        "cognitive_wobble": _cognitive_wobble(features),
        "retrieval_friction": _retrieval_friction_cog(features),
    }

    weights = {
        "filler_burst": 0.12,
        "retrieval_pause": 0.14,
        "self_correction": 0.12,
        "irregular_pacing": 0.12,
        "rate_fluctuation": 0.08,
        "hesitation_cluster": 0.08,
        "semantic_repair": 0.16,
        "cognitive_wobble": 0.10,
        "retrieval_friction": 0.08,
    }

    raw = sum(components[k] * weights[k] for k in components)
    scaled = min(1.0, raw * config.SPONTANEITY_SUPPRESSION_SCALE)
    return round(scaled, 6), {k: round(v, 6) for k, v in components.items()}


def _filler_burst(features: dict[str, float]) -> float:
    rate = features.get("ling_filler_rate_per_30s", 0.0)
    clusters = features.get("ling_filler_clusters", 0.0)
    if rate < 0.5:
        return 0.0
    return min(1.0, rate / 6.0) * 0.5 + min(1.0, clusters / 2.0) * 0.5


def _retrieval_pause(features: dict[str, float]) -> float:
    pause = features.get("ling_retrieval_pause_max", 0.0)
    if pause < 0.35:
        return 0.0
    if pause > 2.0:
        return 0.85
    return min(1.0, (pause - 0.35) / 1.2)


def _self_correction(
    features: dict[str, float],
    breakdown: dict[str, float],
) -> float:
    count = features.get("ling_self_corrections", 0.0)
    reps = features.get("ling_repetition_rate", 0.0)
    bd = breakdown.get("self_correction", 0.0)
    return min(1.0, max(count / 2.0, reps, bd))


def _irregular_pacing(features: dict[str, float]) -> float:
    ent = features.get("ling_pause_entropy", 0.0)
    gv = features.get("ling_gap_variance", 0.0)
    ent_s = min(1.0, ent / max(config.PAUSE_ENTROPY_NORM, 1e-6))
    gv_s = min(1.0, gv / 0.02)
    return 0.55 * ent_s + 0.45 * gv_s


def _rate_fluctuation(features: dict[str, float]) -> float:
    wps = features.get("ling_wps", 0.0)
    if wps <= 0:
        return 0.0
    # Moderate variance from ideal monotone reading pace (~2.0-3.5 wps)
    deviation = abs(wps - 2.8) / 2.8
    return min(1.0, deviation * 1.2)


def _hesitation_cluster(features: dict[str, float]) -> float:
    clusters = features.get("ling_filler_clusters", 0.0)
    rate = features.get("ling_filler_rate_per_30s", 0.0)
    if clusters >= 1 and 1.0 <= rate <= 6.0:
        return min(1.0, clusters / 3.0)
    return 0.0


def _semantic_repair(features: dict[str, float]) -> float:
    return min(1.0, float(features.get("cog_semantic_repair", 0.0)) * 1.15)


def _cognitive_wobble(features: dict[str, float]) -> float:
    return min(1.0, float(features.get("cog_cognitive_wobble", 0.0)))


def _retrieval_friction_cog(features: dict[str, float]) -> float:
    return min(1.0, float(features.get("cog_retrieval_friction", 0.0)))

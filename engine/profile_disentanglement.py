"""Profile purification: separate delivery-style from cognitive-guidance similarity."""

from __future__ import annotations

from typing import Any

import numpy as np

import config
from engine.profile_memory import BehavioralProfile

# Delivery / polish — poor script-reading discriminators (speaker-style leakage).
STYLE_METRIC_WEIGHTS: dict[str, float] = {
    "ling_wps": 0.28,
    "ling_gap_variance": 0.32,
    "ling_pause_entropy": 0.30,
    "ling_filler_rate_per_30s": 0.25,
    "ling_filler_clusters": 0.25,
    "ling_has_words": 0.0,
    "ling_scope_fallback": 0.0,
    "acoustic_jitter_local": 0.22,
    "acoustic_shimmer_local": 0.22,
    "acoustic_hnr": 0.22,
    "acoustic_MeanVoicedSegmentLengthSec": 0.25,
    "acoustic_MeanUnvoicedSegmentLength": 0.25,
    "acoustic_F0semitoneFrom27.5Hz_sma3nz_stddevNorm": 0.28,
    "acoustic_pitch_range_hz": 0.30,
    "acoustic_pitch_delta": 0.32,
}

# Cognitive guidance — script-specific preconstructed delivery patterns.
COGNITIVE_GUIDANCE_WEIGHTS: dict[str, float] = {
    "cog_guided_explanation_index": 1.0,
    "cog_concept_compression": 0.95,
    "cog_retrieval_friction": 0.90,
    "cog_semantic_drift": 0.88,
    "cog_assembly_variance": 0.85,
    "cog_semantic_repair": 0.88,
    "cog_cognitive_wobble": 0.82,
    "cog_fluency_trap": 0.75,
    "cog_spontaneity_index": 0.70,
    "ling_self_corrections": 0.72,
    "ling_repetition_rate": 0.68,
    "ling_retrieval_pause_max": 0.55,
    "ling_technical_density": 0.40,
}

# Stored in SCRIPT profile at calibration — cognitive + acoustic + selective linguistic.
SCRIPT_CALIBRATION_KEEP_PREFIXES = ("cog_", "acoustic_")
SCRIPT_CALIBRATION_KEEP_EXACT = frozenset(
    {
        "ling_self_corrections",
        "ling_repetition_rate",
        "ling_retrieval_pause_max",
        "ling_technical_density",
        "ling_filler_rate_per_30s",
        "ling_filler_clusters",
        "ling_pause_entropy",
        "ling_gap_variance",
    }
)


def _metric_weight(key: str, bundle: dict[str, float], default: float) -> float:
    if key in bundle:
        return bundle[key]
    if key.startswith("cog_"):
        return config.SCRIPT_COGNITIVE_METRIC_WEIGHT
    if key.startswith("ling_"):
        return config.SCRIPT_LINGUISTIC_DELIVERY_WEIGHT
    if key.startswith("acoustic_"):
        return config.SCRIPT_ACOUSTIC_DELIVERY_WEIGHT
    if key.startswith("video_"):
        return 0.0
    return default


def script_metric_weights(feature_keys: set[str] | frozenset[str]) -> dict[str, float]:
    """Purified SCRIPT similarity weights — cognitive high, delivery low."""
    weights: dict[str, float] = {}
    for key in feature_keys:
        if key in STYLE_METRIC_WEIGHTS:
            weights[key] = STYLE_METRIC_WEIGHTS[key] * config.SCRIPT_STYLE_METRIC_WEIGHT
        elif key in COGNITIVE_GUIDANCE_WEIGHTS:
            weights[key] = COGNITIVE_GUIDANCE_WEIGHTS[key] * config.SCRIPT_COGNITIVE_METRIC_WEIGHT
        elif key.startswith("cog_"):
            weights[key] = config.SCRIPT_COGNITIVE_METRIC_WEIGHT
        elif key.startswith("ling_"):
            weights[key] = config.SCRIPT_LINGUISTIC_DELIVERY_WEIGHT
        elif key.startswith("acoustic_"):
            weights[key] = config.SCRIPT_ACOUSTIC_DELIVERY_WEIGHT
        elif key.startswith("video_"):
            weights[key] = 0.0
    return weights


def natural_metric_weights(feature_keys: set[str] | frozenset[str]) -> dict[str, float]:
    """NATURAL anchor weights — broader spontaneous cognition + delivery diversity."""
    weights: dict[str, float] = {}
    for key in feature_keys:
        if key.startswith("cog_"):
            if key == "cog_spontaneity_index":
                weights[key] = config.NATURAL_COGNITIVE_METRIC_WEIGHT * 1.15
            elif key in ("cog_cognitive_wobble", "cog_semantic_repair", "cog_semantic_drift"):
                weights[key] = config.NATURAL_COGNITIVE_METRIC_WEIGHT * 1.08
            else:
                weights[key] = config.NATURAL_COGNITIVE_METRIC_WEIGHT
        elif key.startswith("ling_"):
            weights[key] = config.NATURAL_LINGUISTIC_METRIC_WEIGHT
        elif key.startswith("acoustic_"):
            weights[key] = config.NATURAL_ACOUSTIC_METRIC_WEIGHT
        elif key.startswith("video_"):
            weights[key] = 0.0
    return weights


def filter_features_for_script_calibration(features: dict[str, float]) -> dict[str, float]:
    """Keep cognitive-guidance + acoustic features when building SCRIPT profile from calibration."""
    if not config.SCRIPT_PROFILE_PURIFY_CALIBRATION:
        return {k: v for k, v in features.items() if np.isfinite(v)}

    out: dict[str, float] = {}
    for key, value in features.items():
        if not np.isfinite(value):
            continue
        if key.startswith(SCRIPT_CALIBRATION_KEEP_PREFIXES):
            out[key] = value
        elif key in SCRIPT_CALIBRATION_KEEP_EXACT:
            out[key] = value
    return out


def calibration_feature_row(features: dict[str, float]) -> dict[str, float]:
    """
    Build a non-empty SCRIPT calibration row.
    Purified cognitive/linguistic/acoustic first; fall back to acoustic+ling if empty.
    """
    row = filter_features_for_script_calibration(features)
    if row:
        return row
    # Transcript/cognitive missing — acoustic reading fingerprint still valid
    return {
        k: float(v)
        for k, v in features.items()
        if (k.startswith("acoustic_") or k.startswith("ling_")) and np.isfinite(v)
    }


def compute_profile_similarity_bundle(
    profile: BehavioralProfile,
    features: dict[str, float],
    *,
    metric_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return purified, style, and cognitive-channel similarities."""
    keys = set(profile.metric_stats().keys()) & set(features.keys())
    if not keys:
        return {
            "similarity": 0.0,
            "style_similarity": 0.0,
            "cognitive_similarity": 0.0,
        }

    style_w = {
        k: STYLE_METRIC_WEIGHTS.get(k, _metric_weight(k, {}, config.SCRIPT_ACOUSTIC_DELIVERY_WEIGHT))
        for k in keys
        if k in STYLE_METRIC_WEIGHTS
        or k.startswith("acoustic_")
        or k in ("ling_wps", "ling_gap_variance", "ling_pause_entropy", "ling_filler_rate_per_30s")
    }
    cog_w = {
        k: COGNITIVE_GUIDANCE_WEIGHTS.get(k, config.SCRIPT_COGNITIVE_METRIC_WEIGHT)
        for k in keys
        if k.startswith("cog_") or k in COGNITIVE_GUIDANCE_WEIGHTS
    }

    purified = profile.similarity(features, metric_weights=metric_weights or script_metric_weights(keys))
    style_sim = profile.similarity(features, metric_weights=style_w) if style_w else 0.0
    cog_sim = profile.similarity(features, metric_weights=cog_w) if cog_w else purified

    return {
        "similarity": float(purified),
        "style_similarity": float(style_sim),
        "cognitive_similarity": float(cog_sim),
    }


def adjust_script_similarity_for_disentanglement(
    *,
    script_similarity: float,
    style_similarity: float,
    cognitive_similarity: float,
    cognitive_spontaneity: float = 0.0,
    cognitive_wobble: float = 0.0,
    semantic_repair: float = 0.0,
    naturality_score: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """
    Reduce script authority when similarity is driven by delivery-style overlap
    rather than cognitive-guidance patterns, especially for fluent natural speakers.
    """
    adjusted = float(script_similarity)
    meta: dict[str, float] = {
        "script_similarity_raw": round(script_similarity, 6),
        "style_similarity": round(style_similarity, 6),
        "cognitive_similarity": round(cognitive_similarity, 6),
    }

    style_leak = max(0.0, style_similarity - cognitive_similarity)
    meta["style_leak"] = round(style_leak, 6)

    if style_leak >= config.PROFILE_STYLE_LEAK_MIN:
        leak_dampen = min(
            config.FLUENT_STYLE_LEAK_DAMPEN_MAX,
            style_leak * config.FLUENT_STYLE_LEAK_FACTOR,
        )
        if cognitive_spontaneity >= config.FLUENT_NATURAL_SPONTANEITY_FLOOR * 0.7:
            leak_dampen *= 1.0 + min(0.35, cognitive_spontaneity * 0.4)
        if cognitive_wobble >= config.COGNITIVE_TURBULENCE_WOBBLE_MIN:
            leak_dampen *= 1.0 + min(0.25, cognitive_wobble * 0.5)
        if semantic_repair >= config.COGNITIVE_TURBULENCE_REPAIR_MIN:
            leak_dampen *= 1.08
        if naturality_score >= config.WEAK_SUSPICION_NATURALITY_CAP * 0.85:
            leak_dampen *= 1.06
        adjusted *= max(0.45, 1.0 - leak_dampen)
        meta["style_leak_dampen"] = round(leak_dampen, 6)

    # High script similarity alone should not dominate when turbulence is present.
    if (
        adjusted >= config.STRONG_SCRIPT_THRESHOLD * 0.88
        and cognitive_spontaneity >= config.COGNITIVE_TURBULENCE_SPONTANEITY_MIN
        and (
            cognitive_wobble >= config.COGNITIVE_TURBULENCE_WOBBLE_MIN
            or semantic_repair >= config.COGNITIVE_TURBULENCE_REPAIR_MIN
        )
    ):
        turb_dampen = min(
            config.COGNITIVE_TURBULENCE_SCRIPT_DAMPEN_MAX,
            config.COGNITIVE_TURBULENCE_SCRIPT_DAMPEN
            * (cognitive_wobble + semantic_repair * 0.8),
        )
        adjusted *= max(0.50, 1.0 - turb_dampen)
        meta["turbulence_dampen"] = round(turb_dampen, 6)

    meta["script_similarity_adjusted"] = round(adjusted, 6)
    return float(max(0.0, min(1.0, adjusted))), meta


def fluent_natural_learning_eligible(
    features: dict[str, float],
    nat_breakdown: dict[str, float],
    *,
    script_similarity: float,
    cognitive_spontaneity: float,
    guided_explanation: float,
    naturality_learning: float,
) -> tuple[bool, str]:
    """
    Alternate NATURAL update path for fluent technical spontaneous speech
    (not only hesitant / high-entropy patterns).
    """
    if not config.ENABLE_FLUENT_NATURAL_PROFILE_LEARNING:
        return False, "fluent_path_disabled"

    if script_similarity >= config.FLUENT_NATURAL_UPDATE_SCRIPT_CEILING:
        return False, "fluent_path_script_too_high"

    if cognitive_spontaneity < config.FLUENT_NATURAL_LEARNING_MIN_SPONTANEITY:
        return False, "fluent_path_low_spontaneity"

    if guided_explanation >= config.FLUENT_NATURAL_LEARNING_MAX_GUIDED:
        return False, "fluent_path_guided_too_high"

    if naturality_learning < config.FLUENT_NATURAL_LEARNING_MIN_NATURALITY:
        return False, "fluent_path_low_learning_naturality"

    wobble = float(features.get("cog_cognitive_wobble", 0.0))
    repair = float(features.get("cog_semantic_repair", 0.0))
    drift = float(features.get("cog_semantic_drift", 0.0))
    friction = float(features.get("cog_retrieval_friction", 0.0))

    turbulence = wobble + repair * 0.85 + drift * 0.5 + friction * 0.35
    if turbulence >= config.FLUENT_NATURAL_LEARNING_MIN_TURBULENCE:
        return True, "fluent_natural_turbulence_path"

    if cognitive_spontaneity >= config.FLUENT_NATURAL_LEARNING_STRONG_SPONTANEITY:
        if nat_breakdown.get("self_correction", 0.0) >= 0.30:
            return True, "fluent_natural_repair_path"
        if features.get("ling_technical_density", 0.0) >= 0.06:
            if features.get("ling_pause_entropy", 0.0) >= config.FLUENT_NATURAL_MIN_PAUSE_ENTROPY:
                return True, "fluent_technical_natural_path"
        if features.get("acoustic_pitch_delta", 0.0) and abs(
            float(features.get("acoustic_pitch_delta", 0.0))
        ) >= config.FLUENT_NATURAL_MIN_PITCH_DELTA:
            return True, "fluent_natural_pitch_dynamics_path"

    return False, "fluent_path_insufficient_evidence"

"""Calibrated suspicion tiers, weighted accumulation, and final decision logic."""

from __future__ import annotations

from enum import Enum
from typing import Any

import config


class SuspicionLevel(str, Enum):
    NONE = "NONE"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


def level_weight(level: SuspicionLevel) -> float:
    if level == SuspicionLevel.NONE:
        return 0.0
    if level == SuspicionLevel.WEAK:
        return config.SUSPICION_WEIGHT_WEAK
    if level == SuspicionLevel.MODERATE:
        return config.SUSPICION_WEIGHT_MODERATE
    return config.SUSPICION_WEIGHT_STRONG


def classify_suspicion_level(
    *,
    contrastive_score: float,
    script_similarity: float,
    naturality_score: float = 0.0,
    natural_similarity: float = 0.0,
    cognitive_spontaneity: float = 0.0,
    natural_profile_samples: int = -1,
) -> SuspicionLevel:
    """
    Calibrated tiers — weak scores (0.17–0.23) must not equal strong (0.38+).
    """
    c = float(max(0.0, contrastive_score))
    script = float(script_similarity)

    if c < config.SUSPICION_TIER_WEAK:
        base = SuspicionLevel.NONE
    elif c < config.SUSPICION_TIER_MODERATE:
        base = SuspicionLevel.WEAK
    elif c < config.SUSPICION_TIER_STRONG:
        base = SuspicionLevel.MODERATE
    else:
        base = SuspicionLevel.STRONG

    # Script-dominance can bump one tier, but not from NONE on tiny contrastive alone
    if script >= config.STRONG_SCRIPT_THRESHOLD and c >= config.SUSPICION_SCRIPT_BUMP_MIN:
        bump = SuspicionLevel.MODERATE if base == SuspicionLevel.WEAK else SuspicionLevel.STRONG
        if base == SuspicionLevel.NONE and c >= config.SUSPICION_TIER_WEAK:
            bump = SuspicionLevel.WEAK
        order = [SuspicionLevel.NONE, SuspicionLevel.WEAK, SuspicionLevel.MODERATE, SuspicionLevel.STRONG]
        base = order[max(order.index(base), order.index(bump))]

    # Weak suspicion suppression: high spontaneity caps tier
    spont_cap = (
        naturality_score >= config.WEAK_SUSPICION_NATURALITY_CAP
        or cognitive_spontaneity >= config.WEAK_SUSPICION_COGNITIVE_SPONTANEITY_CAP
    )
    if (
        base in (SuspicionLevel.WEAK, SuspicionLevel.MODERATE)
        and spont_cap
        and natural_similarity >= config.WEAK_SUSPICION_NATURAL_SIM_CAP
        and script < config.STRONG_SCRIPT_THRESHOLD
    ):
        base = SuspicionLevel.WEAK if base == SuspicionLevel.MODERATE else SuspicionLevel.NONE

    # NATURAL profile cold start: do not treat weak-only script prior as suspicious
    if natural_profile_samples == 0:
        tier_boost = config.COLD_START_SUSPICION_TIER_BOOST
        if c < config.SUSPICION_TIER_WEAK + tier_boost:
            base = SuspicionLevel.NONE
        elif (
            base == SuspicionLevel.WEAK
            and script < config.COLD_START_WEAK_SCRIPT_CEILING
        ):
            base = SuspicionLevel.NONE

    return base


def nonlinear_level_contribution(
    level: SuspicionLevel,
    contrastive_score: float,
    script_similarity: float,
) -> float:
    """Nonlinear evidence units contributed by this window."""
    w = level_weight(level)
    if w <= 0:
        return 0.0
    score_factor = contrastive_score ** config.SUSPICION_NONLINEAR_POWER
    script_factor = 1.0
    if script_similarity >= config.STRONG_SCRIPT_THRESHOLD:
        script_factor = 1.0 + config.SUSPICION_STRONG_SCRIPT_FACTOR * (
            (script_similarity - config.STRONG_SCRIPT_THRESHOLD) / 0.32
        )
    return float(w * score_factor * script_factor)


def ewma_input_for_level(
    level: SuspicionLevel,
    contrastive_score: float,
) -> float:
    """Asymmetric EWMA input — tracks behavioral state on every window."""
    c = float(max(0.0, contrastive_score))
    if level == SuspicionLevel.NONE:
        return c * config.EWMA_BEHAVIORAL_TRACK_SCALE
    if level == SuspicionLevel.WEAK:
        return c * config.EWMA_WEAK_INPUT_SCALE
    if level == SuspicionLevel.MODERATE:
        return c * config.EWMA_MODERATE_INPUT_SCALE
    return c * config.EWMA_STRONG_INPUT_SCALE


def aggregate_answer_evidence(
    windows: list[dict[str, Any]],
) -> dict[str, float]:
    """Weighted suspicious density from tiered windows."""
    if not windows:
        return {
            "weighted_evidence": 0.0,
            "strong_count": 0.0,
            "moderate_count": 0.0,
            "weak_count": 0.0,
            "strong_ratio": 0.0,
            "moderate_plus_ratio": 0.0,
            "weak_ratio": 0.0,
            "longest_weak_streak": 0.0,
            "weak_density": 0.0,
            "max_level_rank": 0.0,
        }

    total_w = 0.0
    strong = moderate = weak = 0
    max_rank = 0
    longest_weak = 0
    current_weak = 0
    rank = {
        SuspicionLevel.NONE.value: 0,
        SuspicionLevel.WEAK.value: 1,
        SuspicionLevel.MODERATE.value: 2,
        SuspicionLevel.STRONG.value: 3,
    }

    for w in windows:
        level_str = w.get("suspicion_level", SuspicionLevel.NONE.value)
        level = SuspicionLevel(level_str) if level_str in rank else SuspicionLevel.NONE
        c = float(w.get("contrastive_score", 0))
        s = float(w.get("script_similarity", 0))
        total_w += nonlinear_level_contribution(level, c, s)
        max_rank = max(max_rank, rank.get(level.value, 0))
        if level == SuspicionLevel.STRONG:
            strong += 1
            current_weak = 0
        elif level == SuspicionLevel.MODERATE:
            moderate += 1
            current_weak = 0
        elif level == SuspicionLevel.WEAK:
            weak += 1
            current_weak += 1
            longest_weak = max(longest_weak, current_weak)
        else:
            current_weak = 0

    n = len(windows)
    weak_ratio = weak / n
    # weak density: sustained weak matters more than sparse weak spikes
    weak_density = (weak_ratio * min(1.0, longest_weak / max(3, n // 3))) if weak else 0.0
    return {
        "weighted_evidence": round(total_w, 4),
        "strong_count": float(strong),
        "moderate_count": float(moderate),
        "weak_count": float(weak),
        "strong_ratio": round(strong / n, 4),
        "moderate_plus_ratio": round((strong + moderate) / n, 4),
        "weak_ratio": round(weak_ratio, 4),
        "longest_weak_streak": float(longest_weak),
        "weak_density": round(float(weak_density), 4),
        "weak_only_ratio": round(weak / n, 4) if weak and not strong and not moderate else 0.0,
        "max_level_rank": float(max_rank),
    }


def compute_calibrated_composite(
    *,
    ewma: float,
    peak_ewma: float,
    weighted_evidence: float,
    margin: float,
    horizon: Any | None = None,
    suspicion_momentum: float = 0.0,
    streak_boost: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Composite score driven by evidence quality + momentum-preserved peak."""
    evidence_norm = weighted_evidence / max(config.SUSPICION_EVIDENCE_NORM, 1e-6)
    evidence_component = min(1.0, evidence_norm) * margin * config.EVIDENCE_COMPOSITE_SCALE

    peak_blend = config.PEAK_EWMA_BLEND
    if suspicion_momentum >= config.MOMENTUM_MED_THRESHOLD:
        peak_blend = max(peak_blend, config.PEAK_EWMA_MOMENTUM_BLEND)

    composite_base = float(ewma) * config.COMPOSITE_EWMA_BLEND
    peak_evidence = max(peak_ewma * peak_blend, evidence_component)
    persistence = min(1.0, evidence_norm)
    if suspicion_momentum >= config.MOMENTUM_MED_THRESHOLD:
        persistence = min(1.0, persistence + 0.15 * suspicion_momentum)
    alpha = min(1.0, config.PEAK_CREDIBILITY_GAIN * persistence)
    composite = composite_base + alpha * max(0.0, peak_evidence - composite_base)
    composite += streak_boost

    if horizon and getattr(horizon, "script_dominance_active", False):
        floor = float(getattr(horizon, "answer_contrastive_floor", 0.0) or 0.0)
        composite = max(composite, floor * config.ANSWER_FLOOR_SCALE * 0.9)

    return float(composite), {
        "evidence_component": round(evidence_component, 6),
        "evidence_norm": round(evidence_norm, 4),
        "peak_blend_used": round(peak_blend, 4),
    }


def resolve_answer_status(
    *,
    composite: float,
    weighted_evidence: float,
    strong_ratio: float,
    moderate_plus_ratio: float,
    consecutive_strong: int,
    consecutive_moderate_plus: int,
    margin: float,
    horizon: Any | None = None,
    duration_sec: float = 0.0,
    weak_only_dominant: bool = False,
    longest_strong_streak: int = 0,
    lifetime_strong_ratio: float = 0.0,
    peak_ewma: float = 0.0,
    strong_window_count: int = 0,
    suspicion_momentum: float = 0.0,
    weak_ratio: float = 0.0,
    longest_weak_streak: int = 0,
    avg_script_similarity: float = 0.0,
    weak_consistency_score: float = 0.0,
    persistent_weak_authority_active: bool = False,
    recovery_strength: float = 0.0,
    moderate_window_count: int = 0,
    window_count: int = 0,
    flat_suspicious_flow_active: bool = False,
    consistency_authority_score: float = 0.0,
    natural_breathing_detected: bool = False,
) -> tuple[str, str]:
    """
    Returns (status, confidence).
    AMBIGUOUS is the default for borderline; PROBABLE needs quality evidence.
    """
    script_dom = bool(horizon and getattr(horizon, "script_dominance_active", False))
    low_drift = float(getattr(horizon, "low_drift_score", 0.0) or 0.0) if horizon else 0.0

    # Short answers: require reliable evidence volume before AMBIGUOUS.
    short_unreliable = (
        duration_sec > 0
        and (
            duration_sec <= config.TEMPORAL_SHORT_ANSWER_SEC
            or window_count < config.TEMPORAL_SHORT_MIN_WINDOWS
        )
        and strong_window_count == 0
        and moderate_window_count < config.SHORT_ANSWER_AMBIGUOUS_MIN_MODERATE_WINDOWS
    )
    if short_unreliable and composite < margin * config.SHORT_ANSWER_AMBIGUOUS_COMPOSITE_RATIO:
        return "CLEAR", "LOW"

    # Consistency authority: flat elevated weak flow without strong peaks (Answer 5).
    if flat_suspicious_flow_active and strong_window_count == 0 and not natural_breathing_detected:
        if consistency_authority_score >= config.CONSISTENCY_AUTHORITY_MIN_SCORE + 0.10:
            return "PROBABLE_SCRIPT_READING", "MEDIUM"
        if (
            consistency_authority_score >= config.CONSISTENCY_AUTHORITY_MIN_SCORE
            or weak_consistency_score >= config.PERSISTENT_WEAK_CONSISTENCY_MIN + 0.06
        ):
            return "PROBABLE_SCRIPT_READING", "LOW"

    # Persistent weak authority: sustained low-variance guided delivery (tier + contrastive).
    if persistent_weak_authority_active and strong_window_count == 0:
        if (
            composite >= margin * config.WEAK_CLUSTER_PROBABLE_COMPOSITE_RATIO * 0.85
            or weighted_evidence >= config.AMBIGUOUS_MIN_WEIGHTED_EVIDENCE * 1.1
            or weak_consistency_score >= config.PERSISTENT_WEAK_CONSISTENCY_MIN + 0.08
        ):
            conf = "MEDIUM" if weak_consistency_score >= config.PERSISTENT_WEAK_CONSISTENCY_MIN + 0.12 else "LOW"
            return "PROBABLE_SCRIPT_READING", conf
        if composite >= margin * config.WEAK_CLUSTER_AMBIGUOUS_COMPOSITE_RATIO:
            return "AMBIGUOUS", "MEDIUM"

    if weak_only_dominant and strong_ratio < config.PROBABLE_MIN_STRONG_RATIO:
        if flat_suspicious_flow_active or consistency_authority_score >= config.CONSISTENCY_AUTHORITY_MIN_SCORE:
            weak_only_dominant = False
        elif composite >= margin * config.AMBIGUOUS_EWMA_RATIO:
            return "AMBIGUOUS", "LOW"
        else:
            return "CLEAR", "LOW"

    streak_quality = (
        longest_strong_streak >= config.PROBABLE_LONGEST_STREAK_MIN
        or consecutive_strong >= config.PROBABLE_MIN_CONSECUTIVE_STRONG
    )

    probable = (
        weighted_evidence >= config.PROBABLE_MIN_WEIGHTED_EVIDENCE * 0.82
        and strong_ratio >= config.PROBABLE_MIN_STRONG_RATIO
        and streak_quality
        and composite >= margin * config.PROBABLE_MIN_COMPOSITE_RATIO
    )

    # Quality-over-quantity: several STRONG windows + high peak even if tail EWMA dipped
    probable = probable or (
        strong_window_count >= 3
        and longest_strong_streak >= config.PROBABLE_LONGEST_STREAK_MIN
        and lifetime_strong_ratio >= config.PROBABLE_LIFETIME_STRONG_RATIO
        and peak_ewma >= margin * config.PROBABLE_PEAK_EWMA_RATIO
        and weighted_evidence >= config.PROBABLE_MIN_WEIGHTED_EVIDENCE * 0.75
    )

    if script_dom and low_drift >= config.LOW_DRIFT_SUSPICION_THRESHOLD:
        probable = probable or (
            strong_ratio >= 0.35
            and longest_strong_streak >= 2
            and weighted_evidence >= config.PROBABLE_MIN_WEIGHTED_EVIDENCE * 0.8
            and (composite >= margin * 0.85 or peak_ewma >= margin * 0.9)
            and duration_sec >= config.LONG_HORIZON_MIN_SEC * 0.45
        )

    # Momentum path: sustained suspicious density without requiring trailing consecutive
    probable = probable or (
        suspicion_momentum >= config.MOMENTUM_HIGH_THRESHOLD
        and strong_window_count >= 2
        and peak_ewma >= margin * 0.7
        and weighted_evidence >= config.AMBIGUOUS_MIN_WEIGHTED_EVIDENCE * 1.5
        and not weak_only_dominant
    )

    if probable:
        conf = "HIGH" if (
            strong_ratio >= config.HIGH_CONFIDENCE_MIN_STRONG_RATIO
            and weighted_evidence >= config.HIGH_MIN_WEIGHTED_EVIDENCE
            and composite >= margin
        ) else "MEDIUM"
        return "PROBABLE_SCRIPT_READING", conf

    # Sustained-weak path: clever scripted reading that never spikes to STRONG/MODERATE.
    # Must be persistent (ratio + streak) and have elevated script similarity.
    sustained_weak = (
        weak_ratio >= config.WEAK_CLUSTER_MIN_RATIO
        and longest_weak_streak >= config.WEAK_CLUSTER_MIN_STREAK
        and avg_script_similarity >= config.WEAK_CLUSTER_MIN_AVG_SCRIPT_SIM
        and not weak_only_dominant
    )
    if sustained_weak:
        if (
            composite >= margin * config.WEAK_CLUSTER_PROBABLE_COMPOSITE_RATIO
            and suspicion_momentum >= config.WEAK_CLUSTER_PROBABLE_MIN_MOMENTUM
        ):
            return "PROBABLE_SCRIPT_READING", "MEDIUM"
        if composite >= margin * config.WEAK_CLUSTER_AMBIGUOUS_COMPOSITE_RATIO:
            return "AMBIGUOUS", "MEDIUM"

    ambiguous = (
        composite >= margin * config.AMBIGUOUS_EWMA_RATIO
        or weighted_evidence >= config.AMBIGUOUS_MIN_WEIGHTED_EVIDENCE
    ) and (
        moderate_plus_ratio >= 0.25
        or composite >= margin * 0.75
        or (script_dom and composite >= margin * 0.65)
    )

    if ambiguous:
        return "AMBIGUOUS", "MEDIUM" if composite >= margin else "LOW"

    return "CLEAR", "LOW"


def build_calibration_explanation(
    summary: dict[str, Any],
    evidence: dict[str, float],
    horizon: Any | None,
) -> list[str]:
    reasons: list[str] = []
    if horizon:
        reasons.extend(getattr(horizon, "explanation", []) or [])

    status = summary.get("status")
    if status == "PROBABLE_SCRIPT_READING":
        if evidence.get("strong_ratio", 0) >= config.PROBABLE_MIN_STRONG_RATIO:
            reasons.insert(
                0,
                f"strong suspicious windows ({evidence['strong_ratio']:.0%} STRONG tier)",
            )
        reasons.append(
            f"weighted suspicious evidence {evidence.get('weighted_evidence', 0):.2f} "
            f"(composite {summary.get('composite_score', 0):.3f})"
        )
        if summary.get("consecutive_strong", 0) >= 2:
            reasons.append(
                f"sustained STRONG/MODERATE streak ({summary.get('consecutive_strong', 0)} strong)"
            )
    elif status == "AMBIGUOUS":
        peak = summary.get("peak_ewma", 0)
        if peak and peak >= config.CONTRASTIVE_MARGIN and summary.get("longest_strong_streak", 0) >= 2:
            reasons.append(
                f"strong peaks (peak EWMA {peak:.3f}) diluted by late clear windows — "
                f"weighted evidence {evidence.get('weighted_evidence', 0):.2f}"
            )
        else:
            if (
                evidence.get("weak_ratio", 0) >= config.WEAK_CLUSTER_MIN_RATIO
                and evidence.get("longest_weak_streak", 0) >= config.WEAK_CLUSTER_MIN_STREAK
            ):
                reasons.append(
                    "sustained weak suspicious density (clever script-reading pattern)"
                )
            reasons.append(
                f"borderline weighted evidence ({evidence.get('weighted_evidence', 0):.2f}) — "
                "mostly weak/moderate tiers"
            )
        if summary.get("suspicion_momentum", 0) >= config.MOMENTUM_MED_THRESHOLD:
            reasons.append(
                f"suspicion momentum {summary.get('suspicion_momentum', 0):.2f} preserved partial evidence"
            )
    elif evidence.get("weak_only_ratio", 0) > 0.5:
        reasons.append("weak-only suspicion suppressed — insufficient STRONG evidence")

    return reasons[:10]

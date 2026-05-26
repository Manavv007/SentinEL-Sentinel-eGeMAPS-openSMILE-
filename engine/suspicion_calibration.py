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
    if (
        base in (SuspicionLevel.WEAK, SuspicionLevel.MODERATE)
        and naturality_score >= config.WEAK_SUSPICION_NATURALITY_CAP
        and natural_similarity >= config.WEAK_SUSPICION_NATURAL_SIM_CAP
        and script < config.STRONG_SCRIPT_THRESHOLD
    ):
        base = SuspicionLevel.WEAK if base == SuspicionLevel.MODERATE else SuspicionLevel.NONE

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
    """Asymmetric EWMA input — weak fades fast, strong persists."""
    if level == SuspicionLevel.NONE:
        return 0.0
    if level == SuspicionLevel.WEAK:
        return contrastive_score * config.EWMA_WEAK_INPUT_SCALE
    if level == SuspicionLevel.MODERATE:
        return contrastive_score * config.EWMA_MODERATE_INPUT_SCALE
    return contrastive_score * config.EWMA_STRONG_INPUT_SCALE


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
            "max_level_rank": 0.0,
        }

    total_w = 0.0
    strong = moderate = weak = 0
    max_rank = 0
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
        elif level == SuspicionLevel.MODERATE:
            moderate += 1
        elif level == SuspicionLevel.WEAK:
            weak += 1

    n = len(windows)
    return {
        "weighted_evidence": round(total_w, 4),
        "strong_count": float(strong),
        "moderate_count": float(moderate),
        "weak_count": float(weak),
        "strong_ratio": round(strong / n, 4),
        "moderate_plus_ratio": round((strong + moderate) / n, 4),
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

    composite = max(
        ewma * config.COMPOSITE_EWMA_BLEND,
        peak_ewma * peak_blend,
        evidence_component,
    )
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
) -> tuple[str, str]:
    """
    Returns (status, confidence).
    AMBIGUOUS is the default for borderline; PROBABLE needs quality evidence.
    """
    script_dom = bool(horizon and getattr(horizon, "script_dominance_active", False))
    low_drift = float(getattr(horizon, "low_drift_score", 0.0) or 0.0) if horizon else 0.0

    if weak_only_dominant and strong_ratio < config.PROBABLE_MIN_STRONG_RATIO:
        if composite >= margin * config.AMBIGUOUS_EWMA_RATIO:
            return "AMBIGUOUS", "LOW"
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

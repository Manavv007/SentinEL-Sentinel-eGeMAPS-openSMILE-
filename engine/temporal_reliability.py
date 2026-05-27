"""Duration-aware temporal evidence reliability (focused helpers)."""

from __future__ import annotations

from typing import Any

import numpy as np

import config
from engine.suspicion_calibration import SuspicionLevel


def _longest_elevated_streak(contrastives: list[float], floor: float) -> int:
    best = cur = 0
    for c in contrastives:
        if c >= floor:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _natural_breathing_detected(contrastives: list[float]) -> tuple[bool, float]:
    """
    Natural cognition dips and recovers (e.g. 0.17 → 0.08 → 0.17).
    Returns (detected, breathing_score 0–1).
    """
    n = len(contrastives)
    if n < 4:
        return False, 0.0

    dip_threshold = config.CONSISTENCY_BREATHING_DIP_THRESHOLD
    mid = contrastives[1:-1]
    dips = sum(1 for c in mid if c < dip_threshold)
    score_range = max(contrastives) - min(contrastives)
    range_factor = min(1.0, score_range / max(config.CONSISTENCY_BREATHING_RANGE_MIN, 1e-6))
    dip_factor = dips / max(len(mid), 1)
    breathing_score = range_factor * dip_factor
    detected = (
        dips >= 1
        and score_range >= config.CONSISTENCY_BREATHING_RANGE_MIN
        and breathing_score >= config.CONSISTENCY_BREATHING_MIN_SCORE
    )
    return detected, float(breathing_score)


def _recovery_depth(contrastives: list[float]) -> float:
    if len(contrastives) < 2:
        return 0.0
    drops = [
        contrastives[i - 1] - contrastives[i]
        for i in range(1, len(contrastives))
        if contrastives[i] < contrastives[i - 1]
    ]
    return float(np.mean(drops)) if drops else 0.0


def compute_temporal_reliability(
    window_dicts: list[dict[str, Any]],
    *,
    duration_sec: float,
    suspicious_coverage_ratio: float,
    weak_coverage_ratio: float,
    suspicion_variance: float,
    suspicion_std: float,
    longest_weak_streak: int,
    recovery_strength: float,
    ewma_values: list[float] | None = None,
) -> dict[str, float]:
    """
    Reliability-weighted view of suspiciousness over meaningful duration.

    Uses contrastive scores (not tier labels) for continuity — adaptive fake
    spontaneity often suppresses WEAK tiers while scores stay flatly elevated.
    """
    n = len(window_dicts)
    if n == 0:
        return _empty_reliability()

    contrastives = [float(w.get("contrastive_score", 0)) for w in window_dicts]
    levels = [str(w.get("suspicion_level", SuspicionLevel.NONE.value)) for w in window_dicts]
    floor = config.PERSISTENT_WEAK_MEAN_FLOOR

    elevated = [c for c in contrastives if c >= floor]
    elevated_ratio = len(elevated) / n
    mean_elevated = float(np.mean(elevated)) if elevated else 0.0
    mean_all = float(np.mean(contrastives))

    weak_scores = [
        c for c, lv in zip(contrastives, levels, strict=True) if lv == SuspicionLevel.WEAK.value
    ]
    mod_plus_scores = [
        c
        for c, lv in zip(contrastives, levels, strict=True)
        if lv in (SuspicionLevel.WEAK.value, SuspicionLevel.MODERATE.value)
    ]
    mean_weak = float(np.mean(weak_scores)) if weak_scores else 0.0
    mean_suspicious = float(np.mean(mod_plus_scores)) if mod_plus_scores else mean_elevated

    stability = 1.0 - min(1.0, suspicion_std / max(config.PERSISTENT_WEAK_STABILITY_STD_NORM, 1e-6))
    longest_elevated = _longest_elevated_streak(contrastives, floor)
    continuity_elevated = min(1.0, longest_elevated / max(n, 1))
    continuity_tier = min(1.0, longest_weak_streak / max(n, 1))
    recovery_penalty = 1.0 - min(1.0, max(0.0, recovery_strength))

    natural_breathing, breathing_score = _natural_breathing_detected(contrastives)
    breathing_flatness = 1.0 - min(1.0, breathing_score)

    weak_signal = min(
        1.0,
        max(0.0, (mean_elevated - floor) / max(config.PERSISTENT_WEAK_MEAN_SPAN, 1e-6)),
    )

    weak_consistency = (
        0.28 * max(suspicious_coverage_ratio, elevated_ratio)
        + 0.22 * max(weak_coverage_ratio, elevated_ratio * 0.9)
        + 0.16 * stability
        + 0.14 * max(continuity_elevated, continuity_tier)
        + 0.08 * recovery_penalty
        + 0.07 * weak_signal
        + 0.05 * breathing_flatness
    )
    weak_consistency = float(min(1.0, max(0.0, weak_consistency)))

    suspicious_stability_score = float(
        min(
            1.0,
            0.35 * stability
            + 0.30 * elevated_ratio
            + 0.20 * breathing_flatness
            + 0.15 * recovery_penalty,
        )
    )

    ewma_variance = 0.0
    if ewma_values and len(ewma_values) >= 2:
        ewma_variance = float(np.var(ewma_values))

    recovery_breaks = 0
    for i in range(1, len(levels)):
        prev_susp = levels[i - 1] != SuspicionLevel.NONE.value
        curr_clear = levels[i] == SuspicionLevel.NONE.value
        if prev_susp and curr_clear:
            recovery_breaks += 1
    recovery_frequency = recovery_breaks / max(n - 1, 1)
    recovery_depth = _recovery_depth(contrastives)

    continuity_authority = float(
        min(
            1.0,
            0.40 * continuity_elevated
            + 0.35 * elevated_ratio
            + 0.25 * (1.0 - min(1.0, recovery_frequency * 2.0)),
        )
    )

    consistency_authority = float(
        min(
            1.0,
            0.26 * elevated_ratio
            + 0.22 * suspicious_stability_score
            + 0.20 * continuity_authority
            + 0.16 * breathing_flatness
            + 0.10 * recovery_penalty
            + 0.06 * weak_consistency,
        )
    )

    mod_count = sum(1 for lv in levels if lv == SuspicionLevel.MODERATE.value)
    strong_count = sum(1 for lv in levels if lv == SuspicionLevel.STRONG.value)
    short_answer = (
        duration_sec <= config.TEMPORAL_SHORT_ANSWER_SEC
        or n < config.TEMPORAL_SHORT_MIN_WINDOWS
    )

    short_penalty = 0.0
    if short_answer and strong_count == 0:
        dur_factor = max(0.3, min(1.0, duration_sec / max(config.TEMPORAL_SHORT_ANSWER_SEC, 1e-6)))
        win_factor = max(0.3, min(1.0, n / max(config.TEMPORAL_SHORT_MIN_WINDOWS, 1)))
        short_penalty = min(
            config.TEMPORAL_SHORT_SUPPRESSION_MAX,
            (1.0 - dur_factor) * 0.28 + (1.0 - win_factor) * 0.22,
        )
        if mod_count <= 2 and mean_suspicious < config.SUSPICION_TIER_STRONG:
            short_penalty = min(0.5, short_penalty + 0.12 * (2 - mod_count))

    duration_reliability = min(1.0, duration_sec / max(config.CONSISTENCY_MIN_DURATION_SEC, 1e-6))

    flat_flow_active = _flat_suspicious_flow_active(
        duration_sec=duration_sec,
        window_count=n,
        strong_count=strong_count,
        elevated_ratio=elevated_ratio,
        suspicion_std=suspicion_std,
        suspicion_variance=suspicion_variance,
        recovery_strength=recovery_strength,
        recovery_frequency=recovery_frequency,
        mean_elevated=mean_elevated,
        longest_elevated_streak=longest_elevated,
        consistency_authority=consistency_authority,
        natural_breathing=natural_breathing,
        short_answer=short_answer,
    )

    authority_active = flat_flow_active or persistent_weak_authority_active(
        duration_sec=duration_sec,
        window_count=n,
        strong_count=strong_count,
        weak_consistency=weak_consistency,
        suspicious_coverage_ratio=max(suspicious_coverage_ratio, elevated_ratio),
        suspicion_variance=suspicion_variance,
        recovery_strength=recovery_strength,
        mean_suspicious=max(mean_suspicious, mean_elevated),
        longest_weak_streak=max(longest_weak_streak, longest_elevated),
    )

    authority_score = consistency_authority * duration_reliability

    reliability_evidence_boost = 0.0
    if authority_active:
        reliability_evidence_boost = (
            config.PERSISTENT_WEAK_RELIABILITY_EVIDENCE_SCALE
            * max(weak_consistency, consistency_authority)
            * duration_reliability
        )
        if flat_flow_active:
            reliability_evidence_boost += (
                config.CONSISTENCY_AUTHORITY_EVIDENCE_BOOST * consistency_authority
            )

    return {
        "weak_consistency_score": round(weak_consistency, 6),
        "suspicious_stability_score": round(suspicious_stability_score, 6),
        "suspicious_stability": round(stability, 6),
        "consistency_authority_score": round(consistency_authority, 6),
        "continuity_authority_score": round(continuity_authority, 6),
        "elevated_contrastive_ratio": round(elevated_ratio, 6),
        "elevated_contrastive_mean": round(mean_elevated, 6),
        "suspicion_stddev": round(suspicion_std, 6),
        "ewma_variance": round(ewma_variance, 6),
        "recovery_frequency": round(recovery_frequency, 6),
        "recovery_depth": round(recovery_depth, 6),
        "recovery_strength": round(recovery_strength, 6),
        "natural_breathing_detected": natural_breathing,
        "suspicious_breathing_score": round(breathing_score, 6),
        "suspicious_breathing_flatness": round(breathing_flatness, 6),
        "mean_weak_contrastive": round(mean_weak, 6),
        "mean_suspicious_contrastive": round(mean_suspicious, 6),
        "short_answer_confidence_penalty": round(short_penalty, 6),
        "persistent_weak_authority_score": round(authority_score, 6),
        "persistent_weak_authority_active": authority_active,
        "flat_suspicious_flow_active": flat_flow_active,
        "reliability_evidence_boost": round(reliability_evidence_boost, 6),
    }


def _flat_suspicious_flow_active(
    *,
    duration_sec: float,
    window_count: int,
    strong_count: int,
    elevated_ratio: float,
    suspicion_std: float,
    suspicion_variance: float,
    recovery_strength: float,
    recovery_frequency: float,
    mean_elevated: float,
    longest_elevated_streak: int,
    consistency_authority: float,
    natural_breathing: bool,
    short_answer: bool,
) -> bool:
    """Stabilized cognitive guidance: flat elevated weak flow, no natural breathing."""
    if short_answer or strong_count > 0 or natural_breathing:
        return False
    if duration_sec < config.CONSISTENCY_MIN_DURATION_SEC:
        return False
    if window_count < config.PERSISTENT_WEAK_MIN_WINDOWS:
        return False
    if mean_elevated < config.PERSISTENT_WEAK_MEAN_FLOOR:
        return False
    if elevated_ratio < config.CONSISTENCY_MIN_ELEVATED_RATIO:
        return False
    if suspicion_std > config.CONSISTENCY_FLAT_STD_MAX:
        return False
    if suspicion_variance > config.PERSISTENT_WEAK_VARIANCE_MAX * 1.5:
        return False
    if recovery_strength > config.CONSISTENCY_MAX_RECOVERY_STRENGTH:
        return False
    if recovery_frequency > config.CONSISTENCY_MAX_RECOVERY_FREQUENCY:
        return False
    if longest_elevated_streak < config.PERSISTENT_WEAK_MIN_STREAK:
        return False
    return consistency_authority >= config.CONSISTENCY_AUTHORITY_MIN_SCORE


def persistent_weak_authority_active(
    *,
    duration_sec: float,
    window_count: int,
    strong_count: int,
    weak_consistency: float,
    suspicious_coverage_ratio: float,
    suspicion_variance: float,
    recovery_strength: float,
    mean_suspicious: float,
    longest_weak_streak: int,
) -> bool:
    """Sustained weak suspiciousness with minimal recovery — tier-based fallback."""
    if strong_count > 0:
        return False
    if duration_sec < config.PERSISTENT_WEAK_MIN_DURATION_SEC * 0.80:
        return False
    if window_count < config.PERSISTENT_WEAK_MIN_WINDOWS:
        return False
    if mean_suspicious < config.PERSISTENT_WEAK_MEAN_FLOOR:
        return False
    if suspicious_coverage_ratio < config.PERSISTENT_SUSPICIOUS_COVERAGE_MIN * 0.88:
        return False
    if suspicion_variance > config.PERSISTENT_WEAK_VARIANCE_MAX * 1.35:
        return False
    if recovery_strength > config.PERSISTENT_WEAK_RECOVERY_MAX:
        return False
    if longest_weak_streak < config.PERSISTENT_WEAK_MIN_STREAK:
        return False
    return weak_consistency >= config.PERSISTENT_WEAK_CONSISTENCY_MIN


def short_answer_blocks_ambiguous(
    *,
    duration_sec: float,
    window_count: int,
    strong_count: int,
    moderate_count: int,
    composite: float,
    peak_suspicion: float,
    margin: float,
) -> bool:
    """Small-sample answers need stronger, more reliable evidence before AMBIGUOUS."""
    short = (
        duration_sec <= config.TEMPORAL_SHORT_ANSWER_SEC
        or window_count < config.TEMPORAL_SHORT_MIN_WINDOWS
    )
    if not short or strong_count > 0:
        return False
    if moderate_count >= config.SHORT_ANSWER_AMBIGUOUS_MIN_MODERATE_WINDOWS:
        return False
    if composite >= margin * config.SHORT_ANSWER_AMBIGUOUS_COMPOSITE_RATIO:
        return False
    if peak_suspicion >= config.SHORT_ANSWER_AMBIGUOUS_MIN_PEAK:
        return False
    return True


def _empty_reliability() -> dict[str, float]:
    return {
        "weak_consistency_score": 0.0,
        "suspicious_stability_score": 0.0,
        "suspicious_stability": 0.0,
        "consistency_authority_score": 0.0,
        "continuity_authority_score": 0.0,
        "elevated_contrastive_ratio": 0.0,
        "elevated_contrastive_mean": 0.0,
        "suspicion_stddev": 0.0,
        "ewma_variance": 0.0,
        "recovery_frequency": 0.0,
        "recovery_depth": 0.0,
        "recovery_strength": 0.0,
        "natural_breathing_detected": False,
        "suspicious_breathing_score": 0.0,
        "suspicious_breathing_flatness": 0.0,
        "mean_weak_contrastive": 0.0,
        "mean_suspicious_contrastive": 0.0,
        "short_answer_confidence_penalty": 0.0,
        "persistent_weak_authority_score": 0.0,
        "persistent_weak_authority_active": False,
        "flat_suspicious_flow_active": False,
        "reliability_evidence_boost": 0.0,
    }

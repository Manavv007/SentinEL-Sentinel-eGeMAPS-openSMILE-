"""Answer-level behavioral synthesis — dominant mode over tail EWMA."""

from __future__ import annotations

from typing import Any

import numpy as np

import config
from engine.suspicion_calibration import SuspicionLevel


def compute_answer_behavioral_metrics(
    windows: list[dict[str, Any]],
    *,
    horizon: Any | None = None,
    momentum_summary: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Aggregate full-answer behavioral signature (Layer 3)."""
    if not windows:
        return _empty_metrics()

    momentum_summary = momentum_summary or {}
    levels = [str(w.get("suspicion_level", SuspicionLevel.NONE.value)) for w in windows]
    contrastives = [float(w.get("contrastive_score", 0)) for w in windows]
    scripts = [float(w.get("script_similarity", 0)) for w in windows]
    naturals = [float(w.get("natural_similarity", 0)) for w in windows]
    nats = [float(w.get("naturality_score", 0)) for w in windows]
    cog_spont = [float(w.get("cognitive_spontaneity", 0)) for w in windows]
    cog_guided = [float(w.get("guided_explanation_index", 0)) for w in windows]

    n = len(windows)
    strong = sum(1 for lv in levels if lv == SuspicionLevel.STRONG.value)
    moderate = sum(1 for lv in levels if lv == SuspicionLevel.MODERATE.value)
    weak = sum(1 for lv in levels if lv == SuspicionLevel.WEAK.value)
    none_c = sum(1 for lv in levels if lv == SuspicionLevel.NONE.value)

    suspicious_density = (strong + moderate + weak * 0.35) / n
    peak_suspicion = float(max(contrastives)) if contrastives else 0.0
    p90_suspicion = float(np.percentile(contrastives, 90)) if contrastives else 0.0
    natural_saturation = sum(1 for ns in naturals if ns >= 0.78) / n

    recovery_strength = _recovery_strength(windows, contrastives, nats, naturals)
    behavioral_drift = float(
        getattr(horizon, "low_drift_score", 0.0) if horizon else 0.0
    )

    dominant_score = _dominant_script_reading_score(
        strong_count=strong,
        moderate_count=moderate,
        n_windows=n,
        strong_ratio=strong / n,
        suspicious_density=suspicious_density,
        peak_suspicion=peak_suspicion,
        p90_suspicion=p90_suspicion,
        avg_script=float(np.mean(scripts)) if scripts else 0.0,
        recovery_strength=recovery_strength,
        behavioral_drift=behavioral_drift,
        longest_strong_streak=float(
            momentum_summary.get("longest_strong_streak", 0)
        ),
        peak_ewma=float(momentum_summary.get("peak_ewma", 0) or 0),
        natural_similarity_saturation_ratio=natural_saturation,
    )

    return {
        "strong_window_count": strong,
        "moderate_window_count": moderate,
        "weak_window_count": weak,
        "clear_window_count": none_c,
        "strong_window_ratio": round(strong / n, 4),
        "moderate_plus_ratio": round((strong + moderate) / n, 4),
        "suspicious_density": round(suspicious_density, 4),
        "peak_suspicion": round(peak_suspicion, 6),
        "p90_suspicion": round(p90_suspicion, 6),
        "longest_strong_streak": int(momentum_summary.get("longest_strong_streak", 0)),
        "average_script_similarity": round(float(np.mean(scripts)), 6),
        "average_naturality": round(float(np.mean(nats)), 6),
        "average_natural_similarity": round(float(np.mean(naturals)), 6),
        "behavioral_drift": round(behavioral_drift, 4),
        "recovery_strength": round(recovery_strength, 4),
        "dominant_script_reading_score": round(dominant_score, 4),
        "natural_similarity_saturation_ratio": round(natural_saturation, 4),
        "suspicion_momentum": round(
            float(momentum_summary.get("suspicion_momentum", 0)), 4
        ),
        "cognitive_spontaneity": round(float(np.mean(cog_spont)), 4) if cog_spont else 0.0,
        "guided_explanation_index": round(float(np.mean(cog_guided)), 4) if cog_guided else 0.0,
        "peak_cognitive_spontaneity": round(float(max(cog_spont)), 4) if cog_spont else 0.0,
    }


def synthesize_final_decision(
    temporal_summary: dict[str, Any],
    behavioral: dict[str, Any],
    *,
    margin: float | None = None,
) -> tuple[str, str, list[str]]:
    """
    Layer 3 final authority: dominant behavioral mode overrides tail-sensitive AMBIGUOUS.
    """
    margin = margin or config.CONTRASTIVE_MARGIN
    composite = float(temporal_summary.get("composite_score", 0))
    peak_ewma = float(temporal_summary.get("peak_ewma", 0))
    temporal_status = str(temporal_summary.get("status", "CLEAR"))

    dom = float(behavioral.get("dominant_script_reading_score", 0))
    peak = float(behavioral.get("peak_suspicion", 0))
    strong_n = int(behavioral.get("strong_window_count", 0))
    strong_r = float(behavioral.get("strong_window_ratio", 0))
    density = float(behavioral.get("suspicious_density", 0))
    recovery = float(behavioral.get("recovery_strength", 0))
    drift = float(behavioral.get("behavioral_drift", 0))
    streak = int(behavioral.get("longest_strong_streak", 0))
    weighted = float(temporal_summary.get("weighted_evidence", 0))
    cog_spont = float(behavioral.get("cognitive_spontaneity", 0))
    cog_guided = float(behavioral.get("guided_explanation_index", 0))

    reasons: list[str] = []

    momentum = float(behavioral.get("suspicion_momentum", 0))
    natural_saturation = float(behavioral.get("natural_similarity_saturation_ratio", 0))

    promotion_gate = (
        streak >= config.PROBABLE_LONGEST_STREAK_MIN
        and (
            momentum >= config.MOMENTUM_MED_THRESHOLD
            or peak_ewma >= margin * 2.2
            or composite >= margin
        )
    )

    # --- Dominant scripted reading (Answer 4 target) ---
    scripted_dominant = (
        dom >= config.DOMINANT_SCRIPT_READING_THRESHOLD
        and strong_n >= config.DOMINANT_MIN_STRONG_WINDOWS
        and peak >= config.DOMINANT_MIN_PEAK_SUSPICION
        and recovery < config.DOMINANT_MAX_RECOVERY_STRENGTH
        and promotion_gate
        and natural_saturation < 0.65
    )

    peak_authority = (
        peak >= config.PEAK_SUSPICION_AUTHORITY
        and strong_n >= 2
        and streak >= config.PROBABLE_LONGEST_STREAK_MIN
        and (peak_ewma >= margin or composite >= margin)
        and recovery < 0.55
        and promotion_gate
    )

    density_authority = (
        density >= config.DOMINANT_MIN_SUSPICIOUS_DENSITY
        and strong_r >= config.PROBABLE_MIN_STRONG_RATIO
        and streak >= config.PROBABLE_LONGEST_STREAK_MIN
        and peak >= config.SUSPICION_TIER_MODERATE
        and recovery < 0.5
        and promotion_gate
    )

  # Guided preconstructed flow (scripted answers 2,4,6,7 target)
    guided_scripted = (
        cog_guided >= config.COGNITIVE_GUIDED_HIGH_THRESHOLD
        and cog_spont < config.COGNITIVE_CLEAR_GUIDED_MAX
        and density >= config.DOMINANT_MIN_SUSPICIOUS_DENSITY * 0.85
        and peak >= config.SUSPICION_TIER_MODERATE
    )
    if guided_scripted and promotion_gate and recovery < 0.55:
        reasons.append(
            f"guided explanation flow (guided {cog_guided:.2f}, "
            f"low cognitive spontaneity {cog_spont:.2f})"
        )
        conf = "HIGH" if dom >= config.DOMINANT_HIGH_CONFIDENCE_THRESHOLD else "MEDIUM"
        return "PROBABLE_SCRIPT_READING", conf, reasons

    if scripted_dominant or peak_authority or density_authority:
        conf = "HIGH" if (
            dom >= config.DOMINANT_HIGH_CONFIDENCE_THRESHOLD
            and peak >= config.PEAK_SUSPICION_AUTHORITY
            and strong_n >= 3
        ) else "MEDIUM"
        if peak_authority:
            reasons.append(
                f"dominant mode: peak suspicion {peak:.3f} with {strong_n} STRONG windows "
                f"(tail recovery weak {recovery:.2f})"
            )
        elif scripted_dominant:
            reasons.append(
                f"dominant scripted-reading signature (score {dom:.2f}, "
                f"{strong_r:.0%} STRONG, density {density:.2f})"
            )
        else:
            reasons.append(
                f"sustained suspicious density {density:.2f} across answer "
                f"({streak} max STRONG streak)"
            )
        if drift >= config.LOW_DRIFT_SUSPICION_THRESHOLD:
            reasons.append("low long-term behavioral drift (globally stable delivery)")
        return "PROBABLE_SCRIPT_READING", conf, reasons

    # --- Fluent natural cognition (Answer 3 target) ---
    fluent_natural = (
        cog_spont >= config.COGNITIVE_CLEAR_SPONTANEITY_MIN
        and cog_guided <= config.COGNITIVE_CLEAR_GUIDED_MAX
        and strong_n == 0
        and peak < config.SUSPICION_TIER_STRONG
    )
    if fluent_natural and dom < config.DOMINANT_SCRIPT_READING_THRESHOLD:
        reasons.append(
            f"fluent natural cognition (spontaneity {cog_spont:.2f}, "
            f"guided {cog_guided:.2f}) — delivery regularity not script evidence"
        )
        return "CLEAR", "LOW", reasons

    # --- False-positive guards (Answers 1, 3, 5) ---
    weak_dominant = (
        strong_n == 0
        or (strong_n == 1 and peak < config.SUSPICION_TIER_STRONG)
    ) and peak < config.DOMINANT_MIN_PEAK_SUSPICION

    if weak_dominant and dom < config.DOMINANT_AMBIGUOUS_CEILING:
        if composite >= margin * config.AMBIGUOUS_EWMA_RATIO or weighted >= config.AMBIGUOUS_MIN_WEIGHTED_EVIDENCE:
            reasons.append(
                "only weak/moderate suspicion tiers — no dominant scripted signature"
            )
            return "AMBIGUOUS", "LOW", reasons
        reasons.append("insufficient STRONG suspicious evidence")
        return "CLEAR", "LOW", reasons

    strong_recovery = recovery >= config.STRONG_RECOVERY_THRESHOLD
    if strong_recovery and dom < config.DOMINANT_SCRIPT_READING_THRESHOLD:
        reasons.append(
            f"sustained spontaneous recovery in tail (strength {recovery:.2f}) "
            "— not scripted-dominant"
        )
        return "AMBIGUOUS", "MEDIUM", reasons

    # Fall back to temporal layer but cap: high composite + peaks → at least AMBIGUOUS
    if temporal_status == "PROBABLE_SCRIPT_READING":
        return temporal_status, str(temporal_summary.get("confidence", "MEDIUM")), reasons

    if (
        composite >= margin
        and peak >= config.SUSPICION_TIER_STRONG
        and strong_n >= 2
    ):
        reasons.append(
            f"temporal composite {composite:.3f} with peak {peak:.3f} — borderline scripted"
        )
        return "AMBIGUOUS", "MEDIUM", reasons

    reasons.append(f"temporal layer: {temporal_status.lower().replace('_', ' ')}")
    return temporal_status, str(temporal_summary.get("confidence", "LOW")), reasons


def _dominant_script_reading_score(
    *,
    strong_count: int,
    moderate_count: int,
    n_windows: int,
    strong_ratio: float,
    suspicious_density: float,
    peak_suspicion: float,
    p90_suspicion: float,
    avg_script: float,
    recovery_strength: float,
    behavioral_drift: float,
    longest_strong_streak: int,
    peak_ewma: float,
    natural_similarity_saturation_ratio: float = 0.0,
) -> float:
    margin = config.CONTRASTIVE_MARGIN
    peak_term = min(1.0, peak_suspicion / max(config.PEAK_SUSPICION_AUTHORITY, 1e-6))
    p90_term = min(1.0, p90_suspicion / max(config.SUSPICION_TIER_STRONG, 1e-6))
    streak_term = min(1.0, longest_strong_streak / max(n_windows, 1))
    script_term = min(1.0, max(0.0, (avg_script - 0.58) / 0.2))
    drift_term = min(1.0, behavioral_drift)
    recovery_penalty = recovery_strength * 0.35
    peak_ewma_term = min(1.0, peak_ewma / max(margin * 2, 1e-6))
    sat_penalty = natural_similarity_saturation_ratio * 0.12

    score = (
        0.28 * peak_term
        + 0.14 * p90_term
        + 0.18 * strong_ratio
        + 0.12 * suspicious_density
        + 0.10 * streak_term
        + 0.08 * script_term
        + 0.06 * drift_term
        + 0.04 * peak_ewma_term
        - recovery_penalty
        - sat_penalty
    )
    if strong_count >= 3:
        score += 0.08
    if peak_suspicion >= config.PEAK_SUSPICION_AUTHORITY:
        score += 0.12
    return float(min(1.0, max(0.0, score)))


def _recovery_strength(
    windows: list[dict[str, Any]],
    contrastives: list[float],
    naturality: list[float],
    natural_sim: list[float],
) -> float:
    """
    0 = minor end recovery; 1 = sustained genuine spontaneous recovery in tail.
    """
    n = len(windows)
    if n < 4:
        return 0.0

    tail_n = max(1, n // 4)
    head_n = max(1, n // 4)
    tail_c = float(np.mean(contrastives[-tail_n:]))
    head_c = float(np.mean(contrastives[:head_n]))
    tail_nat = float(np.mean(naturality[-tail_n:]))
    head_nat = float(np.mean(naturality[:head_n]))
    tail_script = float(np.mean([float(w.get("script_similarity", 0)) for w in windows[-tail_n:]]))

    nat_rise = max(0.0, tail_nat - head_nat)
    contrast_drop = max(0.0, head_c - tail_c)

    recovery = 0.45 * min(1.0, nat_rise / 0.25) + 0.45 * min(1.0, contrast_drop / 0.2)
    if tail_script < config.STRONG_SCRIPT_THRESHOLD - 0.08:
        recovery *= 0.5
    return float(min(1.0, recovery))


def _empty_metrics() -> dict[str, Any]:
    return {
        "strong_window_count": 0,
        "moderate_window_count": 0,
        "weak_window_count": 0,
        "clear_window_count": 0,
        "strong_window_ratio": 0.0,
        "moderate_plus_ratio": 0.0,
        "suspicious_density": 0.0,
        "peak_suspicion": 0.0,
        "p90_suspicion": 0.0,
        "longest_strong_streak": 0,
        "average_script_similarity": 0.0,
        "average_naturality": 0.0,
        "average_natural_similarity": 0.0,
        "behavioral_drift": 0.0,
        "recovery_strength": 0.0,
        "dominant_script_reading_score": 0.0,
        "natural_similarity_saturation_ratio": 0.0,
        "suspicion_momentum": 0.0,
        "cognitive_spontaneity": 0.0,
        "guided_explanation_index": 0.0,
        "peak_cognitive_spontaneity": 0.0,
    }

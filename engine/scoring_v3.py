"""Third-pass scoring: modulated script emphasis + nonlinear temporal aggregation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

import config


def nonlinear_script_emphasis(script_similarity: float) -> float:
    """
    Strong script evidence contributes disproportionately.
    0.72 matters much more than repeated 0.58.
    """
    s = float(max(0.0, min(1.0, script_similarity)))
    base = s * config.CONTRASTIVE_SCRIPT_WEIGHT
    if s < config.STRONG_SCRIPT_THRESHOLD:
        return base
    excess = s - config.STRONG_SCRIPT_THRESHOLD
    span = max(1.0 - config.STRONG_SCRIPT_THRESHOLD, 1e-6)
    boost = config.STRONG_SCRIPT_NONLINEAR_GAIN * (excess / span) ** config.STRONG_SCRIPT_NONLINEAR_POWER
    return float(min(1.0, base + boost))


def spontaneity_modulation_factor(
    *,
    naturality_score: float,
    script_similarity: float,
    profile_confidence: float,
    fake_natularity: float,
) -> float:
    """
    Naturality modulates confidence; it does not erase script evidence.
    Returns multiplier in (floor, 1.0].
    """
    spont = float(max(0.0, min(1.0, naturality_score)))
    prof = float(max(0.0, min(1.0, profile_confidence)))
    fake_penalty = min(0.35, fake_natularity * config.FAKE_NATURALITY_MODULATION_PENALTY)

    reduction = (
        config.SPONTANEITY_MODULATION_WEIGHT
        * spont
        * (0.45 + 0.55 * prof)
        + fake_penalty
    )

    if script_similarity >= config.STRONG_SCRIPT_THRESHOLD:
        max_reduction = config.STRONG_SCRIPT_MAX_SPONT_REDUCTION
    elif script_similarity >= 0.62:
        max_reduction = config.MODERATE_SCRIPT_MAX_SPONT_REDUCTION
    else:
        max_reduction = config.WEAK_SCRIPT_MAX_SPONT_REDUCTION

    reduction = min(reduction, max_reduction)
    return float(max(1.0 - max_reduction, config.SPONTANEITY_MODULATION_FLOOR))


def modulated_suspicion_score(
    *,
    script_similarity: float,
    naturality_score: float,
    profile_confidence: float,
    suppression: float,
    technical_density: float,
    fake_natularity: float = 0.0,
) -> float:
    """
    script_emphasis * spontaneity_modulation - bounded suppression.
    Replaces direct (script - natural) subtraction for ranking.
    """
    script_emph = nonlinear_script_emphasis(script_similarity)
    mod = spontaneity_modulation_factor(
        naturality_score=naturality_score,
        script_similarity=script_similarity,
        profile_confidence=profile_confidence,
        fake_natularity=fake_natularity,
    )
    tech_dampen = config.TECHNICAL_FLUENCY_DAMPENING * min(1.0, technical_density * 4.0)
    raw = script_emph * mod
    return float(max(0.0, raw - suppression - tech_dampen))


def window_temporal_weight(script_similarity: float, contrastive_score: float) -> float:
    """Per-window weight for EWMA / persistence (strong script windows weigh more)."""
    s = float(script_similarity)
    w = 1.0
    if s >= config.STRONG_SCRIPT_THRESHOLD:
        w += config.STRONG_WINDOW_WEIGHT_BONUS * ((s - config.STRONG_SCRIPT_THRESHOLD) / 0.32)
    elif s >= 0.62:
        w += 0.15
    if contrastive_score > config.CONTRASTIVE_MARGIN:
        w += 0.1
    return float(min(2.2, w))


def compute_answer_composite_score(
    windows: list[dict[str, Any]],
    *,
    ewma: float,
    peak_ewma: float,
    margin: float,
    horizon: Any | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Blend EWMA with peak and strong-window evidence so one benign tail cannot erase spikes.
    """
    if not windows:
        return ewma, {}

    script_sims = [float(w.get("script_similarity", 0)) for w in windows]
    scores = [float(w.get("contrastive_score", 0)) for w in windows]
    weights = [
        window_temporal_weight(s, c) for s, c in zip(script_sims, scores, strict=True)
    ]

    weighted_scores = [s * w for s, w in zip(scores, weights, strict=True)]
    strong_mask = [s >= config.STRONG_SCRIPT_THRESHOLD for s in script_sims]
    strong_ratio = sum(strong_mask) / len(script_sims)
    suspicious_flags = [bool(w.get("suspicious_flag")) for w in windows]
    susp_ratio = sum(suspicious_flags) / len(windows)

    p90 = float(np.percentile(weighted_scores, 90)) if weighted_scores else 0.0
    strong_mean = (
        float(np.mean([ws for ws, m in zip(weighted_scores, strong_mask, strict=True) if m]))
        if any(strong_mask)
        else 0.0
    )

    composite = max(
        ewma,
        peak_ewma * config.PEAK_EWMA_BLEND,
        p90 * config.P90_WINDOW_BLEND,
        strong_mean * config.STRONG_MEAN_BLEND,
    )

    if horizon and getattr(horizon, "script_dominance_active", False):
        floor = float(getattr(horizon, "answer_contrastive_floor", 0.0) or 0.0)
        composite = max(composite, floor * config.ANSWER_FLOOR_SCALE)

    meta = {
        "composite_score": round(composite, 6),
        "peak_ewma": round(peak_ewma, 6),
        "p90_weighted": round(p90, 6),
        "strong_window_ratio": round(strong_ratio, 4),
        "strong_mean_weighted": round(strong_mean, 6),
    }
    return float(composite), meta


def build_v3_explanation(
    summary: dict[str, Any],
    horizon: Any | None,
    composite_meta: dict[str, float],
) -> list[str]:
    reasons: list[str] = []
    if horizon:
        reasons.extend(getattr(horizon, "explanation", []) or [])

    comp = float(summary.get("composite_score", summary.get("ewma_score", 0)))
    margin = config.CONTRASTIVE_MARGIN
    strong_r = composite_meta.get("strong_window_ratio", 0)

    if summary.get("status") == "PROBABLE_SCRIPT_READING":
        if strong_r and strong_r >= 0.5:
            reasons.insert(
                0,
                f"repeated high-confidence script windows ({strong_r:.0%} ≥ {config.STRONG_SCRIPT_THRESHOLD})",
            )
        if composite_meta.get("peak_ewma", 0) >= margin:
            reasons.append(
                f"peak suspicious momentum {composite_meta['peak_ewma']:.3f} (final composite {comp:.3f})"
            )
        if summary.get("persistent"):
            reasons.append(
                f"sustained suspicious evidence ({summary.get('suspicious_window_ratio', 0):.0%} windows)"
            )
    elif summary.get("status") == "AMBIGUOUS":
        reasons.append(f"elevated composite suspicion {comp:.3f} without full persistence")
    elif strong_r >= 0.5 and comp < margin:
        reasons.append("strong script spikes present but composite score diluted by late benign window")

    return reasons[:10]

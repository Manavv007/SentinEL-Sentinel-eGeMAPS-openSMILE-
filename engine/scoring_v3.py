"""Third-pass scoring: modulated script emphasis + nonlinear temporal aggregation."""

from __future__ import annotations

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
    return float(max(1.0 - reduction, config.SPONTANEITY_MODULATION_FLOOR))


def modulated_suspicion_score(
    *,
    script_similarity: float,
    natural_similarity: float = 0.0,
    naturality_score: float,
    profile_confidence: float,
    suppression: float,
    technical_density: float,
    fake_natularity: float = 0.0,
    cognitive_spontaneity: float = 0.0,
    guided_explanation: float = 0.0,
    fluency_trap: float = 0.0,
    style_similarity: float = 0.0,
    cognitive_guidance_similarity: float = 0.0,
    cognitive_wobble: float = 0.0,
    semantic_repair: float = 0.0,
) -> float:
    """
    script_emphasis * spontaneity_modulation - bounded suppression.
    Replaces direct (script - natural) subtraction for ranking.
    """
    s = float(max(0.0, min(1.0, script_similarity)))
    n_sim = float(max(0.0, min(1.0, natural_similarity)))
    prof = float(max(0.0, min(1.0, profile_confidence)))

    script_emph = nonlinear_script_emphasis(s)
    mod = spontaneity_modulation_factor(
        naturality_score=naturality_score,
        script_similarity=s,
        profile_confidence=profile_confidence,
        fake_natularity=fake_natularity,
    )
    tech_dampen = config.TECHNICAL_FLUENCY_DAMPENING * min(1.0, technical_density * 4.0)

    cog_spont = float(max(0.0, min(1.0, cognitive_spontaneity)))
    guided = float(max(0.0, min(1.0, guided_explanation)))
    trap = float(max(0.0, min(1.0, fluency_trap)))

    # Cognitive spontaneity dampens script emphasis (fluent natural ≠ guided)
    if cog_spont >= config.FLUENT_NATURAL_SPONTANEITY_FLOOR:
        script_emph *= 1.0 - min(
            config.COGNITIVE_SCRIPT_DAMPEN_AT_SPONTANEITY,
            cog_spont * config.COGNITIVE_SCRIPT_DAMPEN_AT_SPONTANEITY,
        )

    # Style-dominated script overlap: delivery polish without cognitive guidance.
    style_leak = max(0.0, style_similarity - cognitive_guidance_similarity)
    if style_leak >= config.PROFILE_STYLE_LEAK_MIN and cog_spont >= 0.35:
        script_emph *= max(
            0.55,
            1.0 - min(config.FLUENT_STYLE_LEAK_DAMPEN_MAX, style_leak * 0.55),
        )

    # Cognitive turbulence (repairs, wobble) — spontaneous cognition safety.
    turbulence = cognitive_wobble + semantic_repair * 0.85
    if turbulence >= config.COGNITIVE_TURBULENCE_WOBBLE_MIN and cog_spont >= 0.32:
        script_emph *= max(
            0.58,
            1.0 - min(config.COGNITIVE_TURBULENCE_SCRIPT_DAMPEN_MAX, turbulence * 0.35),
        )

    # Fluent natural technical explanation safety.
    if (
        cog_spont >= config.COGNITIVE_CLEAR_SPONTANEITY_MIN
        and guided <= config.COGNITIVE_CLEAR_GUIDED_MAX
        and naturality_score >= config.WEAK_SUSPICION_NATURALITY_CAP * 0.88
        and script_similarity < config.STRONG_SCRIPT_THRESHOLD
    ):
        script_emph *= max(0.62, 1.0 - config.FLUENT_LINGUISTIC_DAMPEN * cog_spont)

    # --- Dynamic range recovery (nonlinear amplification) ---
    nat = float(max(0.0, min(1.0, naturality_score)))
    if s >= 0.55 and nat <= 0.82 and cog_spont < 0.50:
        x = (s - 0.55) / 0.45
        amp = config.CONTRASTIVE_NONLINEAR_AMP * (x**config.CONTRASTIVE_NONLINEAR_POWER)
        script_emph = min(1.25, script_emph * (1.0 + amp))

    # --- Relative dominance scoring (script stronger than natural) ---
    ratio = s / max(n_sim, config.DOMINANCE_NATURAL_FLOOR)
    dominance_boost = 0.0
    if (
        prof >= config.DOMINANCE_MIN_PROFILE_CONFIDENCE
        and ratio >= config.DOMINANCE_RATIO_MIN
        and s >= 0.55
        and nat <= 0.85
        and cog_spont < 0.52
    ):
        dominance_boost = min(
            config.DOMINANCE_RATIO_MAX_BOOST,
            (ratio - config.DOMINANCE_RATIO_MIN)
            * config.DOMINANCE_RATIO_BOOST_PER_UNIT,
        )
        dominance_boost *= prof

    # --- Reduce oversuppression asymmetrically when script is high ---
    sup = float(suppression)
    if s >= 0.55:
        soft = min(0.35, (s - 0.55) / 0.45)
        sup = sup * (1.0 - config.SUPPRESSION_SOFTENING_AT_HIGH_SCRIPT * soft)

    from engine.cognitive_spontaneity import guided_explanation_boost as _guided_boost

    raw = script_emph * mod + dominance_boost
    raw += _guided_boost(guided, trap)
    if trap >= 0.55 and cog_spont < 0.45:
        raw += config.COGNITIVE_FLUENCY_TRAP_BOOST * trap
    return float(max(0.0, raw - sup - tech_dampen))


def window_temporal_weight(script_similarity: float, contrastive_score: float) -> float:
    """Per-window weight for EWMA / persistence (strong script windows weigh more)."""
    s = float(script_similarity)
    w = 1.0
    if s >= config.STRONG_SCRIPT_THRESHOLD:
        span = max(1.0 - config.STRONG_SCRIPT_THRESHOLD, 1e-6)
        w += config.STRONG_WINDOW_WEIGHT_BONUS * ((s - config.STRONG_SCRIPT_THRESHOLD) / span)
    elif s >= 0.62:
        w += 0.15
    if contrastive_score > config.CONTRASTIVE_MARGIN:
        w += 0.1
    return float(min(2.2, w))


def weighted_percentile(
    values: list[float],
    weights: list[float],
    q: float,
) -> float:
    """Weighted percentile (q in 0–100); weights are importance, not score multipliers."""
    if not values:
        return 0.0
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size != w.size:
        raise ValueError("values and weights must have the same length")
    total = float(np.sum(w))
    if total <= 0:
        return float(np.percentile(v, q))
    order = np.argsort(v)
    v_sorted, w_sorted = v[order], w[order]
    cum = np.cumsum(w_sorted) - 0.5 * w_sorted
    return float(np.interp(q / 100.0 * total, cum, v_sorted))


def compute_answer_composite_score(
    windows: list[dict[str, Any]],
    *,
    ewma: float,
    peak_ewma: float,
    margin: float,
    horizon: Any | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Blend EWMA with peak and strong-window evidence.
    Benign EWMA can pull the composite down when peak evidence is not widespread.
    """
    del margin  # reserved for horizon floor scaling via config
    if not windows:
        return ewma, {}

    script_sims = [float(w.get("script_similarity", 0)) for w in windows]
    scores = [float(w.get("contrastive_score", 0)) for w in windows]
    weights = [
        window_temporal_weight(s, c) for s, c in zip(script_sims, scores, strict=True)
    ]

    strong_mask = [s >= config.STRONG_SCRIPT_THRESHOLD for s in script_sims]
    strong_ratio = sum(strong_mask) / len(script_sims)
    suspicious_flags = [bool(w.get("suspicious_flag")) for w in windows]
    susp_ratio = sum(suspicious_flags) / len(windows)

    p90 = weighted_percentile(scores, weights, 90.0)
    if any(strong_mask):
        strong_scores = [sc for sc, m in zip(scores, strong_mask, strict=True) if m]
        strong_weights = [wt for wt, m in zip(weights, strong_mask, strict=True) if m]
        strong_mean = float(np.average(strong_scores, weights=strong_weights))
    else:
        strong_mean = 0.0

    peak_evidence = max(
        peak_ewma * config.PEAK_EWMA_BLEND,
        p90 * config.P90_WINDOW_BLEND,
        strong_mean * config.STRONG_MEAN_BLEND,
    )
    persistence = 0.5 * strong_ratio + 0.5 * susp_ratio
    alpha = min(1.0, config.PEAK_CREDIBILITY_GAIN * persistence)
    composite = float(ewma) + alpha * max(0.0, peak_evidence - float(ewma))

    if horizon and getattr(horizon, "script_dominance_active", False):
        floor = float(getattr(horizon, "answer_contrastive_floor", 0.0) or 0.0)
        composite = max(composite, floor * config.ANSWER_FLOOR_SCALE)

    meta = {
        "composite_score": round(composite, 6),
        "peak_ewma": round(peak_ewma, 6),
        "p90_weighted": round(p90, 6),
        "strong_window_ratio": round(strong_ratio, 4),
        "strong_mean_weighted": round(strong_mean, 6),
        "peak_evidence": round(peak_evidence, 6),
        "peak_credibility_alpha": round(alpha, 4),
        "persistence": round(persistence, 4),
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

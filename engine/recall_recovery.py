"""Targeted recall recovery: long-horizon consistency, script dominance, profile confidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config


@dataclass
class WindowScoreRow:
    """One scored window before long-horizon adjustment."""

    window_id: int
    start_sec: float
    end_sec: float
    features: dict[str, float]
    script_similarity: float
    natural_similarity_raw: float
    natural_similarity_effective: float
    naturality_score: float
    naturality_learning: float
    learning_confidence: float
    naturality_breakdown: dict[str, float]
    contrastive_base: float
    suppression: float
    profile_confidence: float
    natural_update_reason: str
    natural_profile_updated: bool
    script_similarity_raw: float = 0.0
    style_similarity: float = 0.0
    cognitive_guidance_similarity: float = 0.0
    cognitive_spontaneity: float = 0.0
    guided_explanation: float = 0.0
    cognitive_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class HorizonAnalysis:
    per_window_boost: list[float] = field(default_factory=list)
    script_dominance_active: bool = False
    low_drift_score: float = 0.0
    cleanliness_score: float = 0.0
    fake_natularity_score: float = 0.0
    sustained_script_ratio: float = 0.0
    answer_contrastive_floor: float = 0.0
    explanation: list[str] = field(default_factory=list)


def strong_spontaneity_categories(
    features: dict[str, float], nat_breakdown: dict[str, float]
) -> int:
    """Count distinct high-confidence spontaneity evidence categories."""
    categories = 0

    if features.get("ling_self_corrections", 0.0) >= 1.0 or nat_breakdown.get(
        "self_correction", 0.0
    ) >= 0.45:
        categories += 1

    if (
        nat_breakdown.get("retrieval_pause", 0.0) >= 0.55
        and features.get("ling_retrieval_pause_max", 0.0) >= 0.5
    ):
        categories += 1

    if (
        nat_breakdown.get("pause_entropy", 0.0) >= 0.55
        and features.get("ling_pause_entropy", 0.0) >= 0.8
    ):
        categories += 1

    if (
        features.get("ling_filler_clusters", 0.0) >= 1.0
        and nat_breakdown.get("filler_dynamics", 0.0) >= 0.5
    ):
        categories += 1

    if (
        nat_breakdown.get("rate_variance", 0.0) >= 0.55
        and features.get("ling_gap_variance", 0.0) >= 0.004
    ):
        categories += 1

    if abs(features.get("acoustic_pitch_delta", 0.0)) >= 12.0:
        categories += 1

    return categories


def should_update_natural_profile(
    features: dict[str, float],
    naturality: float,
    script_sim: float,
    nat_breakdown: dict[str, float],
    *,
    naturality_learning: float | None = None,
    technical_density: float = 0.0,
    learning_confidence: float = 0.0,
    cognitive_spontaneity: float = 0.0,
    guided_explanation: float = 0.0,
) -> tuple[bool, str]:
    """Strict gate: only high-confidence spontaneous cognition updates NATURAL."""
    learn_nat = (
        float(naturality_learning)
        if naturality_learning is not None
        else naturality
    )

    from engine.profile_disentanglement import fluent_natural_learning_eligible

    fluent_ok, fluent_reason = fluent_natural_learning_eligible(
        features,
        nat_breakdown,
        script_similarity=script_sim,
        cognitive_spontaneity=cognitive_spontaneity,
        guided_explanation=guided_explanation,
        naturality_learning=learn_nat,
    )
    if fluent_ok:
        if learning_confidence >= config.FLUENT_NATURAL_LEARNING_CONFIDENCE:
            return True, fluent_reason
        if learn_nat >= config.FLUENT_NATURAL_LEARNING_MIN_NATURALITY + 0.06:
            return True, fluent_reason

    if script_sim >= config.NATURAL_UPDATE_MAX_SCRIPT_SIM:
        return False, "script_similarity_too_high"

    if learn_nat < config.NATURALITY_LEARNING_THRESHOLD:
        return False, "naturality_learning_below_threshold"

    if learning_confidence < config.NATURAL_PROFILE_MIN_LEARNING_CONFIDENCE:
        return False, "learning_confidence_too_low"

    if technical_density >= config.NATURAL_UPDATE_MAX_TECHNICAL_DENSITY:
        strong_tech = strong_spontaneity_categories(features, nat_breakdown)
        need = (
            config.NATURAL_UPDATE_MIN_STRONG_CATEGORIES
            + config.NATURAL_UPDATE_TECHNICAL_EXTRA_STRONG
        )
        if strong_tech < need:
            return False, "technical_fluent_block"

    spontaneity_signals = sum(
        1
        for key in (
            "self_correction",
            "retrieval_pause",
            "pause_entropy",
            "filler_dynamics",
            "rate_variance",
        )
        if nat_breakdown.get(key, 0.0) >= 0.35
    )
    if spontaneity_signals < config.NATURAL_UPDATE_MIN_SPONTANEITY_SIGNALS:
        return False, "insufficient_spontaneity_indicators"

    strong = strong_spontaneity_categories(features, nat_breakdown)
    if strong < config.NATURAL_UPDATE_MIN_STRONG_CATEGORIES:
        return False, "insufficient_diverse_spontaneity"

    if features.get("ling_has_words", 0.0) >= 1.0:
        pause_ent = features.get("ling_pause_entropy", 0.0)
        if pause_ent < config.NATURAL_UPDATE_MIN_PAUSE_ENTROPY:
            return False, "pause_entropy_below_minimum"
        gap_var = features.get("ling_gap_variance", 0.0)
        if gap_var < 1e-6 and pause_ent < 0.05:
            return False, "pacing_too_regular_for_natural_update"

    # Fluent structured delivery without true spontaneity cues
    if (
        script_sim >= config.NATURAL_UPDATE_FLUENT_SCRIPT_CEILING
        and strong < config.NATURAL_UPDATE_MIN_STRONG_CATEGORIES + 1
    ):
        return False, "fluent_structured_not_spontaneous"

    return True, "spontaneous_window_accepted"


def natural_profile_confidence(
    profile_sample_count: int,
    metric_count: int,
    strong_update_count: int,
    total_updates: int,
    *,
    profile_purity: float = 1.0,
) -> float:
    """0.35–1.0 confidence in NATURAL profile; weak/contaminated profiles weigh less."""
    if profile_sample_count <= 0:
        return config.NATURAL_PROFILE_CONFIDENCE_FLOOR

    sample_factor = min(1.0, profile_sample_count / max(config.NATURAL_PROFILE_MIN_SAMPLES * 2, 1))
    diversity_factor = min(1.0, metric_count / 10.0)
    quality_factor = 1.0
    if total_updates > 0:
        quality_factor = 0.5 + 0.5 * min(1.0, strong_update_count / max(total_updates, 1))

    purity_factor = 0.5 + 0.5 * float(max(0.0, min(1.0, profile_purity)))

    conf = (
        config.NATURAL_PROFILE_CONFIDENCE_FLOOR
        + (1.0 - config.NATURAL_PROFILE_CONFIDENCE_FLOOR)
        * sample_factor
        * diversity_factor
        * quality_factor
        * purity_factor
    )
    return float(min(1.0, max(config.NATURAL_PROFILE_CONFIDENCE_FLOOR, conf)))


def effective_natural_similarity(
    raw_similarity: float,
    profile_confidence: float,
    *,
    script_similarity: float,
    sample_count: int,
    profile_purity: float = 1.0,
    saturation_pressure: float = 0.0,
) -> float:
    """Cap and down-weight natural similarity; purity reduces dilution saturation."""
    if sample_count <= 0:
        prior = max(0.0, 1.0 - script_similarity)
        floor = config.NATURAL_SIMILARITY_MATURITY_FLOOR
        raw_similarity = float(prior * (1.0 - floor) + floor * 0.25 * prior)

    purity = float(max(0.0, min(1.0, profile_purity)))
    dynamic_cap = config.NATURAL_SIMILARITY_CAP * (0.52 + 0.48 * purity)
    dynamic_cap -= saturation_pressure * config.NATURAL_SIMILARITY_SATURATION_PENALTY
    dynamic_cap = max(config.NATURAL_SIMILARITY_CAP * 0.45, dynamic_cap)

    capped = min(float(raw_similarity), dynamic_cap)
    spread = dynamic_cap - capped
    softened = capped + spread * (0.42 + 0.38 * profile_confidence) * profile_confidence
    if saturation_pressure > 0.45:
        softened *= 1.0 - min(
            0.22, saturation_pressure * config.NATURAL_SIMILARITY_SATURATION_PENALTY
        )
    if script_similarity >= config.STRONG_SCRIPT_THRESHOLD:
        softened *= max(0.55, 1.0 - (script_similarity - config.STRONG_SCRIPT_THRESHOLD) * 0.8)
    elif script_similarity >= 0.58 and sample_count > 0:
        # Mild high-script dampening before STRONG threshold (style-leak regime).
        softened *= max(0.72, 1.0 - (script_similarity - 0.58) * 0.45)

    return float(max(0.0, min(dynamic_cap, softened * profile_confidence)))


def capped_suppression(
    suppression: float,
    script_similarity: float,
    *,
    fake_natularity: float = 0.0,
) -> float:
    """Do not let fake spontaneity fully suppress script evidence."""
    if script_similarity >= config.SCRIPT_DOMINANCE_THRESHOLD:
        cap = config.SCRIPT_DOMINANCE_MAX_SUPPRESSION
        if fake_natularity >= config.FAKE_NATURALITY_THRESHOLD:
            cap = min(cap, config.FAKE_NATURALITY_MAX_SUPPRESSION)
        return min(suppression, cap)
    return suppression


def is_window_benign(
    *,
    contrastive: float,
    margin: float,
    naturality: float,
    script_similarity: float,
    suppression: float,
    low_drift_local: float,
) -> bool:
    """Benign decay only when script dominance / low drift do not apply."""
    if script_similarity >= config.SCRIPT_DOMINANCE_THRESHOLD and low_drift_local >= 0.45:
        return False
    if script_similarity >= 0.68 and contrastive > margin * 0.35:
        return False
    if contrastive <= margin * 0.5:
        return True
    if suppression >= 0.35 and script_similarity < config.SCRIPT_DOMINANCE_THRESHOLD:
        return True
    if (
        naturality >= config.EWMA_BENIGN_NATURALITY_MIN
        and script_similarity < 0.62
        and contrastive < margin * 0.45
    ):
        return True
    if script_similarity >= config.STRONG_SCRIPT_THRESHOLD and contrastive >= margin * 0.5:
        return False
    return False


class AnswerHorizonAnalyzer:
    """20–40s behavioral consistency: drift, cleanliness, fake naturality, script dominance."""

    def analyze(
        self,
        rows: list[WindowScoreRow],
        *,
        answer_duration_sec: float,
    ) -> HorizonAnalysis:
        n = len(rows)
        out = HorizonAnalysis(per_window_boost=[0.0] * n)
        if n == 0:
            return out

        script_sims = [r.script_similarity for r in rows]
        wps = [r.features.get("ling_wps", 0.0) for r in rows if r.features.get("ling_wps")]
        gaps = [
            r.features.get("ling_gap_variance", 0.0)
            for r in rows
            if r.features.get("ling_has_words")
        ]
        pause_ents = [
            r.features.get("ling_pause_entropy", 0.0)
            for r in rows
            if r.features.get("ling_has_words")
        ]
        pitch_deltas = [
            abs(r.features.get("acoustic_pitch_delta", 0.0))
            for r in rows
            if r.features.get("acoustic_pitch_delta") is not None
        ]

        out.low_drift_score = _low_drift_score(wps, gaps, pause_ents, pitch_deltas)
        out.cleanliness_score = _cleanliness_persistence(wps, gaps, pause_ents, script_sims)
        out.fake_natularity_score = _fake_natularity_score(rows)
        high_script = sum(1 for s in script_sims if s >= config.SCRIPT_DOMINANCE_THRESHOLD)
        out.sustained_script_ratio = high_script / n
        out.script_dominance_active = (
            high_script >= config.SCRIPT_DOMINANCE_MIN_WINDOWS
            and out.sustained_script_ratio >= 0.5
            and out.low_drift_score >= config.LOW_DRIFT_SUSPICION_THRESHOLD
        )

        duration_factor = min(1.0, answer_duration_sec / max(config.LONG_HORIZON_MIN_SEC, 1.0))

        dominance_boost = 0.0
        if out.script_dominance_active:
            dominance_boost = config.SCRIPT_DOMINANCE_BOOST * duration_factor
            out.explanation.append(
                f"sustained high script similarity ({out.sustained_script_ratio:.0%} of windows ≥ "
                f"{config.SCRIPT_DOMINANCE_THRESHOLD})"
            )

        drift_boost = 0.0
        if (
            out.low_drift_score >= config.LOW_DRIFT_SUSPICION_THRESHOLD
            and answer_duration_sec >= config.LONG_HORIZON_MIN_SEC * 0.5
        ):
            drift_boost = config.LOW_DRIFT_BOOST * duration_factor
            out.explanation.append(
                f"extremely low behavioral drift over {answer_duration_sec:.0f}s"
            )

        clean_boost = 0.0
        if out.cleanliness_score >= config.CLEANLINESS_SUSPICION_THRESHOLD:
            clean_boost = config.CLEANLINESS_BOOST * duration_factor
            out.explanation.append("persistently controlled / too-clean delivery")

        fake_boost = 0.0
        if out.fake_natularity_score >= config.FAKE_NATURALITY_THRESHOLD:
            fake_boost = config.FAKE_NATURALITY_BOOST * duration_factor
            out.explanation.append("statistically regular fake-spontaneity patterns")

        base_boost = dominance_boost + drift_boost + clean_boost + fake_boost
        out.answer_contrastive_floor = min(
            config.ANSWER_CONTRASTIVE_FLOOR_MAX,
            base_boost * config.ANSWER_FLOOR_SCALE,
        )

        for i, row in enumerate(rows):
            boost = base_boost
            if row.script_similarity >= config.SCRIPT_DOMINANCE_THRESHOLD:
                boost += config.PER_WINDOW_SCRIPT_BOOST
            if (
                row.script_similarity >= 0.65
                and row.natural_similarity_effective >= 0.75
            ):
                boost += config.SCRIPT_NATURAL_COLLAPSE_BOOST
            out.per_window_boost[i] = round(boost, 6)

        if out.script_dominance_active and out.low_drift_score >= 0.55:
            out.explanation.append("script evidence allowed to dominate despite elevated natural similarity")

        return out


def _low_drift_score(
    wps: list[float],
    gaps: list[float],
    pause_ents: list[float],
    pitch_deltas: list[float],
) -> float:
    """Higher => more suspiciously stable (low drift)."""
    scores: list[float] = []

    if len(wps) >= 2:
        wps_cv = float(np.std(wps, ddof=1) / (np.mean(wps) + 1e-6))
        scores.append(1.0 - min(1.0, wps_cv / 0.25))

    if len(gaps) >= 2:
        gap_cv = float(np.std(gaps, ddof=1) / (np.mean(gaps) + 1e-6))
        scores.append(1.0 - min(1.0, gap_cv / 0.8))

    if len(pause_ents) >= 2:
        ent_range = max(pause_ents) - min(pause_ents)
        scores.append(1.0 - min(1.0, ent_range / 1.5))

    if len(pitch_deltas) >= 2:
        pitch_cv = float(np.std(pitch_deltas, ddof=1) / (np.mean(pitch_deltas) + 1e-6))
        scores.append(1.0 - min(1.0, pitch_cv / 0.6))

    if not scores:
        return 0.0
    return float(np.mean(scores))


def _cleanliness_persistence(
    wps: list[float],
    gaps: list[float],
    pause_ents: list[float],
    script_sims: list[float],
) -> float:
    """Highly stable pacing + controlled delivery over the answer."""
    stability_parts: list[float] = []

    if len(wps) >= 2:
        wps_std = float(np.std(wps, ddof=1))
        stability_parts.append(1.0 - min(1.0, wps_std / 0.35))

    if len(gaps) >= 2:
        gap_std = float(np.std(gaps, ddof=1))
        stability_parts.append(1.0 - min(1.0, gap_std / 0.008))

    if len(pause_ents) >= 2:
        ent_std = float(np.std(pause_ents, ddof=1))
        stability_parts.append(1.0 - min(1.0, ent_std / 0.9))

    if not stability_parts:
        return 0.0

    stability = float(np.mean(stability_parts))
    script_pressure = float(np.mean([1.0 if s >= 0.62 else s / 0.62 for s in script_sims]))
    return float(min(1.0, stability * (0.55 + 0.45 * script_pressure)))


def _fake_natularity_score(rows: list[WindowScoreRow]) -> float:
    """Detect regular fillers/hesitations without true variability."""
    if len(rows) < 2:
        return 0.0

    filler_rates = [r.features.get("ling_filler_rate_per_30s", 0.0) for r in rows]
    pause_maxes = [r.features.get("ling_retrieval_pause_max", 0.0) for r in rows]
    nat_scores = [r.naturality_score for r in rows]

    signals: list[float] = []

    if len(filler_rates) >= 2 and np.mean(filler_rates) > 0.5:
        fr_cv = float(np.std(filler_rates, ddof=1) / (np.mean(filler_rates) + 1e-6))
        if fr_cv < 0.35:
            signals.append(0.7)

    if len(pause_maxes) >= 2:
        pm_cv = float(np.std(pause_maxes, ddof=1) / (np.mean(pause_maxes) + 1e-6))
        if 0.1 < np.mean(pause_maxes) < 1.2 and pm_cv < 0.4:
            signals.append(0.65)

    if len(nat_scores) >= 2:
        nat_std = float(np.std(nat_scores, ddof=1))
        if np.mean(nat_scores) > 0.55 and nat_std < 0.08:
            signals.append(0.75)

    high_nat_high_script = sum(
        1
        for r in rows
        if r.naturality_score >= 0.55
        and r.script_similarity >= 0.62
        and r.natural_similarity_effective >= 0.7
    )
    if high_nat_high_script >= max(2, len(rows) // 2):
        signals.append(0.8)

    return float(max(signals)) if signals else 0.0


def _local_low_drift(features: dict[str, float]) -> float:
    """Per-window heuristic for benign-guard."""
    gv = features.get("ling_gap_variance", 0.0)
    ent = features.get("ling_pause_entropy", 0.0)
    wps = features.get("ling_wps", 0.0)
    score = 0.0
    if gv < 0.003:
        score += 0.35
    if ent > 0.5 and gv < 0.012:
        score += 0.25
    if 2.0 <= wps <= 3.8 and gv < 0.008:
        score += 0.25
    return min(1.0, score)


def build_decision_explanation(
    summary: dict[str, Any],
    horizon: HorizonAnalysis,
) -> list[str]:
    """Human-readable reasons for answer-level contrastive decision."""
    reasons: list[str] = list(horizon.explanation)
    ewma = float(summary.get("ewma_score", 0))
    margin = config.CONTRASTIVE_MARGIN

    if summary.get("status") == "PROBABLE_SCRIPT_READING":
        if horizon.script_dominance_active:
            reasons.insert(0, "script-dominance override engaged")
        if ewma >= margin:
            reasons.append(f"contrastive EWMA {ewma:.3f} ≥ margin {margin}")
        if summary.get("persistent"):
            reasons.append(
                f"sustained suspicious windows ({summary.get('consecutive_suspicious', 0)} consecutive)"
            )
    elif summary.get("status") == "AMBIGUOUS":
        reasons.append(f"moderate sustained contrastive pressure (EWMA {ewma:.3f})")
    elif horizon.script_dominance_active and ewma < margin:
        reasons.append("script-like consistency detected but below alert persistence")

    return reasons[:8]

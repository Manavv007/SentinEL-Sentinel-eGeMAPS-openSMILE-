"""
Cognitive Sourcing Inference — distinguish internally generated vs externally guided cognition.

Models semantic-effort covariance, chunk transitions, segment spontaneity variance,
inter-answer variability, and session-level evidence accumulation (not surface polish).
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config

_STATUS_RANK = {
    "CLEAR": 0,
    "AMBIGUOUS": 1,
    "PROBABLE_SCRIPT_READING": 2,
}


def _normalize_decision_explanation(value: Any) -> list[str]:
    """Ensure explainability field stays a list for API/UI consumers."""
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Repair legacy strings that concatenated list explanations into one blob.
        if "; [" in text:
            head, tail = text.split("; [", 1)
            items: list[str] = []
            if head.strip():
                items.append(head.strip())
            try:
                import ast

                parsed = ast.literal_eval("[" + tail)
                if isinstance(parsed, list):
                    items.extend(str(item).strip() for item in parsed if str(item).strip())
                    return items
            except (SyntaxError, ValueError):
                pass
        return [text]
    return [str(value)]


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _safe_pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return 0.0
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if float(np.std(xa)) < 1e-6 or float(np.std(ya)) < 1e-6:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def _window_series(windows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(w.get(key, 0.0) or 0.0) for w in windows]


def _normalize_windows_for_sourcing(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Accept temporal window exports (nested cognitive_breakdown) or flat window_dicts."""
    out: list[dict[str, Any]] = []
    for w in windows:
        breakdown = w.get("cognitive_breakdown") or {}
        pause = w.get("pause_metrics") or {}
        out.append(
            {
                **w,
                "cognitive_spontaneity": float(
                    w.get("cognitive_spontaneity", breakdown.get("cognitive_spontaneity", 0.0))
                    or 0.0
                ),
                "semantic_complexity": float(
                    w.get("semantic_complexity", breakdown.get("semantic_complexity", 0.0)) or 0.0
                ),
                "acoustic_turbulence": float(
                    w.get("acoustic_turbulence", breakdown.get("acoustic_turbulence", 0.0)) or 0.0
                ),
                "semantic_effort_decoupling": float(
                    w.get(
                        "semantic_effort_decoupling",
                        breakdown.get("semantic_effort_decoupling", 0.0),
                    )
                    or 0.0
                ),
                "pretoken_retrieval_adaptation": float(
                    w.get(
                        "pretoken_retrieval_adaptation",
                        breakdown.get("pretoken_retrieval_adaptation", 0.0),
                    )
                    or 0.0
                ),
                "semantic_acoustic_coherence": float(
                    w.get(
                        "semantic_acoustic_coherence",
                        breakdown.get("semantic_acoustic_coherence", 0.0),
                    )
                    or 0.0
                ),
                "semantic_repair": float(
                    w.get("semantic_repair", breakdown.get("semantic_repair", 0.0)) or 0.0
                ),
                "retrieval_friction": float(
                    w.get("retrieval_friction", breakdown.get("retrieval_friction", 0.0)) or 0.0
                ),
                "ling_gap_variance": float(
                    w.get("ling_gap_variance", pause.get("gap_variance", 0.0)) or 0.0
                ),
                "ling_retrieval_pause_max": float(
                    w.get("ling_retrieval_pause_max", pause.get("retrieval_pause_max", 0.0)) or 0.0
                ),
            }
        )
    return out


def compute_semantic_effort_covariance(windows: list[dict[str, Any]]) -> dict[str, float]:
    """
    Local covariance between semantic complexity and acoustic turbulence.
    Natural cognition tends toward positive coupling; external reading flattens effort.
    """
    complexity = _window_series(windows, "semantic_complexity")
    turbulence = _window_series(windows, "acoustic_turbulence")
    decoupling = _window_series(windows, "semantic_effort_decoupling")
    pretoken = _window_series(windows, "pretoken_retrieval_adaptation")

    raw_corr = _safe_pearson(complexity, turbulence)
    # Normalize correlation to 0..1 where 1 = strong positive coupling (internal-like)
    coupling_strength = _clamp01((raw_corr + 1.0) / 2.0)

    # Variance coupling: do complexity and turbulence co-vary in magnitude?
    if len(complexity) >= 3:
        c_std = float(np.std(complexity))
        t_std = float(np.std(turbulence))
        variance_coupling = _clamp01(min(c_std, t_std) / max(c_std, t_std, 1e-6))
    else:
        variance_coupling = 0.0

    mean_decouple = float(np.mean(decoupling)) if decoupling else 0.0
    mean_pretoken = float(np.mean(pretoken)) if pretoken else 0.0

    # Flat effort under semantic load → external sourcing signal
    flat_effort_under_load = 0.0
    if complexity:
        high_c_idx = [i for i, c in enumerate(complexity) if c >= float(np.percentile(complexity, 60))]
        if high_c_idx:
            high_turb = [turbulence[i] for i in high_c_idx]
            low_turb_ratio = sum(1 for t in high_turb if t < float(np.median(turbulence))) / len(high_turb)
            flat_effort_under_load = _clamp01(low_turb_ratio * (1.0 - coupling_strength))

    semantic_effort_covariance_score = _clamp01(
        0.55 * coupling_strength
        + 0.25 * variance_coupling
        + 0.12 * mean_pretoken
        - 0.18 * mean_decouple
        - 0.10 * flat_effort_under_load
    )

    return {
        "semantic_effort_covariance_score": round(semantic_effort_covariance_score, 4),
        "semantic_effort_raw_correlation": round(raw_corr, 4),
        "variance_coupling_score": round(variance_coupling, 4),
        "flat_effort_under_load_score": round(flat_effort_under_load, 4),
    }


def compute_chunk_transition_analysis(windows: list[dict[str, Any]]) -> dict[str, float]:
    """
    Memory-chunk retrieval shows boundary turbulence; external reading stays globally flat.
    """
    if len(windows) < 2:
        return {
            "chunk_transition_quality_score": 0.0,
            "chunk_boundary_turbulence_score": 0.0,
            "bridge_phrase_signal_score": 0.0,
            "retrieval_boundary_score": 0.0,
        }

    complexity = _window_series(windows, "semantic_complexity")
    turbulence = _window_series(windows, "acoustic_turbulence")
    repair = _window_series(windows, "semantic_repair")
    friction = _window_series(windows, "retrieval_friction")
    pretoken = _window_series(windows, "pretoken_retrieval_adaptation")
    spontaneity = _window_series(windows, "cognitive_spontaneity")

    boundary_turb: list[float] = []
    bridge_signals: list[float] = []
    retrieval_bounds: list[float] = []

    for i in range(1, len(windows)):
        dc = abs(complexity[i] - complexity[i - 1])
        dt = abs(turbulence[i] - turbulence[i - 1])
        boundary_turb.append(_clamp01(dt + dc * 0.35))
        bridge_signals.append(_clamp01(repair[i] * 0.55 + repair[i - 1] * 0.25))
        retrieval_bounds.append(
            _clamp01(friction[i] * 0.45 + pretoken[i] * 0.35 + pretoken[i - 1] * 0.20)
        )

    boundary_mean = float(np.mean(boundary_turb)) if boundary_turb else 0.0
    bridge_mean = float(np.mean(bridge_signals)) if bridge_signals else 0.0
    retrieval_mean = float(np.mean(retrieval_bounds)) if retrieval_bounds else 0.0

    # Global stabilization: low spontaneity variance + low boundary turbulence
    spont_var = float(np.var(spontaneity)) if spontaneity else 0.0
    global_flat_penalty = _clamp01(1.0 - min(1.0, spont_var * 8.0) - boundary_mean * 0.4)

    chunk_transition_quality_score = _clamp01(
        0.38 * boundary_mean
        + 0.28 * bridge_mean
        + 0.24 * retrieval_mean
        - 0.10 * global_flat_penalty
    )

    return {
        "chunk_transition_quality_score": round(chunk_transition_quality_score, 4),
        "chunk_boundary_turbulence_score": round(boundary_mean, 4),
        "bridge_phrase_signal_score": round(bridge_mean, 4),
        "retrieval_boundary_score": round(retrieval_mean, 4),
    }


def compute_segment_spontaneity_variance(windows: list[dict[str, Any]]) -> dict[str, float]:
    """
    Natural prepared speech varies across segments; externally guided speech stabilizes uniformly.
    """
    if not windows:
        return {
            "segment_spontaneity_variance": 0.0,
            "segment_effort_coherence_variance": 0.0,
            "segment_stabilization_uniformity": 0.0,
        }

    seg_size = max(2, min(4, len(windows) // 3 or 2))
    segments: list[list[dict[str, Any]]] = []
    for i in range(0, len(windows), seg_size):
        segments.append(windows[i : i + seg_size])

    seg_spont: list[float] = []
    seg_coherence: list[float] = []
    seg_decouple: list[float] = []
    seg_turb: list[float] = []

    for seg in segments:
        seg_spont.append(float(np.mean(_window_series(seg, "cognitive_spontaneity"))))
        seg_coherence.append(float(np.mean(_window_series(seg, "semantic_acoustic_coherence"))))
        seg_decouple.append(float(np.mean(_window_series(seg, "semantic_effort_decoupling"))))
        seg_turb.append(float(np.mean(_window_series(seg, "acoustic_turbulence"))))

    spont_var = float(np.var(seg_spont)) if len(seg_spont) > 1 else 0.0
    coherence_var = float(np.var(seg_coherence)) if len(seg_coherence) > 1 else 0.0
    decouple_var = float(np.var(seg_decouple)) if len(seg_decouple) > 1 else 0.0
    turb_var = float(np.var(seg_turb)) if len(seg_turb) > 1 else 0.0

    segment_spontaneity_variance = _clamp01(spont_var * 6.0 + coherence_var * 2.5)
    segment_effort_coherence_variance = _clamp01(decouple_var * 5.0 + turb_var * 3.0)

    # High uniformity = low variance across segments (external-like)
    segment_stabilization_uniformity = _clamp01(
        1.0 - segment_spontaneity_variance * 0.55 - segment_effort_coherence_variance * 0.45
    )

    return {
        "segment_spontaneity_variance": round(segment_spontaneity_variance, 4),
        "segment_effort_coherence_variance": round(segment_effort_coherence_variance, 4),
        "segment_stabilization_uniformity": round(segment_stabilization_uniformity, 4),
    }


def compute_answer_sourcing_signals(
    windows: list[dict[str, Any]],
    behavioral: dict[str, Any],
    temporal_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Per-answer cognitive sourcing feature bundle."""
    temporal_summary = temporal_summary or {}
    windows = _normalize_windows_for_sourcing(windows)
    covariance = compute_semantic_effort_covariance(windows)
    chunks = compute_chunk_transition_analysis(windows)
    segments = compute_segment_spontaneity_variance(windows)

    sem_decouple = float(behavioral.get("semantic_effort_decoupling_score", 0.0) or 0.0)
    sem_coh = float(behavioral.get("semantic_acoustic_coherence_score", 0.0) or 0.0)
    pretoken = float(behavioral.get("pretoken_retrieval_adaptation_score", 0.0) or 0.0)
    cog_fluct = float(behavioral.get("cognitive_fluctuation_score", 0.0) or 0.0)
    susp_var = float(temporal_summary.get("suspicion_variance", 0.0) or 0.0)

    effort_dynamics_flatness = _clamp01(
        0.40 * segments["segment_stabilization_uniformity"]
        + 0.30 * covariance["flat_effort_under_load_score"]
        + 0.20 * sem_decouple
        - 0.25 * covariance["semantic_effort_covariance_score"]
        - 0.15 * cog_fluct
    )

    return {
        **covariance,
        **chunks,
        **segments,
        "effort_dynamics_flatness_score": round(effort_dynamics_flatness, 4),
        "behavioral_stabilization_persistence": round(
            _clamp01(
                0.35 * segments["segment_stabilization_uniformity"]
                + 0.25 * (1.0 - min(1.0, susp_var * 12.0))
                + 0.20 * sem_decouple
                + 0.20 * (1.0 - sem_coh)
            ),
            4,
        ),
        "retrieval_adaptation_presence": round(
            _clamp01(0.55 * pretoken + 0.25 * chunks["retrieval_boundary_score"] + 0.20 * cog_fluct),
            4,
        ),
    }


def compute_speaker_style_vector(answer_profiles: list[dict[str, Any]]) -> dict[str, float]:
    """Estimate natural delivery baseline to separate style from cognitive sourcing state."""
    if not answer_profiles:
        return {
            "natural_pacing_tendency": 0.5,
            "natural_fluency_tendency": 0.5,
            "natural_pause_style": 0.5,
            "natural_turbulence_baseline": 0.5,
        }

    pacing = [float(p.get("pacing_baseline", 0.5)) for p in answer_profiles]
    fluency = [float(p.get("fluency_baseline", 0.5)) for p in answer_profiles]
    pause = [float(p.get("pause_style_baseline", 0.5)) for p in answer_profiles]
    turb = [float(p.get("turbulence_baseline", 0.5)) for p in answer_profiles]

    return {
        "natural_pacing_tendency": round(float(np.median(pacing)), 4),
        "natural_fluency_tendency": round(float(np.median(fluency)), 4),
        "natural_pause_style": round(float(np.median(pause)), 4),
        "natural_turbulence_baseline": round(float(np.median(turb)), 4),
    }


def _answer_style_profile(
    windows: list[dict[str, Any]],
    behavioral: dict[str, Any],
) -> dict[str, float]:
    windows = _normalize_windows_for_sourcing(windows)
    gap_var = _window_series(windows, "ling_gap_variance")
    turb = _window_series(windows, "acoustic_turbulence")
    spont = _window_series(windows, "cognitive_spontaneity")

    return {
        "pacing_baseline": float(np.mean(gap_var)) if gap_var else 0.5,
        "fluency_baseline": _clamp01(
            1.0 - float(behavioral.get("suspicious_density", 0.0) or 0.0)
        ),
        "pause_style_baseline": float(np.mean(_window_series(windows, "ling_retrieval_pause_max")))
        if windows
        else 0.5,
        "turbulence_baseline": float(np.mean(turb)) if turb else float(np.mean(spont)) if spont else 0.5,
    }


def compute_interview_variability_profile(
    answer_profiles: list[dict[str, Any]],
) -> dict[str, float]:
    """Cross-answer behavioral modulation — natural speakers adapt; guided speakers stay flat."""
    if len(answer_profiles) < 2:
        return {
            "pacing_variance": 0.0,
            "turbulence_variance": 0.0,
            "hesitation_variance": 0.0,
            "semantic_effort_covariance_variance": 0.0,
            "retrieval_adaptation_variance": 0.0,
            "inter_answer_uniformity_score": 0.5,
            "cross_question_adaptation_score": 0.5,
        }

    def _var(key: str) -> float:
        vals = [float(p.get(key, 0.0)) for p in answer_profiles]
        return float(np.var(vals)) if vals else 0.0

    pacing_var = _var("pacing_baseline")
    turb_var = _var("turbulence_baseline")
    hesit_var = _var("pause_style_baseline")
    cov_var = _var("semantic_effort_covariance_score")
    retr_var = _var("retrieval_adaptation_presence")

    # Low variance across answers → artificially consistent (external-like)
    combined_var = pacing_var + turb_var + hesit_var + cov_var + retr_var
    inter_answer_uniformity = _clamp01(1.0 - min(1.0, combined_var * 4.0))

    # Natural adaptation: covariance and retrieval should vary with question load
    complexity_spread = _var("semantic_complexity_score")
    cross_question_adaptation = _clamp01(
        min(1.0, complexity_spread * 6.0) * 0.5
        + min(1.0, cov_var * 8.0) * 0.3
        + min(1.0, turb_var * 6.0) * 0.2
    )

    return {
        "pacing_variance": round(pacing_var, 6),
        "turbulence_variance": round(turb_var, 6),
        "hesitation_variance": round(hesit_var, 6),
        "semantic_effort_covariance_variance": round(cov_var, 6),
        "retrieval_adaptation_variance": round(retr_var, 6),
        "inter_answer_uniformity_score": round(inter_answer_uniformity, 4),
        "cross_question_adaptation_score": round(cross_question_adaptation, 4),
    }


def compute_sourcing_likelihoods(
    signals: dict[str, Any],
    *,
    style: dict[str, float] | None = None,
    variability: dict[str, float] | None = None,
    behavioral: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Relative plausibility: internal generation vs external sourcing.
    """
    style = style or {}
    variability = variability or {}
    behavioral = behavioral or {}

    cov = float(signals.get("semantic_effort_covariance_score", 0.0))
    chunk = float(signals.get("chunk_transition_quality_score", 0.0))
    seg_var = float(signals.get("segment_spontaneity_variance", 0.0))
    flat = float(signals.get("effort_dynamics_flatness_score", 0.0))
    uniform = float(signals.get("segment_stabilization_uniformity", 0.0))
    stab_persist = float(signals.get("behavioral_stabilization_persistence", 0.0))
    retrieval = float(signals.get("retrieval_adaptation_presence", 0.0))
    inter_uniform = float(variability.get("inter_answer_uniformity_score", 0.5))
    cross_adapt = float(variability.get("cross_question_adaptation_score", 0.5))

    sem_decouple = float(behavioral.get("semantic_effort_decoupling_score", 0.0) or 0.0)
    guidance_dom = float(behavioral.get("guidance_dominance_score", 0.0) or 0.0)
    prepared = bool(behavioral.get("prepared_internal_speech_protection", False))

    internal_generation_likelihood = _clamp01(
        0.28 * cov
        + 0.14 * chunk
        + 0.14 * seg_var
        + 0.18 * retrieval
        + 0.12 * cross_adapt
        + 0.14 * (1.0 - inter_uniform)
    )

    external_sourcing_likelihood = _clamp01(
        0.18 * flat
        + 0.14 * uniform
        + 0.14 * stab_persist
        + 0.16 * sem_decouple
        + 0.12 * guidance_dom
        + 0.10 * inter_uniform
        + 0.10 * (1.0 - cross_adapt)
        - 0.14 * cov
    )

    if prepared and cov >= 0.55 and retrieval >= 0.35:
        internal_generation_likelihood = _clamp01(internal_generation_likelihood + 0.10)
        external_sourcing_likelihood = _clamp01(external_sourcing_likelihood - 0.08)

    # Prepared internalization tolerance — weaken external inference for chunk-like internal speech
    if prepared and config.PREPARED_INTERNALIZATION_PROTECTION:
        weaken = _clamp01(
            0.35 * seg_var
            + 0.30 * chunk
            + 0.25 * retrieval
            + 0.10 * cov
        )
        if (
            uniform >= config.SOURCING_PROTECTION_WEAKEN_UNIFORM_MIN
            and cov <= config.SOURCING_PROTECTION_WEAKEN_COV_MAX
            and retrieval <= config.SOURCING_PROTECTION_WEAKEN_RETRIEVAL_MAX
        ):
            weaken *= 0.35
        external_sourcing_likelihood *= 1.0 - 0.65 * weaken
        external_sourcing_likelihood = _clamp01(external_sourcing_likelihood)
        internal_generation_likelihood = _clamp01(
            internal_generation_likelihood + 0.15 * weaken
        )

    # Style-state disentanglement: deviation from speaker baseline
    turb_base = float(style.get("natural_turbulence_baseline", 0.5))
    turb_now = float(behavioral.get("acoustic_turbulence_score", turb_base) or turb_base)
    state_deviation = _clamp01(abs(turb_now - turb_base) * 1.2)
    if state_deviation < 0.12 and uniform > 0.55:
        external_sourcing_likelihood = _clamp01(external_sourcing_likelihood + 0.06)

    sourcing_margin = external_sourcing_likelihood - internal_generation_likelihood
    external_soft_evidence = _clamp01(
        max(0.0, sourcing_margin) * 0.65 + external_sourcing_likelihood * 0.35
    )

    return {
        "internal_generation_likelihood": round(internal_generation_likelihood, 4),
        "external_sourcing_likelihood": round(external_sourcing_likelihood, 4),
        "sourcing_margin": round(sourcing_margin, 4),
        "external_soft_evidence": round(external_soft_evidence, 4),
    }


def enrich_behavioral_with_sourcing(
    behavioral: dict[str, Any],
    windows: list[dict[str, Any]],
    temporal_summary: dict[str, Any] | None = None,
    *,
    style: dict[str, float] | None = None,
    variability: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Merge answer-level sourcing signals into behavioral synthesis dict."""
    if not config.ENABLE_COGNITIVE_SOURCING:
        return behavioral

    signals = compute_answer_sourcing_signals(windows, behavioral, temporal_summary)
    likelihoods = compute_sourcing_likelihoods(
        signals,
        style=style,
        variability=variability,
        behavioral=behavioral,
    )

    prepared_weakened = bool(
        behavioral.get("prepared_internal_speech_protection", False)
        and signals.get("segment_stabilization_uniformity", 0.0)
        >= config.SOURCING_PROTECTION_WEAKEN_UNIFORM_MIN
        and signals.get("semantic_effort_covariance_score", 1.0)
        <= config.SOURCING_PROTECTION_WEAKEN_COV_MAX
        and signals.get("retrieval_adaptation_presence", 1.0)
        <= config.SOURCING_PROTECTION_WEAKEN_RETRIEVAL_MAX
    )

    out = {**behavioral, **signals, **likelihoods}
    out["prepared_internalization_protection_active"] = bool(
        behavioral.get("prepared_internal_speech_protection", False) and not prepared_weakened
    )
    out["prepared_internalization_protection_weakened"] = prepared_weakened
    return out


def session_level_sourcing_inference(
    results_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Interview-level evidence accumulation — persistent external sourcing patterns.
    """
    if not config.ENABLE_COGNITIVE_SOURCING or not results_answers:
        return {"enabled": False}

    profiles: list[dict[str, Any]] = []
    for ans in results_answers:
        contrastive = ans.get("contrastive") or {}
        behavioral = contrastive.get("behavioral_synthesis") or {}
        windows = contrastive.get("windows") or []
        if not behavioral and not windows:
            continue
        style_prof = _answer_style_profile(windows, behavioral)
        signals = compute_answer_sourcing_signals(windows, behavioral, contrastive)
        profiles.append(
            {
                **style_prof,
                **signals,
                "semantic_complexity_score": float(
                    behavioral.get("semantic_complexity_score", 0.0) or 0.0
                ),
                "answer_id": ans.get("answer_id"),
                "status": ans.get("status", "CLEAR"),
                "external_soft_evidence": float(
                    behavioral.get("external_soft_evidence", 0.0) or 0.0
                ),
                "external_sourcing_likelihood": float(
                    behavioral.get("external_sourcing_likelihood", 0.0) or 0.0
                ),
                "internal_generation_likelihood": float(
                    behavioral.get("internal_generation_likelihood", 0.0) or 0.0
                ),
            }
        )

    style = compute_speaker_style_vector(profiles)
    variability = compute_interview_variability_profile(profiles)

    if not profiles:
        return {"enabled": True, "speaker_style_vector": style, "interview_variability_profile": variability}

    soft_scores = [float(p.get("external_soft_evidence", 0.0)) for p in profiles]
    ext_scores = [float(p.get("external_sourcing_likelihood", 0.0)) for p in profiles]
    stab_scores = [float(p.get("behavioral_stabilization_persistence", 0.0)) for p in profiles]

    mean_soft = float(np.mean(soft_scores)) if soft_scores else 0.0
    mean_ext = float(np.mean(ext_scores)) if ext_scores else 0.0
    persistence = float(np.mean(stab_scores)) if stab_scores else 0.0

    # Probabilistic accumulation: weak signals across many answers add up
    accumulated_soft = float(np.mean([_clamp01(s * 0.85 + mean_soft * 0.15) for s in soft_scores]))
    elevated_count = sum(1 for s in soft_scores if s >= config.SOURCING_SOFT_EVIDENCE_MIN)
    elevated_ratio = elevated_count / max(len(soft_scores), 1)

    session_external_likelihood = _clamp01(
        0.35 * mean_ext
        + 0.30 * accumulated_soft
        + 0.20 * persistence
        + 0.10 * variability["inter_answer_uniformity_score"]
        + 0.05 * elevated_ratio
    )
    session_internal_likelihood = _clamp01(
        1.0
        - session_external_likelihood * 0.55
        + variability["cross_question_adaptation_score"] * 0.25
        + (1.0 - variability["inter_answer_uniformity_score"]) * 0.20
    )

    return {
        "enabled": True,
        "speaker_style_vector": style,
        "interview_variability_profile": variability,
        "session_external_sourcing_likelihood": round(session_external_likelihood, 4),
        "session_internal_generation_likelihood": round(session_internal_likelihood, 4),
        "accumulated_soft_evidence": round(accumulated_soft, 4),
        "stabilization_persistence": round(persistence, 4),
        "elevated_soft_evidence_ratio": round(elevated_ratio, 4),
        "answer_sourcing_profiles": profiles,
    }


def apply_session_sourcing_refinement(
    results_answers: list[dict[str, Any]],
    session: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Soft evidence refinement pass — upgrade/downgrade with explainable session reasoning.
    Does not discard ambiguous signals; accumulates them at interview level.
    """
    if not session.get("enabled"):
        return results_answers

    session_ext = float(session.get("session_external_sourcing_likelihood", 0.0))
    session_int = float(session.get("session_internal_generation_likelihood", 0.0))
    variability = session.get("interview_variability_profile") or {}
    inter_uniform = float(variability.get("inter_answer_uniformity_score", 0.5))

    for ans in results_answers:
        contrastive = ans.get("contrastive")
        if not isinstance(contrastive, dict):
            continue
        behavioral = contrastive.get("behavioral_synthesis") or {}
        ext = float(behavioral.get("external_sourcing_likelihood", 0.0) or 0.0)
        soft = float(behavioral.get("external_soft_evidence", 0.0) or 0.0)
        internal = float(behavioral.get("internal_generation_likelihood", 0.0) or 0.0)
        prepared_active = bool(behavioral.get("prepared_internalization_protection_active", False))

        status = str(ans.get("status", "CLEAR"))
        reasons: list[str] = list(contrastive.get("sourcing_refinement_reasons") or [])
        strong_n = int(behavioral.get("strong_window_count", 0) or 0)
        essay_guided = float(behavioral.get("essay_like_guidedness_score", 0.0) or 0.0)
        intra = ans.get("intra_individual") or {}
        p_intra = float(intra.get("p_external_guidance", 0.5))
        spec = ans.get("semantic_specificity") or {}
        personal_natural = False
        if spec:
            from engine.semantic_specificity import is_personal_natural_answer

            personal_natural = is_personal_natural_answer(spec)
        mem_tech = float(spec.get("memorized_technical_script_score", 0.0))

        # Session-level promotion when persistent external sourcing dominates
        promote_probable = (
            session_ext >= config.SESSION_EXTERNAL_LIKELIHOOD_PROBABLE_MIN
            and ext >= config.ANSWER_EXTERNAL_LIKELIHOOD_PROMOTE_MIN
            and soft >= config.SOURCING_SOFT_EVIDENCE_MIN
            and ext > internal + config.SOURCING_LIKELIHOOD_MARGIN
            and not prepared_active
            and not personal_natural
            and (
                status == "PROBABLE_SCRIPT_READING"
                or strong_n >= config.SESSION_SOURCING_MIN_STRONG_WINDOWS
                or essay_guided >= config.ESSAY_LIKE_GUIDEDNESS_MIN
                or mem_tech >= config.MEMORIZED_TECHNICAL_PROBABLE_MIN
            )
            and p_intra >= config.SESSION_P_AMBIGUOUS_LOW
        )
        promote_ambiguous = (
            not promote_probable
            and session_ext >= config.SESSION_EXTERNAL_LIKELIHOOD_AMBIGUOUS_MIN
            and soft >= config.SOURCING_SOFT_EVIDENCE_MIN * 0.75
            and ext > internal
            and status == "CLEAR"
            and not personal_natural
            and mem_tech < config.MEMORIZED_TECHNICAL_PROBABLE_MIN
        )

        new_status = status
        if promote_probable and _STATUS_RANK.get(status, 0) < _STATUS_RANK["PROBABLE_SCRIPT_READING"]:
            new_status = "PROBABLE_SCRIPT_READING"
            reasons.append(
                f"session sourcing inference: external likelihood {ext:.2f} "
                f"(session {session_ext:.2f}, soft evidence {soft:.2f}, "
                f"uniformity {inter_uniform:.2f})"
            )
        elif promote_ambiguous:
            new_status = "AMBIGUOUS"
            reasons.append(
                f"session soft evidence accumulation: external {ext:.2f} vs internal {internal:.2f} "
                f"(session {session_ext:.2f})"
            )

        # Soften over-promotion when session strongly favors internal generation
        if (
            new_status == "PROBABLE_SCRIPT_READING"
            and session_int >= config.SESSION_INTERNAL_LIKELIHOOD_CLEAR_MIN
            and session_ext < config.SESSION_EXTERNAL_LIKELIHOOD_AMBIGUOUS_MIN
            and soft < config.SOURCING_SOFT_EVIDENCE_MIN
        ):
            new_status = "AMBIGUOUS"
            reasons.append(
                f"session internal-generation prior {session_int:.2f} preserves uncertainty"
            )

        if new_status != status:
            ans["status"] = new_status
            contrastive["status"] = new_status
            if new_status == "PROBABLE_SCRIPT_READING":
                ans["confidence"] = contrastive.get("confidence", "MEDIUM")
            elif new_status == "AMBIGUOUS":
                ans["confidence"] = "MEDIUM" if soft >= config.SOURCING_SOFT_EVIDENCE_MIN else "LOW"

        if reasons:
            contrastive["sourcing_refinement_reasons"] = reasons
            existing = _normalize_decision_explanation(
                contrastive.get("decision_explanation")
            )
            merged = list(reasons)
            for item in existing:
                if item not in merged:
                    merged.append(item)
            contrastive["decision_explanation"] = merged

    return results_answers


def compute_interview_sourcing_context(
    results_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build style + variability context without session likelihood aggregation."""
    profiles: list[dict[str, Any]] = []
    for ans in results_answers:
        contrastive = ans.get("contrastive") or {}
        behavioral = contrastive.get("behavioral_synthesis") or {}
        windows = contrastive.get("windows") or []
        if not behavioral and not windows:
            continue
        style_prof = _answer_style_profile(windows, behavioral)
        signals = {
            key: float(behavioral.get(key, 0.0) or 0.0)
            for key in (
                "semantic_effort_covariance_score",
                "retrieval_adaptation_presence",
                "segment_spontaneity_variance",
                "behavioral_stabilization_persistence",
            )
        }
        if not any(signals.values()):
            signals = compute_answer_sourcing_signals(windows, behavioral, contrastive)
        profiles.append(
            {
                **style_prof,
                **signals,
                "semantic_complexity_score": float(
                    behavioral.get("semantic_complexity_score", 0.0) or 0.0
                ),
            }
        )

    return {
        "enabled": True,
        "speaker_style_vector": compute_speaker_style_vector(profiles),
        "interview_variability_profile": compute_interview_variability_profile(profiles),
        "answer_sourcing_profiles": profiles,
    }


def finalize_interview_sourcing(
    results_answers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Single-pass session sourcing: variability refresh then aggregate + refine."""
    if not config.ENABLE_COGNITIVE_SOURCING or not results_answers:
        return results_answers, {"enabled": False}

    context = compute_interview_sourcing_context(results_answers)
    results_answers = refresh_answer_sourcing_with_variability(results_answers, context)
    session = session_level_sourcing_inference(results_answers)
    session["speaker_style_vector"] = context["speaker_style_vector"]
    session["interview_variability_profile"] = context["interview_variability_profile"]
    results_answers = apply_session_sourcing_refinement(results_answers, session)
    return results_answers, session


def refresh_answer_sourcing_with_variability(
    results_answers: list[dict[str, Any]],
    session: dict[str, Any],
) -> list[dict[str, Any]]:
    """Re-score per-answer likelihoods using interview-level variability + style baseline."""
    if not session.get("enabled"):
        return results_answers

    variability = session.get("interview_variability_profile") or {}
    style = session.get("speaker_style_vector") or {}

    for ans in results_answers:
        contrastive = ans.get("contrastive")
        if not isinstance(contrastive, dict):
            continue
        behavioral = contrastive.get("behavioral_synthesis") or {}
        windows = contrastive.get("windows") or []
        if not behavioral:
            continue
        contrastive["behavioral_synthesis"] = enrich_behavioral_with_sourcing(
            behavioral,
            windows,
            contrastive,
            style=style,
            variability=variability,
        )

    return results_answers

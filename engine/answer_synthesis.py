"""Answer-level behavioral synthesis — dominant mode over tail EWMA."""

from __future__ import annotations

from typing import Any

import numpy as np

import config
from engine.suspicion_calibration import SuspicionLevel
from engine.temporal_reliability import short_answer_blocks_ambiguous


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
    wobble = [float(w.get("cognitive_wobble", 0.0)) for w in windows]
    repair = [float(w.get("semantic_repair", 0.0)) for w in windows]
    friction = [float(w.get("retrieval_friction", 0.0)) for w in windows]
    gap_var = [float(w.get("ling_gap_variance", 0.0)) for w in windows]
    retrieval_pause = [float(w.get("ling_retrieval_pause_max", 0.0)) for w in windows]
    essay_rhythm = [float(w.get("essay_like_rhythm", 0.0)) for w in windows]
    thematic_stability = [float(w.get("thematic_stability", 0.0)) for w in windows]
    emotional_grounding = [float(w.get("emotional_grounding", 0.0)) for w in windows]
    self_reference = [float(w.get("self_reference", 0.0)) for w in windows]
    semantic_complexity = [float(w.get("semantic_complexity", 0.0)) for w in windows]
    acoustic_turbulence = [float(w.get("acoustic_turbulence", 0.0)) for w in windows]
    pretoken_adaptation = [float(w.get("pretoken_retrieval_adaptation", 0.0)) for w in windows]
    semantic_acoustic_coherence = [float(w.get("semantic_acoustic_coherence", 0.0)) for w in windows]
    semantic_effort_decoupling = [float(w.get("semantic_effort_decoupling", 0.0)) for w in windows]

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
    cognitive_fluctuation_score = _cognitive_fluctuation_score(
        wobble=wobble,
        repair=repair,
        friction=friction,
        gap_var=gap_var,
        retrieval_pause=retrieval_pause,
    )
    guidance_dominance_score = _guidance_dominance_score(
        guided=float(np.mean(cog_guided)) if cog_guided else 0.0,
        spontaneity=float(np.mean(cog_spont)) if cog_spont else 0.0,
        density=suspicious_density,
        strong_ratio=strong / n,
        drift=behavioral_drift,
        recovery=recovery_strength,
        fluctuation=cognitive_fluctuation_score,
    )
    prepared_internal_protection = _prepared_internal_speech_protection(
        guidance_dominance_score=guidance_dominance_score,
        cognitive_fluctuation_score=cognitive_fluctuation_score,
        spontaneity=float(np.mean(cog_spont)) if cog_spont else 0.0,
        guided=float(np.mean(cog_guided)) if cog_guided else 0.0,
        recovery_strength=recovery_strength,
        suspicious_density=suspicious_density,
    )
    essay_like_guidedness_score = float(np.mean(essay_rhythm)) if essay_rhythm else 0.0
    thematic_stability_score = float(np.mean(thematic_stability)) if thematic_stability else 0.0
    emotional_grounding_score = _emotional_grounding_score(
        emotional=emotional_grounding,
        self_ref=self_reference,
        spontaneity=cog_spont,
        repair=repair,
    )
    semantic_guidedness_score = _semantic_guidedness_score(
        guided=float(np.mean(cog_guided)) if cog_guided else 0.0,
        essay_like=essay_like_guidedness_score,
        thematic_stability=thematic_stability_score,
        fluctuation=cognitive_fluctuation_score,
        emotional_grounding=emotional_grounding_score,
        recovery=recovery_strength,
        suspicious_density=suspicious_density,
    )
    semantic_complexity_score = float(np.mean(semantic_complexity)) if semantic_complexity else 0.0
    acoustic_turbulence_score = float(np.mean(acoustic_turbulence)) if acoustic_turbulence else 0.0
    pretoken_adaptation_score = float(np.mean(pretoken_adaptation)) if pretoken_adaptation else 0.0
    semantic_acoustic_coherence_score = (
        float(np.mean(semantic_acoustic_coherence)) if semantic_acoustic_coherence else 0.0
    )
    semantic_effort_decoupling_score = (
        float(np.mean(semantic_effort_decoupling)) if semantic_effort_decoupling else 0.0
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
        "cognitive_fluctuation_score": round(cognitive_fluctuation_score, 4),
        "guidance_dominance_score": round(guidance_dominance_score, 4),
        "prepared_internal_speech_protection": bool(prepared_internal_protection),
        "essay_like_guidedness_score": round(essay_like_guidedness_score, 4),
        "thematic_stability_score": round(thematic_stability_score, 4),
        "emotional_grounding_score": round(emotional_grounding_score, 4),
        "semantic_guidedness_score": round(semantic_guidedness_score, 4),
        "semantic_complexity_score": round(semantic_complexity_score, 4),
        "acoustic_turbulence_score": round(acoustic_turbulence_score, 4),
        "pretoken_retrieval_adaptation_score": round(pretoken_adaptation_score, 4),
        "semantic_acoustic_coherence_score": round(semantic_acoustic_coherence_score, 4),
        "semantic_effort_decoupling_score": round(semantic_effort_decoupling_score, 4),
        "weak_consistency_score": 0.0,
        "suspicious_coverage_ratio": 0.0,
        "weak_coverage_ratio": 0.0,
        "suspicion_variance": 0.0,
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
    duration_sec = float(temporal_summary.get("answer_duration_sec", 0.0) or 0.0)
    window_count = int(temporal_summary.get("window_count", 0) or 0)
    suspicious_cov = float(temporal_summary.get("suspicious_coverage_ratio", 0.0) or 0.0)
    weak_cov = float(temporal_summary.get("weak_coverage_ratio", 0.0) or 0.0)
    susp_var = float(temporal_summary.get("suspicion_variance", 0.0) or 0.0)
    weak_consistency = float(temporal_summary.get("weak_consistency_score", 0.0) or 0.0)
    consistency_auth = float(temporal_summary.get("consistency_authority_score", 0.0) or 0.0)
    flat_flow = bool(temporal_summary.get("flat_suspicious_flow_active", False))
    natural_breathing = bool(temporal_summary.get("natural_breathing_detected", False))
    stability_score = float(temporal_summary.get("suspicious_stability_score", 0.0) or 0.0)
    persistent_authority = bool(
        temporal_summary.get("persistent_weak_authority_active", False)
    )
    mod_n = int(temporal_summary.get("moderate_window_count", 0) or 0)
    guidance_dom = float(behavioral.get("guidance_dominance_score", 0.0) or 0.0)
    cog_fluct = float(behavioral.get("cognitive_fluctuation_score", 0.0) or 0.0)
    prepared_guard = bool(behavioral.get("prepared_internal_speech_protection", False))
    semantic_guidedness = float(behavioral.get("semantic_guidedness_score", 0.0) or 0.0)
    essay_guidedness = float(behavioral.get("essay_like_guidedness_score", 0.0) or 0.0)
    emotional_grounding = float(behavioral.get("emotional_grounding_score", 0.0) or 0.0)
    sem_coh = float(behavioral.get("semantic_acoustic_coherence_score", 0.0) or 0.0)
    sem_decouple = float(behavioral.get("semantic_effort_decoupling_score", 0.0) or 0.0)
    pretoken_quality = float(
        behavioral.get("pretoken_retrieval_adaptation_score", 0.0) or 0.0
    )
    ext_source = float(behavioral.get("external_sourcing_likelihood", 0.0) or 0.0)
    int_source = float(behavioral.get("internal_generation_likelihood", 0.0) or 0.0)
    soft_evidence = float(behavioral.get("external_soft_evidence", 0.0) or 0.0)
    sem_cov = float(behavioral.get("semantic_effort_covariance_score", 0.0) or 0.0)
    seg_spont_var = float(behavioral.get("segment_spontaneity_variance", 0.0) or 0.0)
    effort_flat = float(behavioral.get("effort_dynamics_flatness_score", 0.0) or 0.0)
    stab_persist = float(behavioral.get("behavioral_stabilization_persistence", 0.0) or 0.0)
    prepared_int_active = bool(
        behavioral.get("prepared_internalization_protection_active", prepared_guard)
    )
    consensus_score, consensus_active_signals = _behavioral_consensus_reasoning(
        guidance_dom=guidance_dom,
        semantic_guidedness=semantic_guidedness,
        persistence_signal=min(1.0, (streak / max(config.PROBABLE_LONGEST_STREAK_MIN, 1)) * 0.5),
        suspicious_density=density,
        weak_consistency=weak_consistency,
        recovery=recovery,
        cognitive_fluctuation=cog_fluct,
        emotional_grounding=emotional_grounding,
        natural_breathing=natural_breathing,
        semantic_effort_decoupling=sem_decouple,
        semantic_acoustic_coherence=sem_coh,
    )
    human_variability = _human_variability_prior(
        cognitive_fluctuation=cog_fluct,
        emotional_grounding=emotional_grounding,
        recovery=recovery,
        cognitive_spontaneity=cog_spont,
        natural_breathing=natural_breathing,
    )

    reasons: list[str] = []

    momentum = float(behavioral.get("suspicion_momentum", 0))
    natural_saturation = float(behavioral.get("natural_similarity_saturation_ratio", 0))
    nat_profile_samples = int(
        temporal_summary.get("natural_profile_samples")
        or behavioral.get("natural_profile_samples")
        or -1
    )
    natural_profile_cold = nat_profile_samples == 0
    cold_strong_min = config.COLD_START_PROBABLE_MIN_STRONG_WINDOWS

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
        and strong_n >= (cold_strong_min if natural_profile_cold else config.DOMINANT_MIN_STRONG_WINDOWS)
        and peak >= config.DOMINANT_MIN_PEAK_SUSPICION
        and recovery < config.DOMINANT_MAX_RECOVERY_STRENGTH
        and promotion_gate
        and natural_saturation < 0.65
    )

    peak_authority = (
        peak >= config.PEAK_SUSPICION_AUTHORITY
        and strong_n >= (cold_strong_min if natural_profile_cold else 2)
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
        and not natural_profile_cold
    )

  # Guided preconstructed flow (scripted answers 2,4,6,7 target)
    guided_scripted = (
        cog_guided >= config.COGNITIVE_GUIDED_HIGH_THRESHOLD
        and cog_spont < config.COGNITIVE_CLEAR_GUIDED_MAX
        and density >= config.DOMINANT_MIN_SUSPICIOUS_DENSITY * 0.85
        and peak >= config.SUSPICION_TIER_MODERATE
        and semantic_guidedness >= config.SEMANTIC_GUIDEDNESS_MIN_PROBABLE
    )
    # --- Cognitive sourcing inference: externally guided articulation ---
    sourcing_external_dominant = (
        config.ENABLE_COGNITIVE_SOURCING
        and ext_source >= config.SOURCING_EXTERNAL_PROMOTE_MIN
        and ext_source > int_source + config.SOURCING_LIKELIHOOD_MARGIN
        and soft_evidence >= config.SOURCING_SOFT_EVIDENCE_MIN * 0.85
        and effort_flat >= 0.42
        and sem_cov <= config.SOURCING_PROTECTION_WEAKEN_COV_MAX + 0.08
        and not prepared_int_active
        and (not natural_profile_cold or strong_n >= cold_strong_min)
    )
    if sourcing_external_dominant and (
        sem_decouple >= config.SEMANTIC_EFFORT_DECOUPLING_PROBABLE_MIN * 0.88
        or stab_persist >= 0.52
        or guidance_dom >= config.GUIDANCE_DOMINANCE_MIN_PROBABLE * 0.92
    ):
        reasons.append(
            f"cognitive sourcing: external likelihood {ext_source:.2f} vs internal {int_source:.2f} "
            f"(covariance {sem_cov:.2f}, flat effort {effort_flat:.2f}, persistence {stab_persist:.2f})"
        )
        conf = "HIGH" if ext_source >= 0.62 and soft_evidence >= config.SOURCING_SOFT_EVIDENCE_MIN else "MEDIUM"
        return "PROBABLE_SCRIPT_READING", conf, reasons

    if (
        guided_scripted
        and promotion_gate
        and recovery < 0.55
        and guidance_dom >= config.GUIDANCE_DOMINANCE_MIN_PROBABLE
        and consensus_score >= config.GENERALIZATION_CONSENSUS_MIN_SCORE
        and consensus_active_signals >= config.GENERALIZATION_DIVERSITY_MIN_SIGNALS
        and sem_decouple >= config.SEMANTIC_EFFORT_DECOUPLING_PROBABLE_MIN
    ):
        reasons.append(
            f"guided explanation flow (guided {cog_guided:.2f}, "
            f"low cognitive spontaneity {cog_spont:.2f})"
        )
        reasons.append(
            f"behavioral consensus {consensus_score:.2f} from {consensus_active_signals} independent signals"
        )
        reasons.append(
            f"semantic-effort decoupling {sem_decouple:.2f} "
            f"(coherence {sem_coh:.2f}, pretoken adaptation {pretoken_quality:.2f})"
        )
        conf = "HIGH" if dom >= config.DOMINANT_HIGH_CONFIDENCE_THRESHOLD else "MEDIUM"
        return "PROBABLE_SCRIPT_READING", conf, reasons

    # --- Consistency authority: flat persistent low-variance suspicious flow (Answer 5) ---
    if (
        flat_flow
        and strong_n == 0
        and not natural_breathing
        and not short_answer_blocks_ambiguous(
            duration_sec=duration_sec,
            window_count=window_count,
            strong_count=strong_n,
            moderate_count=mod_n,
            composite=composite,
            peak_suspicion=peak,
            margin=margin,
        )
    ):
        reasons.append(
            f"consistency authority: stabilized suspicious flow "
            f"(authority {consistency_auth:.2f}, stability {stability_score:.2f}, "
            f"coverage {suspicious_cov:.0%}, variance {susp_var:.4f}, no natural breathing)"
        )
        conf = (
            "MEDIUM"
            if consistency_auth >= config.CONSISTENCY_AUTHORITY_MIN_SCORE + 0.10
            else "LOW"
        )
        return "PROBABLE_SCRIPT_READING", conf, reasons

    # --- Persistent weak authority (reliability over peaks; no STRONG-streak gate) ---
    if (
        persistent_authority
        and strong_n == 0
        and cog_spont < 0.58
        and not natural_breathing
        and guidance_dom >= config.GUIDANCE_DOMINANCE_STRICT_FOR_WEAK
        and semantic_guidedness >= config.SEMANTIC_GUIDEDNESS_MIN_PROBABLE
        and consensus_score >= config.GENERALIZATION_CONSENSUS_MIN_SCORE
        and consensus_active_signals >= config.GENERALIZATION_DIVERSITY_MIN_SIGNALS
    ):
        reasons.append(
            f"temporal reliability: persistent weak suspiciousness "
            f"({suspicious_cov:.0%} coverage, consistency {weak_consistency:.2f}, "
            f"variance {susp_var:.4f}, recovery {recovery:.2f})"
        )
        conf = "MEDIUM" if weak_consistency >= config.PERSISTENT_WEAK_CONSISTENCY_MIN + 0.1 else "LOW"
        return "PROBABLE_SCRIPT_READING", conf, reasons

    if (scripted_dominant or peak_authority or density_authority) and (
        (guidance_dom >= config.GUIDANCE_DOMINANCE_MIN_PROBABLE and semantic_guidedness >= config.SEMANTIC_GUIDEDNESS_MIN_PROBABLE)
        or strong_n >= (cold_strong_min if natural_profile_cold else 3)
    ) and (
        not natural_profile_cold or strong_n >= cold_strong_min
    ) and (
        consensus_score >= config.GENERALIZATION_CONSENSUS_MIN_SCORE
        and consensus_active_signals >= config.GENERALIZATION_DIVERSITY_MIN_SIGNALS
        and sem_decouple >= config.SEMANTIC_EFFORT_DECOUPLING_PROBABLE_MIN
    ):
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
        if essay_guidedness >= config.ESSAY_LIKE_GUIDEDNESS_MIN:
            reasons.append(
                f"essay-like thematic continuity detected (semantic guidedness {semantic_guidedness:.2f})"
            )
        if drift >= config.LOW_DRIFT_SUSPICION_THRESHOLD:
            reasons.append("low long-term behavioral drift (globally stable delivery)")
        reasons.append(
            f"behavioral consensus {consensus_score:.2f} from {consensus_active_signals} independent signals"
        )
        reasons.append(
            f"semantic-effort decoupling {sem_decouple:.2f} "
            f"(coherence {sem_coh:.2f}, pretoken adaptation {pretoken_quality:.2f})"
        )
        return "PROBABLE_SCRIPT_READING", conf, reasons

    if prepared_int_active and config.PREPARED_BEHAVIOR_SOFTEN_TO_AMBIGUOUS:
        if temporal_status == "PROBABLE_SCRIPT_READING":
            reasons.append(
                f"prepared_internal_speech_protection: human turbulence present "
                f"(fluctuation {cog_fluct:.2f}, recovery {recovery:.2f}, emotional grounding {emotional_grounding:.2f}) "
                f"and guidance dominance {guidance_dom:.2f}"
            )
            return "AMBIGUOUS", "MEDIUM", reasons
        if temporal_status == "AMBIGUOUS":
            reasons.append(
                "prepared_internal_speech_protection: polished but internally generated behavior likely"
            )
            return "AMBIGUOUS", "LOW", reasons

    if (
        config.GENERALIZATION_ENABLE_UNCERTAINTY_PRESERVE
        and temporal_status == "PROBABLE_SCRIPT_READING"
        and human_variability >= config.GENERALIZATION_SAFETY_MAX_HUMAN_VARIABILITY
    ):
        reasons.append(
            f"generalization safety: human variability prior {human_variability:.2f} "
            "keeps uncertainty (likely fluent/prepared natural variability)"
        )
        return "AMBIGUOUS", "MEDIUM", reasons

    if (
        temporal_status == "PROBABLE_SCRIPT_READING"
        and sem_coh >= config.SEMANTIC_ACOUSTIC_COHERENCE_CLEAR_MIN
        and pretoken_quality >= config.PRETOKEN_ADAPTATION_CLEAR_MIN
        and sem_decouple < config.SEMANTIC_EFFORT_DECOUPLING_PROBABLE_MIN
    ):
        reasons.append(
            f"semantic-acoustic coupling appears naturally plausible "
            f"(coherence {sem_coh:.2f}, pretoken adaptation {pretoken_quality:.2f})"
        )
        return "AMBIGUOUS", "MEDIUM", reasons

    # --- Soft ambiguous evidence (not a discard bucket) ---
    if (
        config.ENABLE_COGNITIVE_SOURCING
        and soft_evidence >= config.SOURCING_SOFT_EVIDENCE_MIN
        and ext_source > int_source
        and temporal_status in ("CLEAR", "AMBIGUOUS")
        and not prepared_int_active
    ):
        reasons.append(
            f"cognitive sourcing soft evidence: external {ext_source:.2f} "
            f"(covariance {sem_cov:.2f}, segment variance {seg_spont_var:.2f})"
        )
        return "AMBIGUOUS", "MEDIUM" if soft_evidence >= config.SOURCING_SOFT_EVIDENCE_MIN else "LOW", reasons

    # --- Fluent natural cognition (Answer 3 target) ---
    fluent_natural = (
        cog_spont >= config.COGNITIVE_CLEAR_SPONTANEITY_MIN
        and cog_guided <= config.COGNITIVE_CLEAR_GUIDED_MAX
        and strong_n == 0
        and peak < config.SUSPICION_TIER_STRONG
        and not flat_flow
        and not natural_breathing
        and consistency_auth < config.CONSISTENCY_AUTHORITY_MIN_SCORE
    )
    if (
        fluent_natural
        and dom < config.DOMINANT_SCRIPT_READING_THRESHOLD
        and not (
            config.ENABLE_COGNITIVE_SOURCING
            and ext_source >= config.SOURCING_EXTERNAL_PROMOTE_MIN
            and soft_evidence >= config.SOURCING_SOFT_EVIDENCE_MIN
        )
    ):
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
        if short_answer_blocks_ambiguous(
            duration_sec=duration_sec,
            window_count=window_count,
            strong_count=strong_n,
            moderate_count=mod_n,
            composite=composite,
            peak_suspicion=peak,
            margin=margin,
        ):
            reasons.append(
                f"short answer reliability guard ({duration_sec:.0f}s, {window_count} windows, "
                f"{mod_n} moderate) — insufficient sustained evidence"
            )
            return "CLEAR", "LOW", reasons
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
        if natural_profile_cold and strong_n < cold_strong_min:
            reasons.append(
                f"NATURAL profile cold start: {strong_n} STRONG windows "
                f"(need {cold_strong_min}) — preserving uncertainty"
            )
            return "AMBIGUOUS", "MEDIUM", reasons
        if (
            (guidance_dom < config.GUIDANCE_DOMINANCE_MIN_PROBABLE or semantic_guidedness < config.SEMANTIC_GUIDEDNESS_MIN_PROBABLE)
            and strong_n < 3
        ):
            reasons.append(
                f"guidedness authority insufficient (guidance dominance {guidance_dom:.2f}, "
                f"semantic guidedness {semantic_guidedness:.2f}) for externally guided flow"
            )
            return "AMBIGUOUS", "MEDIUM", reasons
        if (
            consensus_score < config.GENERALIZATION_CONSENSUS_MIN_SCORE
            or consensus_active_signals < config.GENERALIZATION_DIVERSITY_MIN_SIGNALS
        ):
            reasons.append(
                f"generalization-first safeguard: consensus {consensus_score:.2f}, "
                f"signal diversity {consensus_active_signals} — insufficient multi-dimensional agreement"
            )
            return "AMBIGUOUS", "MEDIUM", reasons
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
        "cognitive_fluctuation_score": 0.0,
        "guidance_dominance_score": 0.0,
        "prepared_internal_speech_protection": False,
        "essay_like_guidedness_score": 0.0,
        "thematic_stability_score": 0.0,
        "emotional_grounding_score": 0.0,
        "semantic_guidedness_score": 0.0,
        "semantic_complexity_score": 0.0,
        "acoustic_turbulence_score": 0.0,
        "pretoken_retrieval_adaptation_score": 0.0,
        "semantic_acoustic_coherence_score": 0.0,
        "semantic_effort_decoupling_score": 0.0,
        "semantic_effort_covariance_score": 0.0,
        "segment_spontaneity_variance": 0.0,
        "chunk_transition_quality_score": 0.0,
        "effort_dynamics_flatness_score": 0.0,
        "behavioral_stabilization_persistence": 0.0,
        "internal_generation_likelihood": 0.0,
        "external_sourcing_likelihood": 0.0,
        "external_soft_evidence": 0.0,
        "prepared_internalization_protection_active": False,
        "prepared_internalization_protection_weakened": False,
    }


def _cognitive_fluctuation_score(
    *,
    wobble: list[float],
    repair: list[float],
    friction: list[float],
    gap_var: list[float],
    retrieval_pause: list[float],
) -> float:
    if not wobble:
        return 0.0
    wobble_m = float(np.mean(wobble))
    repair_m = float(np.mean(repair)) if repair else 0.0
    friction_m = float(np.mean(friction)) if friction else 0.0
    gap_m = float(np.mean(gap_var)) if gap_var else 0.0
    pause_m = float(np.mean(retrieval_pause)) if retrieval_pause else 0.0
    pause_term = min(1.0, pause_m / 1.2)
    pacing_term = min(1.0, gap_m / 0.01)
    return float(
        min(
            1.0,
            0.34 * wobble_m + 0.28 * repair_m + 0.18 * friction_m + 0.12 * pacing_term + 0.08 * pause_term,
        )
    )


def _guidance_dominance_score(
    *,
    guided: float,
    spontaneity: float,
    density: float,
    strong_ratio: float,
    drift: float,
    recovery: float,
    fluctuation: float,
) -> float:
    return float(
        min(
            1.0,
            max(
                0.0,
                0.34 * guided
                + 0.20 * density
                + 0.14 * strong_ratio
                + 0.12 * drift
                - 0.14 * spontaneity
                - 0.10 * recovery
                - 0.14 * fluctuation,
            ),
        )
    )


def _prepared_internal_speech_protection(
    *,
    guidance_dominance_score: float,
    cognitive_fluctuation_score: float,
    spontaneity: float,
    guided: float,
    recovery_strength: float,
    suspicious_density: float,
) -> bool:
    if guidance_dominance_score >= config.GUIDANCE_DOMINANCE_MIN_PROBABLE:
        return False
    if spontaneity < config.REHEARSAL_PROTECTION_MIN_SPONTANEITY:
        return False
    if guided > config.REHEARSAL_PROTECTION_MAX_GUIDED:
        return False
    if cognitive_fluctuation_score < config.REHEARSAL_PROTECTION_MIN_TURBULENCE:
        return False
    if recovery_strength < config.REHEARSAL_PROTECTION_MIN_RECOVERY:
        return False
    if guidance_dominance_score >= config.GUIDANCE_DOMINANCE_MIN_PROBABLE * 0.95:
        return False
    return suspicious_density <= config.REHEARSAL_PROTECTION_MAX_STABILITY


def _emotional_grounding_score(
    *,
    emotional: list[float],
    self_ref: list[float],
    spontaneity: list[float],
    repair: list[float],
) -> float:
    emo = float(np.mean(emotional)) if emotional else 0.0
    self_m = float(np.mean(self_ref)) if self_ref else 0.0
    spont = float(np.mean(spontaneity)) if spontaneity else 0.0
    rep = float(np.mean(repair)) if repair else 0.0
    return float(min(1.0, 0.38 * emo + 0.27 * self_m + 0.20 * spont + 0.15 * rep))


def _semantic_guidedness_score(
    *,
    guided: float,
    essay_like: float,
    thematic_stability: float,
    fluctuation: float,
    emotional_grounding: float,
    recovery: float,
    suspicious_density: float,
) -> float:
    return float(
        min(
            1.0,
            max(
                0.0,
                0.34 * guided
                + 0.24 * essay_like
                + 0.22 * thematic_stability
                + 0.12 * suspicious_density
                - 0.12 * fluctuation
                - 0.10 * emotional_grounding
                - 0.08 * recovery,
            ),
        )
    )


def _behavioral_consensus_reasoning(
    *,
    guidance_dom: float,
    semantic_guidedness: float,
    persistence_signal: float,
    suspicious_density: float,
    weak_consistency: float,
    recovery: float,
    cognitive_fluctuation: float,
    emotional_grounding: float,
    natural_breathing: bool,
    semantic_effort_decoupling: float = 0.0,
    semantic_acoustic_coherence: float = 0.0,
) -> tuple[float, int]:
    """Generalization-first consensus over independent dimensions."""
    scores = {
        "guidedness": 0.5 * guidance_dom + 0.5 * semantic_guidedness,
        "persistence": 0.6 * persistence_signal + 0.4 * weak_consistency,
        "continuity": suspicious_density,
        "low_recovery": max(0.0, 1.0 - recovery),
        "low_fluctuation": max(0.0, 1.0 - cognitive_fluctuation),
        "low_emotional_anchor": max(0.0, 1.0 - emotional_grounding),
        "flow_stabilization": 0.0 if natural_breathing else min(1.0, 0.55 * guidance_dom + 0.45 * (1.0 - recovery)),
        "cross_modal_decoupling": semantic_effort_decoupling,
        "low_cross_modal_coherence": max(0.0, 1.0 - semantic_acoustic_coherence),
    }
    active = sum(1 for v in scores.values() if v >= 0.55)
    consensus = float(np.mean(list(scores.values()))) if scores else 0.0
    return consensus, active


def _human_variability_prior(
    *,
    cognitive_fluctuation: float,
    emotional_grounding: float,
    recovery: float,
    cognitive_spontaneity: float,
    natural_breathing: bool,
) -> float:
    """High means behavior is plausibly explained by natural human variability."""
    breathing = 1.0 if natural_breathing else 0.0
    return float(
        min(
            1.0,
            0.30 * cognitive_fluctuation
            + 0.22 * emotional_grounding
            + 0.20 * recovery
            + 0.18 * cognitive_spontaneity
            + 0.10 * breathing,
        )
    )

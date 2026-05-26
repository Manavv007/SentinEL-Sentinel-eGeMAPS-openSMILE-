"""Temporal persistence with calibrated suspicion tiers and weighted evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import config
from engine.suspicion_calibration import (
    SuspicionLevel,
    aggregate_answer_evidence,
    build_calibration_explanation,
    classify_suspicion_level,
    compute_calibrated_composite,
    ewma_input_for_level,
    level_weight,
    nonlinear_level_contribution,
    resolve_answer_status,
)
from engine.answer_synthesis import (
    compute_answer_behavioral_metrics,
    synthesize_final_decision,
)
from engine.temporal_persistence import SuspicionMomentumTracker


@dataclass
class WindowEvidence:
    window_id: int
    start_sec: float
    end_sec: float
    script_similarity: float
    natural_similarity: float
    contrastive_score: float
    naturality_score: float
    suspicion_level: str
    suspicious: bool
    confidence: str
    debug: dict[str, Any] = field(default_factory=dict)


class TemporalEvidenceTracker:
    """Tiered suspicion + weighted accumulation; weak evidence fades quickly."""

    def __init__(self) -> None:
        self._ewma: float | None = None
        self._peak_ewma: float = 0.0
        self._consecutive_strong = 0
        self._consecutive_moderate_plus = 0
        self._total_windows = 0
        self._windows: list[WindowEvidence] = []
        self._momentum = SuspicionMomentumTracker()

    def reset_answer(self) -> None:
        self._ewma = None
        self._peak_ewma = 0.0
        self._consecutive_strong = 0
        self._consecutive_moderate_plus = 0
        self._momentum = SuspicionMomentumTracker()

    def observe(
        self,
        *,
        window_id: int,
        start_sec: float,
        end_sec: float,
        script_similarity: float,
        natural_similarity: float,
        contrastive_score: float,
        naturality_score: float,
        margin: float,
        is_benign: bool = False,
        debug: dict[str, Any],
    ) -> WindowEvidence:
        self._total_windows += 1
        ewma_before = self._ewma

        cog_spont = float(debug.get("cognitive_spontaneity", 0.0))
        level = classify_suspicion_level(
            contrastive_score=contrastive_score,
            script_similarity=script_similarity,
            naturality_score=naturality_score,
            natural_similarity=natural_similarity,
            cognitive_spontaneity=cog_spont,
        )
        evidence_units = nonlinear_level_contribution(
            level, contrastive_score, script_similarity
        )
        ewma_input = ewma_input_for_level(level, contrastive_score)

        if level == SuspicionLevel.STRONG:
            self._consecutive_strong += 1
            self._consecutive_moderate_plus += 1
        elif level in (SuspicionLevel.MODERATE, SuspicionLevel.WEAK):
            self._consecutive_strong = 0
            if level == SuspicionLevel.MODERATE:
                self._consecutive_moderate_plus += 1
            else:
                self._consecutive_moderate_plus = max(0, self._consecutive_moderate_plus - 1)
        else:
            self._consecutive_strong = 0
            self._consecutive_moderate_plus = max(0, self._consecutive_moderate_plus - 1)

        momentum_metrics = self._momentum.observe(level)

        self._update_ewma_tiered(
            ewma_input,
            level=level,
            is_benign=is_benign,
            script_similarity=script_similarity,
        )
        self._peak_ewma = max(self._peak_ewma, self._ewma or 0.0)

        suspicious = level != SuspicionLevel.NONE
        confidence = self._window_confidence(level=level, script_similarity=script_similarity)

        ev = WindowEvidence(
            window_id=window_id,
            start_sec=start_sec,
            end_sec=end_sec,
            script_similarity=script_similarity,
            natural_similarity=natural_similarity,
            contrastive_score=contrastive_score,
            naturality_score=naturality_score,
            suspicion_level=level.value,
            suspicious=suspicious,
            confidence=confidence,
            debug={
                **debug,
                "suspicion_level": level.value,
                "suspicion_weight": level_weight(level),
                "evidence_units": round(evidence_units, 4),
                "ewma_input": round(ewma_input, 6),
                "ewma_before": round(ewma_before, 6) if ewma_before is not None else None,
                "ewma_after": round(self._ewma or 0.0, 6),
                "peak_ewma": round(self._peak_ewma, 6),
                "consecutive_strong": self._consecutive_strong,
                "consecutive_moderate_plus": self._consecutive_moderate_plus,
                "is_benign_window": is_benign,
                **momentum_metrics,
            },
        )
        self._windows.append(ev)
        return ev

    def answer_summary(
        self,
        *,
        horizon: Any | None = None,
        answer_duration_sec: float = 0.0,
    ) -> dict[str, Any]:
        duration = answer_duration_sec
        if not duration and self._windows:
            duration = self._windows[-1].end_sec - self._windows[0].start_sec

        margin = config.CONTRASTIVE_MARGIN
        ewma = self._ewma or 0.0

        window_dicts = [
            {
                "suspicion_level": w.suspicion_level,
                "contrastive_score": w.contrastive_score,
                "script_similarity": w.script_similarity,
                "natural_similarity": w.natural_similarity,
                "naturality_score": w.naturality_score,
                "cognitive_spontaneity": float(w.debug.get("cognitive_spontaneity", 0.0)),
                "guided_explanation_index": float(
                    w.debug.get("guided_explanation_index", 0.0)
                ),
            }
            for w in self._windows
        ]
        evidence = aggregate_answer_evidence(window_dicts)
        avg_script_sim = (
            float(sum(w["script_similarity"] for w in window_dicts)) / max(len(window_dicts), 1)
            if window_dicts
            else 0.0
        )

        momentum_summary = self._momentum.summary()
        streak_boost = self._momentum.streak_composite_boost()

        composite, composite_meta = compute_calibrated_composite(
            ewma=ewma,
            peak_ewma=self._peak_ewma,
            weighted_evidence=evidence["weighted_evidence"],
            margin=margin,
            horizon=horizon,
            suspicion_momentum=momentum_summary.get("suspicion_momentum", 0.0),
            streak_boost=streak_boost,
        )
        composite_meta.update(momentum_summary)
        composite_meta["streak_boost"] = round(streak_boost, 6)

        strong_count = int(evidence["strong_count"])
        mod_count = int(evidence["moderate_count"])
        weak_count = int(evidence["weak_count"])

        weak_only = (
            strong_count == 0
            and mod_count == 0
            and weak_count > 0
        ) or (
            evidence["strong_ratio"] < 0.1
            and evidence["weighted_evidence"] < config.AMBIGUOUS_MIN_WEIGHTED_EVIDENCE * 1.5
        )
        # If weak suspicion is dense and script similarity is elevated, do not treat as "weak-only benign".
        if (
            weak_only
            and evidence.get("weak_ratio", 0.0) >= config.WEAK_CLUSTER_MIN_RATIO
            and evidence.get("longest_weak_streak", 0.0) >= config.WEAK_CLUSTER_MIN_STREAK
            and avg_script_sim >= config.WEAK_CLUSTER_MIN_AVG_SCRIPT_SIM
        ):
            weak_only = False

        status, confidence = resolve_answer_status(
            composite=composite,
            weighted_evidence=evidence["weighted_evidence"],
            strong_ratio=evidence["strong_ratio"],
            moderate_plus_ratio=evidence["moderate_plus_ratio"],
            consecutive_strong=self._consecutive_strong,
            consecutive_moderate_plus=self._consecutive_moderate_plus,
            longest_strong_streak=int(momentum_summary.get("longest_strong_streak", 0)),
            lifetime_strong_ratio=float(momentum_summary.get("lifetime_strong_ratio", 0)),
            peak_ewma=self._peak_ewma,
            strong_window_count=strong_count,
            margin=margin,
            horizon=horizon,
            duration_sec=duration,
            weak_only_dominant=weak_only,
            suspicion_momentum=float(momentum_summary.get("suspicion_momentum", 0)),
            weak_ratio=float(evidence.get("weak_ratio", 0.0)),
            longest_weak_streak=int(evidence.get("longest_weak_streak", 0.0)),
            avg_script_similarity=float(avg_script_sim),
        )

        susp_ratio = (strong_count + mod_count + weak_count) / max(self._total_windows, 1)

        temporal_layer = {
            "ewma_score": round(ewma, 6),
            "composite_score": round(composite, 6),
            "peak_ewma": round(self._peak_ewma, 6),
            "weighted_evidence": evidence["weighted_evidence"],
            "status": status,
            "confidence": confidence,
        }

        momentum_for_synthesis = {
            **momentum_summary,
            "peak_ewma": self._peak_ewma,
        }
        behavioral = compute_answer_behavioral_metrics(
            window_dicts,
            horizon=horizon,
            momentum_summary=momentum_for_synthesis,
        )
        final_status, final_conf, synthesis_reasons = synthesize_final_decision(
            temporal_layer,
            behavioral,
            margin=margin,
        )

        summary = {
            **temporal_layer,
            "suspicious_window_ratio": round(susp_ratio, 4),
            "strong_suspicious_ratio": evidence["strong_ratio"],
            "moderate_plus_ratio": evidence["moderate_plus_ratio"],
            "strong_window_count": strong_count,
            "moderate_window_count": mod_count,
            "weak_window_count": weak_count,
            "consecutive_suspicious": self._consecutive_moderate_plus,
            "consecutive_strong": self._consecutive_strong,
            "persistent": (
                self._consecutive_strong >= config.PROBABLE_MIN_CONSECUTIVE_STRONG
                or int(momentum_summary.get("longest_strong_streak", 0))
                >= config.PROBABLE_LONGEST_STREAK_MIN
                or (
                    evidence["strong_ratio"] >= config.PROBABLE_MIN_STRONG_RATIO
                    and evidence["weighted_evidence"] >= config.PROBABLE_MIN_WEIGHTED_EVIDENCE
                )
            ),
            "suspicion_momentum": momentum_summary.get("suspicion_momentum", 0),
            "longest_strong_streak": int(momentum_summary.get("longest_strong_streak", 0)),
            "recent_suspicious_density": momentum_summary.get("recent_suspicious_density", 0),
            "confidence": final_conf,
            "status": final_status,
            "temporal_status": status,
            "behavioral_synthesis": behavioral,
            "composite_meta": {**evidence, **composite_meta},
        }
        summary["decision_explanation"] = build_calibration_explanation(
            summary, evidence, horizon
        )
        if synthesis_reasons and final_status != status:
            summary["decision_explanation"] = synthesis_reasons + summary[
                "decision_explanation"
            ]
        elif synthesis_reasons and final_status == "PROBABLE_SCRIPT_READING":
            summary["decision_explanation"] = synthesis_reasons + summary[
                "decision_explanation"
            ]
        summary["windows"] = [
            {
                "window_id": w.window_id,
                "start_sec": w.start_sec,
                "end_sec": w.end_sec,
                "script_similarity": w.script_similarity,
                "natural_similarity": w.natural_similarity,
                "contrastive_score": w.contrastive_score,
                "naturality_score": w.naturality_score,
                "suspicion_level": w.suspicion_level,
                "suspicious_flag": w.suspicious,
                "confidence_level": w.confidence,
                **w.debug,
            }
            for w in self._windows
        ]
        return summary

    def _update_ewma_tiered(
        self,
        value: float,
        *,
        level: SuspicionLevel,
        is_benign: bool,
        script_similarity: float,
    ) -> None:
        if self._ewma is None:
            self._ewma = value
            return

        if level == SuspicionLevel.NONE or value <= 1e-9:
            mult = self._momentum.decay_multiplier(
                level=SuspicionLevel.NONE,
                peak_ewma=self._peak_ewma,
                is_benign=is_benign,
            )
            self._ewma *= mult
            return

        if is_benign and value <= self._ewma:
            mult = self._momentum.decay_multiplier(
                level=level,
                peak_ewma=self._peak_ewma,
                is_benign=True,
            )
            self._ewma *= mult
            return

        if value > self._ewma:
            alpha = config.EWMA_ALPHA_ATTACK
            if level == SuspicionLevel.STRONG:
                alpha = max(alpha, config.EWMA_STRONG_SCRIPT_ATTACK_ALPHA)
            elif level == SuspicionLevel.WEAK:
                alpha = config.EWMA_ALPHA_ATTACK * 0.55
        else:
            alpha = config.EWMA_ALPHA_DECAY
            if level == SuspicionLevel.WEAK:
                alpha = min(0.95, alpha + 0.15)

        self._ewma = alpha * value + (1.0 - alpha) * self._ewma

    @staticmethod
    def _window_confidence(*, level: SuspicionLevel, script_similarity: float) -> str:
        if level == SuspicionLevel.STRONG:
            return "HIGH"
        if level == SuspicionLevel.MODERATE:
            return "MEDIUM" if script_similarity >= 0.62 else "LOW"
        if level == SuspicionLevel.WEAK:
            return "LOW"
        return "LOW"

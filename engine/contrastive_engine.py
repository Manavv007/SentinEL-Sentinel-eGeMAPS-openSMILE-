"""Dual-profile contrastive behavioral scoring engine."""

from __future__ import annotations

from typing import Any

import config
from engine.feature_extraction import build_window_features
from engine.naturality_scorer import NaturalityScorer
from engine.profile_health import profile_health
from engine.profile_memory import BehavioralProfile
from engine.profile_purity import (
    NaturalProfileStore,
    naturality_for_profile_learning,
    profile_learning_confidence,
)
from engine.recall_recovery import (
    AnswerHorizonAnalyzer,
    WindowScoreRow,
    capped_suppression,
    effective_natural_similarity,
    is_window_benign,
    natural_profile_confidence,
    should_update_natural_profile,
)
from engine.cognitive_spontaneity import (
    cognitive_spontaneity_suppression,
    compute_answer_cognitive_profile,
    score_cognitive_dimensions,
)
from engine.scoring_v3 import modulated_suspicion_score
from engine.spontaneity_evidence import spontaneity_suppression
from engine.temporal_evidence import TemporalEvidenceTracker
from engine.transition_detector import TransitionDetector


class ContrastiveEngine:
    """
    Compare each window to SCRIPT profile (calibration reading) vs NATURAL profile
    (built dynamically from high-confidence spontaneous windows).
    """

    def __init__(self) -> None:
        self.script_profile = BehavioralProfile.empty("script")
        self._natural_store = NaturalProfileStore()
        self.naturality = NaturalityScorer()
        self._interview_window_logs: list[dict[str, Any]] = []
        self._profile_update_logs: list[dict[str, Any]] = []
        self._strong_natural_updates = 0

    @property
    def natural_profile(self) -> BehavioralProfile:
        return self._natural_store.profile

    def build_script_profile_from_calibration(
        self,
        calibration_answers: list[dict[str, Any]],
        *,
        transcripts: list[dict[str, Any]] | None = None,
        timeline: list[dict[str, Any]] | None = None,
    ) -> BehavioralProfile:
        """Build SCRIPT_PROFILE from calibration video windows (intentional reading)."""
        transcripts = transcripts or []
        timeline = timeline or []
        rows: list[dict[str, float]] = []
        prev_pitch: float | None = None

        for answer in calibration_answers:
            tid = int(answer.get("answer_id", 0))
            transcript = transcripts[tid] if tid < len(transcripts) else {}
            a_start = float(answer.get("start_sec", 0))
            a_end = float(answer.get("end_sec", a_start + 30))
            for w in answer.get("windows", []):
                start = float(w.get("window_start", 0))
                end = start + 4.0
                t_slice = _slice_timeline(timeline, start, end)
                pitch = w.get("opensmile", {}).get("pitch_range_hz")
                pitch_delta = None
                if pitch is not None and prev_pitch is not None:
                    pitch_delta = float(pitch) - prev_pitch
                if pitch is not None:
                    prev_pitch = float(pitch)
                rows.append(
                    build_window_features(
                        audio_window=w,
                        transcript=transcript,
                        timeline_slice=t_slice,
                        pitch_delta=pitch_delta,
                        answer_start_sec=a_start,
                        answer_end_sec=a_end,
                    )
                )

        self.script_profile = BehavioralProfile.empty("script")
        self.script_profile.bulk_build(rows)
        return self.script_profile

    def reset_interview(self) -> None:
        """Start a new interview — NATURAL profile starts empty."""
        self._natural_store = NaturalProfileStore()
        self._interview_window_logs = []
        self._profile_update_logs = []
        self._strong_natural_updates = 0

    def _profile_confidence(self) -> float:
        purity = self._natural_store.purity_state().purity_score
        return natural_profile_confidence(
            self.natural_profile.sample_count,
            len(self.natural_profile.metric_stats()),
            self._strong_natural_updates,
            len(self._profile_update_logs),
            profile_purity=purity,
        )

    def _natural_similarity(
        self, features: dict[str, float], script_sim: float
    ) -> tuple[float, float]:
        """Return (raw_mature, effective_weighted) natural similarity."""
        purity_state = self._natural_store.compute_purity(self.script_profile)
        conf = self._profile_confidence()
        if self.natural_profile.sample_count > 0:
            raw = self.natural_profile.similarity_mature(features)
        else:
            raw = 0.0
        self._natural_store.record_raw_similarity(raw)
        effective = effective_natural_similarity(
            raw,
            conf,
            script_similarity=script_sim,
            sample_count=self.natural_profile.sample_count,
            profile_purity=purity_state.purity_score,
            saturation_pressure=purity_state.saturation_pressure,
        )
        return raw, effective

    def process_answer(
        self,
        answer: dict[str, Any],
        transcript: dict[str, Any],
        timeline: list[dict[str, Any]],
    ) -> dict[str, Any]:
        transition = TransitionDetector()
        transition.reset()
        self._natural_store.begin_answer()

        windows = answer.get("windows", [])
        answer_start = float(answer.get("start_sec", 0))
        answer_end = float(answer.get("end_sec", answer_start))
        answer_duration = max(answer_end - answer_start, 0.0)
        answer_tech_density = _answer_technical_density(transcript)
        answer_cognitive = (
            compute_answer_cognitive_profile(transcript)
            if config.ENABLE_COGNITIVE_SPONTANEITY
            else {}
        )

        prev_pitch: float | None = None
        pending: list[WindowScoreRow] = []

        for window_id, w in enumerate(windows):
            start = float(w.get("window_start", 0))
            end = start + 4.0
            t_slice = _slice_timeline(timeline, start, end)

            pitch = w.get("opensmile", {}).get("pitch_range_hz")
            pitch_delta = None
            if pitch is not None and prev_pitch is not None:
                pitch_delta = float(pitch) - prev_pitch
            if pitch is not None:
                prev_pitch = float(pitch)

            features = build_window_features(
                audio_window=w,
                transcript=transcript,
                timeline_slice=t_slice,
                pitch_delta=pitch_delta,
                answer_start_sec=answer_start,
                answer_end_sec=answer_end,
                answer_cognitive_profile=answer_cognitive or None,
            )
            tech_density = float(
                features.get("ling_technical_density", answer_tech_density)
            )

            script_sim = self.script_profile.similarity(features)
            natural_raw, natural_eff = self._natural_similarity(features, script_sim)

            naturality, nat_breakdown = self.naturality.score(
                features, script_similarity=script_sim
            )
            naturality_learning = naturality_for_profile_learning(
                features,
                naturality,
                nat_breakdown,
                script_similarity=script_sim,
            )
            learn_conf = profile_learning_confidence(
                features,
                naturality_learning,
                script_sim,
                nat_breakdown,
                technical_density=tech_density,
            )
            suppression, _sup_breakdown = spontaneity_suppression(features, nat_breakdown)
            cog_spont, cog_guided, cog_breakdown = (0.0, 0.0, {})
            if config.ENABLE_COGNITIVE_SPONTANEITY:
                cog_spont, cog_guided, cog_breakdown = score_cognitive_dimensions(
                    features, answer_cognitive
                )
                suppression = min(
                    1.0,
                    suppression
                    + cognitive_spontaneity_suppression(cog_spont, cog_guided, features),
                )
                naturality = min(
                    1.0,
                    naturality + config.COGNITIVE_NATURALITY_BLEND * cog_spont,
                )
            if script_sim >= config.SCRIPT_DOMINANCE_THRESHOLD:
                suppression = min(suppression, config.SCRIPT_DOMINANCE_MAX_SUPPRESSION)

            update_ok, update_reason = should_update_natural_profile(
                features,
                naturality,
                script_sim,
                nat_breakdown,
                naturality_learning=naturality_learning,
                technical_density=tech_density,
                learning_confidence=learn_conf,
            )
            profile_before = self.natural_profile.sample_count
            profile_updated = False
            if config.ENABLE_DYNAMIC_NATURAL_PROFILE and update_ok:
                added, store_reason = self._natural_store.try_add_sample(
                    features,
                    learning_confidence=learn_conf,
                    window_id=window_id,
                )
                if added:
                    profile_updated = True
                    update_reason = store_reason
                    self._strong_natural_updates += 1
                    self._profile_update_logs.append(
                        {
                            "window_id": window_id,
                            "answer_id": answer.get("answer_id"),
                            "naturality_score": naturality,
                            "naturality_learning": round(naturality_learning, 6),
                            "learning_confidence": round(learn_conf, 6),
                            "reason_added": update_reason,
                            "profile_size_before": profile_before,
                            "profile_size_after": self.natural_profile.sample_count,
                            "script_similarity": round(script_sim, 6),
                            "technical_density": round(tech_density, 6),
                        }
                    )
                    natural_raw, natural_eff = self._natural_similarity(features, script_sim)
                else:
                    update_reason = store_reason

            contrastive_base = modulated_suspicion_score(
                script_similarity=script_sim,
                natural_similarity=natural_eff,
                naturality_score=naturality,
                profile_confidence=self._profile_confidence(),
                suppression=suppression,
                technical_density=tech_density,
                cognitive_spontaneity=cog_spont,
                guided_explanation=cog_guided,
                fluency_trap=float(features.get("cog_fluency_trap", 0.0)),
            )

            pending.append(
                WindowScoreRow(
                    window_id=window_id,
                    start_sec=start,
                    end_sec=end,
                    features=features,
                    script_similarity=script_sim,
                    natural_similarity_raw=natural_raw,
                    natural_similarity_effective=natural_eff,
                    naturality_score=naturality,
                    naturality_learning=naturality_learning,
                    learning_confidence=learn_conf,
                    naturality_breakdown=nat_breakdown,
                    contrastive_base=contrastive_base,
                    suppression=suppression,
                    profile_confidence=self._profile_confidence(),
                    natural_update_reason=update_reason,
                    natural_profile_updated=profile_updated,
                    cognitive_spontaneity=cog_spont,
                    guided_explanation=cog_guided,
                    cognitive_breakdown=cog_breakdown,
                )
            )

        horizon = AnswerHorizonAnalyzer().analyze(
            pending, answer_duration_sec=answer_duration
        )

        tracker = TemporalEvidenceTracker()
        tracker.reset_answer()
        margin = config.CONTRASTIVE_MARGIN

        for row, boost in zip(pending, horizon.per_window_boost):
            from engine.recall_recovery import _local_low_drift

            suppression = capped_suppression(
                row.suppression,
                row.script_similarity,
                fake_natularity=horizon.fake_natularity_score,
            )
            contrastive = max(
                modulated_suspicion_score(
                    script_similarity=row.script_similarity,
                    natural_similarity=row.natural_similarity_effective,
                    naturality_score=row.naturality_score,
                    profile_confidence=row.profile_confidence,
                    suppression=suppression,
                    technical_density=float(
                        row.features.get("ling_technical_density", answer_tech_density)
                    ),
                    fake_natularity=horizon.fake_natularity_score,
                    cognitive_spontaneity=row.cognitive_spontaneity,
                    guided_explanation=row.guided_explanation,
                    fluency_trap=float(row.features.get("cog_fluency_trap", 0.0)),
                )
                + boost,
                horizon.answer_contrastive_floor * 0.5,
            )
            low_drift_local = _local_low_drift(row.features)
            benign = is_window_benign(
                contrastive=contrastive,
                margin=margin,
                naturality=row.naturality_score,
                script_similarity=row.script_similarity,
                suppression=row.suppression,
                low_drift_local=low_drift_local,
            )

            health = profile_health(
                self.script_profile, self.natural_profile, sample_features=row.features
            )
            trans_score = transition.observe(row.features)

            debug = {
                "pitch_metrics": {
                    "pitch_range_hz": row.features.get("acoustic_pitch_range_hz"),
                    "pitch_delta": row.features.get("acoustic_pitch_delta"),
                },
                "pause_metrics": {
                    "pause_entropy": row.features.get("ling_pause_entropy"),
                    "gap_variance": row.features.get("ling_gap_variance"),
                    "retrieval_pause_max": row.features.get("ling_retrieval_pause_max"),
                },
                "filler_metrics": {
                    "filler_rate_per_30s": row.features.get("ling_filler_rate_per_30s"),
                    "filler_clusters": row.features.get("ling_filler_clusters"),
                },
                "naturality_breakdown": row.naturality_breakdown,
                "suppression_total": suppression,
                "technical_density": round(
                    float(row.features.get("ling_technical_density", answer_tech_density)),
                    4,
                ),
                "natural_update_reason": row.natural_update_reason,
                "natural_profile_updated": row.natural_profile_updated,
                "naturality_for_scoring": round(row.naturality_score, 6),
                "naturality_for_learning": round(row.naturality_learning, 6),
                "learning_confidence": round(row.learning_confidence, 6),
                "natural_similarity_raw": round(row.natural_similarity_raw, 6),
                "profile_confidence": round(row.profile_confidence, 6),
                "profile_purity": round(
                    self._natural_store.purity_state().purity_score, 6
                ),
                "recall_recovery_boost": round(boost, 6),
                "long_horizon": {
                    "low_drift_score": round(horizon.low_drift_score, 4),
                    "cleanliness_score": round(horizon.cleanliness_score, 4),
                    "fake_natularity_score": round(horizon.fake_natularity_score, 4),
                    "script_dominance_active": horizon.script_dominance_active,
                    "sustained_script_ratio": round(horizon.sustained_script_ratio, 4),
                },
                "transition_score": trans_score,
                "natural_profile_samples": self.natural_profile.sample_count,
                "profile_health": health,
                "scoring_formula": (
                    "nonlinear_script * (1 - spontaneity_modulation) - suppression + boost"
                ),
                "cognitive_spontaneity": round(row.cognitive_spontaneity, 6),
                "guided_explanation_index": round(row.guided_explanation, 6),
                "cognitive_breakdown": row.cognitive_breakdown,
            }

            ev = tracker.observe(
                window_id=row.window_id,
                start_sec=row.start_sec,
                end_sec=row.end_sec,
                script_similarity=round(row.script_similarity, 6),
                natural_similarity=round(row.natural_similarity_effective, 6),
                contrastive_score=round(contrastive, 6),
                naturality_score=row.naturality_score,
                margin=margin,
                is_benign=benign,
                debug=debug,
            )
            self._interview_window_logs.append(
                {
                    "answer_id": answer.get("answer_id"),
                    **ev.debug,
                    "window_id": ev.window_id,
                    "start_time": row.start_sec,
                    "end_time": row.end_sec,
                    "script_similarity": ev.script_similarity,
                    "natural_similarity": ev.natural_similarity,
                    "contrastive_score": ev.contrastive_score,
                    "naturality_score": ev.naturality_score,
                    "suspicious_flag": ev.suspicious,
                    "confidence_level": ev.confidence,
                }
            )

        summary = tracker.answer_summary(
            horizon=horizon,
            answer_duration_sec=answer_duration,
        )
        summary["transition_peak"] = transition.answer_transition_peak()
        summary["natural_profile_ready"] = self.natural_profile.is_ready()
        summary["natural_profile_samples"] = self.natural_profile.sample_count
        summary["profile_update_count"] = len(
            [u for u in self._profile_update_logs if u.get("answer_id") == answer.get("answer_id")]
        )
        summary["profile_health"] = profile_health(self.script_profile, self.natural_profile)
        if answer_cognitive:
            summary["cognitive_profile"] = answer_cognitive
        purity_state = self._natural_store.compute_purity(self.script_profile)
        summary["natural_profile_purity"] = {
            "purity_score": purity_state.purity_score,
            "diversity_score": purity_state.diversity_score,
            "saturation_pressure": purity_state.saturation_pressure,
            "stored_samples": purity_state.stored_samples,
            "warnings": purity_state.warnings,
        }
        if purity_state.warnings:
            summary.setdefault("decision_explanation", [])
            for w in purity_state.warnings[:2]:
                if w not in summary["decision_explanation"]:
                    summary["decision_explanation"].append(w)
        if "decision_explanation" not in summary:
            summary["decision_explanation"] = []
        summary["recall_recovery"] = {
            "script_dominance_active": horizon.script_dominance_active,
            "low_drift_score": round(horizon.low_drift_score, 4),
            "cleanliness_score": round(horizon.cleanliness_score, 4),
            "fake_natularity_score": round(horizon.fake_natularity_score, 4),
            "answer_contrastive_floor": round(horizon.answer_contrastive_floor, 4),
        }

        if summary["transition_peak"] >= config.TRANSITION_ALERT_THRESHOLD:
            if summary["confidence"] == "MEDIUM":
                summary["confidence"] = "HIGH"
            elif summary["confidence"] == "LOW" and summary["ewma_score"] > margin * 0.8:
                summary["confidence"] = "MEDIUM"

        return summary

    def export_window_logs(self) -> list[dict[str, Any]]:
        return list(self._interview_window_logs)

    def export_profile_update_logs(self) -> list[dict[str, Any]]:
        return list(self._profile_update_logs)

    def export_profiles(self) -> dict[str, Any]:
        return {
            "script_profile": self.script_profile.to_dict(),
            "natural_profile": self.natural_profile.to_dict(),
            "profile_update_logs": self.export_profile_update_logs(),
            "natural_profile_purity": self._natural_store.export_stats(),
        }


def _answer_technical_density(transcript: dict[str, Any]) -> float:
    text = str(transcript.get("transcript", "")).lower()
    if not text.strip():
        return 0.0
    from engine.feature_extraction import TECHNICAL_RE

    tokens = text.split()
    if not tokens:
        return 0.0
    hits = len(TECHNICAL_RE.findall(text))
    return hits / max(len(tokens), 1)


def _slice_timeline(
    timeline: list[dict[str, Any]], start_sec: float, end_sec: float
) -> list[dict[str, Any]]:
    return [
        f
        for f in timeline
        if float(f.get("timestamp_sec", 0)) >= start_sec
        and float(f.get("timestamp_sec", 0)) < end_sec
        and f.get("face_detected")
    ]

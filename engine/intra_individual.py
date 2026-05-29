"""
Intra-individual behavioral modeling — session orchestrator.

Primary question: does this answer deviate from THIS PERSON'S natural baseline?
"""

from __future__ import annotations

from typing import Any

import config
from engine.cognitive_cost import cognitive_cost_profile
from engine.cross_answer_drift import CrossAnswerDriftTracker
from engine.cross_modal_correlation import cross_modal_correlation_analysis
from engine.intra_answer_turbulence import intra_answer_turbulence_metrics
from engine.personal_baseline import (
    PersonalBaselineModel,
    build_personal_baseline_from_calibration,
    extract_answer_window_rows,
)
from engine.recovery_arc import recovery_arc_analysis
from engine.session_probabilistic import SessionProbabilityState


class IntraIndividualSession:
    """Per-interview intra-individual analysis state."""

    def __init__(self, baseline: PersonalBaselineModel) -> None:
        self.baseline = baseline
        self.drift = CrossAnswerDriftTracker()
        self.probability = SessionProbabilityState()
        self._prev_answer_end: float | None = None
        self._early_bootstrap_count = 0

    @classmethod
    def from_profile(cls, profile: dict[str, Any]) -> IntraIndividualSession:
        data = profile.get("personal_baseline")
        if data:
            return cls(PersonalBaselineModel.from_dict(data))
        return cls(PersonalBaselineModel())

    def process_answer(
        self,
        answer: dict[str, Any],
        transcript: dict[str, Any],
        timeline: list[dict[str, Any]],
        *,
        contrastive_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute person-relative evidence for one answer."""
        window_rows = extract_answer_window_rows(answer, transcript, timeline)

        # Seed baseline before deviation when empty or still in early bootstrap window
        if (
            window_rows
            and (
                self.baseline.is_unseeded()
                or self._early_bootstrap_count < config.PERSONAL_BASELINE_EARLY_ANSWERS
            )
        ):
            self.baseline.ingest_window_rows(
                window_rows, alpha=config.PERSONAL_BASELINE_BOOTSTRAP_ALPHA
            )
            if self.baseline.is_unseeded():
                self.baseline.source = "interview_bootstrap"
            self._early_bootstrap_count += 1

        rel_agg = self.baseline.answer_aggregate_relative(window_rows)

        turb = intra_answer_turbulence_metrics(window_rows, self.baseline)
        cost = cognitive_cost_profile(
            answer=answer,
            transcript=transcript,
            window_rows=window_rows,
            baseline=self.baseline,
            prev_answer_end=self._prev_answer_end,
        )
        cross_modal = cross_modal_correlation_analysis(window_rows, timeline=timeline)

        contrastive_scores = None
        if contrastive_summary:
            contrastive_scores = [
                float(w.get("contrastive_score", 0.0))
                for w in (contrastive_summary.get("windows") or [])
            ]
        recovery = recovery_arc_analysis(window_rows, contrastive_scores=contrastive_scores)

        # Semantic-effort from contrastive behavioral synthesis if present
        beh = (contrastive_summary or {}).get("behavioral_synthesis") or {}
        sem_cov = float(beh.get("semantic_effort_covariance_score", 0.5))

        answer_metrics = {
            **rel_agg,
            **turb,
            **cost,
            **cross_modal,
            **recovery,
            "semantic_effort_covariance_score": sem_cov,
            "ling_wps_mean": float(
                sum(r.get("ling_wps", 0.0) for r in window_rows) / max(len(window_rows), 1)
            ),
        }

        drift_snapshot = self.drift.session_drift_profile()
        evidence = {**answer_metrics, **drift_snapshot}

        aid = int(answer.get("answer_id", 0))
        self.probability.update_from_answer_evidence(aid, evidence)

        provisional_status = str((contrastive_summary or {}).get("status", "CLEAR"))
        intra_status, intra_conf, intra_reasons = self.probability.status_for_answer(
            contrastive_status=provisional_status,
            answer_p_external=self.probability.p_external,
        )

        self.drift.record_answer(answer_metrics)
        self._prev_answer_end = float(answer.get("end_sec", 0))

        intra_block = {
            "person_relative": rel_agg,
            "intra_answer_turbulence": turb,
            "cognitive_cost": cost,
            "cross_modal": cross_modal,
            "recovery_arc": recovery,
            "p_external_guidance": round(self.probability.p_external, 4),
            "p_internal_generation": round(1.0 - self.probability.p_external, 4),
            "intra_status": intra_status,
            "intra_confidence": intra_conf,
            "intra_reasons": intra_reasons,
            "baseline_snapshot": self.baseline.summary(),
        }

        return intra_block

    def finalize_answer(
        self,
        answer_result: dict[str, Any],
        intra_block: dict[str, Any],
        *,
        answer: dict[str, Any],
        transcript: dict[str, Any],
        timeline: list[dict[str, Any]],
        contrastive_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Apply intra-individual verdict and update baseline on CLEAR."""
        if not config.ENABLE_INTRA_INDIVIDUAL:
            return answer_result

        intra_status = intra_block.get("intra_status", "AMBIGUOUS")
        intra_conf = intra_block.get("intra_confidence", "LOW")
        intra_reasons = list(intra_block.get("intra_reasons") or [])

        contrastive_status = str((contrastive_summary or {}).get("status", "CLEAR"))

        if config.INTRA_INDIVIDUAL_AUTHORITY:
            final_status = intra_status
            final_conf = intra_conf
        elif intra_status == "PROBABLE_SCRIPT_READING" and contrastive_status != "CLEAR":
            final_status = intra_status
            final_conf = intra_conf
        elif (
            contrastive_status == "PROBABLE_SCRIPT_READING"
            and intra_status == "AMBIGUOUS"
            and float(intra_block.get("p_external_guidance", 0.5))
            < config.SESSION_P_PROBABLE_MIN
            and config.INTRA_INDIVIDUAL_PRESERVE_UNCERTAINTY
        ):
            final_status = "AMBIGUOUS"
            final_conf = intra_conf
            intra_reasons.append(
                "intra-individual uncertainty preserved over contrastive promotion "
                f"(P(external)={float(intra_block.get('p_external_guidance', 0)):.2f})"
            )
        elif intra_status == "CLEAR" and contrastive_status == "PROBABLE_SCRIPT_READING":
            if config.INTRA_INDIVIDUAL_PRESERVE_UNCERTAINTY:
                final_status = "AMBIGUOUS"
                final_conf = "MEDIUM"
                intra_reasons.append(
                    "intra-individual uncertainty preserved over contrastive promotion"
                )
            else:
                final_status = contrastive_status
                final_conf = str((contrastive_summary or {}).get("confidence", "MEDIUM"))
        else:
            final_status = contrastive_status
            final_conf = str((contrastive_summary or {}).get("confidence", "MEDIUM"))

        answer_result["status"] = final_status
        answer_result["confidence"] = final_conf
        answer_result["intra_individual"] = intra_block

        if contrastive_summary is not None:
            contrastive_summary["intra_individual"] = intra_block
            contrastive_summary["status"] = final_status
            contrastive_summary["confidence"] = final_conf
            existing = contrastive_summary.get("decision_explanation") or []
            if not isinstance(existing, list):
                from engine.cognitive_sourcing import _normalize_decision_explanation

                existing = _normalize_decision_explanation(existing)
            for r in intra_reasons:
                if r not in existing:
                    existing.append(r)
            contrastive_summary["decision_explanation"] = existing
            answer_result["contrastive"] = contrastive_summary

        # Slow baseline update from CLEAR answers only
        if final_status == "CLEAR":
            rows = extract_answer_window_rows(answer, transcript, timeline)
            self.baseline.slow_update_from_answer(rows, status="CLEAR")

        return answer_result

    def finalize_session(
        self,
        results_answers: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Session-level drift + probability export."""
        drift_profile = self.drift.session_drift_profile()
        session_export = {
            "enabled": True,
            "personal_baseline": self.baseline.to_dict(),
            "session_probability": self.probability.export(),
            "cross_answer_drift": drift_profile,
            "session_behavior_profile": {
                **drift_profile,
                **self.probability.export(),
            },
        }

        if config.INTRA_INDIVIDUAL_SESSION_REFINEMENT and not self.baseline.is_unseeded():
            p_final = self.probability.p_external
            for ans in results_answers:
                intra = ans.get("intra_individual") or {}
                p_ans = float(intra.get("p_external_guidance", p_final))
                c = ans.get("contrastive") or {}
                beh = c.get("behavioral_synthesis") or {}
                strong_n = int(beh.get("strong_window_count", 0) or 0)
                if (
                    p_final >= config.SESSION_P_PROBABLE_MIN
                    and p_ans >= config.SESSION_P_PROBABLE_MIN
                    and strong_n >= 2
                    and ans.get("status") not in ("CLEAR",)
                ):
                    if ans.get("status") != "PROBABLE_SCRIPT_READING":
                        ans["status"] = "PROBABLE_SCRIPT_READING"
                        ans["confidence"] = "MEDIUM"
                        c = ans.get("contrastive") or {}
                        c["status"] = ans["status"]
                        reasons = c.get("decision_explanation") or []
                        if isinstance(reasons, list):
                            msg = (
                                f"session intra-individual: P(external)={p_final:.2f} "
                                f"persistent personal-baseline deviation"
                            )
                            if msg not in reasons:
                                reasons.append(msg)
                            c["decision_explanation"] = reasons

        return results_answers, session_export


def build_calibration_personal_baseline(
    calibration_answers: list[dict[str, Any]],
    *,
    transcripts: list[dict[str, Any]] | None = None,
    timeline: list[dict[str, Any]] | None = None,
) -> PersonalBaselineModel:
    return build_personal_baseline_from_calibration(
        calibration_answers,
        transcripts=transcripts,
        timeline=timeline,
    )

"""
Session-level probabilistic evidence accumulation.

P(external_guidance | evidence) updated gradually — not binary threshold voting.
"""

from __future__ import annotations

from typing import Any

import config


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _logit(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, p))
    import math

    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    import math

    return 1.0 / (1.0 + math.exp(-x))


class SessionProbabilityState:
    """Interview-wide P(external_guidance) with explainable log."""

    def __init__(self, p0: float | None = None) -> None:
        self.p_external = p0 if p0 is not None else config.SESSION_P_PRIOR
        self.history: list[dict[str, Any]] = []
        self.answer_probs: list[float] = []

    def update_from_answer_evidence(
        self,
        answer_id: int,
        evidence: dict[str, float],
        *,
        reasons: list[str] | None = None,
    ) -> float:
        """
        Likelihood-style update from person-relative evidence bundle.
        Returns updated session P(external).
        """
        # Evidence toward external guidance (all person-relative / session-relative)
        ext_signals = (
            0.18 * evidence.get("rel_mean_deviation", 0.0)
            + 0.16 * evidence.get("intra_turbulence_suppression", 0.0)
            + 0.14 * evidence.get("cognitive_cost_flatness", 0.0)
            + 0.12 * evidence.get("cross_answer_uniformity", 0.0)
            + 0.10 * evidence.get("cross_modal_sync_score", 0.0)
            + 0.12 * evidence.get("recovery_instant_flag", 0.0)
            + 0.10 * evidence.get("variance_of_variance_score", 0.0) * -1.0
            + 0.08 * evidence.get("rel_max_deviation", 0.0)
        )
        int_signals = (
            0.20 * evidence.get("intra_turbulence_burst_score", 0.0)
            + 0.18 * (1.0 - evidence.get("cognitive_cost_flatness", 0.5))
            + 0.15 * evidence.get("adaptation_present", 0.0)
            + 0.12 * (1.0 - evidence.get("cross_modal_sync_score", 0.0))
            + 0.15 * evidence.get("recovery_arc_quality", 0.5)
            + 0.10 * (1.0 - evidence.get("rel_mean_deviation", 0.0))
            + 0.10 * (1.0 - evidence.get("cross_answer_uniformity", 0.5))
        )

        likelihood_ratio = (0.55 + ext_signals) / max(0.15, 0.45 + int_signals)
        lr = max(0.65, min(1.45, likelihood_ratio))
        strength = config.SESSION_EVIDENCE_UPDATE_STRENGTH

        logit = _logit(self.p_external) + strength * (lr - 1.0)
        self.p_external = _clamp01(_sigmoid(logit))
        self.answer_probs.append(self.p_external)

        entry = {
            "answer_id": answer_id,
            "p_external_after": round(self.p_external, 4),
            "likelihood_ratio": round(lr, 4),
            "evidence": {k: round(float(v), 4) for k, v in evidence.items()},
            "reasons": reasons or [],
        }
        self.history.append(entry)
        return self.p_external

    def status_for_answer(
        self,
        *,
        contrastive_status: str,
        answer_p_external: float | None = None,
    ) -> tuple[str, str, list[str]]:
        """
        Map probability to CLEAR / AMBIGUOUS / PROBABLE with honest uncertainty band.
        May preserve contrastive when intra-individual is uncertain.
        """
        p = answer_p_external if answer_p_external is not None else self.p_external
        reasons: list[str] = []

        low = config.SESSION_P_AMBIGUOUS_LOW
        high = config.SESSION_P_AMBIGUOUS_HIGH
        probable = config.SESSION_P_PROBABLE_MIN

        if p >= probable:
            reasons.append(
                f"intra-individual: P(external|evidence)={p:.2f} — sustained deviation from personal baseline"
            )
            return "PROBABLE_SCRIPT_READING", "MEDIUM", reasons

        if p <= (1.0 - probable) and p < low:
            reasons.append(
                f"intra-individual: P(external|evidence)={p:.2f} — consistent with personal natural baseline"
            )
            return "CLEAR", "LOW", reasons

        if low <= p <= high:
            reasons.append(
                f"intra-individual: P(external|evidence)={p:.2f} — honest uncertainty (conflicting or weak evidence)"
            )
            return "AMBIGUOUS", "MEDIUM", reasons

        # Borderline — lean on soft ambiguous rather than hard collapse
        if p > 0.5:
            reasons.append(
                f"intra-individual: P(external|evidence)={p:.2f} — weak external-guidance lean"
            )
            return "AMBIGUOUS", "LOW", reasons

        reasons.append(
            f"intra-individual: P(external|evidence)={p:.2f} — weak internal-generation lean"
        )
        return "AMBIGUOUS", "LOW", reasons

    def export(self) -> dict[str, Any]:
        return {
            "p_external_final": round(self.p_external, 4),
            "p_prior": config.SESSION_P_PRIOR,
            "answer_probabilities": [round(x, 4) for x in self.answer_probs],
            "history": self.history,
        }

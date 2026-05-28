"""
Cross-answer behavioral drift — session-level consistency vs personal adaptation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config


def _var(vals: list[float]) -> float:
    return float(np.var(vals)) if len(vals) > 1 else 0.0


class CrossAnswerDriftTracker:
    """Accumulate per-answer behavioral summaries for session drift analysis."""

    def __init__(self) -> None:
        self._answers: list[dict[str, float]] = []

    def record_answer(self, answer_metrics: dict[str, float]) -> None:
        if answer_metrics:
            self._answers.append(dict(answer_metrics))

    def session_drift_profile(self) -> dict[str, Any]:
        if len(self._answers) < 2:
            return {
                "cross_answer_drift_score": 0.5,
                "cross_answer_uniformity": 0.5,
                "adaptation_present": 0.0,
                "answer_count": len(self._answers),
            }

        keys = (
            "rel_mean_deviation",
            "intra_turbulence_burst_score",
            "semantic_effort_covariance_score",
            "cognitive_cost_flatness",
            "ling_wps_mean",
        )
        variances: list[float] = []
        for key in keys:
            vals = [float(a.get(key, 0.0)) for a in self._answers if key in a]
            if len(vals) >= 2:
                variances.append(_var(vals))

        mean_var = float(np.mean(variances)) if variances else 0.0
        # Low variance across semantically different answers → artificially consistent
        uniformity = float(max(0.0, min(1.0, 1.0 - mean_var * config.CROSS_DRIFT_VAR_SCALE)))
        drift_score = float(min(1.0, mean_var * config.CROSS_DRIFT_VAR_SCALE))
        adaptation = float(min(1.0, drift_score * 1.2))

        return {
            "cross_answer_drift_score": round(drift_score, 4),
            "cross_answer_uniformity": round(uniformity, 4),
            "adaptation_present": round(adaptation, 4),
            "pacing_variance": round(_var([a.get("ling_wps_mean", 0.0) for a in self._answers]), 6),
            "turbulence_variance": round(
                _var([a.get("intra_turbulence_burst_score", 0.0) for a in self._answers]), 6
            ),
            "answer_count": len(self._answers),
        }

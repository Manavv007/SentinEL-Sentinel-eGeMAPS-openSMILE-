"""Fuse multi-channel suspicion scores into a single answer-level score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import config

BASE_CHANNELS = ("acoustic", "linguistic", "gaze", "lip")
GPU_WEIGHT = 0.30
NON_GPU_SCALE = 0.70

BASE_WEIGHTS: dict[str, float] = {
    "acoustic": config.WEIGHT_ACOUSTIC,
    "linguistic": config.WEIGHT_LINGUISTIC,
    "gaze": config.WEIGHT_GAZE,
    "lip": config.WEIGHT_LIP,
}


@dataclass
class FuseResult:
    """Legacy-compatible result object for CLI/report helpers."""

    index: int
    start_sec: float
    end_sec: float
    fused_score: float
    ewma_score: float
    status: str
    signals: dict[str, float] = field(default_factory=dict)
    raw_score: float = 0.0
    signal_breakdown: dict[str, dict[str, float | None]] = field(default_factory=dict)
    strongest_signal: str = ""


class FusedScorer:
    """
    Combine acoustic, linguistic, gaze, lip (and optional gpu) scores per answer.

    EWMA is applied across the sequence of answers (not within a single answer).
    Per-window EWMA for acoustic lives in AnalysisEngine only.
    """

    def __init__(self) -> None:
        self.smoothed: float | None = None

    def reset_ewma(self) -> None:
        """Reset cross-answer EWMA state (call before a new interview)."""
        self.smoothed = None

    def score_answer(
        self,
        *,
        answer_id: int,
        scores: dict[str, float | None],
        start_sec: float = 0.0,
        end_sec: float = 0.0,
    ) -> dict[str, Any]:
        """
        Fuse channel scores for one answer and update answer-level EWMA.

        Parameters
        ----------
        answer_id:
            Answer identifier.
        scores:
            Channel scores in [0, 1], or None if unavailable.
            Keys: acoustic, linguistic, gaze, lip, gpu (optional).
        """
        weights = self._effective_weights(scores)
        breakdown: dict[str, dict[str, float | None]] = {}
        contributions: list[tuple[str, float, float]] = []

        for channel in (*BASE_CHANNELS, "gpu"):
            weight = weights.get(channel, 0.0)
            raw = scores.get(channel)
            clamped = self._clamp(raw) if raw is not None else None
            breakdown[channel] = {"score": clamped, "weight": round(weight, 6)}
            if clamped is not None and weight > 0:
                contributions.append((channel, clamped, weight))

        if not contributions:
            raw_fused = 0.0
            strongest = ""
        else:
            raw_fused = sum(score * weight for _, score, weight in contributions)
            strongest = max(contributions, key=lambda x: x[1])[0]

        smoothed = self._update_ewma(raw_fused)
        status = (
            "PROBABLE_SCRIPT_READING"
            if smoothed >= config.ALERT_THRESHOLD
            else "CLEAR"
        )

        signal_scores = {
            ch: breakdown[ch]["score"]
            for ch in breakdown
            if breakdown[ch]["score"] is not None
        }

        return {
            "answer_id": answer_id,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "raw_score": round(raw_fused, 6),
            "smoothed_score": round(smoothed, 6),
            "status": status,
            "signal_breakdown": breakdown,
            "strongest_signal": strongest,
            # Legacy fields for report/CLI compatibility
            "index": answer_id,
            "fused_score": round(raw_fused, 6),
            "ewma_score": round(smoothed, 6),
            "signals": {
                ch: round(float(signal_scores[ch]), 4)  # type: ignore[arg-type]
                for ch in signal_scores
            },
        }

    def score_answer_legacy(
        self,
        *,
        index: int,
        start_sec: float,
        end_sec: float,
        features: dict[str, float],
        gpu_score: float | None = None,
    ) -> FuseResult:
        """Wrap score_answer() as a FuseResult dataclass."""
        payload = self.score_answer(
            answer_id=index,
            scores={
                "acoustic": features.get("acoustic"),
                "linguistic": features.get("linguistic"),
                "gaze": features.get("gaze"),
                "lip": features.get("lip"),
                "gpu": gpu_score,
            },
            start_sec=start_sec,
            end_sec=end_sec,
        )
        return FuseResult(
            index=index,
            start_sec=start_sec,
            end_sec=end_sec,
            fused_score=float(payload["fused_score"]),
            ewma_score=float(payload["ewma_score"]),
            status=str(payload["status"]),
            signals=dict(payload.get("signals", {})),
            raw_score=float(payload["raw_score"]),
            signal_breakdown=dict(payload["signal_breakdown"]),
            strongest_signal=str(payload["strongest_signal"]),
        )

    @staticmethod
    def _clamp(value: float | None) -> float | None:
        if value is None:
            return None
        return min(max(float(value), 0.0), 1.0)

    @staticmethod
    def _effective_weights(scores: dict[str, float | None]) -> dict[str, float]:
        """
        Build weights that sum to 1.0 over available signals only.

        GPU present → gpu=0.30, other base weights × 0.70.
        None scores → weight redistributed proportionally to remaining channels.
        """
        if scores.get("gpu") is not None:
            weights: dict[str, float] = {
                ch: w * NON_GPU_SCALE for ch, w in BASE_WEIGHTS.items()
            }
            weights["gpu"] = GPU_WEIGHT
        else:
            weights = dict(BASE_WEIGHTS)

        for ch in list(weights.keys()):
            if scores.get(ch) is None:
                weights[ch] = 0.0

        total = sum(weights.values())
        if total <= 0:
            return {}
        return {ch: w / total for ch, w in weights.items()}

    def _update_ewma(self, raw_fused: float) -> float:
        if self.smoothed is None:
            self.smoothed = raw_fused
            return raw_fused

        if raw_fused > self.smoothed:
            alpha = config.EWMA_ALPHA_ATTACK
        else:
            alpha = config.EWMA_ALPHA_DECAY

        self.smoothed = alpha * raw_fused + (1.0 - alpha) * self.smoothed
        return self.smoothed

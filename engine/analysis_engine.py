"""Acoustic reading-baseline calibration and Gaussian similarity scoring."""

from __future__ import annotations

from typing import Any

import numpy as np

import config

# Flat metric name -> weight within acoustic channel (sums to 1.0)
ACOUSTIC_METRIC_WEIGHTS: dict[str, float] = {
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": 0.25,
    "MeanVoicedSegmentLengthSec": 0.15,
    "MeanUnvoicedSegmentLength": 0.10,
    "jitter_local": 0.20,
    "shimmer_local": 0.10,
    "hnr": 0.10,
    "pitch_range_hz": 0.10,
}

OPENSMILE_KEYS = frozenset(
    {
        "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
        "MeanVoicedSegmentLengthSec",
        "MeanUnvoicedSegmentLength",
    }
)


class AnalysisEngine:
    """Build reading profiles from calibration windows and score interview answers."""

    def __init__(self) -> None:
        self._ewma: float | None = None

    @staticmethod
    def _metric_value(window: dict[str, Any], metric: str) -> float:
        if metric in OPENSMILE_KEYS:
            return float(window.get("opensmile", {}).get(metric, 0.0))
        return float(window.get("parselmouth", {}).get(metric, 0.0))

    @staticmethod
    def _robust_std(values: list[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return config.STD_FLOOR
        if arr.size == 1:
            return config.STD_FLOOR
        q75, q25 = np.percentile(arr, [75, 25])
        iqr_std = float((q75 - q25) / 1.349)
        return max(iqr_std, config.STD_FLOOR)

    def calibrate(self, windows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        """
        Build reading_profile from calibration feature windows.

        Each metric stores mean and IQR-based robust std (floored at STD_FLOOR).
        """
        buckets: dict[str, list[float]] = {m: [] for m in ACOUSTIC_METRIC_WEIGHTS}

        for window in windows:
            for metric in ACOUSTIC_METRIC_WEIGHTS:
                buckets[metric].append(self._metric_value(window, metric))

        profile: dict[str, dict[str, float]] = {}
        for metric, values in buckets.items():
            clean = [v for v in values if np.isfinite(v)]
            if not clean:
                profile[metric] = {"mean": 0.0, "std": config.STD_FLOOR}
                continue
            profile[metric] = {
                "mean": float(np.mean(clean)),
                "std": self._robust_std(clean),
            }
        return profile

    def score_window(
        self,
        window: dict[str, Any],
        reading_profile: dict[str, dict[str, float]],
    ) -> tuple[float, dict[str, float]]:
        """Score a single 4-second window; return (raw_acoustic_score, per-metric breakdown)."""
        breakdown: dict[str, float] = {}
        score = 0.0

        for metric, weight in ACOUSTIC_METRIC_WEIGHTS.items():
            x = self._metric_value(window, metric)
            stats = reading_profile.get(metric, {"mean": 0.0, "std": config.STD_FLOOR})
            mean = float(stats["mean"])
            std = max(float(stats["std"]), config.STD_FLOOR)
            z = abs(x - mean) / std
            similarity = float(np.exp(-0.5 * z * z))
            contribution = similarity * weight
            breakdown[metric] = round(contribution, 6)
            score += contribution

        score = round(score, 6)
        return score, breakdown

    @staticmethod
    def calibrate_channel_score(
        score: float,
        *,
        duration_sec: float,
        technical_density: float = 0.0,
    ) -> float:
        """Reduce acoustic false positives for fluent / technical / short answers."""
        s = float(score) * config.ACOUSTIC_SENSITIVITY_SCALE
        if duration_sec < config.ACOUSTIC_SHORT_ANSWER_SEC:
            s *= config.ACOUSTIC_SHORT_ANSWER_EXTRA_SCALE
        tech_dampen = config.TECHNICAL_FLUENCY_DAMPENING * min(1.0, technical_density * 4.0)
        s *= max(0.0, 1.0 - tech_dampen)
        return round(min(max(s, 0.0), 1.0), 6)

    def score_answer(
        self,
        windows: list[dict[str, Any]],
        reading_profile: dict[str, dict[str, float]],
    ) -> tuple[float, dict[str, Any]]:
        """
        Score all windows in one answer with asymmetric EWMA (reset per call).

        Returns (smoothed_score, breakdown) where breakdown includes per-window
        raw scores and final EWMA.
        """
        self.reset_ewma()

        window_scores: list[float] = []
        window_breakdowns: list[dict[str, float]] = []

        for window in windows:
            raw, bd = self.score_window(window, reading_profile)
            window_scores.append(raw)
            window_breakdowns.append(bd)
            self._update_ewma(raw)

        smoothed = self._ewma if self._ewma is not None else 0.0
        return round(smoothed, 6), {
            "window_scores": window_scores,
            "window_breakdowns": window_breakdowns,
            "ewma_score": round(smoothed, 6),
            "raw_ewma_before_calibration": round(smoothed, 6),
        }

    def _update_ewma(self, value: float) -> float:
        if self._ewma is None:
            self._ewma = value
            return value

        alpha = (
            config.EWMA_ALPHA_ATTACK
            if value > self._ewma
            else config.EWMA_ALPHA_DECAY
        )
        self._ewma = alpha * value + (1.0 - alpha) * self._ewma
        return self._ewma

    def reset_ewma(self) -> None:
        self._ewma = None

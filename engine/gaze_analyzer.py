"""Gaze feature analysis over answer windows from a gaze/lip timeline."""

from __future__ import annotations

from typing import Any

import numpy as np

import config

GAZE_WEIGHTS: dict[str, float] = {
    "horizontal_scan_regularity": 0.35,
    "gaze_x_variance_regularity": 0.20,
    "blink_rate_similarity": 0.30,
    "gaze_down_fraction": 0.15,
}

BLINK_EAR_THRESHOLD = 0.2
GAZE_DOWN_Y_THRESHOLD = 0.6
SUSPICIOUS_DOWN_FRACTION = 0.3


class GazeAnalyzer:
    """Score gaze behaviour against calibration reading baselines."""

    def calibrate(self, timeline: list[dict[str, Any]]) -> dict[str, float]:
        """Build blink-rate baseline from calibration timeline."""
        blink_rate = self._blink_rate_per_min(timeline)
        return {
            "blink_rate_mean": blink_rate,
            "blink_rate_std": max(config.STD_FLOOR, blink_rate * 0.25),
        }

    def analyze(
        self,
        window: list[dict[str, Any]],
        calibration: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Return (gaze_score, per-feature breakdown)."""
        if not window:
            return 0.0, {k: 0.0 for k in GAZE_WEIGHTS}

        duration_sec = max(
            float(window[-1]["timestamp_sec"]) - float(window[0]["timestamp_sec"]),
            1e-6,
        )

        reversals_per_sec = self._reversals_per_sec(window, duration_sec)
        scan_regularity = float(np.exp(-reversals_per_sec / 2.0))

        variance_regularity = self._gaze_x_variance_regularity(window)
        blink_rate = self._blink_rate_per_min(window)
        cal_mean = float(calibration.get("blink_rate_mean", blink_rate))
        cal_std = max(float(calibration.get("blink_rate_std", config.STD_FLOOR)), config.STD_FLOOR)
        z_blink = (blink_rate - cal_mean) / cal_std
        blink_similarity = float(np.exp(-0.5 * z_blink * z_blink))

        down_fraction = self._gaze_down_fraction(window)
        down_score = max(0.0, 1.0 - down_fraction / SUSPICIOUS_DOWN_FRACTION)

        components = {
            "horizontal_scan_regularity": round(scan_regularity, 6),
            "gaze_x_variance_regularity": round(variance_regularity, 6),
            "blink_rate_similarity": round(blink_similarity, 6),
            "gaze_down_fraction": round(down_score, 6),
        }

        score = sum(components[k] * GAZE_WEIGHTS[k] for k in GAZE_WEIGHTS)
        return round(score, 6), components

    @staticmethod
    def _reversals_per_sec(window: list[dict[str, Any]], duration_sec: float) -> float:
        xs = [float(f["gaze_x_ratio"]) for f in window if f.get("face_detected")]
        if len(xs) < 3:
            return 0.0

        reversals = 0
        prev_delta = 0.0
        for i in range(1, len(xs)):
            delta = xs[i] - xs[i - 1]
            if abs(delta) < 1e-4:
                continue
            if prev_delta != 0.0 and np.sign(delta) != np.sign(prev_delta):
                reversals += 1
            prev_delta = delta

        return reversals / duration_sec

    @staticmethod
    def _gaze_x_variance_regularity(window: list[dict[str, Any]]) -> float:
        """Low CV of rolling 1-sec gaze_x variance → reading-like."""
        xs = np.asarray(
            [float(f["gaze_x_ratio"]) for f in window if f.get("face_detected")],
            dtype=np.float64,
        )
        if xs.size < 4:
            return 0.5

        # ~10 samples per second at 10 fps timeline
        window_size = 10
        variances: list[float] = []
        for start in range(0, len(xs) - window_size + 1):
            chunk = xs[start : start + window_size]
            variances.append(float(np.var(chunk)))

        if len(variances) < 2:
            return 0.5

        var_arr = np.asarray(variances, dtype=np.float64)
        mean_v = float(var_arr.mean())
        if mean_v < 1e-8:
            return 1.0
        cv = float(var_arr.std(ddof=0) / mean_v)
        return float(np.exp(-cv))

    @staticmethod
    def _blink_rate_per_min(window: list[dict[str, Any]]) -> float:
        if not window:
            return 0.0

        duration_min = max(
            (float(window[-1]["timestamp_sec"]) - float(window[0]["timestamp_sec"])) / 60.0,
            1e-6,
        )

        blinks = 0
        in_blink = False
        for frame in window:
            if not frame.get("face_detected"):
                continue
            is_blink = float(frame.get("ear", 1.0)) < BLINK_EAR_THRESHOLD
            if is_blink and not in_blink:
                blinks += 1
                in_blink = True
            elif not is_blink:
                in_blink = False

        return blinks / duration_min

    @staticmethod
    def _gaze_down_fraction(window: list[dict[str, Any]]) -> float:
        detected = [f for f in window if f.get("face_detected")]
        if not detected:
            return 0.0
        down = sum(1 for f in detected if float(f["gaze_y_ratio"]) > GAZE_DOWN_Y_THRESHOLD)
        return down / len(detected)

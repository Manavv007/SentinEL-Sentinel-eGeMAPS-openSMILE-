"""Lip movement feature analysis over answer windows from a gaze/lip timeline."""

from __future__ import annotations

from typing import Any

import numpy as np

import config

LIP_WEIGHTS: dict[str, float] = {
    "aperture_variance_similarity": 0.50,
    "movement_rate_cv": 0.30,
    "symmetry_score": 0.20,
}


class LipAnalyzer:
    """Score lip behaviour against calibration reading baselines."""

    def calibrate(self, timeline: list[dict[str, Any]]) -> dict[str, float]:
        variance = self._lip_aperture_variance(timeline)
        cv = self._lip_movement_rate_cv(timeline)
        asymmetry = self._mean_lip_asymmetry(timeline)
        return {
            "aperture_variance_mean": variance,
            "aperture_variance_std": max(config.STD_FLOOR, variance * 0.25 if variance > 0 else config.STD_FLOOR),
            "movement_rate_cv_baseline": cv,
            "asymmetry_baseline": asymmetry,
        }

    def analyze(
        self,
        window: list[dict[str, Any]],
        calibration: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Return (lip_score, per-feature breakdown)."""
        if not window:
            return 0.0, {k: 0.0 for k in LIP_WEIGHTS}

        variance = self._lip_aperture_variance(window)
        cal_mean = float(calibration.get("aperture_variance_mean", variance))
        cal_std = max(
            float(calibration.get("aperture_variance_std", config.STD_FLOOR)),
            config.STD_FLOOR,
        )
        z = (variance - cal_mean) / cal_std
        aperture_sim = float(np.exp(-0.5 * z * z))

        movement_cv = self._lip_movement_rate_cv(window)
        movement_score = float(np.exp(-movement_cv * 2.0))

        asymmetry = self._mean_lip_asymmetry(window)
        cal_asym = float(calibration.get("asymmetry_baseline", asymmetry))
        # Lower asymmetry vs calibration → more reading-like
        asym_diff = max(0.0, asymmetry - cal_asym)
        symmetry_score = float(np.exp(-asym_diff * 4.0))

        components = {
            "aperture_variance_similarity": round(aperture_sim, 6),
            "movement_rate_cv": round(movement_score, 6),
            "symmetry_score": round(symmetry_score, 6),
        }

        score = sum(components[k] * LIP_WEIGHTS[k] for k in LIP_WEIGHTS)
        return round(score, 6), components

    @staticmethod
    def _lip_aperture_variance(window: list[dict[str, Any]]) -> float:
        apertures = [
            float(f["lip_aperture"])
            for f in window
            if f.get("face_detected") and float(f.get("lip_aperture", 0)) > 0
        ]
        if len(apertures) < 2:
            return 0.0
        return float(np.var(np.asarray(apertures, dtype=np.float64), ddof=1))

    @staticmethod
    def _lip_movement_rate_cv(window: list[dict[str, Any]]) -> float:
        """CV of inter-opening intervals from lip aperture peaks."""
        apertures = np.asarray(
            [
                float(f["lip_aperture"])
                for f in window
                if f.get("face_detected")
            ],
            dtype=np.float64,
        )
        if apertures.size < 4:
            return 0.0

        threshold = float(np.percentile(apertures, 75))
        peak_indices: list[int] = []
        for i in range(1, len(apertures) - 1):
            if apertures[i] >= threshold and apertures[i] >= apertures[i - 1] and apertures[i] >= apertures[i + 1]:
                peak_indices.append(i)

        if len(peak_indices) < 2:
            return 0.0

        times = np.asarray(
            [float(window[i]["timestamp_sec"]) for i in peak_indices],
            dtype=np.float64,
        )
        intervals = np.diff(times)
        if intervals.size == 0:
            return 0.0

        mean_i = float(intervals.mean())
        if mean_i < 1e-6:
            return 0.0
        return float(intervals.std(ddof=0) / mean_i)

    @staticmethod
    def _mean_lip_asymmetry(window: list[dict[str, Any]]) -> float:
        """Mean per-frame asymmetry: |left-right| / (left+right)."""
        ratios: list[float] = []
        for f in window:
            if not f.get("face_detected"):
                continue
            left = float(f.get("lip_left_disp", 0))
            right = float(f.get("lip_right_disp", 0))
            denom = left + right
            if denom < 1e-6:
                continue
            ratios.append(abs(left - right) / denom)
        if not ratios:
            return 0.0
        return float(np.mean(ratios))

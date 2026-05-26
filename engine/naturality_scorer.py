"""Estimate cognitive spontaneity (naturality) for interview windows."""

from __future__ import annotations

import math
from typing import Any

import config


def _stretch(score: float, *, center: float = 0.35, steepness: float = 4.0) -> float:
    """Sigmoid stretch to spread compressed raw scores into usable range."""
    x = (float(score) - center) * steepness
    return 1.0 / (1.0 + math.exp(-x))


class NaturalityScorer:
    """
    Higher score => more likely spontaneous / unscripted cognition.

    Does NOT assume early interview windows are natural.
    """

    WEIGHTS = {
        "self_correction": 0.18,
        "retrieval_pause": 0.16,
        "pause_entropy": 0.18,
        "filler_dynamics": 0.16,
        "rate_variance": 0.14,
        "acoustic_dynamics": 0.08,
        "low_script_overlap": 0.10,
    }

    def score(
        self,
        features: dict[str, float],
        *,
        script_similarity: float,
    ) -> tuple[float, dict[str, float]]:
        components = {
            "self_correction": self._self_correction(features),
            "retrieval_pause": self._retrieval_pause(features),
            "pause_entropy": self._pause_entropy(features),
            "filler_dynamics": self._filler_dynamics(features),
            "rate_variance": self._rate_variance(features),
            "acoustic_dynamics": self._acoustic_dynamics(features),
            "low_script_overlap": max(0.0, 1.0 - script_similarity),
        }

        raw = sum(components[k] * self.WEIGHTS[k] for k in self.WEIGHTS)

        # Baseline: structured fluent speech still has some spontaneity cues
        if not features.get("ling_has_words"):
            raw = max(raw, config.NATURALITY_NO_WORDS_FLOOR)

        stretched = _stretch(
            raw,
            center=config.NATURALITY_SIGMOID_CENTER,
            steepness=config.NATURALITY_SIGMOID_STEEPNESS,
        )
        total = min(max(stretched, 0.0), 1.0)
        return round(total, 6), {k: round(v, 6) for k, v in components.items()}

    @staticmethod
    def _self_correction(features: dict[str, float]) -> float:
        count = features.get("ling_self_corrections", 0.0)
        reps = features.get("ling_repetition_rate", 0.0)
        return min(1.0, (count * 0.5 + reps * 1.5) / 2.0)

    @staticmethod
    def _retrieval_pause(features: dict[str, float]) -> float:
        pause = features.get("ling_retrieval_pause_max", 0.0)
        if pause < 0.15:
            return 0.25
        if pause > 2.5:
            return 0.75
        return min(1.0, 0.25 + pause / 1.0)

    @staticmethod
    def _pause_entropy(features: dict[str, float]) -> float:
        ent = features.get("ling_pause_entropy", 0.0)
        norm = max(config.PAUSE_ENTROPY_NORM, 1e-6)
        return min(1.0, math.sqrt(ent / norm))

    @staticmethod
    def _filler_dynamics(features: dict[str, float]) -> float:
        rate = features.get("ling_filler_rate_per_30s", 0.0)
        clusters = features.get("ling_filler_clusters", 0.0)
        # Zero fillers != scripted; technical fluency often has few fillers
        if rate == 0.0:
            return 0.45
        if rate > 10.0 and clusters < 1.0:
            return 0.35
        rate_s = min(1.0, rate / 5.0)
        cluster_s = min(1.0, clusters / 2.0)
        return 0.55 * rate_s + 0.45 * cluster_s

    @staticmethod
    def _rate_variance(features: dict[str, float]) -> float:
        gap_var = features.get("ling_gap_variance", 0.0)
        wps = features.get("ling_wps", 0.0)
        gv = min(1.0, math.sqrt(gap_var / 0.008))
        if wps > 0:
            wps_score = 1.0 - min(1.0, abs(wps - 2.5) / 3.0)
        else:
            wps_score = 0.4
        return 0.55 * gv + 0.45 * wps_score

    @staticmethod
    def _acoustic_dynamics(features: dict[str, float]) -> float:
        pitch_delta = abs(features.get("acoustic_pitch_delta", 0.0))
        pitch_range = features.get("acoustic_pitch_range_hz", 0.0)
        delta_s = min(1.0, pitch_delta / 40.0)
        range_s = min(1.0, pitch_range / 120.0) if pitch_range else 0.3
        return 0.6 * delta_s + 0.4 * range_s

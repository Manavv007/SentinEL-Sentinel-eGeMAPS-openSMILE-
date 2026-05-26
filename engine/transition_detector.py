"""Detect sudden shifts from spontaneous to script-like behavior."""

from __future__ import annotations

from typing import Any

import numpy as np

import config


class TransitionDetector:
    """Track abrupt stabilization of pacing, fillers, and pause structure."""

    def __init__(self) -> None:
        self._prev_features: dict[str, float] | None = None
        self._transition_scores: list[float] = []

    def reset(self) -> None:
        self._prev_features = None
        self._transition_scores = []

    def observe(self, features: dict[str, float]) -> float:
        """
        Return transition score for this window (0-1, higher = sharper shift toward reading).
        """
        if self._prev_features is None:
            self._prev_features = dict(features)
            return 0.0

        prev = self._prev_features
        cur = features
        signals: list[float] = []

        # Filler disappearance
        prev_fill = prev.get("ling_filler_rate_per_30s", 0.0)
        cur_fill = cur.get("ling_filler_rate_per_30s", 0.0)
        if prev_fill > 1.0 and cur_fill < 0.5:
            signals.append(min(1.0, (prev_fill - cur_fill) / max(prev_fill, 1e-6)))

        # Pause entropy collapse (rhythmic reading)
        prev_ent = prev.get("ling_pause_entropy", 0.0)
        cur_ent = cur.get("ling_pause_entropy", 0.0)
        if prev_ent > 0.3 and cur_ent < prev_ent * 0.5:
            signals.append(min(1.0, (prev_ent - cur_ent) / max(prev_ent, 1e-6)))

        # Lip aperture stabilization
        prev_lip = prev.get("video_lip_aperture_std", 0.0)
        cur_lip = cur.get("video_lip_aperture_std", 0.0)
        if prev_lip > 0.02 and cur_lip < prev_lip * 0.5:
            signals.append(0.7)

        # Pitch dynamics flattening
        prev_pitch = prev.get("acoustic_pitch_range_hz", 0.0)
        cur_pitch = cur.get("acoustic_pitch_range_hz", 0.0)
        if prev_pitch > 50 and cur_pitch < prev_pitch * 0.6:
            signals.append(0.6)

        self._prev_features = dict(features)
        score = float(np.mean(signals)) if signals else 0.0
        self._transition_scores.append(score)
        return round(score, 6)

    def answer_transition_peak(self) -> float:
        if not self._transition_scores:
            return 0.0
        return round(float(max(self._transition_scores)), 6)

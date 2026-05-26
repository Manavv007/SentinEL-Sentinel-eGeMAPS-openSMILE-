"""Suspicion momentum, density tracking, and context-aware EWMA decay."""

from __future__ import annotations

from collections import deque
from typing import Any

import config
from engine.suspicion_calibration import SuspicionLevel


def level_rank(level: SuspicionLevel | str) -> int:
    if isinstance(level, str):
        level = SuspicionLevel(level) if level in {e.value for e in SuspicionLevel} else SuspicionLevel.NONE
    return {
        SuspicionLevel.NONE: 0,
        SuspicionLevel.WEAK: 1,
        SuspicionLevel.MODERATE: 2,
        SuspicionLevel.STRONG: 3,
    }[level]


class SuspicionMomentumTracker:
    """Rolling suspicious density + streak memory for context-aware decay."""

    def __init__(self, *, lookback: int | None = None) -> None:
        self._lookback = lookback or config.MOMENTUM_LOOKBACK_WINDOWS
        self._history: deque[SuspicionLevel] = deque(maxlen=self._lookback)
        self._longest_strong_streak = 0
        self._current_strong_streak = 0
        self._total_strong = 0
        self._total_moderate_plus = 0
        self._total_windows = 0

    def observe(self, level: SuspicionLevel) -> dict[str, float]:
        self._total_windows += 1
        self._history.append(level)

        if level == SuspicionLevel.STRONG:
            self._current_strong_streak += 1
            self._total_strong += 1
            self._total_moderate_plus += 1
            self._longest_strong_streak = max(
                self._longest_strong_streak, self._current_strong_streak
            )
        elif level in (SuspicionLevel.MODERATE, SuspicionLevel.WEAK):
            self._current_strong_streak = 0
            if level == SuspicionLevel.MODERATE:
                self._total_moderate_plus += 1
        else:
            self._current_strong_streak = 0

        density = self._recent_density()
        momentum = self._compute_momentum(density)

        return {
            "suspicion_momentum": round(momentum, 4),
            "recent_suspicious_density": round(density, 4),
            "longest_strong_streak": float(self._longest_strong_streak),
            "current_strong_streak": float(self._current_strong_streak),
            "lifetime_strong_ratio": round(
                self._total_strong / max(self._total_windows, 1), 4
            ),
            "lifetime_moderate_plus_ratio": round(
                self._total_moderate_plus / max(self._total_windows, 1), 4
            ),
        }

    def _recent_density(self) -> float:
        if not self._history:
            return 0.0
        weighted = 0.0
        for lvl in self._history:
            r = level_rank(lvl)
            if r >= 3:
                weighted += 1.0
            elif r == 2:
                weighted += 0.55
            elif r == 1:
                weighted += 0.2
        return weighted / len(self._history)

    def _compute_momentum(self, density: float) -> float:
        """0–1: higher = preserve EWMA longer after strong streaks."""
        streak_factor = min(1.0, self._longest_strong_streak / max(config.MOMENTUM_STREAK_NORM, 1))
        density_factor = min(1.0, density / max(config.MOMENTUM_DENSITY_NORM, 1e-6))
        lifetime_strong = self._total_strong / max(self._total_windows, 1)
        return float(
            min(
                1.0,
                0.45 * density_factor
                + 0.35 * streak_factor
                + 0.20 * min(1.0, lifetime_strong / 0.35),
            )
        )

    def decay_multiplier(
        self,
        *,
        level: SuspicionLevel,
        peak_ewma: float,
        is_benign: bool,
    ) -> float:
        """
        Context-aware decay: slow when momentum high, fast when only weak history.
        """
        density = self._recent_density()
        momentum = self._compute_momentum(density)
        margin = config.CONTRASTIVE_MARGIN

        if level == SuspicionLevel.NONE or (is_benign and level in (SuspicionLevel.NONE, SuspicionLevel.WEAK)):
            if momentum >= config.MOMENTUM_HIGH_THRESHOLD and peak_ewma >= margin * 0.65:
                return config.EWMA_DECAY_MOMENTUM_HIGH
            if momentum >= config.MOMENTUM_MED_THRESHOLD or self._longest_strong_streak >= 2:
                return config.EWMA_DECAY_MOMENTUM_MED
            if density < 0.15:
                return config.EWMA_DECAY_LOW_DENSITY
            return config.EWMA_WEAK_DECAY_MULTIPLIER

        if level == SuspicionLevel.WEAK and is_benign:
            if momentum >= config.MOMENTUM_MED_THRESHOLD:
                return config.EWMA_DECAY_MOMENTUM_MED
            return config.EWMA_WEAK_DECAY_MULTIPLIER

        return 1.0

    def streak_composite_boost(self) -> float:
        """Boost final composite when STRONG streaks were sustained."""
        if self._longest_strong_streak < 2:
            return 0.0
        boost = config.STREAK_BOOST_PER_STRONG * min(
            self._longest_strong_streak, config.STREAK_BOOST_MAX_WINDOWS
        )
        if self._total_strong >= 3:
            boost += config.STREAK_BOOST_STRONG_COUNT_BONUS
        return float(boost)

    def summary(self) -> dict[str, float]:
        return {
            "suspicion_momentum": round(self._compute_momentum(self._recent_density()), 4),
            "recent_suspicious_density": round(self._recent_density(), 4),
            "longest_strong_streak": float(self._longest_strong_streak),
            "current_strong_streak": float(self._current_strong_streak),
            "lifetime_strong_ratio": round(
                self._total_strong / max(self._total_windows, 1), 4
            ),
            "lifetime_moderate_plus_ratio": round(
                self._total_moderate_plus / max(self._total_windows, 1), 4
            ),
        }

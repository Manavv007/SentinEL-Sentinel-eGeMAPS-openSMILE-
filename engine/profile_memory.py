"""Dual behavioral profile memory: SCRIPT (calibration) and NATURAL (dynamic)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config


@dataclass
class MetricStats:
    """Robust rolling statistics for one behavioral metric."""

    mean: float = 0.0
    median: float = 0.0
    mad: float = 0.0
    std: float = 0.0
    n: int = 0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean": round(self.mean, 6),
            "median": round(self.median, 6),
            "mad": round(self.mad, 6),
            "std": round(self.std, 6),
            "n": self.n,
        }

    @classmethod
    def from_values(cls, values: list[float]) -> MetricStats:
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if arr.size == 0:
            return cls()
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        return cls(
            mean=float(arr.mean()),
            median=med,
            mad=max(mad, config.STD_FLOOR),
            std=max(float(arr.std(ddof=1) if arr.size > 1 else config.STD_FLOOR), config.STD_FLOOR),
            n=int(arr.size),
        )


class BehavioralProfile:
    """
    Stores per-metric statistics for script or natural speech behavior.

    Built from calibration (SCRIPT) or updated opportunistically during interview (NATURAL).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._samples: dict[str, list[float]] = {}

    @classmethod
    def empty(cls, name: str = "natural") -> BehavioralProfile:
        return cls(name)

    @property
    def sample_count(self) -> int:
        if not self._samples:
            return 0
        return max(len(v) for v in self._samples.values())

    def is_ready(self) -> bool:
        return self.sample_count >= config.NATURAL_PROFILE_MIN_SAMPLES

    def profile_strength(self) -> float:
        """0-1 heuristic: more samples and more metrics => stronger profile."""
        if not self._samples:
            return 0.0
        n_metrics = len(self._samples)
        n_samples = self.sample_count
        metric_factor = min(1.0, n_metrics / 12.0)
        sample_factor = min(1.0, n_samples / max(config.NATURAL_PROFILE_MIN_SAMPLES, 1))
        return float(metric_factor * sample_factor)

    def profile_distance(self, other: BehavioralProfile) -> float:
        """Mean absolute median distance across shared metrics (higher = more separated)."""
        a = self.metric_stats()
        b = other.metric_stats()
        shared = set(a.keys()) & set(b.keys())
        if not shared:
            return 1.0
        dists: list[float] = []
        for key in shared:
            scale = max(a[key].mad, a[key].std, config.STD_FLOOR)
            dists.append(abs(a[key].median - b[key].median) / scale)
        return float(np.mean(dists)) if dists else 1.0

    def update(self, features: dict[str, float]) -> int:
        """Add one observation; returns sample_count after update."""
        max_keep = (
            config.NATURAL_PROFILE_MAX_SAMPLES_PER_METRIC
            if self.name == "natural"
            else config.PROFILE_MAX_SAMPLES_PER_METRIC
        )
        for key, value in features.items():
            if not np.isfinite(value):
                continue
            bucket = self._samples.setdefault(key, [])
            bucket.append(float(value))
            if len(bucket) > max_keep:
                del bucket[: len(bucket) - max_keep]
        return self.sample_count

    def trim_to_max_per_metric(self, max_keep: int | None = None) -> None:
        """Drop oldest values when per-metric buckets exceed limit."""
        limit = max_keep or (
            config.NATURAL_PROFILE_MAX_SAMPLES_PER_METRIC
            if self.name == "natural"
            else config.PROFILE_MAX_SAMPLES_PER_METRIC
        )
        for bucket in self._samples.values():
            if len(bucket) > limit:
                del bucket[: len(bucket) - limit]

    def bulk_build(self, feature_rows: list[dict[str, float]]) -> None:
        """Build profile from many windows (calibration script profile)."""
        for row in feature_rows:
            self.update(row)

    def metric_stats(self) -> dict[str, MetricStats]:
        return {k: MetricStats.from_values(v) for k, v in self._samples.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "metrics": {k: s.to_dict() for k, s in self.metric_stats().items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BehavioralProfile:
        profile = cls(str(data.get("name", "script")))
        metrics = data.get("metrics", {})
        for key, stats in metrics.items():
            mean = float(stats.get("mean", 0))
            for _ in range(int(stats.get("n", 1))):
                profile._samples.setdefault(key, []).append(mean)
        return profile

    def similarity(self, features: dict[str, float]) -> float:
        """
        Gaussian similarity of features to this profile (0-1, higher = more alike).

        Uses MAD-based scale per metric; unknown metrics are skipped.
        """
        stats = self.metric_stats()
        if not stats:
            return 0.0

        scores: list[float] = []
        for key, value in features.items():
            if key not in stats or not np.isfinite(value):
                continue
            st = stats[key]
            scale = max(st.mad, st.std, config.STD_FLOOR)
            z = abs(float(value) - st.median) / scale
            scores.append(float(np.exp(-0.5 * z * z)))

        if not scores:
            return 0.0
        return float(np.mean(scores))

    def similarity_mature(self, features: dict[str, float]) -> float:
        """
        Similarity scaled by profile maturity (non-zero after first sample).

        Avoids hard zero until MIN_SAMPLES while still down-weighting early estimates.
        """
        if self.sample_count <= 0:
            return 0.0
        raw = self.similarity(features)
        maturity = min(1.0, self.sample_count / max(config.NATURAL_PROFILE_MIN_SAMPLES, 1))
        floor = config.NATURAL_SIMILARITY_MATURITY_FLOOR
        scale = floor + (1.0 - floor) * maturity
        return float(raw * scale)

"""High-purity NATURAL profile learning: bounded memory, diversity, anti-dilution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
from engine.profile_memory import BehavioralProfile
from engine.recall_recovery import strong_spontaneity_categories


@dataclass
class StoredNaturalSample:
    features: dict[str, float]
    learning_confidence: float
    answer_index: int
    window_id: int
    diversity_key: tuple[float, ...]
    age_answers: int = 0


@dataclass
class ProfilePurityState:
    purity_score: float = 1.0
    diversity_score: float = 1.0
    saturation_pressure: float = 0.0
    update_rate: float = 0.0
    stored_samples: int = 0
    warnings: list[str] = field(default_factory=list)


def naturality_for_profile_learning(
    features: dict[str, float],
    scoring_naturality: float,
    nat_breakdown: dict[str, float],
    *,
    script_similarity: float,
) -> float:
    """
    Stricter than scoring naturality — requires multiple strong spontaneity cues.
    """
    if script_similarity >= config.NATURAL_UPDATE_MAX_SCRIPT_SIM:
        return 0.0

    strong_cats = strong_spontaneity_categories(features, nat_breakdown)
    if strong_cats < config.NATURAL_UPDATE_MIN_STRONG_CATEGORIES:
        return min(scoring_naturality * 0.55, 0.45)

    component_floor = min(
        nat_breakdown.get("retrieval_pause", 0.0),
        nat_breakdown.get("pause_entropy", 0.0),
        nat_breakdown.get("rate_variance", 0.0),
    )
    if component_floor < 0.4:
        return min(scoring_naturality * 0.65, 0.5)

    pause_ent = features.get("ling_pause_entropy", 0.0)
    if pause_ent < config.NATURAL_UPDATE_MIN_PAUSE_ENTROPY:
        return min(scoring_naturality * 0.7, 0.52)

    gap_var = features.get("ling_gap_variance", 0.0)
    if gap_var < 0.002 and features.get("ling_filler_clusters", 0.0) < 1.0:
        return min(scoring_naturality * 0.75, 0.55)

    learning = scoring_naturality * (0.55 + 0.12 * strong_cats)
    learning = min(1.0, learning + component_floor * 0.08)
    return float(min(1.0, max(0.0, learning)))


def profile_learning_confidence(
    features: dict[str, float],
    naturality_learning: float,
    script_sim: float,
    nat_breakdown: dict[str, float],
    *,
    technical_density: float,
) -> float:
    """0–1 confidence that this window is safe to add to NATURAL profile."""
    if naturality_learning < config.NATURALITY_LEARNING_THRESHOLD:
        return 0.0

    strong = strong_spontaneity_categories(features, nat_breakdown)
    min_strong = config.NATURAL_UPDATE_MIN_STRONG_CATEGORIES
    if technical_density >= config.NATURAL_UPDATE_MAX_TECHNICAL_DENSITY:
        min_strong += config.NATURAL_UPDATE_TECHNICAL_EXTRA_STRONG
    if strong < min_strong:
        return 0.0

    spont_signals = sum(
        1
        for key in (
            "self_correction",
            "retrieval_pause",
            "pause_entropy",
            "filler_dynamics",
            "rate_variance",
        )
        if nat_breakdown.get(key, 0.0) >= 0.45
    )
    if spont_signals < config.NATURAL_UPDATE_MIN_SPONTANEITY_SIGNALS:
        return 0.0

    conf = 0.35
    conf += 0.25 * min(1.0, naturality_learning)
    conf += 0.15 * min(1.0, strong / max(config.NATURAL_UPDATE_MIN_STRONG_CATEGORIES, 1))
    conf += 0.1 * max(0.0, 1.0 - script_sim / max(config.NATURAL_UPDATE_MAX_SCRIPT_SIM, 1e-6))
    if technical_density >= config.NATURAL_UPDATE_MAX_TECHNICAL_DENSITY:
        conf *= 0.35
    elif technical_density >= config.NATURAL_UPDATE_MAX_TECHNICAL_DENSITY * 0.7:
        conf *= 0.65

    pause_ent = features.get("ling_pause_entropy", 0.0)
    if pause_ent >= config.NATURAL_UPDATE_MIN_PAUSE_ENTROPY:
        conf += 0.08
    if features.get("ling_self_corrections", 0.0) >= 1.0:
        conf += 0.05
    if features.get("ling_filler_clusters", 0.0) >= 1.0:
        conf += 0.05

    return float(min(1.0, max(0.0, conf)))


def _diversity_key(features: dict[str, float]) -> tuple[float, ...]:
    return (
        round(features.get("ling_wps", 0.0), 2),
        round(features.get("ling_pause_entropy", 0.0), 2),
        round(features.get("ling_gap_variance", 0.0), 4),
        round(features.get("ling_filler_rate_per_30s", 0.0), 1),
        round(features.get("acoustic_pitch_range_hz", 0.0) or 0.0, 0),
    )


def _feature_distance(a: dict[str, float], b: dict[str, float]) -> float:
    keys = (
        "ling_wps",
        "ling_pause_entropy",
        "ling_gap_variance",
        "ling_filler_rate_per_30s",
        "acoustic_pitch_range_hz",
    )
    dists: list[float] = []
    for key in keys:
        va = float(a.get(key, 0.0) or 0.0)
        vb = float(b.get(key, 0.0) or 0.0)
        scale = max(abs(va), abs(vb), 0.05)
        dists.append(abs(va - vb) / scale)
    return float(np.mean(dists)) if dists else 1.0


class NaturalProfileStore:
    """
    Bounded, high-confidence NATURAL profile with diversity and aging.
    Rebuilds BehavioralProfile from curated samples only.
    """

    def __init__(self) -> None:
        self.profile = BehavioralProfile.empty("natural")
        self._samples: list[StoredNaturalSample] = []
        self._raw_similarity_history: deque[float] = deque(
            maxlen=config.NATURAL_SIMILARITY_SATURATION_WINDOW
        )
        self._windows_seen = 0
        self._windows_offered_update = 0
        self._answer_index = 0
        self._warnings: list[str] = []
        self._last_purity = ProfilePurityState()

    def begin_answer(self) -> None:
        self._answer_index += 1
        for sample in self._samples:
            sample.age_answers += 1
        self._apply_aging()

    @property
    def sample_count(self) -> int:
        return self.profile.sample_count

    def purity_state(self) -> ProfilePurityState:
        return self._last_purity

    def recent_warnings(self) -> list[str]:
        return list(self._warnings[-6:])

    def record_raw_similarity(self, raw: float) -> None:
        self._raw_similarity_history.append(float(raw))

    def saturation_pressure(self) -> float:
        if len(self._raw_similarity_history) < 6:
            return 0.0
        mean_raw = float(np.mean(self._raw_similarity_history))
        if mean_raw < config.NATURAL_SIMILARITY_SATURATION_MEAN:
            return 0.0
        excess = mean_raw - config.NATURAL_SIMILARITY_SATURATION_MEAN
        span = max(config.NATURAL_SIMILARITY_CAP - config.NATURAL_SIMILARITY_SATURATION_MEAN, 1e-6)
        return float(min(1.0, excess / span))

    def compute_purity(
        self, script_profile: BehavioralProfile
    ) -> ProfilePurityState:
        warnings: list[str] = []
        n = len(self._samples)
        if n == 0:
            state = ProfilePurityState(purity_score=1.0, diversity_score=1.0)
            self._last_purity = state
            return state

        dist_script = script_profile.profile_distance(self.profile)
        diversity = self._diversity_score()
        update_rate = (
            self._windows_offered_update / max(self._windows_seen, 1)
            if self._windows_seen
            else 0.0
        )
        saturation = self.saturation_pressure()

        purity = 1.0
        purity *= min(1.0, dist_script / max(config.PROFILE_COLLAPSE_DISTANCE * 2.5, 1e-6))
        purity *= 0.55 + 0.45 * diversity
        purity *= 1.0 - saturation * config.NATURAL_SIMILARITY_SATURATION_PENALTY
        if update_rate > config.NATURAL_PROFILE_UPDATE_RATE_WARN:
            purity *= 0.75
            warnings.append(
                f"WARNING: NATURAL profile update rate high ({update_rate:.0%}) — possible dilution"
            )
        if saturation > 0.55:
            warnings.append(
                "WARNING: natural_similarity saturation detected — profile may be over-generalized"
            )
        if diversity < config.NATURAL_PROFILE_PURITY_COLLAPSE:
            warnings.append(
                "WARNING: NATURAL profile diversity collapsed — rejecting repetitive samples"
            )
        if dist_script < config.PROFILE_COLLAPSE_DISTANCE:
            warnings.append("WARNING: NATURAL profile dilution — too close to SCRIPT profile")

        purity = float(min(1.0, max(config.NATURAL_PROFILE_PURITY_COLLAPSE * 0.5, purity)))
        state = ProfilePurityState(
            purity_score=round(purity, 4),
            diversity_score=round(diversity, 4),
            saturation_pressure=round(saturation, 4),
            update_rate=round(update_rate, 4),
            stored_samples=n,
            warnings=warnings,
        )
        self._warnings.extend(warnings)
        self._last_purity = state
        return state

    def try_add_sample(
        self,
        features: dict[str, float],
        *,
        learning_confidence: float,
        window_id: int,
    ) -> tuple[bool, str]:
        self._windows_seen += 1
        if learning_confidence < config.NATURAL_PROFILE_MIN_LEARNING_CONFIDENCE:
            return False, "learning_confidence_too_low"

        self._windows_offered_update += 1
        dkey = _diversity_key(features)

        if self._is_redundant(features, dkey):
            return False, "insufficient_behavioral_diversity"

        sample = StoredNaturalSample(
            features=dict(features),
            learning_confidence=learning_confidence,
            answer_index=self._answer_index,
            window_id=window_id,
            diversity_key=dkey,
        )
        self._samples.append(sample)
        self._enforce_capacity()
        self._rebuild_profile()
        return True, "high_confidence_spontaneous_sample_added"

    def _is_redundant(self, features: dict[str, float], dkey: tuple[float, ...]) -> bool:
        if not self._samples:
            return False
        same_key = sum(1 for s in self._samples if s.diversity_key == dkey)
        if same_key >= 2:
            return True
        min_dist = min(_feature_distance(features, s.features) for s in self._samples)
        return min_dist < config.NATURAL_PROFILE_DIVERSITY_MIN_DISTANCE

    def _diversity_score(self) -> float:
        if len(self._samples) < 2:
            return 1.0
        keys = [s.diversity_key for s in self._samples]
        unique_ratio = len(set(keys)) / len(keys)
        dists: list[float] = []
        for i, a in enumerate(self._samples):
            for b in self._samples[i + 1 :]:
                dists.append(_feature_distance(a.features, b.features))
        mean_dist = float(np.mean(dists)) if dists else 0.0
        dist_factor = min(1.0, mean_dist / 0.35)
        return float(0.5 * unique_ratio + 0.5 * dist_factor)

    def _enforce_capacity(self) -> None:
        max_n = config.NATURAL_PROFILE_MAX_STORED_SAMPLES
        while len(self._samples) > max_n:
            weakest = min(
                range(len(self._samples)),
                key=lambda i: (
                    self._samples[i].learning_confidence,
                    -self._samples[i].age_answers,
                ),
            )
            del self._samples[weakest]

    def _apply_aging(self) -> None:
        max_age = config.NATURAL_PROFILE_SAMPLE_MAX_AGE_ANSWERS
        forget_age = config.NATURAL_PROFILE_FORGET_LOW_CONF_AGE
        kept: list[StoredNaturalSample] = []
        for s in self._samples:
            if s.age_answers > max_age:
                continue
            if (
                s.age_answers > forget_age
                and s.learning_confidence < config.NATURAL_PROFILE_MIN_LEARNING_CONFIDENCE + 0.08
            ):
                continue
            kept.append(s)
        if len(kept) != len(self._samples):
            self._samples = kept
            self._rebuild_profile()

    def _rebuild_profile(self) -> None:
        self.profile = BehavioralProfile.empty("natural")
        max_per = config.NATURAL_PROFILE_MAX_SAMPLES_PER_METRIC
        sorted_samples = sorted(
            self._samples, key=lambda s: s.learning_confidence, reverse=True
        )
        for sample in sorted_samples:
            self.profile.update(sample.features)
        self.profile.trim_to_max_per_metric(max_per)

    def export_stats(self) -> dict[str, Any]:
        p = self._last_purity
        return {
            "stored_samples": len(self._samples),
            "windows_seen": self._windows_seen,
            "purity_score": p.purity_score,
            "diversity_score": p.diversity_score,
            "saturation_pressure": p.saturation_pressure,
            "update_rate": p.update_rate,
            "warnings": self.recent_warnings(),
        }

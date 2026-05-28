"""
Personal speaking baseline — intra-individual reference for all behavioral reasoning.

The baseline models how THIS PERSON normally sounds (pacing, turbulence, pauses, etc.).
Suspicion is derived from deviation from this baseline, not population norms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
from engine.feature_extraction import build_window_features

# Metrics tracked in the personal baseline (handcrafted features only).
PERSONAL_METRIC_KEYS: tuple[str, ...] = (
    "ling_wps",
    "ling_filler_rate_per_30s",
    "ling_gap_variance",
    "ling_pause_entropy",
    "ling_retrieval_pause_max",
    "ling_self_corrections",
    "acoustic_jitter_local",
    "acoustic_shimmer_local",
    "acoustic_hnr",
    "acoustic_pitch_range_hz",
    "acoustic_F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "acoustic_MeanVoicedSegmentLengthSec",
    "acoustic_MeanUnvoicedSegmentLength",
    "cog_acoustic_turbulence",
    "cog_semantic_complexity",
    "cog_semantic_acoustic_coherence",
    "cog_pretoken_retrieval_adaptation",
    "video_gaze_x_std",
    "video_lip_aperture_std",
)


@dataclass
class MetricBaseline:
    median: float = 0.0
    mad: float = 0.1
    n: int = 0

    def update(self, value: float, *, alpha: float) -> None:
        if not np.isfinite(value):
            return
        if self.n == 0:
            self.median = float(value)
            self.mad = max(abs(value) * 0.15, 0.05)
            self.n = 1
            return
        self.median = (1.0 - alpha) * self.median + alpha * float(value)
        dev = abs(float(value) - self.median)
        self.mad = (1.0 - alpha) * self.mad + alpha * max(dev, 0.02)
        self.n += 1

    def relative_deviation(self, value: float) -> float:
        """0 = at personal norm; higher = more unusual for this person."""
        if not np.isfinite(value):
            return 0.0
        scale = max(self.mad, config.PERSONAL_BASELINE_MAD_FLOOR)
        z = abs(float(value) - self.median) / scale
        return float(min(1.0, z / config.PERSONAL_DEVIATION_Z_CAP))


class PersonalBaselineModel:
    """Slow-adaptive personal baseline initialized from calibration + early interview."""

    def __init__(self) -> None:
        self.metrics: dict[str, MetricBaseline] = {
            k: MetricBaseline() for k in PERSONAL_METRIC_KEYS
        }
        self.bootstrap_answers_remaining: int = config.PERSONAL_BASELINE_EARLY_ANSWERS
        self.update_count: int = 0
        self.source: str = "empty"

    def ingest_feature_row(self, row: dict[str, float], *, alpha: float | None = None) -> None:
        alpha = alpha if alpha is not None else config.PERSONAL_BASELINE_UPDATE_ALPHA
        for key in PERSONAL_METRIC_KEYS:
            if key in row:
                self.metrics[key].update(float(row[key]), alpha=alpha)

    def ingest_window_rows(
        self,
        rows: list[dict[str, float]],
        *,
        alpha: float | None = None,
    ) -> None:
        if not rows:
            return
        alpha = alpha if alpha is not None else config.PERSONAL_BASELINE_BOOTSTRAP_ALPHA
        for row in rows:
            self.ingest_feature_row(row, alpha=alpha)

    def person_relative_features(self, row: dict[str, float]) -> dict[str, float]:
        """Convert absolute features to person-relative deviation magnitudes."""
        out: dict[str, float] = {}
        devs: list[float] = []
        for key in PERSONAL_METRIC_KEYS:
            if key not in row:
                continue
            rel = self.metrics[key].relative_deviation(float(row[key]))
            out[f"rel_{key}"] = round(rel, 4)
            devs.append(rel)
        out["rel_mean_deviation"] = round(float(np.mean(devs)), 4) if devs else 0.0
        out["rel_max_deviation"] = round(float(max(devs)), 4) if devs else 0.0
        return out

    def answer_aggregate_relative(
        self,
        window_rows: list[dict[str, float]],
    ) -> dict[str, float]:
        if not window_rows:
            return {"rel_mean_deviation": 0.0, "rel_max_deviation": 0.0}
        rel_rows = [self.person_relative_features(r) for r in window_rows]
        keys = [k for k in rel_rows[0] if k.startswith("rel_")]
        agg: dict[str, float] = {}
        for k in keys:
            vals = [r[k] for r in rel_rows if k in r]
            if vals:
                agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{k}_max"] = round(float(np.max(vals)), 4)
        agg["rel_mean_deviation"] = round(
            float(np.mean([r.get("rel_mean_deviation", 0.0) for r in rel_rows])), 4
        )
        agg["rel_max_deviation"] = round(
            float(np.max([r.get("rel_max_deviation", 0.0) for r in rel_rows])), 4
        )
        return agg

    def slow_update_from_answer(
        self,
        window_rows: list[dict[str, float]],
        *,
        status: str,
    ) -> None:
        """Update baseline only from CLEAR answers (natural internal generation)."""
        if status != "CLEAR" or not window_rows:
            return
        alpha = config.PERSONAL_BASELINE_UPDATE_ALPHA
        if self.bootstrap_answers_remaining > 0:
            alpha = config.PERSONAL_BASELINE_BOOTSTRAP_ALPHA
            self.bootstrap_answers_remaining -= 1
        for row in window_rows:
            self.ingest_feature_row(row, alpha=alpha)
        self.update_count += 1

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "update_count": self.update_count,
            "bootstrap_remaining": self.bootstrap_answers_remaining,
            "metrics": {
                k: {"median": round(v.median, 4), "mad": round(v.mad, 4), "n": v.n}
                for k, v in self.metrics.items()
                if v.n > 0
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "update_count": self.update_count,
            "bootstrap_answers_remaining": self.bootstrap_answers_remaining,
            "metrics": {
                k: {"median": v.median, "mad": v.mad, "n": v.n}
                for k, v in self.metrics.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonalBaselineModel:
        model = cls()
        model.source = str(data.get("source", "loaded"))
        model.update_count = int(data.get("update_count", 0))
        model.bootstrap_answers_remaining = int(
            data.get(
                "bootstrap_answers_remaining",
                config.PERSONAL_BASELINE_EARLY_ANSWERS,
            )
        )
        for key, stats in (data.get("metrics") or {}).items():
            if key in model.metrics:
                model.metrics[key] = MetricBaseline(
                    median=float(stats.get("median", 0.0)),
                    mad=float(stats.get("mad", 0.1)),
                    n=int(stats.get("n", 0)),
                )
        return model


def extract_answer_window_rows(
    answer: dict[str, Any],
    transcript: dict[str, Any],
    timeline: list[dict[str, Any]],
) -> list[dict[str, float]]:
    """Build per-window feature rows for one answer."""
    from engine.feature_extraction import slice_timeline

    rows: list[dict[str, float]] = []
    a_start = float(answer.get("start_sec", 0))
    a_end = float(answer.get("end_sec", a_start + 30))
    prev_pitch: float | None = None

    for w in answer.get("windows", []):
        start = float(w.get("window_start", 0))
        end = start + 4.0
        pitch = w.get("opensmile", {}).get("F0semitoneFrom27.5Hz_sma3nz_stddevNorm")
        if pitch is None:
            pitch = w.get("parselmouth", {}).get("pitch_range_hz")
        pitch_delta = None
        if pitch is not None and prev_pitch is not None:
            pitch_delta = float(pitch) - prev_pitch
        if pitch is not None:
            prev_pitch = float(pitch)
        rows.append(
            build_window_features(
                audio_window=w,
                transcript=transcript,
                timeline_slice=slice_timeline(timeline, start, end),
                pitch_delta=pitch_delta,
                answer_start_sec=a_start,
                answer_end_sec=a_end,
            )
        )
    return rows


def build_personal_baseline_from_calibration(
    calibration_answers: list[dict[str, Any]],
    *,
    transcripts: list[dict[str, Any]] | None = None,
    timeline: list[dict[str, Any]] | None = None,
) -> PersonalBaselineModel:
    """
    Seed personal baseline from calibration clip (person's voice/delivery tendencies).
    Refined during interview from CLEAR answers.
    """
    transcripts = transcripts or []
    timeline = timeline or []
    model = PersonalBaselineModel()
    model.source = "calibration"
    all_rows: list[dict[str, float]] = []

    for answer in calibration_answers:
        tid = int(answer.get("answer_id", 0))
        transcript = transcripts[tid] if tid < len(transcripts) else {}
        all_rows.extend(extract_answer_window_rows(answer, transcript, timeline))

    model.ingest_window_rows(all_rows, alpha=config.PERSONAL_BASELINE_BOOTSTRAP_ALPHA)
    return model

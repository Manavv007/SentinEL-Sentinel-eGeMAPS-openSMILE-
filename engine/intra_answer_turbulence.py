"""
Intra-answer turbulence — micro-variability bursts relative to personal baseline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config
from engine.personal_baseline import PersonalBaselineModel


def _rolling_std(series: list[float], window: int = 3) -> list[float]:
    if len(series) < window:
        return [float(np.std(series))] if series else []
    out: list[float] = []
    for i in range(len(series) - window + 1):
        out.append(float(np.std(series[i : i + window])))
    return out


def intra_answer_turbulence_metrics(
    window_rows: list[dict[str, float]],
    baseline: PersonalBaselineModel,
) -> dict[str, float]:
    """
    Measure within-answer micro-variability vs this person's typical turbulence.
    Scripted delivery often suppresses rolling variance bursts.
    """
    if len(window_rows) < 2:
        return {
            "intra_turbulence_burst_score": 0.0,
            "intra_turbulence_suppression": 0.5,
            "variance_of_variance_score": 0.0,
        }

    def _series(key: str, fallback: str | None = None) -> list[float]:
        vals = []
        for r in window_rows:
            v = r.get(key)
            if v is None and fallback:
                v = r.get(fallback)
            vals.append(float(v or 0.0))
        return vals

    wps = _series("ling_wps")
    gaps = _series("ling_gap_variance")
    pitch = _series("acoustic_pitch_range_hz")
    turb = _series("cog_acoustic_turbulence", "acoustic_jitter_local")

    roll_stds = []
    for s in (wps, gaps, pitch, turb):
        roll_stds.extend(_rolling_std(s))

    burst = float(np.mean(roll_stds)) if roll_stds else 0.0
    vov = float(np.var(roll_stds)) if len(roll_stds) > 1 else 0.0

    # Person-relative: compare burst level to baseline turbulence metric
    base_turb = baseline.metrics.get("cog_acoustic_turbulence")
    if base_turb and base_turb.n > 0:
        expected = base_turb.median * 0.35 + base_turb.mad
        suppression = float(max(0.0, min(1.0, 1.0 - burst / max(expected, 1e-6))))
    else:
        suppression = float(max(0.0, min(1.0, 1.0 - burst * 4.0)))

    return {
        "intra_turbulence_burst_score": round(burst, 4),
        "intra_turbulence_suppression": round(suppression, 4),
        "variance_of_variance_score": round(min(1.0, vov * 8.0), 4),
    }

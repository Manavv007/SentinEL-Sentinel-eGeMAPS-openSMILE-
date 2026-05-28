"""
Recovery arc analysis — trajectory after disfluency / hesitation events.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config


def recovery_arc_analysis(
    window_rows: list[dict[str, float]],
    *,
    contrastive_scores: list[float] | None = None,
) -> dict[str, float]:
    """
    Model recovery shape after turbulence spikes.
    Instant perfect recovery or no recovery can indicate scripted delivery.
    """
    if len(window_rows) < 3:
        return {
            "recovery_arc_quality": 0.5,
            "recovery_instant_flag": 0.0,
            "recovery_collapse_flag": 0.0,
        }

    turb = [float(r.get("cog_acoustic_turbulence", 0.0)) for r in window_rows]
    repair = [float(r.get("cog_semantic_repair", 0.0)) for r in window_rows]
    contrastives = contrastive_scores or [0.0] * len(window_rows)

    events: list[int] = []
    for i, t in enumerate(turb):
        if t >= float(np.percentile(turb, 75)) and repair[i] > 0.1:
            events.append(i)

    if not events:
        return {
            "recovery_arc_quality": 0.55,
            "recovery_instant_flag": 0.0,
            "recovery_collapse_flag": 0.0,
        }

    instant = 0
    collapse = 0
    slopes: list[float] = []

    for idx in events:
        tail = turb[idx : min(len(turb), idx + 4)]
        if len(tail) < 2:
            continue
        drop = tail[0] - tail[-1]
        slope = drop / max(len(tail) - 1, 1)
        slopes.append(slope)
        if slope > 0.25 and len(tail) <= 2:
            instant += 1
        if tail[-1] >= tail[0] * 0.95:
            collapse += 1

    n = max(len(events), 1)
    instant_flag = instant / n
    collapse_flag = collapse / n
    quality = float(np.mean(slopes)) if slopes else 0.5
    quality_norm = max(0.0, min(1.0, 0.5 + quality))

    return {
        "recovery_arc_quality": round(quality_norm, 4),
        "recovery_instant_flag": round(instant_flag, 4),
        "recovery_collapse_flag": round(collapse_flag, 4),
        "recovery_slope_mean": round(float(np.mean(slopes)) if slopes else 0.0, 4),
    }

"""
Cross-modal correlation — coupling between gaze, prosody, pacing, lip motion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import config


def cross_modal_correlation_analysis(
    window_rows: list[dict[str, float]],
    *,
    timeline: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    """
    Sustained high cross-channel synchronization suggests externally guided delivery.
    Returns neutral scores when video timeline is unavailable.
    """
    if len(window_rows) < 3:
        return {
            "cross_modal_sync_score": 0.0,
            "cross_modal_available": 0.0,
        }

    series_map = {
        "gaze": [float(r.get("video_gaze_x_std", 0.0)) for r in window_rows],
        "lip": [float(r.get("video_lip_aperture_std", 0.0)) for r in window_rows],
        "pitch": [float(r.get("acoustic_pitch_range_hz", 0.0)) for r in window_rows],
        "pace": [float(r.get("ling_wps", 0.0)) for r in window_rows],
        "turb": [float(r.get("cog_acoustic_turbulence", 0.0)) for r in window_rows],
    }

    has_video = any(any(v > 1e-6 for v in s) for k, s in series_map.items() if k in ("gaze", "lip"))
    if not has_video and not timeline:
        return {
            "cross_modal_sync_score": 0.0,
            "cross_modal_available": 0.0,
        }

    corrs: list[float] = []
    keys = [k for k, s in series_map.items() if np.std(s) > 1e-6]
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            sa, sb = series_map[a], series_map[b]
            if len(sa) < 3:
                continue
            c = float(np.corrcoef(np.asarray(sa), np.asarray(sb))[0, 1])
            if np.isfinite(c):
                corrs.append(abs(c))

    sync = float(np.mean(corrs)) if corrs else 0.0
    # High absolute correlation across modalities → artificial coupling
    sync_score = float(max(0.0, min(1.0, (sync - 0.35) / 0.45)))

    return {
        "cross_modal_sync_score": round(sync_score, 4),
        "cross_modal_mean_abs_corr": round(sync, 4),
        "cross_modal_available": 1.0,
    }
